"""
Four-colourings of complete graphs without monochromatic C6 (R_4(C6)).
Goal: maximize n such that K_n can be 4-coloured with no monochromatic 6-cycle.
Uses constraint-based backtracking with efficient C6 detection.
"""

import time
import numpy as np
from itertools import combinations, permutations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _elapsed():
    return time.time() - _START_TIME

def _edge_index(i, j, n):
    """Index of edge (i,j) in lexicographic order (i<j)."""
    if i > j:
        i, j = j, i
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_adj(color_array, n):
    """Build 4 adjacency matrices from flat color array."""
    adj = [np.zeros((n, n), dtype=np.uint8) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = color_array[idx]
            adj[c][i, j] = 1
            adj[c][j, i] = 1
            idx += 1
    return adj

def _has_c6_fast(adj, n):
    """
    Check if any color class contains a C6.
    Uses the fact that a C6 exists iff there are 6 vertices with a Hamiltonian cycle.
    For each color, check all 6-subsets.
    """
    for c in range(4):
        A = adj[c]
        # For each 6-subset, check if induced subgraph has a Hamiltonian cycle
        for combo in combinations(range(n), 6):
            if _induced_has_c6(A, combo):
                return True
    return False

def _induced_has_c6(A, combo):
    """Check if the induced subgraph on combo has a Hamiltonian cycle (C6)."""
    # combo has 6 vertices. Check all 60 distinct cycles.
    first = combo[0]
    rest = list(combo[1:])
    for perm in permutations(rest):
        cycle = (first,) + perm
        ok = True
        for k in range(6):
            u = cycle[k]
            v = cycle[(k + 1) % 6]
            if A[u, v] == 0:
                ok = False
                break
        if ok:
            return True
    return False

def _count_c6s(adj, n):
    """Count total monochromatic C6s across all 4 colors."""
    total = 0
    for c in range(4):
        A = adj[c]
        for combo in combinations(range(n), 6):
            if _induced_has_c6(A, combo):
                total += 1
    return total

def _verify(color_array, n):
    """Verify that no color class contains a C6."""
    adj = _build_adj(color_array, n)
    return not _has_c6_fast(adj, n)

def _random_coloring(n, rng):
    m = n * (n - 1) // 2
    return rng.integers(0, 4, size=m)

def _backtrack(n, max_nodes=500000, rng=None):
    """
    Backtracking search for a valid 4-coloring of K_n with no monochromatic C6.
    Assigns edges in lexicographic order, with forward checking.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    m = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute all C6s as frozensets of edge indices
    c6_list = []
    seen_c6 = set()
    for combo in combinations(range(n), 6):
        first = combo[0]
        rest = list(combo[1:])
        for perm in permutations(rest):
            cycle = (first,) + perm
            edge_set = set()
            for k in range(6):
                u = cycle[k]
                v = cycle[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                edge_set.add((u, v))
            if len(edge_set) == 6:
                edge_idx_set = frozenset(edge_idx[e] for e in edge_set)
                if edge_idx_set not in seen_c6:
                    seen_c6.add(edge_idx_set)
                    c6_list.append(edge_idx_set)
    
    # For each edge, which C6s does it belong to?
    edge_to_c6s = [[] for _ in range(m)]
    for idx, c6 in enumerate(c6_list):
        for e in c6:
            edge_to_c6s[e].append(idx)
    
    # State for backtracking
    colors = [-1] * m
    # For each C6, track: number of edges assigned, and count per color
    c6_counts = [0] * len(c6_list)  # number of edges assigned in this C6
    c6_color_counts = [[0, 0, 0, 0] for _ in range(c6_list)]
    
    nodes_visited = 0
    deadline = _DEADLINE - 3
    
    def _is_valid_assignment(edge_idx_val, color):
        """Check if assigning color to edge_idx_val doesn't complete any C6."""
        for c6_idx in edge_to_c6s[edge_idx_val]:
            if c6_color_counts[c6_idx][color] >= 5:
                # This would make 6 edges of the same color in this C6
                return False
        return True
    
    def _assign(edge_idx_val, color):
        colors[edge_idx_val] = color
        for c6_idx in edge_to_c6s[edge_idx_val]:
            c6_counts[c6_idx] += 1
            c6_color_counts[c6_idx][color] += 1
    
    def _unassign(edge_idx_val, color):
        colors[edge_idx_val] = -1
        for c6_idx in edge_to_c6s[edge_idx_val]:
            c6_counts[c6_idx] -= 1
            c6_color_counts[c6_idx][color] -= 1
    
    def _search(depth):
        nonlocal nodes_visited
        if _elapsed() > deadline:
            return False
        if depth == m:
            return True
        
        nodes_visited += 1
        if nodes_visited % 10000 == 0 and _elapsed() > deadline:
            return False
        
        # Try colors in order
        for color in range(4):
            if _is_valid_assignment(depth, color):
                _assign(depth, color)
                if _search(depth + 1):
                    return True
                _unassign(depth, color)
        
        return False
    
    if _search(0):
        return colors
    return None

def _local_search(n, max_iters=20000, rng=None):
    """
    Local search: minimize number of monochromatic C6s.
    Start with random coloring, flip edges to reduce C6 count.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    m = n * (n - 1) // 2
    colors = rng.integers(0, 4, size=m)
    
    deadline = _DEADLINE - 5
    
    def _count_c6s_from_colors(col):
        adj = _build_adj(col, n)
        return _count_c6s(adj, n)
    
    current_count = _count_c6s_from_colors(colors)
    
    for _ in range(max_iters):
        if _elapsed() > deadline:
            break
        if current_count == 0:
            break
        
        # Pick a random edge to flip
        edge_pos = rng.integers(0, m)
        old_color = colors[edge_pos]
        new_color = (old_color + 1 + rng.integers(0, 3)) % 4
        
        if new_color == old_color:
            continue
        
        colors[edge_pos] = new_color
        new_count = _count_c6s_from_colors(colors)
        
        if new_count < current_count:
            current_count = new_count
        else:
            colors[edge_pos] = old_color
    
    if current_count == 0:
        return colors.tolist()
    return None

def _greedy_with_restarts(n, num_restarts=50, rng=None):
    """
    Try multiple random restarts with local search.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    deadline = _DEADLINE - 5
    
    for _ in range(num_restarts):
        if _elapsed() > deadline:
            break
        result = _local_search(n, max_iters=5000, rng=rng)
        if result is not None:
            return result
    return None

def construct():
    """
    Construct a 4-coloring of K_n with no monochromatic C6, maximizing n.
    """
    # Start with a known small valid coloring
    best_n = 1
    best_colors = [0]  # K_1 has no edges, trivially valid
    
    # For n=2,3,4,5: any coloring works (no C6 possible)
    # For n=6: need to check
    # Let's build up from small n
    
    # A simple valid coloring for small n: use a pattern
    # For n <= 5, no C6 is possible, so any coloring works
    for n in range(2, 6):
        m = n * (n - 1) // 2
        # Use a simple pattern: color = (i+j) % 4
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((i + j) % 4)
        best_n = n
        best_colors = colors
    
    # Try to extend to larger n
    rng = np.random.default_rng(12345)
    
    for n in range(6, 21):
        if _elapsed() > _DEADLINE - 10:
            break
        
        found = False
        
        # Try backtracking for small n
        if n <= 12:
            result = _backtrack(n, max_nodes=200000, rng=rng)
            if result is not None:
                # Verify
                if _verify(result, n):
                    best_n = n
                    best_colors = result
                    found = True
        
        # Try local search
        if not found:
            result = _greedy_with_restarts(n, num_restarts=20, rng=rng)
            if result is not None:
                if _verify(result, n):
                    best_n = n
                    best_colors = result
                    found = True
        
        # If we found a valid coloring, continue to try larger n
        if found:
            continue
        else:
            # If we couldn't find one for this n, break
            break
    
    # Final verification
    if best_n > 1:
        if not _verify(best_colors, best_n):
            # Fallback to a known valid small coloring
            best_n = 5
            best_colors = []
            for i in range(5):
                for j in range(i + 1, 5):
                    best_colors.append((i + j) % 4)
    
    return {'n': best_n, 'colors': best_colors}
