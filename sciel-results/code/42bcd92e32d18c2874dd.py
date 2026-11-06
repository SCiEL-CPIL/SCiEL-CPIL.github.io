import random
import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_adj(n, colors, c):
    """Build adjacency matrix for color c in K_n."""
    adj = np.zeros((n, n), dtype=bool)
    # colors is in lexicographic order: for i<j, index is sum_{k=0}^{i-1}(n-1-k) + (j-i-1)
    # Let's build it properly
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == c:
                adj[i, j] = True
                adj[j, i] = True
            idx += 1
    return adj

def _count_c6_fast(adj, n):
    """Count 6-cycles in adjacency matrix (undirected, exact count).
    A 6-cycle v0-v1-v2-v3-v4-v5-v0 is counted once if we fix v0 < v1 < ... < v5? No.
    We count all 6-cycles. Each undirected 6-cycle is counted 12 times (2 directions * 6 starting points).
    We return count // 12."""
    # Use matrix multiplication approach for small n
    # A 6-cycle exists if there are paths of length 3 that form a cycle
    # Actually, let's just enumerate for small n
    if n < 6:
        return 0
    
    # For efficiency, use the fact that a 6-cycle v0..v5 requires edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0)
    # Count by fixing v0 = 0, then v1 > 0, v2 > v1, ... but that's complex.
    # Simpler: for n <= 20, use DFS-based enumeration with pruning
    
    count = 0
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                adj_list[i].append(j)
                adj_list[j].append(i)
    
    # Enumerate 6-cycles: fix vertex 0, find cycles through it
    # A cycle through vertex 0: 0-a-b-c-d-e-0
    for a in adj_list[0]:
        for b in adj_list[a]:
            if b == 0:
                continue
            for c in adj_list[b]:
                if c in (0, a):
                    continue
                for d in adj_list[c]:
                    if d in (0, a, b):
                        continue
                    for e in adj_list[d]:
                        if e in (0, a, b, c):
                            continue
                        # Check if e connects back to 0
                        if adj[e, 0]:
                            # We have a cycle 0-a-b-c-d-e-0
                            # But we need to count each cycle once
                            # Normalize: the cycle is determined by the set of 6 vertices and the cyclic order
                            # To avoid overcounting, require 0 = min vertex in cycle
                            verts = [0, a, b, c, d, e]
                            if min(verts) == 0:
                                # Also need to avoid counting the same cycle from different a
                                # The cycle 0-a-b-c-d-e-0 is the same as 0-e-d-c-b-a-0
                                # To count each undirected cycle once, require a < e (the two neighbors of 0 in the cycle)
                                if a < e:
                                    count += 1
    return count

def _has_c6(adj, n):
    """Return True if there's any 6-cycle."""
    return _count_c6_fast(adj, n) > 0

def _count_violations(n, colors):
    """Count total 6-cycles across all 4 color classes."""
    total = 0
    for c in range(4):
        adj = _build_adj(n, colors, c)
        total += _count_c6_fast(adj, n)
    return total

def _greedy_init(n):
    """Initialize coloring greedily to avoid C6 as much as possible."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            best_c = 0
            best_v = None
            for c in range(4):
                # Tentatively assign color c to edge (i,j)
                colors.append(c)
                v = _count_violations(n, colors)
                colors.pop()
                if best_v is None or v < best_v:
                    best_v = v
                    best_c = c
            colors.append(best_c)
    return colors

def _local_search(n, colors, time_limit):
    """Run local search to minimize C6 count."""
    colors = list(colors)
    best_v = _count_violations(n, colors)
    if best_v == 0:
        return colors
    
    start = time.time()
    deadline = start + time_limit
    
    # Precompute edge indices
    edge_count = n * (n - 1) // 2
    
    while time.time() < deadline:
        # Pick a random edge
        idx = random.randint(0, edge_count - 1)
        old_c = colors[idx]
        new_c = random.randint(0, 3)
        if old_c == new_c:
            continue
        
        colors[idx] = new_c
        new_v = _count_violations(n, colors)
        
        if new_v < best_v:
            best_v = new_v
            if best_v == 0:
                return colors
        else:
            # Revert if it got worse (or equal, to avoid plateaus)
            colors[idx] = old_c
    
    return colors

def _try_n(n, time_limit):
    """Try to find a valid coloring for K_n."""
    # Greedy init
    colors = _greedy_init(n)
    v = _count_violations(n, colors)
    if v == 0:
        return colors
    
    # Local search
    colors = _local_search(n, colors, time_limit)
    v = _count_violations(n, colors)
    if v == 0:
        return colors
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Try from largest n down
    for n in range(19, 8, -1):
        if time.time() > deadline - 5:
            break
        
        # Estimate time per n based on edge count
        edge_count = n * (n - 1) // 4  # Rough estimate
        time_per_n = min(15.0, max(2.0, 60.0 / max(1, n)))
        
        if time.time() + time_per_n > deadline - 2:
            break
        
        sol = _try_n(n, time_per_n)
        if sol is not None:
            return {'n': n, 'colors': sol}
    
    # Fallback: known n=9 coloring
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
