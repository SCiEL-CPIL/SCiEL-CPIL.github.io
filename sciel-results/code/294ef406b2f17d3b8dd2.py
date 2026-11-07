"""Improved program for packing edge-disjoint Hamiltonian cycles on the 21-vertex host."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_hc(adj, n, rng, deadline, start=0):
    """Fast Hamiltonian cycle finder with Warnsdorff + forced moves + connectivity check."""
    # Connectivity check
    if n <= 1:
        return None
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if len(seen) != n:
        return None
    
    # Iterative DFS with Warnsdorff heuristic
    path = [start]
    visited = {start}
    # Pre-compute neighbor lists for speed
    nbrs = {v: sorted(adj[v]) for v in range(n)}
    
    # Use a stack of (node, neighbor_iterator)
    # For each node, we track which neighbors we've tried
    # Warnsdorff: sort by degree in remaining graph
    
    def warnsdorff_sort(cur, path, visited):
        """Sort neighbors of cur by Warnsdorff criterion (fewest onward options)."""
        unvisited_nbrs = [nb for nb in nbrs[cur] if nb not in visited]
        if not unvisited_nbrs:
            return []
        # Count onward degree for each candidate
        scored = []
        for nb in unvisited_nbrs:
            onward = len([x for x in nbrs[nb] if x not in visited and x != start or x == start and len(path) == n-1])
            scored.append((onward, nb))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [nb for _, nb in scored]
    
    # Iterative backtracking
    # Stack: list of (node, list_of_candidates, index)
    stack = [(start, warnsdorff_sort(start, path, visited), 0)]
    
    while stack:
        if time.time() > deadline:
            return None
        
        node, candidates, idx = stack[-1]
        
        if idx >= len(candidates):
            # Backtrack
            stack.pop()
            if not stack:
                return None
            if len(path) == n:
                return None
            path.pop()
            visited.discard(node)
            # After backtracking, re-sort parent's candidates
            if stack:
                parent_node = stack[-1][0]
                stack[-1] = (parent_node, warnsdorff_sort(parent_node, path, visited), stack[-1][2])
            continue
        
        next_node = candidates[idx]
        # Update index
        stack[-1] = (node, candidates, idx + 1)
        
        if next_node in visited:
            continue
        
        # Check if this is the last node - must connect back to start
        if len(path) == n - 1:
            if start not in adj[next_node]:
                continue
            # Found it!
            return path + [next_node]
        
        visited.add(next_node)
        path.append(next_node)
        
        # Check for forced moves: if any unvisited node has only 1 unvisited neighbor, must go there
        new_candidates = warnsdorff_sort(next_node, path, visited)
        stack.append((next_node, new_candidates, 0))
        
        # Forced move check: look for unvisited nodes with degree 1 in remaining
        # This is expensive, so only do it sometimes
        if len(path) < n - 2 and rng.random() < 0.3:
            forced = None
            for v in range(n):
                if v not in visited and v != next_node:
                    unv_nbrs = [x for x in nbrs[v] if x not in visited]
                    if len(unv_nbrs) == 1:
                        forced = unv_nbrs[0]
                        break
            if forced is not None and forced != next_node:
                # We should go to forced from next_node if connected
                if forced in adj[next_node]:
                    # Force the move
                    visited.add(forced)
                    path.append(forced)
                    new_candidates = warnsdorff_sort(forced, path, visited)
                    stack.append((forced, new_candidates, 0))
    
    return None

def _find_hc_multi(adj, n, rng, deadline):
    """Try multiple starting points."""
    starts = list(range(n))
    rng.shuffle(starts)
    for start in starts:
        if time.time() > deadline:
            return None
        result = _find_hc(adj, n, rng, deadline, start=start)
        if result is not None:
            return result
    return None

def _build_adj(edge_list, n):
    adj = defaultdict(set)
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _cycle_edges(cycle):
    """Get set of edge keys for a cycle."""
    edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        edges.add(_edge_key(u, v))
    return edges

def _validate(cycles, n, edge_set):
    """Validate all cycles are valid and edge-disjoint."""
    if not cycles:
        return False
    used_edges = set()
    all_vertices = None
    for cycle in cycles:
        if len(cycle) != n:
            return False
        if len(set(cycle)) != n:
            return False
        verts = set(cycle)
        if all_vertices is None:
            all_vertices = verts
        elif verts != all_vertices:
            return False
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in edge_set:
                return False
            if ek in used_edges:
                return False
            used_edges.add(ek)
    return True

def _search_three(edges, n, rng, deadline):
    """Search for 3 edge-disjoint Hamiltonian cycles."""
    original_edges = set()
    for u, v in edges:
        original_edges.add(_edge_key(u, v))
    
    cycles = []
    remaining = set(original_edges)
    
    for cidx in range(3):
        if time.time() > deadline:
            break
        
        edge_list = list(remaining)
        adj = _build_adj(edge_list, n)
        
        # Check min degree
        min_deg = min(len(adj[v]) for v in range(n)) if adj else 0
        if min_deg < 2:
            break
        
        found = None
        # Multiple attempts with different randomizations
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            # Shuffle neighbor orderings
            shuffled_adj = {}
            for v in range(n):
                lst = list(adj[v])
                rng.shuffle(lst)
                shuffled_adj[v] = set(lst)
            
            cycle = _find_hc(shuffled_adj, n, rng, deadline)
            if cycle is not None:
                found = cycle
                break
        
        if found is None:
            # Try multi-start
            adj2 = _build_adj(list(remaining), n)
            found = _find_hc_multi(adj2, n, rng, deadline)
        
        if found is None:
            break
        
        cycles.append(found)
        
        # Remove used edges
        for ek in _cycle_edges(found):
            remaining.discard(ek)
    
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))
    
    best = []
    
    # Try many random seeds
    for seed in range(2000):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        cycles = _search_three(edges, n, rng, deadline)
        if len(cycles) > len(best):
            best = [c[:] for c in cycles]
            if len(best) == 3 and _validate(best, n, edge_set):
                return [[int(v) for v in c] for c in best]
    
    if len(best) == 3 and _validate(best, n, edge_set):
        return [[int(v) for v in c] for c in best]
    
    if best:
        return [[int(v) for v in c] for c in best]
    
    # Fallback
    return [list(range(n))]
# EVOLVE-BLOCK-END
