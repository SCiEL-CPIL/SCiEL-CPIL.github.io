"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj_from_edges(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(n, adj):
    for u in range(n):
        for v in adj[u]:
            if v > u:
                if adj[u] & adj[v]:
                    return False
    return True

def _diameter_leq(n, adj, max_diam):
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_diam:
                        return False
                    queue.append(v)
        if -1 in dist:
            return False
    return True

def _is_valid(n, base_edges, added_edges):
    all_edges = list(base_edges) + list(added_edges)
    adj = _build_adj_from_edges(n, all_edges)
    if not _is_triangle_free(n, adj):
        return False
    if not _diameter_leq(n, adj, 4):
        return False
    return True

def _get_all_candidate_edges(n, base_edges, current_added):
    """Get all edges that could be added without creating triangles or duplicates."""
    all_edges = set((min(u,v), max(u,v)) for u,v in base_edges)
    for u,v in current_added:
        all_edges.add((min(u,v), max(u,v)))
    adj = _build_adj_from_edges(n, list(base_edges) + list(current_added))
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u,v) in all_edges:
                continue
            # Check triangle-free: u and v must not share a common neighbor
            if adj[u] & adj[v]:
                continue
            candidates.append((u,v))
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n = P['n']
    base_edges = [(u, v) for u, v in P['edges']]
    base_edge_set = set((min(u, v), max(u, v)) for u, v in base_edges)
    
    # Initial solution: connect endpoints to middle
    initial_solution = [
        (0, 12), (23, 12), (0, 11), (23, 13),
        (1, 12), (22, 12), (2, 12), (21, 12),
        (3, 12), (20, 12)
    ]
    
    # Verify and reduce initial
    if not _is_valid(n, base_edges, initial_solution):
        # Try simpler initial
        initial_solution = [(0, 12), (23, 12), (0, 11), (23, 13), (1, 12), (22, 12), (2, 12), (21, 12)]
        if not _is_valid(n, base_edges, initial_solution):
            return []
    
    best = list(initial_solution)
    
    # Phase 1: Greedy removal
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(len(best)):
            if (time.time() - start_time) > deadline:
                break
            new_edges = [e for j, e in enumerate(best) if j != i]
            if _is_valid(n, base_edges, new_edges):
                best = new_edges
                improved = True
                break
    
    # Phase 2: 2-opt swaps - remove one edge, add another
    for iteration in range(50):
        if (time.time() - start_time) > deadline:
            break
        made_swap = False
        current_added_set = set((min(u,v), max(u,v)) for u,v in best)
        candidates = _get_all_candidate_edges(n, base_edges, best)
        
        for i in range(len(best)):
            if (time.time() - start_time) > deadline:
                break
            for cand in candidates:
                if (time.time() - start_time) > deadline:
                    break
                # Try: remove best[i], add cand
                new_edges = [e for j, e in enumerate(best) if j != i] + [cand]
                # Check no duplicate
                if len(set((min(u,v), max(u,v)) for u,v in new_edges)) != len(new_edges):
                    continue
                if _is_valid(n, base_edges, new_edges):
                    best = new_edges
                    made_swap = True
                    break
            if made_swap:
                break
        if not made_swap:
            break
    
    # Phase 3: After swaps, try more removals
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(len(best)):
            if (time.time() - start_time) > deadline:
                break
            new_edges = [e for j, e in enumerate(best) if j != i]
            if _is_valid(n, base_edges, new_edges):
                best = new_edges
                improved = True
                break
    
    # Phase 4: Try different initial solutions
    alt_solutions = [
        [(0, 12), (23, 12), (1, 12), (22, 12), (2, 12), (21, 12), (3, 12), (20, 12)],
        [(0, 11), (23, 13), (1, 12), (22, 12), (2, 12), (21, 12), (3, 12), (20, 12)],
        [(0, 12), (23, 12), (0, 10), (23, 14), (1, 12), (22, 12), (2, 12), (21, 12)],
        [(0, 12), (23, 12), (1, 11), (22, 13), (2, 12), (21, 12), (3, 12), (20, 12)],
    ]
    
    for alt in alt_solutions:
        if (time.time() - start_time) > deadline:
            break
        if not _is_valid(n, base_edges, alt):
            continue
        reduced = list(alt)
        imp = True
        while imp and (time.time() - start_time) < deadline:
            imp = False
            for i in range(len(reduced)):
                if (time.time() - start_time) > deadline:
                    break
                new_edges = [e for j, e in enumerate(reduced) if j != i]
                if _is_valid(n, base_edges, new_edges):
                    reduced = new_edges
                    imp = True
                    break
        
        # Try 2-opt on this reduced solution
        for iteration in range(20):
            if (time.time() - start_time) > deadline:
                break
            made_swap = False
            candidates = _get_all_candidate_edges(n, base_edges, reduced)
            for i in range(len(reduced)):
                if (time.time() - start_time) > deadline:
                    break
                for cand in candidates:
                    if (time.time() - start_time) > deadline:
                        break
                    new_edges = [e for j, e in enumerate(reduced) if j != i] + [cand]
                    if len(set((min(u,v), max(u,v)) for u,v in new_edges)) != len(new_edges):
                        continue
                    if _is_valid(n, base_edges, new_edges):
                        reduced = new_edges
                        made_swap = True
                        break
                if made_swap:
                    break
            if not made_swap:
                break
        
        # More removals after swaps
        imp = True
        while imp and (time.time() - start_time) < deadline:
            imp = False
            for i in range(len(reduced)):
                if (time.time() - start_time) > deadline:
                    break
                new_edges = [e for j, e in enumerate(reduced) if j != i]
                if _is_valid(n, base_edges, new_edges):
                    reduced = new_edges
                    imp = True
                    break
        
        if len(reduced) < len(best):
            best = reduced
    
    # Final verification
    if _is_valid(n, base_edges, best):
        return [[u, v] for u, v in best]
    else:
        return [[u, v] for u, v in initial_solution]

# EVOLVE-BLOCK-END
