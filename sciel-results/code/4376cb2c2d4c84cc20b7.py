"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _build_graph(n, edges):
    """Build adjacency, degree, and triangle count structures."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edge_set = set()
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            continue
        edge_set.add((a, b))
        adj[a].add(b)
        adj[b].add(a)
        deg[a] += 1
        deg[b] += 1
    
    # Count triangles per vertex
    tri_count = [0] * n
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = len(adj[u] & adj[v])
                tri_count[u] += common
                tri_count[v] += common
    # Each triangle counted 3 times (once per vertex), so divide by 3... 
    # Actually tri_count[v] = number of triangles containing v
    # For max degree sum over triangles, we need per-triangle sums.
    # Let's track it differently: for each edge (u,v), the triangles containing it
    # are the common neighbors. The degree sum for triangle (u,v,w) is deg[u]+deg[v]+deg[w].
    
    # Better approach: track max triangle degree sum directly
    # For incremental updates, when we add/remove edge (u,v):
    # - Triangles removed: for each common neighbor w of u and v, triangle (u,v,w) is gone
    # - Triangles added: for each common neighbor w of u and v, triangle (u,v,w) is new
    # But we need the MAX over all triangles, so we need to track all triangle sums.
    
    # Alternative: maintain a list/set of all triangle degree sums
    # When edge (u,v) is added: new triangles are (u,v,w) for w in adj[u]&adj[v]
    # When edge (u,v) is removed: triangles (u,v,w) for w in adj[u]&adj[v] are removed
    
    # For efficiency, let's just recompute from scratch but faster,
    # or maintain a multiset of triangle sums.
    
    return adj, deg, edge_set

def _compute_score_fast(adj, deg, n, edge_set):
    """Compute max triangle degree sum efficiently."""
    max_sum = 0
    for u, v in edge_set:
        if u > v:
            continue
        common = adj[u] & adj[v]
        for w in common:
            s = deg[u] + deg[v] + deg[w]
            if s > max_sum:
                max_sum = s
    return max_sum

def _baseline_edges():
    """K_{8,8} plus one edge inside a part."""
    return [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]

def _local_search(n, max_iter, deadline, start_edges=None):
    """Local search with fast incremental scoring."""
    if start_edges is None:
        start_edges = _baseline_edges()
    
    adj, deg, edge_set = _build_graph(n, start_edges)
    current_score = _compute_score_fast(adj, deg, n, edge_set)
    best_score = current_score
    best_edges = set(edge_set)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_possible_set = set(all_possible)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try multiple swaps per iteration
        for _ in range(30):
            if time.time() > deadline:
                break
            
            # Pick random edge to remove
            existing = list(edge_set)
            if not existing:
                break
            u1, v1 = random.choice(existing)
            
            # Pick random edge to add
            u2, v2 = random.choice(all_possible)
            
            if (u2, v2) == (u1, v1) or (u2, v2) in edge_set:
                continue
            
            # Perform swap
            edge_set.discard((u1, v1))
            edge_set.add((u2, v2))
            
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            deg[u1] -= 1
            deg[v1] -= 1
            
            adj[u2].add(v2)
            adj[v2].add(u2)
            deg[u2] += 1
            deg[v2] += 1
            
            new_score = _compute_score_fast(adj, deg, n, edge_set)
            
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(edge_set)
            else:
                # Revert
                edge_set.discard((u2, v2))
                edge_set.add((u1, v1))
                
                adj[u2].discard(v2)
                adj[v2].discard(u2)
                deg[u2] -= 1
                deg[v2] -= 1
                
                adj[u1].add(v1)
                adj[v1].add(u1)
                deg[u1] += 1
                deg[v1] += 1
    
    return list(best_edges), best_score

def _polish(n, edges, deadline):
    """Final polish: try all single-edge swaps exhaustively."""
    adj, deg, edge_set = _build_graph(n, edges)
    current_score = _compute_score_fast(adj, deg, n, edge_set)
    best_score = current_score
    best_edges = set(edge_set)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_possible_set = set(all_possible)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u1, v1 in list(edge_set):
            if time.time() > deadline:
                break
            for u2, v2 in all_possible:
                if (u2, v2) == (u1, v1) or (u2, v2) in edge_set:
                    continue
                
                # Perform swap
                edge_set.discard((u1, v1))
                edge_set.add((u2, v2))
                
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                deg[u1] -= 1
                deg[v1] -= 1
                
                adj[u2].add(v2)
                adj[v2].add(u2)
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = _compute_score_fast(adj, deg, n, edge_set)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(edge_set)
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    edge_set.discard((u2, v2))
                    edge_set.add((u1, v1))
                    
                    adj[u2].discard(v2)
                    adj[v2].discard(u2)
                    deg[u2] -= 1
                    deg[v2] -= 1
                    
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    deg[u1] += 1
                    deg[v1] += 1
    
    return list(best_edges), best_score

def construct():
    """Construct a graph on 16 vertices with 65 edges minimizing max triangle degree sum."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    # Start with baseline
    best_edges = _baseline_edges()
    best_score = _compute_score_fast(*_build_graph(n, best_edges)[:2], n, _build_graph(n, best_edges)[2])
    
    # Run local search with multiple restarts
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            start_e = best_edges
        else:
            # Perturb the best solution
            adj, deg, edge_set = _build_graph(n, best_edges)
            perturbed = list(edge_set)
            for _ in range(5):
                u1, v1 = random.choice(perturbed)
                u2, v2 = random.choice([(i, j) for i in range(n) for j in range(i+1, n)])
                if (u2, v2) not in edge_set:
                    perturbed.remove((u1, v1))
                    perturbed.append((u2, v2))
                    break
            start_e = perturbed
        
        ls_edges, ls_score = _local_search(n, max_iter=3000, deadline=deadline - 30, start_edges=start_e)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
    
    # Final polish pass
    if time.time() < deadline:
        polish_edges, polish_score = _polish(n, best_edges, deadline)
        if polish_score < best_score:
            best_score = polish_score
            best_edges = polish_edges
    
    # Ensure exactly 65 edges
    edge_set = set()
    for u, v in best_edges:
        a, b = (u, v) if u < v else (v, u)
        edge_set.add((a, b))
    
    if len(edge_set) != target_edges:
        best_edges = _baseline_edges()
        edge_set = set()
        for u, v in best_edges:
            a, b = (u, v) if u < v else (v, u)
            edge_set.add((a, b))
    
    return {'n': n, 'edges': [list(e) for e in sorted(edge_set)]}
# EVOLVE-BLOCK-END
