"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_START_TIME = time.time()
_TIME_LIMIT = 75.0  # safety margin before 90s kill

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _bfs_distances(adj, n, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _all_pairs_dist(adj, n):
    """Return list of (dist, u, v) for all u<v."""
    result = []
    for u in range(n):
        dist = _bfs_distances(adj, n, u)
        for v in range(u+1, n):
            if dist[v] == -1:
                result.append((999, u, v))
            else:
                result.append((dist[v], u, v))
    return result

def _check_solution(n, base_edges, added_edges):
    edge_set = set()
    for u, v in base_edges:
        edge_set.add((min(u, v), max(u, v)))
    for u, v in added_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Triangle-free check
    for i in range(n):
        for j in adj[i]:
            for k in adj[i]:
                if j < k and k in adj[j]:
                    return False
    
    # Connected + diameter
    for start in range(n):
        dist = _bfs_distances(adj, n, start)
        if -1 in dist or max(dist) > 4:
            return False
    
    return True

def _build_adj(n, base_edges, added_edges):
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degrees(adj):
    return [len(adj[i]) for i in range(len(adj))]

def _is_triangle_free_add(adj, u, v):
    """Check if adding edge (u,v) keeps graph triangle-free."""
    for k in adj[u]:
        if k != v and k in adj[v]:
            return False
    return True

def _greedy_construct(n, base_edges):
    """Greedy: repeatedly add edge that best reduces diameter, with degree balance."""
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    added = []
    edge_set = set(base_set)
    
    # All candidate edges
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                candidates.append((i, j))
    
    while _time_ok():
        # Compute current distances
        pairs = _all_pairs_dist(adj, n)
        max_d = max(d for d, _, _ in pairs)
        if max_d <= 4:
            break
        
        # Count how many pairs are at each distance
        # We want to add edge (u,v) that minimizes the resulting diameter
        # Approximation: count pairs that would be shortened
        best_edge = None
        best_score = None
        
        deg = _degrees(adj)
        
        for u, v in candidates:
            if (u, v) in edge_set:
                continue
            if not _is_triangle_free_add(adj, u, v):
                continue
            
            # Score: prefer edges that connect low-degree vertices
            # and are "long" in the current graph (helps most)
            du = _bfs_distances(adj, n, u)
            dv = _bfs_distances(adj, n, v)
            
            # Count pairs that would be improved
            improved = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                duw = du[w]
                dvw = dv[w]
                if duw == -1 or dvw == -1:
                    continue
                # New distance via (u,v): min(duw, dvw) + 1
                new_d = min(duw, dvw) + 1
                old_d = min(duw, dvw)  # wait, old distance from w to... 
                # Actually we need dist from each node to farthest
                # Simpler: just count how many nodes have dist > 4 to u or v
                if duw > 3 or dvw > 3:
                    improved += 1
            
            # Degree penalty: prefer lower degree endpoints
            deg_penalty = deg[u] + deg[v]
            
            score = (improved, -deg_penalty)
            
            if best_score is None or score > best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
    
    return added

def _local_search(n, base_edges, added_edges, max_iters=500):
    """Try to remove or swap edges to reduce count."""
    if not _time_ok():
        return added_edges
    
    adj = _build_adj(n, base_edges, added_edges)
    current = list(added_edges)
    
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    all_cand = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in base_set:
                all_cand.append((i, j))
    
    improved = True
    iters = 0
    while improved and iters < max_iters and _time_ok():
        improved = False
        iters += 1
        
        # Try removing each edge
        for idx in range(len(current)):
            if not _time_ok():
                break
            trial = current[:idx] + current[idx+1:]
            trial_adj = _build_adj(n, base_edges, trial)
            
            # Check triangle-free
            tf = True
            for i in range(n):
                for j in trial_adj[i]:
                    for k in trial_adj[i]:
                        if j < k and k in trial_adj[j]:
                            tf = False
                            break
                    if not tf:
                        break
                if not tf:
                    break
            
            if not tf:
                continue
            
            # Check diameter
            ok = True
            for start in range(n):
                dist = _bfs_distances(trial_adj, n, start)
                if -1 in dist or max(dist) > 4:
                    ok = False
                    break
            
            if ok:
                current = trial
                adj = trial_adj
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another
        for idx in range(len(current)):
            if not _time_ok():
                break
            for new_edge in all_cand:
                if (new_edge[0], new_edge[1]) in set(tuple(sorted(e)) for e in current):
                    continue
                trial = current[:idx] + current[idx+1:] + [list(new_edge)]
                
                # Check no duplicates
                seen = set()
                for u, v in trial:
                    key = (min(u,v), max(u,v))
                    if key in seen:
                        break
                    seen.add(key)
                else:
                    continue
                break
            
            trial_adj = _build_adj(n, base_edges, trial)
            
            # Triangle-free
            tf = True
            for i in range(n):
                for j in trial_adj[i]:
                    for k in trial_adj[i]:
                        if j < k and k in trial_adj[j]:
                            tf = False
                            break
                    if not tf:
                        break
                if not tf:
                    break
            
            if not tf:
                continue
            
            # Diameter
            ok = True
            for start in range(n):
                dist = _bfs_distances(trial_adj, n, start)
                if -1 in dist or max(dist) > 4:
                    ok = False
                    break
            
            if ok:
                current = trial
                adj = trial_adj
                improved = True
                break
        
        if improved:
            continue
    
    return current

def construct():
    n = P['n']
    base = P['edges']
    
    # Phase 1: Greedy construction
    added = _greedy_construct(n, base)
    
    if not _check_solution(n, base, added):
        # Fallback: direct pattern
        candidates = []
        base_set = set((min(u,v), max(u,v)) for u, v in base)
        for i in range(0, 18):
            j = i + 6
            if j < n and (i, j) not in base_set:
                candidates.append([i, j])
        for i in range(0, 12):
            j = i + 12
            if j < n and (i, j) not in base_set:
                candidates.append([i, j])
        if _check_solution(n, base, candidates):
            added = candidates
    
    if not _check_solution(n, base, added):
        return []
    
    # Phase 2: Local search to reduce edge count
    if _time_ok():
        added = _local_search(n, base, added)
        if _check_solution(n, base, added):
            pass
        else:
            # Revert to greedy result
            added = _greedy_construct(n, base)
    
    # Ensure proper types
    result = []
    for u, v in added:
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
