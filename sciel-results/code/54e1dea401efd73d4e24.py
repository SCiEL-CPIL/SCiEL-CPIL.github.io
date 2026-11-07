"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
import math

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, n):
    dist = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = [i]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
    return dist

def _get_diameter(dist):
    max_d = 0
    for i in range(len(dist)):
        for j in range(len(dist)):
            if dist[i][j] > max_d:
                max_d = dist[i][j]
    return max_d

def _is_triangle_free(adj, u, v, n):
    neighbors_u = set(adj[u])
    neighbors_v = set(adj[v])
    common = neighbors_u.intersection(neighbors_v)
    if common:
        return False
    return True

def _verify_solution(added_edges):
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    seen_added = set()
    for u, v in added_edges:
        if (u, v) in base_edges or (v, u) in base_edges:
            return False
        if (u, v) in seen_added or (v, u) in seen_added:
            return False
        seen_added.add((u, v))
        seen_added.add((v, u))
        adj[u].append(v)
        adj[v].append(u)
        
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
                    
    dist = _bfs_distances(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return False
                
    diam = _get_diameter(dist)
    if diam > 4:
        return False
        
    return True

def _exact_search(start_time, deadline):
    n = P['n']
    base_adj = [[] for _ in range(n)]
    base_edges_set = set()
    for u, v in P['edges']:
        base_adj[u].append(v)
        base_adj[v].append(u)
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # Generate all valid candidate edges (not in base, triangle-free with base)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in base_edges_set or (v, u) in base_edges_set:
                continue
            if _is_triangle_free(base_adj, u, v, n):
                candidates.append((u, v))
                
    # Sort candidates to improve pruning (optional, but helps)
    # Heuristic: prioritize edges that connect distant nodes?
    # For exact search, order matters less than pruning, but let's keep it simple.
    
    best_solution = None
    best_k = len(candidates) + 1
    
    # Pre-compute base distances
    base_dist = _bfs_distances(base_adj, n)
    base_diam = _get_diameter(base_dist)
    
    # If base is already valid (diam <= 4), return empty
    if base_diam <= 4:
        return []
        
    # Branch and Bound
    # We try to find a solution of size k.
    # We iterate k from lower bound to upper bound.
    
    # Lower bound: 
    # In a path of 24 nodes, to get diameter 4, we need to "jump".
    # Max distance in path is 23.
    # An edge (u,v) reduces distance between u and v to 1.
    # Roughly, we need enough edges to cover the "gaps".
    # Let's start from k=1 and go up.
    
    # Upper bound: we can always add a valid spanning tree? No, must be triangle free.
    # A simple valid solution: add edges (0, 12), (12, 23)? 
    # Let's just use the greedy result as an upper bound if available, or a large number.
    # Greedy from previous code is a good upper bound generator.
    
    def _greedy_upper_bound():
        adj = [list(x) for x in base_adj]
        added = []
        dist = _bfs_distances(adj, n)
        curr_diam = _get_diameter(dist)
        
        max_iters = 50
        for _ in range(max_iters):
            if curr_diam <= 4:
                break
            # Find furthest pair
            max_d = -1
            bu, bv = -1, -1
            for i in range(n):
                for j in range(i+1, n):
                    if dist[i][j] > max_d:
                        max_d = dist[i][j]
                        bu, bv = i, j
                        
            if bu == -1:
                break
                
            # Try to add edge (bu, bv) if valid
            if (bu, bv) not in base_edges_set and (bv, bu) not in base_edges_set:
                if _is_triangle_free(adj, bu, bv, n):
                    adj[bu].append(bv)
                    adj[bv].append(bu)
                    added.append([bu, bv])
                    dist = _bfs_distances(adj, n)
                    curr_diam = _get_diameter(dist)
                    continue
            
            # If direct edge fails, find any edge that reduces diameter
            found = False
            for u, v in candidates:
                if (u, v) in [(x, y) for x, y in added] or (v, u) in [(x, y) for x, y in added]:
                    continue
                if _is_triangle_free(adj, u, v, n):
                    adj[u].append(v)
                    adj[v].append(u)
                    temp_dist = _bfs_distances(adj, n)
                    temp_diam = _get_diameter(temp_dist)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    if temp_diam < curr_diam:
                        adj[u].append(v)
                        adj[v].append(u)
                        added.append([u, v])
                        dist = temp_dist
                        curr_diam = temp_diam
                        found = True
                        break
            if not found:
                break
        return added

    ub_added = _greedy_upper_bound()
    if _verify_solution(ub_added):
        ub_k = len(ub_added)
        best_solution = ub_added
    else:
        ub_k = len(candidates)
        best_solution = None
        
    # If we found a solution with 0 edges, return it
    if ub_k == 0:
        return []
        
    # Search for k = 1 to ub_k - 1
    # If we find a solution with k edges, we can stop because we search in increasing order.
    # Wait, we want MINIMUM. So we search k=1, k=2, etc.
    # If we find a valid solution for k, that is the minimum.
    
    for k in range(1, ub_k):
        if time.time() > deadline:
            break
            
        # Backtracking search for exactly k edges
        found = False
        current_edges = []
        
        # State for backtracking
        # adj: current adjacency list
        # dist: current distances (recomputed on change? or incremental?)
        # For n=24, recomputing BFS is O(n^2) ~ 600 ops. Very fast.
        
        # We need to ensure triangle-free property is maintained.
        # We can pre-check candidates against current adj.
        
        def backtrack(idx, depth, adj, dist, diam):
            nonlocal found, best_solution, best_k
            
            if time.time() > deadline:
                return True # Signal to stop
            
            if depth == k:
                if diam <= 4:
                    found = True
                    best_solution = [e[:] for e in current_edges]
                    best_k = k
                return found
            
            # Pruning: if even adding all remaining edges can't reduce diameter enough?
            # Hard to estimate.
            
            # Heuristic: if current diameter is already <= 4, we can stop adding?
            # No, we must add exactly k edges? 
            # The problem asks to minimize the number of added edges.
            # So if we have a valid solution with d < k edges, we would have found it in previous iterations.
            # In this iteration, we are looking for a solution of EXACTLY k edges?
            # Actually, if we find a solution with < k edges, it would have been found earlier.
            # So we can assume we need to add k edges.
            # But wait, if we add edges and diameter becomes <=4 before depth k, 
            # we could just stop? No, because we are in the loop for k.
            # If we find a valid solution with depth < k, it contradicts the fact that we didn't find it in smaller k loops.
            # Unless the search for smaller k failed due to pruning? No, backtracking is exhaustive for a fixed k.
            # So if we reach depth < k and diam <= 4, it means a solution with < k edges exists.
            # This should have been found in the loop for that smaller k.
            # So this case shouldn't happen if the search is correct.
            # However, to be safe, if diam <= 4, we can return found?
            # No, because we are searching for k. If we return found, we might accept a solution with < k edges?
            # Actually, if diam <= 4 at depth d < k, then the first d edges form a valid solution.
            # So we should have found it in the k=d loop.
            # So if we are in k loop and diam <= 4, we can stop searching for this k?
            # Yes, because a smaller solution exists.
            
            if diam <= 4:
                # This implies a solution with < k edges exists.
                # We should have found it earlier.
                # To be robust, let's mark found and return.
                found = True
                best_solution = [e[:] for e in current_edges]
                best_k = depth
                return True
            
            # Iterate through candidates
            # To reduce symmetry, we can restrict choices?
            # Candidates are undirected edges.
            # We can just iterate through the list.
            
            # Optimization: Only consider candidates that are "useful"
            # i.e., they reduce the distance between some pair that is currently > 4?
            # Or simply, any candidate.
            
            # To speed up, we can sort candidates by potential benefit.
            # But for exact search, we just need to explore.
            
            for i in range(idx, len(candidates)):
                u, v = candidates[i]
                
                # Check if edge (u,v) is triangle-free with current adj
                if not _is_triangle_free(adj, u, v, n):
                    continue
                    
                # Check if edge is already in current_edges
                # Since we pick from candidates list which doesn't have duplicates, and we don't reuse indices?
                # We are using idx to avoid permutations of the same set.
                # So (u,v) is not in current_edges.
                
                # Add edge
                adj[u].append(v)
                adj[v].append(u)
                current_edges.append([u, v])
                
                # Recompute distances and diameter
                new_dist = _bfs_distances(adj, n)
                new_diam = _get_diameter(new_dist)
                
                # Recurse
                if backtrack(i + 1, depth + 1, adj, new_dist, new_diam):
                    return True
                    
                # Undo
                adj[u].remove(v)
                adj[v].remove(u)
                current_edges.pop()
                
                if time.time() > deadline:
                    return True
                    
            return False
            
        # Initialize state
        init_adj = [list(x) for x in base_adj]
        init_dist = _bfs_distances(init_adj, n)
        init_diam = _get_diameter(init_dist)
        
        if backtrack(0, 0, init_adj, init_dist, init_diam):
            if found:
                return best_solution
            else:
                # Timeout or other stop
                break
                
    # If no better solution found, return the upper bound
    if best_solution is not None:
        return best_solution
    else:
        return []

def construct():
    start_time = time.time()
    deadline = start_time + 80 # Safety margin
    
    # 1. Try Exact Search
    # The exact search might be slow if the solution is large, but for n=24 and diam=4, 
    # the number of edges should be small (likely < 10).
    # So k=1, 2, ... up to ~10 is very fast.
    
    result = _exact_search(start_time, deadline)
    
    # 2. Verify result
    if _verify_solution(result):
        res = []
        for u, v in result:
            res.append([int(u), int(v)])
        return res
    else:
        # Fallback to greedy if exact search failed or returned invalid
        # This should rarely happen
        n = P['n']
        adj = [[] for _ in range(n)]
        base_edges = set()
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
            base_edges.add((u, v))
            base_edges.add((v, u))
            
        added = []
        dist = _bfs_distances(adj, n)
        current_diam = _get_diameter(dist)
        
        max_iters = 100
        for _ in range(max_iters):
            if current_diam <= 4:
                break
            max_d = -1
            best_u, best_v = -1, -1
            for i in range(n):
                for j in range(i+1, n):
                    if dist[i][j] > max_d:
                        max_d = dist[i][j]
                        best_u, best_v = i, j
                        
            if best_u == -1:
                break
                
            if (best_u, best_v) not in base_edges and (best_v, best_u) not in base_edges:
                if _is_triangle_free(adj, best_u, best_v, n):
                    adj[best_u].append(best_v)
                    adj[best_v].append(best_u)
                    base_edges.add((best_u, best_v))
                    base_edges.add((best_v, best_u))
                    added.append([best_u, best_v])
                    dist = _bfs_distances(adj, n)
                    current_diam = _get_diameter(dist)
                    continue
                    
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    if i == best_u and j == best_v:
                        continue
                    if (i, j) in base_edges or (j, i) in base_edges:
                        continue
                    if _is_triangle_free(adj, i, j, n):
                        adj[i].append(j)
                        adj[j].append(i)
                        temp_dist = _bfs_distances(adj, n)
                        temp_diam = _get_diameter(temp_dist)
                        adj[i].remove(j)
                        adj[j].remove(i)
                        
                        if temp_diam < current_diam:
                            adj[i].append(j)
                            adj[j].append(i)
                            base_edges.add((i, j))
                            base_edges.add((j, i))
                            added.append([i, j])
                            dist = temp_dist
                            current_diam = temp_diam
                            found = True
                            break
                if found:
                    break
            if not found:
                break
                
        if _verify_solution(added):
            res = []
            for u, v in added:
                res.append([int(u), int(v)])
            return res
        else:
            return []

# EVOLVE-BLOCK-END
