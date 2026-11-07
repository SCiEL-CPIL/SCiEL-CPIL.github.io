"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
import itertools

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
        
    for u, v in added_edges:
        if (u, v) in base_edges:
            return False
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
                    
    # Check connectivity and diameter
    dist = _bfs_distances(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return False
                
    diam = _get_diameter(dist)
    if diam > 4:
        return False
        
    return True

def _construct_greedy():
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    added_edges = []
    
    dist = _bfs_distances(adj, n)
    current_diam = _get_diameter(dist)
    
    max_iters = 50
    iter_count = 0
    
    while current_diam > 4 and iter_count < max_iters:
        iter_count += 1
        max_d = -1
        best_u, best_v = -1, -1
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
                    best_u, best_v = i, j
        
        if best_u == -1:
            break
            
        if (best_u, best_v) in base_edges:
            pass
        else:
            if _is_triangle_free(adj, best_u, best_v, n):
                adj[best_u].append(best_v)
                adj[best_v].append(best_u)
                base_edges.add((best_u, best_v))
                base_edges.add((best_v, best_u))
                added_edges.append([best_u, best_v])
                dist = _bfs_distances(adj, n)
                current_diam = _get_diameter(dist)
            else:
                found = False
                for i in range(n):
                    for j in range(i+1, n):
                        if i == best_u and j == best_v:
                            continue
                        if (i, j) in base_edges:
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
                                added_edges.append([i, j])
                                dist = temp_dist
                                current_diam = temp_diam
                                found = True
                                break
                    if found:
                        break
                if not found:
                    break

    return added_edges

def _local_search(added_edges, deadline):
    n = P['n']
    current_solution = [e[:] for e in added_edges]
    
    if not _verify_solution(current_solution):
        return current_solution
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to remove any edge
        for i in range(len(current_solution)):
            if time.time() > deadline:
                return current_solution
            test_solution = current_solution[:i] + current_solution[i+1:]
            if _verify_solution(test_solution):
                current_solution = test_solution
                improved = True
                break
                
    return current_solution

def _random_search(deadline):
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Generate candidate edges
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_edges:
                candidates.append([u, v])
                
    random.shuffle(candidates)
    
    best_solution = None
    best_len = float('inf')
    
    # Try to find a solution with k edges
    for k in range(3, 15):
        if time.time() > deadline:
            break
            
        attempts = 0
        max_attempts = 200000
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            subset = random.sample(candidates, k)
            if _verify_solution(subset):
                best_solution = subset
                best_len = k
                break
                
        if best_solution is not None:
            break
            
    if best_solution is None:
        return None
        
    # Local search to improve
    current = [e[:] for e in best_solution]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            test_solution = current[:i] + current[i+1:]
            if _verify_solution(test_solution):
                current = test_solution
                improved = True
                break
                
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85 # Safety margin
    
    # 1. Generate initial solution with greedy
    added = _construct_greedy()
    
    # 2. Local Search to minimize edges
    optimized = _local_search(added, deadline)
    
    best_solution = optimized
    if _verify_solution(optimized):
        best_len = len(optimized)
    else:
        best_len = float('inf')
        
    # 3. Random search if time permits
    if time.time() < deadline - 10:
        random.seed(42) # Deterministic for reproducibility in evaluation? Or just let it be random.
        # Actually, deterministic is better for debugging, but random is better for finding different local minima.
        # Let's use a fixed seed for consistency in the "random" part, or just rely on system time.
        # The prompt doesn't require determinism, but it helps with debugging.
        # Let's try a few different seeds or just one random search run.
        
        random_sol = _random_search(deadline)
        if random_sol is not None and _verify_solution(random_sol):
            if len(random_sol) < best_len:
                best_solution = random_sol
                best_len = len(random_sol)
                
    # 4. Final check and return
    if _verify_solution(best_solution):
        res = []
        for u, v in best_solution:
            res.append([int(u), int(v)])
        return res
    else:
        # Fallback to greedy if it was valid
        if _verify_solution(added):
            res = []
            for u, v in added:
                res.append([int(u), int(v)])
            return res
        else:
            return []

# EVOLVE-BLOCK-END
