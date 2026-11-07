"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import deque, defaultdict
from itertools import combinations

sys.setrecursionlimit(100000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Precompute all-pairs shortest paths and predecessors
    INF = float('inf')
    dist = [[INF]*n for _ in range(n)]
    pred = [[-1]*n for _ in range(n)]
    for start in range(n):
        dist[start][start] = 0
        pred[start][start] = -1
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[start][v] == INF:
                    dist[start][v] = dist[start][u] + 1
                    pred[start][v] = u
                    q.append(v)
    
    def get_shortest_path(start, end):
        if dist[start][end] == INF:
            return None
        path = []
        cur = end
        while cur != -1:
            path.append(cur)
            cur = pred[start][cur]
        path.reverse()
        return path

    # Find all simple paths between u and v up to max_len
    path_cache = {}
    def get_all_paths(u, v, max_len=5):
        key = (u, v, max_len)
        if key in path_cache:
            return path_cache[key]
        
        if u == v:
            path_cache[key] = [[u]]
            return path_cache[key]
            
        paths = []
        def dfs(curr, path, visited):
            if len(path) - 1 > max_len:
                return
            if curr == v:
                paths.append(list(path))
                return
            for nxt in adj_list[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, path, visited)
                    path.pop()
                    visited.remove(nxt)
        
        dfs(u, [u], {u})
        paths.sort(key=lambda p: (len(p), p))
        path_cache[key] = paths
        return paths

    # Bron-Kerbosch for Maximum Clique
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
            
    bron_kerbosch(set(), set(range(n)), set())
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_k = 0
    best_B = []
    best_paths = {}
    
    def check_subdivision(B, time_limit=2.0):
        start_t = time.time()
        B_set = set(B)
        k = len(B)
        
        non_adj_pairs = []
        for i in range(k):
            for j in range(i+1, k):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj_pairs.append((u, v))
        
        if not non_adj_pairs:
            return True, {}
        
        domains = {}
        for u, v in non_adj_pairs:
            paths = []
            for l in range(2, 7):
                all_p = get_all_paths(u, v, l)
                valid_p = [p for p in all_p if not any(x in B_set for x in p[1:-1])]
                if valid_p:
                    paths.extend(valid_p)
                    break
            if not paths:
                all_p = get_all_paths(u, v, 8)
                valid_p = [p for p in all_p if not any(x in B_set for x in p[1:-1])]
                if not valid_p:
                    return False, None
                paths = valid_p
            
            domains[(u, v)] = paths
        
        sorted_pairs = sorted(non_adj_pairs, key=lambda p: len(domains[p]))
        
        used_interiors = set()
        assignment = {}
        
        def backtrack(idx):
            if time.time() - start_t > time_limit:
                return False
            if idx == len(sorted_pairs):
                return True
            
            u, v = sorted_pairs[idx]
            for path in domains[(u, v)]:
                interior = path[1:-1]
                if any(w in used_interiors for w in interior):
                    continue
                used_interiors.update(interior)
                assignment[(u, v)] = path
                if backtrack(idx + 1):
                    return True
                used_interiors.difference_update(interior)
                del assignment[(u, v)]
            return False
        
        if backtrack(0):
            return True, dict(assignment)
        return False, None

    # Strategy: Systematic search over subsets of increasing size
    # Use randomized restarts with bias toward high-degree vertices
    
    # First, try the max clique as a starting point
    current_B = list(best_clique)
    ok, paths_dict = check_subdivision(current_B, time_limit=5.0)
    if ok:
        best_k = len(current_B)
        best_B = current_B
        best_paths = paths_dict
    
    # Try to find larger sets by random sampling
    # For each target size from best_k+1 to n, try multiple random samples
    for target_k in range(best_k + 1, n + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        attempts = 0
        max_attempts = 200
        
        while attempts < max_attempts and not found:
            if time.time() > deadline - 5:
                break
            
            attempts += 1
            
            # Generate a random subset of size target_k
            # Bias: include high-degree vertices with higher probability
            B_rand = []
            remaining = list(range(n))
            
            for _ in range(target_k):
                if not remaining:
                    break
                # Weight by degree
                weights = [degrees[v] + 1 for v in remaining]
                total_weight = sum(weights)
                r = random.random() * total_weight
                cumsum = 0
                for i, v in enumerate(remaining):
                    cumsum += weights[i]
                    if r <= cumsum:
                        B_rand.append(v)
                        remaining.remove(v)
                        break
                else:
                    # Fallback if random didn't select
                    B_rand.append(remaining.pop())
            
            if len(B_rand) < target_k:
                continue
                
            ok, paths_rand = check_subdivision(B_rand, time_limit=1.5)
            if ok:
                best_k = target_k
                best_B = B_rand
                best_paths = paths_rand
                found = True
                break
        
        if not found:
            # Try a few more with different strategies
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Try including some vertices from current best_B
                if best_B and len(best_B) >= target_k - 2:
                    # Keep most of best_B, replace a few
                    B_try = list(best_B)
                    to_remove = min(2, len(B_try))
                    for _ in range(to_remove):
                        idx = random.randint(0, len(B_try) - 1)
                        B_try.pop(idx)
                    
                    # Add new vertices
                    candidates = [v for v in range(n) if v not in B_try]
                    random.shuffle(candidates)
                    for v in candidates:
                        if len(B_try) >= target_k:
                            break
                        B_try.append(v)
                    
                    if len(B_try) == target_k:
                        ok, paths_try = check_subdivision(B_try, time_limit=1.5)
                        if ok:
                            best_k = target_k
                            best_B = B_try
                            best_paths = paths_try
                            found = True
                            break
            
            if not found:
                break  # Can't find this size, stop
    
    # Final verification
    B = best_B
    k = len(B)
    
    if k < 2:
        for u, v in P['edges']:
            if u < v:
                B = [u, v]
                k = 2
                break
        if k < 2:
            B = [0, 1]
            k = 2

    final_ok, final_paths_dict = check_subdivision(B, time_limit=3.0)
    
    if not final_ok:
        # Shrink B until valid
        while len(B) > 2 and not final_ok:
            B.pop()
            final_ok, final_paths_dict = check_subdivision(B, time_limit=2.0)
        if not final_ok:
            for u, v in P['edges']:
                B = [u, v]
                break
            final_ok, final_paths_dict = check_subdivision(B, time_limit=1.0)
            if not final_ok:
                B = [0, 1]
                final_paths_dict = {}
    
    paths_dict = final_paths_dict
    
    paths = []
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            u, v = B[i], B[j]
            if v in adj[u]:
                paths.append([u, v])
            else:
                if (u, v) in paths_dict:
                    paths.append(list(paths_dict[(u, v)]))
                elif (v, u) in paths_dict:
                    paths.append(list(reversed(paths_dict[(v, u)])))
                else:
                    p = get_shortest_path(u, v)
                    if p:
                        paths.append(p)
                    else:
                        paths.append([u, v])
    
    branches_out = [int(b) for b in B]
    paths_out = [[int(x) for x in p] for p in paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
