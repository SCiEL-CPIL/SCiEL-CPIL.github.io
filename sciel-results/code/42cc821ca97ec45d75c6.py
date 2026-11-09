import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    # Find max clique for initial best
    def find_max_clique():
        # Simple Bron-Kerbosch variant or greedy for small n
        # Since n=24, we can use a recursive approach
        def expand(clique, candidates):
            if not candidates:
                return clique
            # Heuristic: pick vertex with max degree in candidates
            v = max(candidates, key=lambda x: len(adj[x] & candidates))
            res1 = expand(clique + [v], candidates & adj[v])
            res2 = expand(clique, candidates - {v})
            return res1 if len(res1) > len(res2) else res2
        return expand([], set(range(n)))

    best_clique = find_max_clique()
    best_k = len(best_clique)
    best_result = {'branches': best_clique, 'paths': [[best_clique[i], best_clique[j]] for i in range(len(best_clique)) for j in range(i+1, len(best_clique))]}

    # Function to check if a set of branches forms a valid subdivision
    def check_subdivision(branches):
        k = len(branches)
        if k < 2:
            return False, None
        
        branch_set = set(branches)
        # Pairs to connect
        pairs = [(branches[i], branches[j]) for i in range(k) for j in range(i+1, k)]
        
        # Sort pairs by "difficulty" - e.g., number of common neighbors or path length
        # For now, just process in order.
        
        used_internal = set()
        paths = {}
        
        def backtrack(idx):
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            # Find a path from u to v avoiding used_internal and branch_set (except u,v)
            forbidden = used_internal.union(branch_set - {u, v})
            
            # BFS to find shortest path
            visited = {u}
            queue = deque([(u, [u])])
            found_path = None
            
            while queue:
                node, path = queue.popleft()
                if node == v:
                    found_path = path
                    break
                for neighbor in adj[node]:
                    if neighbor == v:
                        found_path = path + [v]
                        break
                    if neighbor not in visited and neighbor not in forbidden:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
                if found_path:
                    break
            
            if found_path is None:
                return False
            
            internal_nodes = set(found_path[1:-1])
            if internal_nodes.isdisjoint(used_internal):
                new_used = used_internal.union(internal_nodes)
                paths[(u, v)] = found_path
                if backtrack(idx + 1):
                    return True
                del paths[(u, v)]
            
            return False

        if backtrack(0):
            # Construct result
            res_paths = []
            for u, v in pairs:
                if (u, v) in paths:
                    res_paths.append(paths[(u, v)])
                else:
                    # Should be an edge if not in paths dict? 
                    # Wait, the backtrack assigns ALL pairs.
                    # If (u,v) is an edge, BFS returns [u,v] immediately.
                    # So it should be in paths.
                    res_paths.append(paths[(u, v)])
            
            # Verify disjointness one last time (safety check)
            all_internals = set()
            for p in res_paths:
                internals = set(p[1:-1])
                if all_internals.intersection(internals):
                    return False, None
                all_internals.update(internals)
                
            return True, {'branches': list(branches), 'paths': res_paths}
        else:
            return False, None

    # Search for larger k
    # Pool of candidate vertices: top degree vertices
    degrees = [len(adj[i]) for i in range(n)]
    candidates = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try k from best_k + 1 up to 12
    for k in range(best_k + 1, 13):
        if time.time() > deadline:
            break
            
        # Generate combinations from the top candidates
        # To avoid combinatorial explosion, limit the pool size
        pool_size = min(n, 15) 
        pool = candidates[:pool_size]
        
        # If k > pool_size, we must include some lower degree vertices
        # But let's start with the pool
        
        # We can use a randomized approach or systematic
        # Systematic: combinations of pool of size k
        # If pool is 15 and k=10, C(15,10) = 3003. Manageable.
        # If k=12, C(15,12) = 455.
        
        found = False
        for comb in combinations(pool, k):
            if time.time() > deadline:
                break
                
            # Quick check: internal edges count
            # Not strictly necessary for subdivision, but helps prune
            valid, res = check_subdivision(list(comb))
            if valid:
                best_k = k
                best_result = res
                found = True
                # We can break if we found one for this k, as we are maximizing k
                # But maybe there's a better one? No, k is the value.
                # We want the max k. So if we find one, we can move to k+1.
                break
        
        # If we didn't find in the top pool, try a wider pool or random samples
        if not found and pool_size < n:
            # Try random samples from all n vertices
            import random
            random.seed(42) # Deterministic for reproducibility in tests, but here we just need one good one
            attempts = 0
            while attempts < 50 and time.time() < deadline:
                sample = random.sample(range(n), k)
                valid, res = check_subdivision(sample)
                if valid:
                    best_k = k
                    best_result = res
                    found = True
                    break
                attempts += 1

    # Ensure types are correct
    if best_result:
        best_result['branches'] = list(best_result['branches'])
        best_result['paths'] = [list(p) for p in best_result['paths']]
    else:
        # Fallback
        best_result = {'branches': [0, 1], 'paths': [[0, 1]]}

    return best_result
