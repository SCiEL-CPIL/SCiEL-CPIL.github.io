import time
import itertools
import random
from collections import deque

def construct():
    P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 85.0

    def time_left():
        return deadline - time.time()

    def find_paths_between(u, v, branch_set, used_internal, max_paths=50):
        """
        Find simple paths from u to v that do not use other branches 
        and do not use vertices in used_internal.
        """
        paths = []
        def dfs(path, visited):
            if len(paths) >= max_paths:
                return
            curr = path[-1]
            if curr == v:
                if len(path) > 1:
                    internal = set(path[1:-1])
                    if not (internal & used_internal) and not (internal & branch_set):
                        paths.append(list(path))
                return
            
            if len(path) - 1 >= 10: # Depth limit
                return
            
            for nb in sorted(adj[curr]):
                if nb in visited:
                    continue
                if nb in branch_set and nb != v:
                    continue
                if nb in used_internal:
                    continue
                
                visited.add(nb)
                dfs(path + [nb], visited)
                visited.remove(nb)
        
        dfs([u], {u})
        return paths

    def check_subdivision(branches, node_limit=10000):
        if len(branches) < 2:
            return None
        
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        used_internal = set()
        assignment = {}
        nodes_visited = 0
        
        def get_candidates(u, v):
            return find_paths_between(u, v, branch_set, used_internal, max_paths=20)
        
        def backtrack(remaining_pairs):
            nonlocal nodes_visited
            if time.time() > deadline:
                return False
            if nodes_visited > node_limit:
                return False
            nodes_visited += 1
            
            if not remaining_pairs:
                return True
            
            u, v = remaining_pairs[0]
            rest = remaining_pairs[1:]
            
            candidates = get_candidates(u, v)
            candidates.sort(key=lambda x: len(x))
            
            for path in candidates:
                internal = set(path[1:-1])
                assignment[(u, v)] = path
                used_internal.update(internal)
                
                if backtrack(rest):
                    return True
                
                used_internal.difference_update(internal)
                del assignment[(u, v)]
            
            return False
        
        if backtrack(pairs):
            return [assignment[p] for p in pairs]
        
        return None

    vertices = list(range(n))
    vertices.sort(key=lambda v: -len(adj[v]))
    
    best_branches = None
    best_paths = None
    best_size = 0
    
    for target_size in [5, 4, 3, 2]:
        if time.time() > deadline - 5:
            break
            
        found = False
        pool = vertices[:15]
        
        current_branches = pool[:target_size]
        
        paths = check_subdivision(current_branches)
        if paths:
            best_branches = current_branches
            best_paths = paths
            best_size = target_size
            found = True
            break
        
        num_tries = 100
        for _ in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            rand_branches = random.sample(pool, target_size)
            paths = check_subdivision(rand_branches)
            if paths:
                best_branches = rand_branches
                best_paths = paths
                best_size = target_size
                found = True
                break
        
        if found:
            break
        
    if best_size == 0:
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    best_branches = [u, v]
                    best_paths = [[u, v]]
                    best_size = 2
                    break
            if best_size > 0:
                break

    if best_size >= 2 and best_paths is not None:
        branches = [int(b) for b in best_branches]
        formatted_paths = []
        for p in best_paths:
            formatted_paths.append([int(x) for x in p])
        
        return {'branches': branches, 'paths': formatted_paths}
    
    if P['edges']:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    else:
        return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
