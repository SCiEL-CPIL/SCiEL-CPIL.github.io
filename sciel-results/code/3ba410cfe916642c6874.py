import time
from itertools import combinations
import sys
from collections import deque

sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def bron_kerbosch(R, P, X):
        if not P and not X:
            yield R
        P_union_X = P.union(X)
        if not P_union_X:
            return
        u = max(P_union_X, key=lambda x: len(adj[x].intersection(P)))
        for v in list(P - adj[u]):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            yield from bron_kerbosch(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
            
    maximal_cliques = []
    initial_P = set(range(n))
    for clique in bron_kerbosch(set(), initial_P, set()):
        maximal_cliques.append(clique)
        
    maximal_cliques.sort(key=len, reverse=True)
    
    best_branches = []
    best_paths = []
    
    def can_add_branch(v, B, used_internal):
        targets = list(B)
        
        def find_paths(current_targets, current_used, current_paths):
            if not current_targets:
                return True, current_used, current_paths
            
            u = current_targets[0]
            remaining_targets = current_targets[1:]
            
            forbidden = current_used.union(B - {v, u})
            
            # BFS for shortest path
            q = deque()
            q.append((v, [v]))
            visited_bfs = {v}
            found_path = None
            while q:
                curr, path = q.popleft()
                if curr == u:
                    found_path = path
                    break
                for neighbor in adj[curr]:
                    if neighbor in visited_bfs:
                        continue
                    if neighbor in forbidden and neighbor != u:
                        continue
                    if neighbor in B and neighbor != u and neighbor != v:
                        continue
                    visited_bfs.add(neighbor)
                    q.append((neighbor, path + [neighbor]))
            
            if found_path:
                internal = set(found_path[1:-1])
                if internal.isdisjoint(current_used):
                    new_used = current_used.union(internal)
                    new_paths = current_paths + [(u, found_path)]
                    res, new_used, new_paths = find_paths(remaining_targets, new_used, new_paths)
                    if res:
                        return True, new_used, new_paths
            
            return False, None, None
        
        res, new_used, new_paths = find_paths(targets, used_internal, [])
        return res, new_used, new_paths

    for clique in maximal_cliques:
        if time.time() > deadline:
            break
            
        B = set(clique)
        used_internal = set()
        paths = []
        
        valid_clique = True
        for u, v in combinations(clique, 2):
            if v in adj[u]:
                paths.append((u, v, [u, v]))
            else:
                valid_clique = False
                break
        
        if not valid_clique:
            continue
            
        candidates = [v for v in range(n) if v not in B]
        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        
        for v in candidates:
            if time.time() > deadline:
                break
            res, new_used, new_paths = can_add_branch(v, B, used_internal)
            if res:
                B.add(v)
                used_internal = new_used
                for u, path in new_paths:
                    paths.append((u, v, path))
                    
        if len(B) > len(best_branches):
            best_branches = list(B)
            best_paths = paths

    if not best_branches:
        if edges:
            u, v = edges[0]
            return {'branches': [u, v], 'paths': [[u, v]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
            
    final_paths = []
    for u, v, path in best_paths:
        final_paths.append(path)
        
    final_branches = [int(b) for b in best_branches]
    final_paths_list = [list(map(int, p)) for p in final_paths]
    
    return {'branches': final_branches, 'paths': final_paths_list}

# EVOLVE-BLOCK-END
