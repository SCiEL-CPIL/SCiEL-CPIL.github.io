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

    def get_shortest_path(u, v, forbidden):
        """Find shortest path from u to v avoiding forbidden vertices (except u, v)."""
        if u == v:
            return [u]
        if (u, v) in edge_set:
            return [u, v]
        visited = {u}
        queue = deque([(u, [u])])
        while queue:
            node, path = queue.popleft()
            for neighbor in adj[node]:
                if neighbor == v:
                    return path + [v]
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def get_paths_kshortest(u, v, forbidden, limit=30):
        """Find up to 'limit' shortest paths from u to v avoiding forbidden."""
        if u == v:
            return [[u]]
        if (u, v) in edge_set:
            return [[u, v]]
        
        paths = []
        visited = {u}
        queue = deque([(u, [u])])
        while queue and len(paths) < limit:
            node, path = queue.popleft()
            for neighbor in adj[node]:
                if neighbor == v:
                    paths.append(path + [v])
                    if len(paths) >= limit:
                        return paths
                    continue
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return paths

    def try_construct_subdivision(branches):
        """Try to find internally disjoint paths for all pairs in branches using backtracking."""
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        used_internal = set()
        paths = {}
        
        # Identify missing edges
        missing_edges = []
        for i in range(k):
            for j in range(i+1, k):
                u, v = branches[i], branches[j]
                if (u, v) not in edge_set:
                    missing_edges.append((u, v))
        
        # Pre-compute candidate paths for each missing edge
        # This helps with backtracking as we don't recompute BFS every time
        candidate_paths = {}
        for u, v in missing_edges:
            forbidden = branch_set
            # For initial candidate generation, we allow using all non-branch vertices
            # The backtracking will handle the 'used_internal' constraint
            candidate_paths[(u, v)] = get_paths_kshortest(u, v, forbidden, limit=50)
            candidate_paths[(v, u)] = candidate_paths[(u, v)]
            
        # Sort missing edges by number of available candidate paths (ascending)
        # This is a heuristic: constrain the most limited pairs first
        missing_edges_sorted = sorted(missing_edges, key=lambda p: len(candidate_paths[p]))
        
        def backtrack(idx, used_internal, paths):
            if idx == len(missing_edges_sorted):
                return True
            
            u, v = missing_edges_sorted[idx]
            # Try each candidate path
            for path in candidate_paths[(u, v)]:
                internal_nodes = set(path[1:-1])
                if internal_nodes.isdisjoint(used_internal):
                    new_used = used_internal.union(internal_nodes)
                    paths[(u, v)] = path
                    paths[(v, u)] = path[::-1]
                    
                    if backtrack(idx + 1, new_used, paths):
                        return True
                    
                    # Backtrack
                    del paths[(u, v)]
                    del paths[(v, u)]
            
            return False

        if backtrack(0, used_internal, paths):
            result_paths = []
            for i in range(k):
                for j in range(i+1, k):
                    u, v = branches[i], branches[j]
                    if (u, v) in edge_set:
                        result_paths.append([u, v])
                    else:
                        if (u, v) in paths:
                            result_paths.append(paths[(u, v)])
                        elif (v, u) in paths:
                            result_paths.append(paths[(v, u)])
                        else:
                            return None
            return {'branches': list(branches), 'paths': result_paths}
        else:
            return None

    # 1. Find Max Clique using Bron-Kerbosch with pivot
    def find_max_clique():
        def expand(clique, candidates):
            if not candidates:
                return clique
            # Pivot heuristic
            pivot = max(candidates, key=lambda v: len(adj[v] & candidates))
            candidates_to_try = candidates - adj[pivot]
            
            best = clique
            for v in candidates_to_try:
                new_clique = expand(clique + [v], candidates & adj[v])
                if len(new_clique) > len(best):
                    best = new_clique
            return best
        return expand([], set(range(n)))

    best_clique = find_max_clique()
    best_result = None
    if len(best_clique) >= 2:
        branches = best_clique
        paths = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                paths.append([branches[i], branches[j]])
        best_result = {'branches': branches, 'paths': paths}
    
    # 2. Try to find larger subdivisions
    # Start from len(best_clique) + 1 up to 10
    start_k = len(best_clique) + 1
    if start_k < 3: start_k = 3
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Pre-compute degrees for sorting candidates
    degrees = [len(adj[i]) for i in range(n)]
    
    for k in range(start_k, 11):
        if time.time() > deadline:
            break
            
        req_edges = k + k*(k-1)//2 - n
        
        # Generate combinations, prioritizing high-degree vertices
        # Simple approach: iterate all, but skip if not enough edges
        for comb in combinations(range(n), k):
            if time.time() > deadline:
                break
                
            internal_edges = 0
            for i in range(k):
                for j in range(i+1, k):
                    if (comb[i], comb[j]) in edge_set:
                        internal_edges += 1
            
            if internal_edges < req_edges:
                continue
            
            res = try_construct_subdivision(list(comb))
            if res:
                if best_result is None or k > len(best_result['branches']):
                    best_result = res
        
    if best_result is None:
        u, v = edges[0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
        
    best_result['branches'] = list(best_result['branches'])
    best_result['paths'] = [list(p) for p in best_result['paths']]
    
    return best_result

# EVOLVE-BLOCK-END
