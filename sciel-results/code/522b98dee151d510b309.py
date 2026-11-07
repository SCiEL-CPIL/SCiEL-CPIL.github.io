"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def time_check():
        return time.time() > deadline
    
    # 1. Find Max Clique using Bron-Kerbosch with pivoting
    max_clique = []
    
    def bron_kerbosch(R, P_set, X):
        nonlocal max_clique
        if time_check():
            return
        if not P_set and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        # Pivot selection
        if P_set or X:
            pivot = max(list(P_set | X), key=lambda v: len(adj[v] & P_set))
            candidates = [v for v in P_set if v not in adj[pivot]]
        else:
            candidates = list(P_set)
            
        for v in candidates:
            bron_kerbosch(R + [v], P_set & adj[v], X & adj[v])
            P_set.discard(v)
            X.add(v)
    
    bron_kerbosch([], set(range(n)), set())
    
    # Base result: max clique with direct edge paths
    if not max_clique:
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    base_clique = [int(v) for v in max_clique]
    base_k = len(base_clique)
    
    def build_direct_paths(clique):
        paths = []
        for u, v in combinations(clique, 2):
            paths.append([int(u), int(v)])
        return paths
    
    best_branches = base_clique
    best_paths = build_direct_paths(best_branches)
    
    # 2. Try to find larger clique subdivisions
    # For k from base_k+1 up to n, try to find k vertices with disjoint paths
    # Use high-degree vertices as candidates first
    
    def find_disjoint_paths_for_branches(B, timeout_remaining=None):
        """Find pairwise disjoint paths between all pairs in B, avoiding B in interiors."""
        B_set = set(B)
        pairs = list(combinations(B, 2))
        used_interiors = set()
        found_paths = {}
        
        # Sort pairs by shortest path length to process easy ones first
        def get_shortest_path_len(u, v, forbidden):
            # BFS for shortest path avoiding forbidden set (except endpoints)
            if u == v:
                return 0
            visited = {u}
            queue = deque([(u, [u])])
            while queue:
                node, path = queue.popleft()
                if node == v:
                    return len(path) - 1
                for neighbor in adj[node]:
                    if neighbor in visited:
                        continue
                    if neighbor in forbidden and neighbor != v:
                        continue
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
            return float('inf')
        
        # Sort pairs by estimated path length
        pair_order = sorted(pairs, key=lambda p: get_shortest_path_len(p[0], p[1], B_set))
        
        for u, v in pair_order:
            if timeout_remaining is not None and time.time() > timeout_remaining:
                return None
            
            # BFS to find path from u to v avoiding used_interiors and B_set (except v)
            visited = {u}
            queue = deque([(u, [u])])
            found = None
            
            while queue:
                node, path = queue.popleft()
                if node == v:
                    internal = path[1:-1]
                    if all(x not in used_interiors and x not in B_set for x in internal):
                        found = path
                        break
                for neighbor in adj[node]:
                    if neighbor in visited:
                        continue
                    if neighbor in B_set and neighbor != v:
                        continue
                    if neighbor in used_interiors:
                        continue
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
            
            if found is None:
                return None
            
            for x in found[1:-1]:
                used_interiors.add(x)
            found_paths[(u, v)] = found
        
        # Build result in consistent order
        res_paths = []
        for u, v in combinations(B, 2):
            if (u, v) in found_paths:
                res_paths.append(found_paths[(u, v)])
            elif (v, u) in found_paths:
                res_paths.append(found_paths[(v, u)][::-1])
            else:
                return None
        return res_paths
    
    # Try increasing k values
    # Candidate selection: prefer high-degree vertices
    degree = {i: len(adj[i]) for i in range(n)}
    sorted_vertices = sorted(range(n), key=lambda v: -degree[v])
    
    # Try k from base_k+1 up to min(n, base_k+5) to limit search
    max_k_to_try = min(n, base_k + 6)
    
    for k in range(base_k + 1, max_k_to_try + 1):
        if time_check():
            break
        
        # Generate candidate sets of size k from high-degree vertices
        # Use combinations but limit the number of candidates tried
        candidates_tried = 0
        max_candidates = 500  # Limit to avoid too much work
        
        for combo in combinations(sorted_vertices, k):
            if time_check() or candidates_tried >= max_candidates:
                break
            candidates_tried += 1
            
            B = list(combo)
            B_set = set(B)
            
            # Quick check: for each pair, there must be a path avoiding B_set
            feasible = True
            for u, v in combinations(B, 2):
                if get_shortest_path_len(u, v, B_set) == float('inf'):
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Try to find disjoint paths
            timeout_rem = start_time + 85.0
            paths = find_disjoint_paths_for_branches(B, timeout_rem)
            
            if paths is not None:
                # Validate
                valid = True
                all_interiors = set()
                for path in paths:
                    for x in path[1:-1]:
                        if x in B_set:
                            valid = False
                            break
                        if x in all_interiors:
                            valid = False
                            break
                        all_interiors.add(x)
                    # Check adjacency
                    for i in range(len(path) - 1):
                        if path[i+1] not in adj[path[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                    # Check endpoints
                    pair = None
                    for a, b in combinations(B, 2):
                        if (path[0] == a and path[-1] == b) or (path[0] == b and path[-1] == a):
                            pair = (a, b)
                            break
                    if pair is None:
                        valid = False
                        break
                if valid:
                    best_branches = [int(v) for v in B]
                    best_paths = paths
                    break  # Found a valid k-subdivision, try larger k
        
        # If we found a valid subdivision of size k, continue to try k+1
    
    # Final result
    if len(best_branches) < 2:
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    return {'branches': best_branches, 'paths': best_paths}

# EVOLVE-BLOCK-END
