"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]

    # 1. Find Max Clique using Bron-Kerbosch
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        u = None
        max_neighbors = -1
        for v in list(P) + list(X):
            count = len(adj_set[v] & P)
            if count > max_neighbors:
                max_neighbors = count
                u = v
        if u is None:
            candidates = list(P)
        else:
            candidates = [v for v in P if v not in adj_set[u]]
        candidates.sort(key=lambda x: degrees[x], reverse=True)
        for v in candidates:
            if time.time() > deadline:
                return
            bron_kerbosch(R | {v}, P & adj_set[v], X & adj_set[v])
            P = P - {v}
            X = X | {v}

    bron_kerbosch(set(), set(range(n)), set())
    
    # 2. Try to find a larger Topological Minor (Clique Subdivision)
    # We try to find K_k subdivision for k > len(best_clique)
    # Start from k = len(best_clique) + 1 up to some limit (e.g., 8)
    # If found, return it.
    
    def find_subdivision(k):
        """
        Try to find a K_k subdivision.
        Returns (branches, paths) if found, else None.
        """
        # Candidates for branches: degree >= k-1
        candidates = [v for v in range(n) if degrees[v] >= k - 1]
        
        # If not enough candidates, return None
        if len(candidates) < k:
            return None
            
        # Iterate through combinations of candidates
        for branches in combinations(candidates, k):
            if time.time() > deadline - 5:
                return None
            b_set = set(branches)
            
            # Check if valid subdivision exists
            # We need to find pairwise internally disjoint paths for all pairs in branches
            # Paths must be in G - B
            # Path for (u,v) connects N(u) to N(v) in G-B
            
            # Pre-compute neighbors in G-B for each branch
            # neighbors_in_G_B[branch] = {x in adj[branch] if x not in B}
            neighbors_in_G_B = {}
            for b in branches:
                neighbors_in_G_B[b] = [x for x in adj[b] if x not in b_set]
                if not neighbors_in_G_B[b]:
                    return None # Can't even start a path
            
            # List of pairs to connect
            pairs = list(combinations(branches, 2))
            
            # Backtracking search for disjoint paths
            # State: index of pair, set of used internal vertices, list of found paths
            # We use a recursive helper
            
            def backtrack(pair_idx, used_vertices, paths_found):
                if time.time() > deadline - 5:
                    return False
                
                if pair_idx == len(pairs):
                    # All paths found
                    # Construct final paths
                    # We need to map back to actual vertex sequences
                    # For now, we just know it's possible. We need to extract the paths.
                    # This requires storing the actual path vertices.
                    # Let's modify backtrack to store paths.
                    return True
                    
                u, v = pairs[pair_idx]
                # Find a path from N(u) to N(v) in G - B - used_vertices
                # Start nodes: neighbors_in_G_B[u] not in used
                # End nodes: neighbors_in_G_B[v] not in used
                
                start_nodes = [x for x in neighbors_in_G_B[u] if x not in used_vertices]
                end_nodes = [x for x in neighbors_in_G_B[v] if x not in used_vertices]
                
                if not start_nodes or not end_nodes:
                    return False
                
                # BFS/DFS to find a path
                # We want to try different paths to allow backtracking
                # For simplicity, we'll find one path, try it, and if it fails later, 
                # we'd need to try another. 
                # Given the small size, we can enumerate simple paths?
                # Or just find one shortest path and hope?
                # Better: Find all simple paths up to a certain length?
                # Let's just find one path using BFS, and if the overall search fails, 
                # we might miss a solution. 
                # To be robust, we should iterate through possible paths.
                
                # Let's use a helper to generate simple paths from start to end in available graph
                # Since n is small, we can do a DFS for paths.
                
                available = set(range(n)) - b_set - used_vertices
                
                def find_path(start, end, current_path, visited):
                    if time.time() > deadline - 5:
                        return []
                    if start == end:
                        return [start]
                    for neighbor in adj[start]:
                        if neighbor in available and neighbor not in visited:
                            visited.add(neighbor)
                            path = find_path(neighbor, end, current_path + [neighbor], visited)
                            if path:
                                return path
                            visited.remove(neighbor)
                    return []

                # To allow backtracking on path choice, we need to yield paths or try multiple.
                # Let's try to find a path, use it, and if backtrack fails, we'd need to try next path.
                # This is complex. 
                # Simplification: Try to find a set of disjoint paths using a flow-like approach or 
                # just greedy with randomization?
                
                # Let's try a simpler check: 
                # Just check if max flow from {N(u)} to {N(v)} is at least 1? 
                # No, we need disjointness across ALL pairs.
                
                # Let's stick to the recursive backtracking but we need to handle path selection.
                # For the first attempt, we will just find ONE path (shortest) and use it.
                # If this fails, we return None for this branch set.
                # This might miss valid subdivisions, but it's a good heuristic.
                
                # Find shortest path
                from collections import deque
                def bfs_shortest_path(start, end, available):
                    if start not in available or end not in available:
                        return None
                    if start == end:
                        return [start]
                    queue = deque()
                    queue.append((start, [start]))
                    visited = {start}
                    while queue:
                        node, path = queue.popleft()
                        for neighbor in adj[node]:
                            if neighbor in available and neighbor not in visited:
                                if neighbor == end:
                                    return path + [neighbor]
                                visited.add(neighbor)
                                queue.append((neighbor, path + [neighbor]))
                    return None

                # We need to try different paths. 
                # Let's just pick the first one found by BFS.
                # To improve, we could shuffle start/end nodes.
                
                found_path = None
                # Try all start-end combinations?
                # For small degree, this is fine.
                for s in start_nodes:
                    for e in end_nodes:
                        p = bfs_shortest_path(s, e, available)
                        if p:
                            found_path = p
                            break
                    if found_path:
                        break
                
                if not found_path:
                    return False
                
                # The internal vertices of the path are p[1:-1]
                internal = set(found_path[1:-1])
                
                # Check if internal vertices conflict with used
                # They are in 'available', so they are not in used_vertices or b_set.
                # But they might conflict with OTHER paths found later.
                # So we add them to used_vertices for the next recursion.
                
                new_used = used_vertices | internal
                new_paths = paths_found + [(u, v, found_path)]
                
                if backtrack(pair_idx + 1, new_used, new_paths):
                    return True
                    
                return False

            # We need to store the result paths if found.
            # Let's modify the structure to return the paths.
            
            result_paths = []
            
            def backtrack_with_result(pair_idx, used_vertices, paths_found):
                if time.time() > deadline - 5:
                    return False
                
                if pair_idx == len(pairs):
                    result_paths.extend(paths_found)
                    return True
                    
                u, v = pairs[pair_idx]
                start_nodes = [x for x in neighbors_in_G_B[u] if x not in used_vertices]
                end_nodes = [x for x in neighbors_in_G_B[v] if x not in used_vertices]
                
                if not start_nodes or not end_nodes:
                    return False
                
                available = set(range(n)) - b_set - used_vertices
                
                from collections import deque
                def bfs_shortest_path(start, end, avail):
                    if start not in avail or end not in avail:
                        return None
                    if start == end:
                        return [start]
                    queue = deque()
                    queue.append((start, [start]))
                    visited = {start}
                    while queue:
                        node, path = queue.popleft()
                        for neighbor in adj[node]:
                            if neighbor in avail and neighbor not in visited:
                                if neighbor == end:
                                    return path + [neighbor]
                                visited.add(neighbor)
                                queue.append((neighbor, path + [neighbor]))
                    return None

                # Try to find a path
                # To allow more flexibility, we can try multiple paths.
                # Here we just try the first one found.
                path_found = None
                for s in start_nodes:
                    for e in end_nodes:
                        p = bfs_shortest_path(s, e, available)
                        if p:
                            path_found = p
                            break
                    if path_found:
                        break
                
                if not path_found:
                    return False
                
                internal = set(path_found[1:-1])
                new_used = used_vertices | internal
                new_paths = paths_found + [(u, v, path_found)]
                
                if backtrack_with_result(pair_idx + 1, new_used, new_paths):
                    return True
                return False

            if backtrack_with_result(0, set(), []):
                # Reconstruct final paths including branches
                final_paths = []
                for u, v, p in result_paths:
                    # p is internal path from neighbor of u to neighbor of v
                    # Full path: u -> p[0] -> ... -> p[-1] -> v
                    full_path = [u] + p + [v]
                    final_paths.append(full_path)
                return (list(branches), final_paths)
                
        return None

    # Try to find larger subdivisions
    current_best_size = len(best_clique)
    found_larger = None
    
    # Max possible size is limited by n
    # We try k from current_best_size + 1 to min(n, 10)
    for k in range(current_best_size + 1, min(n, 10) + 1):
        if time.time() > deadline - 10:
            break
        res = find_subdivision(k)
        if res:
            found_larger = res
            break # Found a valid one, stop searching for larger? 
                   # Actually, we want MAX. So we should continue to see if k+1 is possible?
                   # But find_subdivision is expensive. 
                   # Let's just take the first one found which is > clique.
                   # Or better, loop from high to low?
                   # No, we don't know the max.
                   # Let's assume if we find K_5, it's better than K_4.
                   # If we find K_6, it's better.
                   # So we should try to go as high as possible.
                   # But finding K_6 is harder.
                   # Let's just return the first one found that is > clique.
                   # This is a heuristic.

    if found_larger:
        branches, paths = found_larger
        return {
            'branches': list(branches),
            'paths': [list(p) for p in paths]
        }
    else:
        # Fallback to clique
        branches = best_clique
        paths = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                u = branches[i]
                v = branches[j]
                paths.append([u, v])
        
        return {
            'branches': list(branches),
            'paths': [list(p) for p in paths]
        }

# EVOLVE-BLOCK-END
