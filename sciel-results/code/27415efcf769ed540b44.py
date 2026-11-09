import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    n = P['n']
    edges = P['edges']
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all cliques of size 3
    cliques_3 = []
    for i, j, k in combinations(range(n), 3):
        if j in adj[i] and k in adj[i] and k in adj[j]:
            cliques_3.append((i, j, k))
            
    # Fallback result: best clique found
    best_branches = cliques_3[0] if cliques_3 else (0, 1, 2)
    best_paths = None
    
    def make_paths_from_clique(clique):
        return {tuple(sorted(pair)): list(pair) for pair in combinations(clique, 2)}
    
    # Initial best is a clique
    if cliques_3:
        best_paths = make_paths_from_clique(best_branches)
    else:
        # Should not happen for this graph, but safe fallback
        u, v = edges[0]
        best_branches = (u, v)
        best_paths = {(u, v): [u, v]}

    # Try to find a K4 subdivision
    # We will iterate over all 4-subsets
    # To optimize, we can prioritize 4-subsets that contain a triangle
    # Or just all 4-subsets if fast enough
    
    def check_k4(b1, b2, b3, b4):
        branches = (b1, b2, b3, b4)
        pairs = list(combinations(branches, 2))
        
        # Order pairs by "difficulty" (heuristic: fewer neighbors available?)
        # For now, fixed order
        # We use backtracking to find disjoint paths
        
        used_internal = set()
        paths_found = {}
        
        def find_path(u, v, forbidden):
            # BFS for shortest paths, return list of a few shortest paths
            # forbidden: set of vertices that cannot be used (internal vertices of other paths + branches)
            # Note: branches are allowed as endpoints but not as internal vertices.
            # The 'forbidden' set should contain all branches and all used internal vertices.
            
            from collections import deque
            q = deque()
            q.append((u, [u]))
            visited = {u}
            paths = []
            limit = 5  # Max paths to consider
            max_len = 6  # Max path length (edges)
            
            while q and len(paths) < limit:
                node, path = q.popleft()
                if len(path) > max_len:
                    continue
                if node == v:
                    paths.append(path)
                    continue
                for nb in adj[node]:
                    if nb in forbidden:
                        continue
                    if nb in visited:
                        continue
                    visited.add(nb)
                    q.append((nb, path + [nb]))
            return paths

        def backtrack(idx):
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            # Forbidden set: all branches + all used internal vertices
            forbidden = set(branches) | used_internal
            
            candidate_paths = find_path(u, v, forbidden)
            
            # Shuffle to add randomness? Or sort by length?
            # Sort by length
            candidate_paths.sort(key=len)
            
            for p in candidate_paths:
                # p is a list of vertices [u, ..., v]
                internal = set(p[1:-1])
                if internal & used_internal:
                    continue
                
                # Try this path
                used_internal.update(internal)
                paths_found[(u, v)] = p
                if backtrack(idx + 1):
                    return True
                
                # Undo
                used_internal.difference_update(internal)
                del paths_found[(u, v)]
                
            return False

        if backtrack(0):
            return branches, paths_found
        return None, None

    # Iterate through 4-subsets
    # We can prioritize subsets that contain a triangle to save time?
    # Let's just iterate all 4-subsets. 10626 is small.
    found_k4 = False
    for quad in combinations(range(n), 4):
        if time.time() > deadline:
            break
        res_branches, res_paths = check_k4(*quad)
        if res_branches is not None:
            found_k4 = True
            best_branches = res_branches
            best_paths = res_paths
            break  # Found one, try to find larger? Or just return?
            # Let's try to find K5 if time allows, but K4 is a big jump.
            # We'll continue searching for K5 in a separate phase if time permits.
            
    if found_k4:
        # Try to find K5?
        # We can try to expand the found K4 by adding a vertex
        # Or just search all 5-subsets
        # Given time constraints, K4 is a solid improvement (4 vs 3).
        # Let's see if we can quickly check some 5-subsets.
        pass

    # Prepare output
    # Convert to required format
    out_branches = list(best_branches)
    out_paths = []
    for pair in combinations(out_branches, 2):
        # Ensure key matches
        key = tuple(sorted(pair))
        # The path in best_paths is stored with sorted key?
        # In check_k4, keys are from combinations, which are sorted if input is sorted.
        # But in make_paths_from_clique, keys are sorted.
        # In check_k4, pairs are from combinations(branches, 2). If branches is sorted, pairs are sorted.
        # We passed sorted quads from combinations(range(n), 4), so branches are sorted.
        # So keys are sorted tuples.
        path = best_paths[key]
        out_paths.append(list(path))
        
    return {'branches': out_branches, 'paths': out_paths}

# EVOLVE-BLOCK-END
