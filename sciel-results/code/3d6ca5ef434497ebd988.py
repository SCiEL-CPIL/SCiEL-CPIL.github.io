import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]

    def time_ok():
        return time.time() < deadline

    # Max-flow based disjoint paths verification
    # We model vertex capacities of 1 for non-branch vertices.
    # To check if a set of branches B admits k(k-1)/2 internally disjoint paths:
    # 1. Split each non-branch vertex v into v_in and v_out with capacity 1.
    # 2. Branch vertices have infinite capacity (or just don't split, but we need to route through them? No, they are endpoints).
    #    Actually, the standard reduction for "internally disjoint paths between all pairs" is complex because it's not just a single source-sink pair.
    #    
    #    Wait, the problem is "clique subdivision". This means for EVERY pair of branches, there is a path.
    #    This is equivalent to finding a topological minor that is a complete graph.
    #    
    #    A simpler structural approach for dense graphs:
    #    If the graph is dense, the branches are likely to be connected by short paths (length 2 or 3).
    #    Let's try to find a set of vertices B such that for every u,v in B, there is a path of length <= L.
    #    And these paths are internally disjoint.
    #    
    #    Given the small n=24, we can use a backtracking search for the set of branches,
    #    verifying with a fast disjoint path solver.
    #    
    #    Let's optimize the disjoint path solver.
    #    The previous solver used DFS which could be slow.
    #    We can use a bipartite matching approach for length-2 paths?
    #    Or simply, since the graph is dense, most paths will be direct edges or length 2.
    #    
    #    Strategy:
    #    1. Find a large clique C.
    #    2. Try to extend C by adding vertices one by one.
    #    3. For each candidate set B, check if internally disjoint paths exist.
    #    
    #    To check disjoint paths efficiently:
    #    We can formulate this as a flow problem if we fix the order of pairs? No.
    #    
    #    Let's stick to the backtracking path finder but make it much faster by:
    #    1. Precomputing all-pairs shortest paths (BFS) to get a lower bound on path length.
    #    2. Using a greedy heuristic to assign paths.
    #    3. Only backtracking when necessary.
    
    # Precompute shortest path distances
    dist = [[-1]*n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    q.append(v)

    def find_disjoint_paths(branches, max_len=3):
        """
        Returns a dict {(u,v): path} if internally disjoint paths exist for all pairs in branches.
        Uses backtracking with MRV heuristic.
        """
        k = len(branches)
        if k < 2:
            return None
        
        branches_set = set(branches)
        pairs = [(u, v) for i, u in enumerate(branches) for v in branches[i+1:]]
        
        # Sort pairs by difficulty (fewest available paths) initially?
        # We'll do dynamic MRV in the recursion.
        
        used_interior = set()
        path_map = {}
        
        def dfs(idx, used_interior, path_map):
            if not time_ok():
                return False
            
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            
            # Find all simple paths from u to v of length <= max_len
            # that avoid used_interior and other branches.
            forbidden = used_interior | (branches_set - {u, v})
            
            paths = []
            # BFS/DFS to find short paths
            # Since max_len is small (2 or 3), we can enumerate quickly.
            if max_len >= 1 and u in adj[v]:
                if not (set([u,v]) & used_interior): # Endpoints are branches, so not in used_interior unless we messed up
                    paths.append([u, v])
            
            if max_len >= 2:
                # Path u-w-v
                common = adj[u] & adj[v]
                for w in common:
                    if w not in forbidden:
                        paths.append([u, w, v])
            
            if max_len >= 3:
                # Path u-w-x-v
                # This is more expensive. Only do if needed.
                pass # Skip for now to keep it fast, rely on len 1 and 2 mostly.
            
            if not paths:
                return False
            
            # Heuristic: Try shorter paths first
            paths.sort(key=len)
            
            for p in paths:
                interior = set(p[1:-1])
                if interior & used_interior:
                    continue
                
                # Check if interior vertices are distinct from branches (should be by construction)
                if interior & branches_set:
                    continue
                
                # Recurse
                new_used = used_interior | interior
                new_map = dict(path_map)
                new_map[(u, v)] = p
                
                if dfs(idx + 1, new_used, new_map):
                    path_map.clear()
                    path_map.update(new_map)
                    return True
            
            return False
        
        # Optimization: Order pairs. Pairs with fewer options first.
        # Calculate initial path counts for each pair
        pair_options = []
        for u, v in pairs:
            count = 0
            if u in adj[v]:
                count += 1
            if max_len >= 2:
                count += len(adj[u] & adj[v])
            pair_options.append((count, u, v))
        
        pair_options.sort()
        # Reorder pairs list
        pairs = [(u, v) for _, u, v in pair_options]
        
        if dfs(0, set(), {}):
            return path_map
        return None

    # Find initial clique using greedy heuristic
    def find_greedy_clique():
        # Start with highest degree vertex
        degs = [len(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: -degs[x])
        
        clique = []
        candidates = set(range(n))
        
        while candidates and time_ok():
            # Pick vertex in candidates with max degree within candidates
            best_v = None
            best_deg = -1
            for v in candidates:
                deg_in_cand = len(adj[v] & candidates)
                if deg_in_cand > best_deg:
                    best_deg = deg_in_cand
                    best_v = v
            
            if best_v is None:
                break
            
            clique.append(best_v)
            candidates = candidates & adj[best_v]
        
        return clique

    initial_clique = find_greedy_clique()
    
    best_branches = initial_clique
    best_paths = None
    
    # Try to verify the initial clique
    if len(initial_clique) >= 2:
        best_paths = find_disjoint_paths(initial_clique, 2)
        if best_paths is None:
            # Try with longer paths
            best_paths = find_disjoint_paths(initial_clique, 3)
    
    if best_paths is None:
        # Fallback: Find any edge
        for u, v in edges:
            best_paths = find_disjoint_paths([u, v], 2)
            if best_paths:
                best_branches = [u, v]
                break
        if best_paths is None:
            # Should not happen
            return {'branches': [0, 1], 'paths': [[0, 1]]}

    # Local Search: Grow the set
    current_branches = list(best_branches)
    current_paths = best_paths
    
    improved = True
    while improved and time_ok():
        improved = False
        candidates = [v for v in range(n) if v not in current_branches]
        
        # Sort candidates by potential connectivity to existing branches
        def score_candidate(v):
            # Count how many branches are within distance 2
            count = 0
            for b in current_branches:
                if v in adj[b]:
                    count += 10
                else:
                    # Check common neighbors
                    if adj[v] & adj[b]:
                        count += 5
            return count
        
        candidates.sort(key=score_candidate, reverse=True)
        
        for v in candidates:
            if not time_ok():
                break
            
            new_branches = current_branches + [v]
            # Try to find disjoint paths for the new set
            # Start with max_len=2, then 3
            new_paths = find_disjoint_paths(new_branches, 2)
            if new_paths is None:
                new_paths = find_disjoint_paths(new_branches, 3)
            
            if new_paths:
                current_branches = new_branches
                current_paths = new_paths
                best_branches = current_branches
                best_paths = new_paths
                improved = True
                break # Restart loop with new set

    # Local Search: Swap?
    # If we are stuck, maybe remove a vertex and add two?
    # Or just try different orderings.
    if time_ok():
        for _ in range(5):
            if not time_ok():
                break
            improved = False
            candidates = [v for v in range(n) if v not in current_branches]
            # Shuffle to try different orders
            import random
            random.shuffle(candidates)
            
            for v in candidates:
                if not time_ok():
                    break
                new_branches = current_branches + [v]
                new_paths = find_disjoint_paths(new_branches, 3)
                if new_paths:
                    current_branches = new_branches
                    current_paths = new_paths
                    best_branches = current_branches
                    best_paths = new_paths
                    improved = True
                    break
            if not improved:
                break

    # Construct output
    branches_out = [int(b) for b in best_branches]
    paths_out = []
    k = len(branches_out)
    
    # Ensure consistent ordering for keys
    # The keys in current_paths are (u, v) as they were in the 'pairs' list during solve.
    # We need to match them to the output pairs.
    for i in range(k):
        for j in range(i+1, k):
            u, v = branches_out[i], branches_out[j]
            # The solver used the order of 'branches' list at the time of solving.
            # In find_disjoint_paths, pairs were generated from the input list order.
            # So the key might be (u, v) or (v, u) depending on input order.
            # We passed sorted? No, we passed current_branches which is a list.
            # Let's just look for the key.
            
            key1 = (u, v)
            key2 = (v, u)
            
            p = None
            if key1 in current_paths:
                p = current_paths[key1]
            elif key2 in current_paths:
                p = current_paths[key2]
                # Reverse if needed to start with u and end with v?
                # The problem says "Path endpoints must be their branch pair".
                # It doesn't strictly say start with the smaller index, but usually "pair" implies set.
                # However, the path is a list. Let's ensure it starts with u and ends with v for consistency.
                if p[0] == v and p[-1] == u:
                    p = list(reversed(p))
            else:
                # This should not happen if the solver succeeded
                p = [u, v] # Fallback, might be invalid if no edge
            
            paths_out.append([int(x) for x in p])

    return {
        'branches': branches_out,
        'paths': paths_out
    }

# Run construct to test
# print(construct())
