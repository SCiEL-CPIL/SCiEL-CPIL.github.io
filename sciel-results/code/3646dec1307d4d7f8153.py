"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations, permutations
    import random

    t_start = time.time()
    DEADLINE = 85.0
    random.seed(42)

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_path_bfs(src, dst, forbidden_interior):
        """BFS from src to dst, interior vertices avoid forbidden_interior."""
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            if cur == dst:
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in sorted(adj[cur]):
                if nb not in parent:
                    if nb != dst and nb in forbidden_interior:
                        continue
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def try_add_vertex(branches, paths_dict, new_v):
        """
        Try to add new_v to the clique subdivision defined by branches and paths_dict.
        Returns (new_branches, new_paths_dict) or None.
        """
        if time_up():
            return None
            
        bset = set(branches)
        # Forbidden interior vertices: all interior vertices of existing paths + all branch vertices except new_v
        # Note: new_v is not in bset yet.
        forbidden = set()
        for p in paths_dict.values():
            forbidden.update(p[1:-1])
        forbidden.update(bset) # All existing branches are forbidden as interior for new paths
        
        # We need to find paths from new_v to each b in branches.
        # These paths must be internally disjoint from each other and from 'forbidden'.
        # Let's try to find them one by one.
        # The order of targets might matter. Try all permutations if size is small, else heuristic.
        targets = list(branches)
        
        # Heuristic: sort targets by distance from new_v? Or just try a few random orders.
        # Since k is small, let's try a few specific orders:
        # 1. Original order
        # 2. Reverse
        # 3. Random shuffles
        
        orders_to_try = [targets, list(reversed(targets))]
        for _ in range(5):
            shuffled = targets[:]
            random.shuffle(shuffled)
            orders_to_try.append(shuffled)
            
        # Also try sorting by degree of target (lower degree first might be easier to reach without conflict?)
        deg = {v: len(adj[v]) for v in targets}
        orders_to_try.append(sorted(targets, key=lambda x: deg[x]))
        orders_to_try.append(sorted(targets, key=lambda x: -deg[x]))

        for order in orders_to_try:
            if time_up():
                return None
                
            new_paths = dict(paths_dict)
            used_interiors_new = set(forbidden)
            success = True
            temp_paths = {}
            
            for target in order:
                # Find path from new_v to target
                # Forbidden interior: used_interiors_new
                # Note: new_v is allowed as endpoint. target is allowed as endpoint.
                # Other branches are in used_interiors_new (from 'forbidden' init) so they are blocked as interior.
                
                # We need to ensure the path doesn't use any interior vertex already in used_interiors_new.
                # Also, the path must not use any OTHER branch as interior.
                
                # Simple BFS might not be enough if we need to reserve vertices for other targets.
                # But for small k, greedy BFS often works. If it fails, we might need backtracking.
                # Let's start with greedy BFS.
                
                path = find_path_bfs(new_v, target, used_interiors_new)
                if path is None:
                    success = False
                    break
                
                interior = set(path[1:-1])
                # Check if interior intersects with existing forbidden or other new paths
                if interior & used_interiors_new:
                    success = False
                    break
                
                # Update used_interiors_new with this path's interior
                used_interiors_new |= interior
                temp_paths[(new_v, target)] = path
                temp_paths[(target, new_v)] = path # Symmetry for lookup if needed, but we only store once per pair

            if success:
                # We found paths for all targets.
                # Add new_v to branches
                new_branches = branches + [new_v]
                # Update paths_dict
                for target in order:
                    new_paths[(new_v, target)] = temp_paths[(new_v, target)]
                    new_paths[(target, new_v)] = temp_paths[(new_v, target)]
                
                return new_branches, new_paths

        return None

    def greedy_grow(initial_branches, initial_paths):
        """
        Start with initial_branches and initial_paths, try to add vertices one by one.
        """
        branches = list(initial_branches)
        paths = dict(initial_paths)
        
        # Candidates: all vertices not in branches
        candidates = [v for v in range(n) if v not in branches]
        
        # Sort candidates by degree (high degree first? or low?)
        # High degree might offer more options.
        candidates.sort(key=lambda v: -len(adj[v]))
        
        improved = True
        while improved:
            improved = False
            # Try to add each candidate
            # We should try to add the "best" candidate first.
            # Let's iterate through candidates and try to add the first one that works.
            # Or try all and pick the one that allows further growth? Too expensive.
            # Just try in order.
            
            new_candidates = []
            for v in candidates:
                if time_up():
                    break
                if v in branches:
                    continue
                res = try_add_vertex(branches, paths, v)
                if res is not None:
                    branches, paths = res
                    improved = True
                    # Restart candidate list? Or just remove v?
                    # If we add v, the structure changes.
                    # Let's reset candidates to all non-branches.
                    candidates = [x for x in range(n) if x not in branches]
                    candidates.sort(key=lambda x: -len(adj[x]))
                    break # Restart loop with new state
            
        return branches, paths

    # Find initial seeds
    # 1. Find a triangle
    def find_triangle():
        for a, b, c in combinations(range(n), 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                return [a, b, c], {(a,b):[a,b], (a,c):[a,c], (b,c):[b,c]}
        return None

    # 2. Find an edge
    def find_edge():
        for u, v in edges:
            return [u, v], {(u,v):[u,v]}
        return None

    best_size = 0
    best_result = None

    # Try multiple random restarts of the greedy process
    # Different initial seeds and different random shuffles in try_add_vertex
    
    num_restarts = 20
    for r in range(num_restarts):
        if time_up():
            break
        
        # Choose an initial seed
        # Mix of triangles and edges
        if r % 2 == 0:
            seed = find_triangle()
        else:
            seed = find_edge()
            
        if seed is None:
            continue
            
        init_b, init_p = seed
        
        # Randomize the order of vertices in the seed?
        # The seed is small, so it doesn't matter much.
        
        # Run greedy growth
        # We can also randomize the candidate order in greedy_grow
        # But greedy_grow uses degree. Let's add some randomness there.
        
        # To incorporate randomness, we can shuffle the candidates in greedy_grow
        # Let's modify greedy_grow to accept a random seed or use global random
        # The current greedy_grow sorts by degree. 
        # Let's make a version that shuffles candidates occasionally.
        
        # For now, just run it.
        # To improve, we can try different initial seeds (different triangles).
        
        # Let's find a few distinct triangles
        # For simplicity, just use the first one found, but maybe vary the starting point?
        # Actually, find_triangle returns the first one. 
        # Let's collect a few triangles.
        
        # Optimization: Instead of just one triangle, try all triangles?
        # There are at most ~24 choose 3 = 2024 triangles.
        # Greedy growth is fast.
        
        # Let's collect up to 10 triangles
        triangles = []
        for a, b, c in combinations(range(n), 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                triangles.append(([a, b, c], {(a,b):[a,b], (a,c):[a,c], (b,c):[b,c]}))
                if len(triangles) >= 10:
                    break
                    
        if not triangles:
            # Fallback to edge
            init_b, init_p = find_edge()
            if init_b is None:
                break
            b, p = greedy_grow(init_b, init_p)
            if len(b) > best_size:
                best_size = len(b)
                best_result = {'branches': b, 'paths': p}
            continue

        # Try growing from each triangle
        for tb, tp in triangles:
            if time_up():
                break
            b, p = greedy_grow(tb, tp)
            if len(b) > best_size:
                best_size = len(b)
                best_result = {'branches': b, 'paths': p}
                # If we found a very large one, we might stop early?
                # Max possible is n.
                if best_size == n:
                    break

    # If we haven't found anything, fallback to single edge
    if best_result is None:
        init_b, init_p = find_edge()
        if init_b is not None:
            best_result = {'branches': init_b, 'paths': init_p}
        else:
            # Should not happen for connected graph
            best_result = {'branches': [0], 'paths': []} # Invalid, but safe
            # Actually, task requires at least 2 branches.
            # If no edge, graph is empty? P has edges.
            u, v = edges[0]
            best_result = {'branches': [u, v], 'paths': [[u, v]]}
            # Wait, paths dict in my code is {pair: path}. The return format is list of paths.
            # I need to convert best_result to the required format at the end.

    # Convert best_result to required format
    # best_result['branches'] is list of ints
    # best_result['paths'] is dict {(u,v): path}
    # Required: 'paths': list of paths for each unordered pair.
    
    branches = [int(x) for x in best_result['branches']]
    paths_list = []
    for u, v in combinations(branches, 2):
        # The path might be stored as (u,v) or (v,u)
        if (u, v) in best_result['paths']:
            path = best_result['paths'][(u, v)]
        else:
            path = best_result['paths'][(v, u)]
        paths_list.append([int(x) for x in path])
        
    return {'branches': branches, 'paths': paths_list}
