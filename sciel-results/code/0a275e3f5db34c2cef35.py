import time
import sys

# Increase recursion limit for DFS
sys.setrecursionlimit(20000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    # Precompute degrees
    deg = [0] * n
    for i in range(n):
        deg[i] = bin(adj[i]).count('1')
        
    ALL = (1 << n) - 1
    deadline = time.time() + 85.0 # 5s safety margin

    # --- 1. Find Connected Components ---
    comp_of = [-1] * n
    comps = []
    for s in range(n):
        if comp_of[s] != -1:
            continue
        cid = len(comps)
        mask = 0
        stack = [s]
        comp_of[s] = cid
        mask |= 1 << s
        while stack:
            x = stack.pop()
            nb = adj[x]
            while nb:
                lsb = nb & (-nb)
                nb ^= lsb
                y = lsb.bit_length() - 1
                if comp_of[y] == -1:
                    comp_of[y] = cid
                    mask |= lsb
                    stack.append(y)
        comps.append(mask)
    
    # Sort components by size descending
    comps.sort(key=lambda m: bin(m).count('1'), reverse=True)
    
    best_len = 0
    best_path = []

    # --- 2. Greedy Initialization (Warnsdorff) ---
    def greedy_path(start, target_len=None):
        path = [start]
        seen = 1 << start
        while True:
            cand = adj[path[-1]] & ~seen
            if not cand:
                break
            # Choose neighbor with fewest unvisited neighbors (Warnsdorff)
            best_u = -1
            best_score = float('inf')
            m = cand
            while m:
                lsb = m & (-m)
                m ^= lsb
                u = lsb.bit_length() - 1
                # Score: number of unvisited neighbors of u
                score = bin(adj[u] & ~seen & ~(1 << u)).count('1')
                if score < best_score:
                    best_score = score
                    best_u = u
            path.append(best_u)
            seen |= 1 << best_u
            if target_len and len(path) >= target_len:
                break
        return path

    # Try greedy from all vertices in the largest component
    if comps:
        main_comp = comps[0]
        verts_in_comp = []
        m = main_comp
        while m:
            lsb = m & (-m)
            m ^= lsb
            verts_in_comp.append(lsb.bit_length() - 1)
        
        # Sort by degree ascending (likely endpoints)
        verts_in_comp.sort(key=lambda x: deg[x])
        
        for s in verts_in_comp:
            p = greedy_path(s)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
            if time.time() > deadline:
                return best_path if best_path else [0]

    # --- 3. Local Improvement ---
    def try_insert(path, seen_mask):
        """Try to insert an unvisited vertex into the path."""
        improved = False
        # Try inserting at each position
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            # Find unvisited vertex w adjacent to both u and v
            cand = adj[u] & adj[v] & ~seen_mask
            if cand:
                # Pick one (e.g., first one or one with min degree)
                lsb = cand & (-cand)
                w = lsb.bit_length() - 1
                # Check if inserting w breaks anything? No, w is unvisited.
                # Just insert
                new_path = path[:i+1] + [w] + path[i+1:]
                return new_path, seen_mask | (1 << w), True
        return path, seen_mask, False

    def local_improve(path):
        if not path:
            return path
        seen_mask = 0
        for v in path:
            seen_mask |= 1 << v
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to extend ends
            # Extend right
            cand = adj[path[-1]] & ~seen_mask
            if cand:
                best_u = -1
                best_score = float('inf')
                m = cand
                while m:
                    lsb = m & (-m)
                    m ^= lsb
                    u = lsb.bit_length() - 1
                    score = bin(adj[u] & ~seen_mask & ~(1 << u)).count('1')
                    if score < best_score:
                        best_score = score
                        best_u = u
                if best_u != -1:
                    path.append(best_u)
                    seen_mask |= 1 << best_u
                    improved = True
                    continue
            
            # Extend left
            cand = adj[path[0]] & ~seen_mask
            if cand:
                best_u = -1
                best_score = float('inf')
                m = cand
                while m:
                    lsb = m & (-m)
                    m ^= lsb
                    u = lsb.bit_length() - 1
                    score = bin(adj[u] & ~seen_mask & ~(1 << u)).count('1')
                    if score < best_score:
                        best_score = score
                        best_u = u
                if best_u != -1:
                    path.insert(0, best_u)
                    seen_mask |= 1 << best_u
                    improved = True
                    continue

            # 2. Try to insert a vertex
            new_path, new_seen, ins_ok = try_insert(path, seen_mask)
            if ins_ok:
                path = new_path
                seen_mask = new_seen
                improved = True
                continue

            # 3. Try 2-opt (reverse segment) to change endpoints
            # This is expensive, so do it sparingly or only if stuck
            if len(path) > 2:
                # Try reversing a segment to see if it allows extension
                # Heuristic: try reversing segments that include the current "stuck" end
                # For simplicity, try a few random or specific reversals
                # Let's try reversing the whole path (symmetric, doesn't help extension unless we track direction)
                # Actually, reversing path doesn't change set of edges, just order.
                # It changes which vertex is at the end.
                # If we are stuck at the right end, reversing makes the old left end the new right end.
                # If the old left end has neighbors, we can extend.
                
                # Check if reversing helps
                # We don't need to actually reverse if we just check adj[path[0]]
                # We already checked adj[path[0]] in "Extend left".
                # So if we couldn't extend left or right, reversing doesn't immediately help extension.
                # However, it might help *insertion* if the structure is different? No, insertion depends on internal edges.
                
                # Let's try a "shaking" move: pick a random internal vertex, remove it, try to re-insert elsewhere?
                # Or just break if we can't improve.
                pass

        return path

    # Apply local improvement to the best greedy path
    if best_path:
        improved_path = local_improve(list(best_path))
        if len(improved_path) > best_len:
            best_len = len(improved_path)
            best_path = improved_path

    # --- 4. Branch and Bound ---
    # Use the largest component for DFS
    if comps:
        main_comp = comps[0]
        comp_size = bin(main_comp).count('1')
        
        # If best_len is already close to comp_size, we might skip DFS or prune heavily
        # But let's try DFS from the best path's start or a low-degree vertex
        
        # Reconstruct adjacency for the component only? 
        # The bitmasks already handle this implicitly if we restrict search to main_comp.
        
        # We will run DFS starting from the endpoints of the best_path found so far, 
        # or from vertices in the component that are not in best_path?
        # No, DFS builds a path from scratch. 
        # We can use the best_path as a lower bound for pruning.
        
        # To make DFS effective, we should start from vertices that are likely to be part of the longest path.
        # Start from the vertices of the best_path? Or just low degree vertices.
        
        start_verts = []
        m = main_comp
        while m:
            lsb = m & (-m)
            m ^= lsb
            v = lsb.bit_length() - 1
            start_verts.append(v)
            
        # Sort start vertices by degree
        start_verts.sort(key=lambda x: deg[x])
        
        # We will limit the number of DFS nodes to avoid timeout
        node_count = 0
        NODE_LIMIT = 5000000 # 5 million nodes
        
        def dfs(v, visited_mask, path):
            nonlocal best_len, best_path, node_count, deadline
            
            node_count += 1
            if time.time() > deadline or node_count > NODE_LIMIT:
                return False
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
                
            # Pruning: 
            # 1. If we can't reach all remaining vertices in the component, we can't beat best_len
            #    unless best_len is small.
            #    Let's check connectivity of unvisited vertices from v.
            unvisited_in_comp = main_comp & ~visited_mask
            if unvisited_in_comp == 0:
                return True # Found a Hamiltonian path in the component
            
            # Check if any unvisited vertex is isolated from v
            # BFS/DFS from v through unvisited vertices
            seen = 0
            frontier = adj[v] & unvisited_in_comp
            seen |= frontier
            while frontier:
                lsb = frontier & (-frontier)
                frontier ^= lsb
                u = lsb.bit_length() - 1
                new_frontier = adj[u] & unvisited_in_comp & ~seen
                seen |= new_frontier
                frontier |= new_frontier
            
            # If we can't reach all unvisited vertices, the max possible path length is 
            # len(path) + popcount(seen)
            if len(path) + bin(seen).count('1') <= best_len:
                return True # Prune
            
            # 2. Check for isolated unvisited vertices (degree 0 in unvisited subgraph + v)
            # If an unvisited vertex w has no neighbors in unvisited_in_comp and w != v (v is current, not unvisited)
            # Actually, w must be reachable from v. If w is not in 'seen', it's unreachable.
            # We already handled this with the reachability bound.
            
            # 3. Degree constraint: 
            # If an unvisited vertex has degree 1 in the remaining graph (including v), it must be visited next or it will be stranded.
            # This is complex to implement efficiently with bitmasks. Skip for now.

            # Generate candidates
            cand = adj[v] & unvisited_in_comp
            if not cand:
                return True
            
            # Order candidates by Warnsdorff
            lst = []
            m = cand
            while m:
                lsb = m & (-m)
                m ^= lsb
                u = lsb.bit_length() - 1
                # Score: number of unvisited neighbors
                score = bin(adj[u] & unvisited_in_comp & ~(1 << u)).count('1')
                lst.append((score, u))
            
            lst.sort()
            
            for score, u in lst:
                path.append(u)
                if not dfs(u, visited_mask | (1 << u), path):
                    path.pop()
                    return False
                path.pop()
                
            return True

        # Run DFS from a few promising starts
        # To avoid redundant work, we can try starts that are in the best_path or low degree
        # Let's try the first 10 low-degree vertices in the component
        for s in start_verts[:10]:
            if time.time() > deadline or node_count > NODE_LIMIT:
                break
            # If s is already in best_path, we might still want to explore different paths from it
            dfs(s, 1 << s, [s])

    if not best_path:
        # Fallback: return any valid path, e.g., a single vertex
        best_path = [0]
        
    return best_path
