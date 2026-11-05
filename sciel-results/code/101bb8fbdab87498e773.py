import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    ALL = (1 << n) - 1

    def popcount(x):
        return bin(x).count('1')

    # --- connected components ---
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
    comps.sort(key=popcount, reverse=True)

    deadline = time.time() + 75.0
    best = 0
    best_path = []

    # --- greedy warm start (Warnsdorff) from every vertex ---
    def greedy(start):
        path = [start]
        seen = 1 << start
        while True:
            cand = adj[path[-1]] & ~seen
            if not cand:
                break
            bu = -1
            bon = None
            m = cand
            while m:
                lsb = m & (-m)
                m ^= lsb
                u = lsb.bit_length() - 1
                on = popcount(adj[u] & ~seen & ~(1 << u))
                if bon is None or on < bon:
                    bon = on
                    bu = u
            path.append(bu)
            seen |= 1 << bu
        return path

    for s in range(n):
        p = greedy(s)
        if len(p) > best:
            best = len(p)
            best_path = p

    # --- 2-opt style local improvement: try to extend path by swapping endpoints or inserting vertices ---
    def try_improve(path):
        """Try to extend the path by finding a vertex that can be inserted or an endpoint that can be swapped."""
        if not path:
            return path
        improved = True
        while improved and time.time() < deadline:
            improved = False
            seen = 0
            for v in path:
                seen |= 1 << v
            
            # Try to extend from the right end
            right_end = path[-1]
            cand_right = adj[right_end] & ~seen
            if cand_right:
                m = cand_right
                best_u = -1
                best_on = None
                while m:
                    lsb = m & (-m)
                    m ^= lsb
                    u = lsb.bit_length() - 1
                    on = popcount(adj[u] & ~seen & ~(1 << u))
                    if best_on is None or on < best_on:
                        best_on = on
                        best_u = u
                if best_u >= 0:
                    path.append(best_u)
                    improved = True
                    continue
            
            # Try to extend from the left end
            left_end = path[0]
            cand_left = adj[left_end] & ~seen
            if cand_left:
                m = cand_left
                best_u = -1
                best_on = None
                while m:
                    lsb = m & (-m)
                    m ^= lsb
                    u = lsb.bit_length() - 1
                    on = popcount(adj[u] & ~seen & ~(1 << u))
                    if best_on is None or on < best_on:
                        best_on = on
                        best_u = u
                if best_u >= 0:
                    path.insert(0, best_u)
                    improved = True
                    continue
            
            # Try 2-opt style: for each pair of consecutive edges, check if we can insert a vertex between them
            # Or try to find a vertex v not in path such that v is adjacent to some path[i] and path[i+1] is not in path
            # Actually, let's try a different approach: look for a vertex that can replace an endpoint
            # to allow further extension.
            
            # Simpler: try all unvisited vertices adjacent to any path vertex, and see if inserting it helps
            # This is expensive. Instead, let's just try extending from both ends with Warnsdorff.
            # If we can't extend, break.
            
            # Another idea: try to find a 2-opt improvement.
            # For a path, 2-opt means reversing a segment. But for simple paths, this doesn't change the set of vertices.
            # What we want is to change the order to allow extension.
            
            # Let's try: for each unvisited vertex v, check if it's adjacent to exactly one vertex in the path.
            # If so, we might be able to extend from that end.
            
            # Actually, the greedy extension above should handle this. If we can't extend, we're stuck.
            # Let's try a more aggressive approach: for each unvisited vertex v, check if it has at least 2 neighbors in the path.
            # If so, we can insert it into the path.
            
            for v in range(n):
                if seen & (1 << v):
                    continue
                nb_in_path = adj[v] & seen
                if nb_in_path == 0:
                    continue
                # Count how many neighbors v has in the path
                cnt = popcount(nb_in_path)
                if cnt < 2:
                    continue
                # Find two consecutive neighbors in the path
                # Get the indices of v's neighbors in the path
                indices = []
                for i, u in enumerate(path):
                    if nb_in_path & (1 << u):
                        indices.append(i)
                # Check if any two are consecutive
                for j in range(len(indices) - 1):
                    if indices[j+1] == indices[j] + 1:
                        # Insert v between path[indices[j]] and path[indices[j+1]]
                        a = path[indices[j]]
                        b = path[indices[j+1]]
                        # Check if v is adjacent to both a and b
                        if (adj[a] & (1 << v)) and (adj[b] & (1 << v)):
                            new_path = path[:indices[j]+1] + [v] + path[indices[j+1]:]
                            # Verify it's a valid path
                            valid = True
                            for k in range(len(new_path)-1):
                                if not (adj[new_path[k]] & (1 << new_path[k+1])):
                                    valid = False
                                    break
                            if valid:
                                path = new_path
                                improved = True
                                break
                if improved:
                    break
        
        return path

    # Apply local improvement to the best path found
    if best_path and time.time() < deadline:
        improved_path = try_improve(list(best_path))
        if len(improved_path) > best:
            best = len(improved_path)
            best_path = improved_path

    # --- exact DFS with connectivity (reachability) pruning ---
    nodes = 0
    NODE_CAP = 50_000_000

    def dfs(v, visited_mask, path):
        nonlocal best, best_path, nodes
        nodes += 1
        if time.time() > deadline or nodes > NODE_CAP:
            return False
        if len(path) > best:
            best = len(path)
            best_path = list(path)
        unvis = ALL & ~visited_mask
        if unvis == 0:
            return True
        # reachable unvisited vertices from v through unvisited vertices
        seen = adj[v] & unvis
        frontier = seen
        while frontier:
            lsb = frontier & (-frontier)
            frontier ^= lsb
            x = lsb.bit_length() - 1
            new = adj[x] & unvis & ~seen
            seen |= new
            frontier |= new
        # upper bound: can only add vertices still reachable from the open end
        if len(path) + popcount(seen) <= best:
            return True
        cand = adj[v] & unvis
        if cand == 0:
            return True
        # Warnsdorff ordering: fewest onward options first
        lst = []
        m = cand
        while m:
            lsb = m & (-m)
            m ^= lsb
            u = lsb.bit_length() - 1
            on = popcount(adj[u] & unvis & ~(1 << u))
            lst.append((on, u))
        lst.sort()
        for on, u in lst:
            path.append(u)
            if not dfs(u, visited_mask | (1 << u), path):
                path.pop()
                return False
            path.pop()
        return True

    for comp in comps:
        csz = popcount(comp)
        if best >= csz:
            break
        if time.time() > deadline:
            break
        m = comp
        verts = []
        while m:
            lsb = m & (-m)
            m ^= lsb
            verts.append(lsb.bit_length() - 1)
        # low-degree vertices first (likely endpoints of long paths)
        verts.sort(key=lambda w: popcount(adj[w]))
        for s in verts:
            if time.time() > deadline:
                break
            dfs(s, 1 << s, [s])
            if time.time() > deadline:
                break

    if not best_path:
        best_path = [0]
    return best_path
