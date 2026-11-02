import random
import time

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_fallback(k, g)

    base_vertices = 10
    base_edges = []
    for i in range(base_vertices):
        for j in range(i + 1, base_vertices):
            if (j - i) % 10 != 5:
                base_edges.append((i, j))
                
    mod = 32
    vol = [[0] * base_vertices for _ in range(base_vertices)]
    
    def get_vol(u, v):
        return vol[u][v]
        
    def set_vol(u, v, val):
        vol[u][v] = val % mod
        vol[v][u] = (-val) % mod

    edge_list = base_edges
    
    # Precompute all closed walks of length 4, 6, 8
    # A walk is a sequence of vertices v0...vk where v0=vk
    # and (vi, vi+1) is an edge.
    # We store the list of edges (u, v) in the walk.
    
    def gen_walks(length):
        walks = []
        # Start from each vertex
        for start in range(base_vertices):
            # DFS/BFS to find walks of given length
            # Since graph is small, simple recursion
            def dfs(current, remaining, path_edges):
                if remaining == 0:
                    if current == start:
                        walks.append(list(path_edges))
                    return
                for v in range(base_vertices):
                    if (current, v) in edge_set or (v, current) in edge_set:
                        # Avoid backtracking? 
                        # In a voltage graph, backtracking walk u-v-u has sum 0 always.
                        # We must exclude backtracking walks from the check?
                        # No, backtracking walks correspond to 2-cycles in the lift?
                        # No, 2-cycles are not in simple graphs.
                        # The walk u-v-u in base corresponds to edge (u,v) and (v,u).
                        # Sum = vol[u][v] + vol[v][u] = 0.
                        # This is ALWAYS 0.
                        # So we must NOT include backtracking walks in the check,
                        # OR we must ensure the lift doesn't have 2-cycles.
                        # The lift is simple, so no 2-cycles.
                        # So backtracking walks in base do NOT correspond to cycles in the lift.
                        # They correspond to traversing an edge and going back.
                        # In the lift, (u,z) -> (v, z+vol) -> (u, z+vol+vol[v][u]) = (u, z).
                        # This is a closed walk of length 2 in the lift.
                        # But the lift is simple, so (u,z) and (v, z+vol) are distinct.
                        # Is it a cycle? A cycle must not repeat vertices.
                        # (u,z) is repeated. So it's not a simple cycle.
                        # So we only care about closed walks in base that do NOT backtrack?
                        # Actually, any closed walk in base with sum 0 corresponds to a closed walk in lift.
                        # If the closed walk in base is "simple" (no repeated vertices), the lift walk is a cycle.
                        # If the base walk backtracks, the lift walk repeats vertices immediately.
                        # So we only need to check **simple** closed walks in the base.
                        pass
            # For simplicity, we will just check all closed walks and ignore those that backtrack?
            # No, it's easier to just check the lift directly if the number of walks is small.
            # But building the lift is expensive.
            # Let's stick to the lift BFS but optimize it.
            pass
        return walks

    # Given the complexity of precomputing simple walks, 
    # and the speed of Python, 
    # I will use the Lift BFS but only on a subset of starting vertices 
    # or use a faster C-like implementation via NumPy if possible?
    # No, standard library.
    
    # Let's try the Local Search with Lift BFS again, but limit the search.
    # If we just find ONE short cycle, we perturb.
    # The BFS stops as soon as it finds one.
    # So it's fast if a short cycle exists.
    # If no short cycle, it's slow.
    # But as we converge, short cycles become rare.
    
    # To handle the "no short cycle" case faster:
    # We can check a random sample of starting vertices.
    # If 10 random starts yield no short cycle, it's likely valid.
    # Then do a full check.

    def randomize():
        for u, v in edge_list:
            set_vol(u, v, random.randint(0, mod - 1))

    def build_lift():
        n = base_vertices * mod
        edges = []
        for u, v in base_edges:
            vu = get_vol(u, v)
            for z in range(mod):
                u_idx = u * mod + z
                v_idx = v * mod + (z + vu) % mod
                edges.append((u_idx, v_idx))
        return n, edges

    def check_girth_sampled(n, edges, target_girth, sample_size=20):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Sample starts
        starts = random.sample(range(n), min(sample_size, n))
        for start in starts:
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = [start]
            while q:
                u = q.pop(0)
                if dist[u] >= target_girth // 2 - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < target_girth:
                                return False, (u, v, start)
        return True, None

    def check_girth_full(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = [start]
            while q:
                u = q.pop(0)
                if dist[u] >= target_girth // 2 - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < target_girth:
                                return False, (u, v, start)
        return True, None

    random.seed(42)
    deadline = time.time() + 100

    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            randomize()
        else:
            # Perturb from previous state
            # Pick a random edge and change its voltage
            u, v = random.choice(edge_list)
            old = get_vol(u, v)
            set_vol(u, v, (old + 1) % mod)

        n, edges = build_lift()
        
        # Quick simplicity check
        edge_set = set()
        simple = True
        for u, v in edges:
            if u == v:
                simple = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                simple = False
                break
            edge_set.add(key)
        if not simple:
            continue

        # Sampled check
        is_valid, conflict = check_girth_sampled(n, edges, g)
        
        if is_valid:
            # Full check
            is_valid_full, _ = check_girth_full(n, edges, g)
            if is_valid_full:
                return n, edges
        
        # If not valid, we could use the conflict to guide perturbation,
        # but random perturbation is simpler.

    return _luw_fallback(k, g)

def _luw_fallback(k, g):
    # ... same as before ...
    pass
