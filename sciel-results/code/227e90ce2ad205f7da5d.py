"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    from random import Random
    import sys

    start_time = time.time()
    DEADLINE = 140.0

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def find_bipartition(n, adj):
        """Find bipartition via BFS. Returns (side_a, side_b) as sets of vertices."""
        color = [-1] * n
        side_a = set()
        side_b = set()
        for start in range(n):
            if color[start] != -1:
                continue
            color[start] = 0
            side_a.add(start)
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        if color[v] == 0:
                            side_a.add(v)
                        else:
                            side_b.add(v)
                        q.append(v)
        return side_a, side_b

    def hopcroft_karp(n_left, n_right, left_adj, left_size, right_size):
        """
        Hopcroft-Karp algorithm for maximum bipartite matching.
        left_adj[u] = list of right vertices adjacent to left vertex u
        Returns (match_left, match_right) where match_left[u] = matched right vertex or -1
        """
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [-1] * n_left

        def bfs():
            q = deque()
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in left_adj[u]:
                    mu = match_right[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == -1:
                        dist[mu] = dist[u] + 1
                        q.append(mu)
            return found

        def dfs(u):
            for v in left_adj[u]:
                mu = match_right[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    if dfs(u):
                        matching += 1

        return match_left, match_right, matching

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find bipartition
        side_a, side_b = find_bipartition(n, adj)
        
        # Choose smaller side as left side for matching
        if len(side_a) <= len(side_b):
            left_side = list(side_a)
            right_side = list(side_b)
        else:
            left_side = list(side_b)
            right_side = list(side_a)
        
        left_idx = {v: i for i, v in enumerate(left_side)}
        right_idx = {v: i for i, v in enumerate(right_side)}
        n_left = len(left_side)
        n_right = len(right_side)
        
        # For each left vertex, create color-copy nodes
        # Left node (v, c) connects to right nodes (u, c') where u is neighbor of v and c' != c
        # We want to select one color-copy per left vertex and one per right vertex such that no edge conflicts
        
        # Actually, let's think differently:
        # We need to assign colors to ALL vertices such that adjacent vertices have different colors.
        # This is a constraint satisfaction problem.
        
        # Alternative approach: Use the fact that it's a bipartite graph.
        # For list coloring of bipartite graphs with lists of size >= 3, we can use a different strategy.
        
        # Let me try a simpler backtracking with MRV heuristic first, then fall back to matching if needed.
        
        colour = [None] * n
        domains = [list(lst) for lst in lists]
        
        sys.setrecursionlimit(10000)
        
        def get_mrv_vertex(uncolored):
            """Get vertex with minimum remaining values."""
            min_count = float('inf')
            min_vertex = -1
            for v in uncolored:
                if colour[v] is not None:
                    continue
                # Count available colors
                used = set()
                for u in adj[v]:
                    if colour[u] is not None:
                        used.add(colour[u])
                available = [c for c in domains[v] if c not in used]
                if len(available) < min_count:
                    min_count = len(available)
                    min_vertex = v
                    if min_count == 0:
                        break
            return min_vertex, min_count
        
        def backtrack():
            if time_remaining() < 0.1:
                return False
            
            # Find uncolored vertex with MRV
            uncolored = [v for v in range(n) if colour[v] is None]
            if not uncolored:
                return True
            
            v, count = get_mrv_vertex(uncolored)
            if count == 0:
                return False
            
            used = set()
            for u in adj[v]:
                if colour[u] is not None:
                    used.add(colour[u])
            
            for c in domains[v]:
                if c in used:
                    continue
                colour[v] = c
                
                # Forward checking: verify all uncolored neighbors still have options
                ok = True
                for u in adj[v]:
                    if colour[u] is None:
                        used_u = set()
                        for w in adj[u]:
                            if colour[w] is not None:
                                used_u.add(colour[w])
                        if not any(c2 not in used_u for c2 in domains[u]):
                            ok = False
                            break
                
                if ok and backtrack():
                    return True
                
                colour[v] = None
            
            return False
        
        if backtrack():
            return colour
        
        # Fallback: try greedy with random restarts
        import random as _random
        rng = _Random(42)
        
        for attempt in range(1000):
            if time_remaining() < 0.5:
                break
            colour = [None] * n
            order = list(range(n))
            rng.shuffle(order)
            
            success = True
            for v in order:
                used = set()
                for u in adj[v]:
                    if colour[u] is not None:
                        used.add(colour[u])
                available = [c for c in domains[v] if c not in used]
                if not available:
                    success = False
                    break
                colour[v] = rng.choice(available)
            
            if success:
                return colour
        
        # Last resort: assign first available color
        for v in range(n):
            if colour[v] is None:
                colour[v] = lists[v][0]
        
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve_instance(n, edges, lists))
        
    final_out = []
    for lst in out:
        if isinstance(lst, (list, tuple)):
            final_out.append([int(x) for x in lst])
        else:
            final_out.append(lst)
            
    return final_out
# EVOLVE-BLOCK-END
