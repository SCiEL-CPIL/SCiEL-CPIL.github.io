import time
import random
import sys
import numpy as np

def construct(k, g):
    deadline = time.time() + 140.0

    def _lw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def _fast_girth_check(adj, n, g_target):
        """Check if girth >= g_target using non-backtracking walks."""
        if n < 3:
            return True
        max_depth = g_target - 1
        for s in range(n):
            first_seen = {}
            level = [(s, nb, 1) for nb in adj[s]]
            for nb in adj[s]:
                first_seen[(s, nb)] = 1
            d = 1
            while level and d < max_depth:
                next_level = []
                for prev, cur, dd in level:
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        key = (cur, nb)
                        if key in first_seen:
                            cyc = dd + 1 - first_seen[key]
                            if cyc < g_target and cyc >= 3:
                                return False
                        else:
                            first_seen[key] = dd + 1
                            next_level.append((cur, nb, dd + 1))
                level = next_level
                d += 1
        return True

    def _count_short_cycles(adj, n, g_target):
        """Count the number of non-backtracking closed walks of length < g_target."""
        if n < 3:
            return 0
        max_depth = g_target - 1
        penalty = 0
        for s in range(n):
            first_seen = {}
            level = [(s, nb, 1) for nb in adj[s]]
            for nb in adj[s]:
                first_seen[(s, nb)] = 1
            d = 1
            while level and d < max_depth:
                next_level = []
                for prev, cur, dd in level:
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        key = (cur, nb)
                        if key in first_seen:
                            cyc = dd + 1 - first_seen[key]
                            if cyc < g_target and cyc >= 3:
                                penalty += 1
                        else:
                            first_seen[key] = dd + 1
                            next_level.append((cur, nb, dd + 1))
                level = next_level
                d += 1
                if penalty > 100:
                    break
        return penalty

    def _random_4_reg(n):
        """Generate a random 4-regular graph on n vertices using configuration model."""
        half_edges = []
        for i in range(n):
            for _ in range(4):
                half_edges.append(i)
        random.shuffle(half_edges)
        adj = [set() for _ in range(n)]
        for i in range(0, n * 4, 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
            else:
                for _ in range(10):
                    v = random.randrange(n)
                    if u != v and v not in adj[u]:
                        adj[u].add(v)
                        adj[v].add(u)
                        break
        # Check if we got a valid 4-regular graph
        for i in range(n):
            if len(adj[i]) != 4:
                return None
        return [list(s) for s in adj]

    def _search(n, k, g, time_budget, seed=42):
        rng = random.Random(seed)
        adj = _random_4_reg(n)
        if adj is None:
            return None
        
        cur_pen = _count_short_cycles(adj, n, g)
        best_pen = cur_pen
        best_adj = [lst[:] for lst in adj]
        
        t_end = time.time() + time_budget
        cur_temp = 1.0
        
        # Track which vertices are in short cycles for targeted moves
        def _get_short_cycle_vertices(adj, n, g_target):
            """Return set of vertices that participate in cycles shorter than g_target."""
            if n < 3:
                return set()
            max_depth = g_target - 1
            involved = set()
            for s in range(n):
                first_seen = {}
                level = [(s, nb, 1) for nb in adj[s]]
                for nb in adj[s]:
                    first_seen[(s, nb)] = 1
                d = 1
                while level and d < max_depth:
                    next_level = []
                    for prev, cur, dd in level:
                        for nb in adj[cur]:
                            if nb == prev:
                                continue
                            key = (cur, nb)
                            if key in first_seen:
                                cyc = dd + 1 - first_seen[key]
                                if cyc < g_target and cyc >= 3:
                                    involved.add(cur)
                                    involved.add(nb)
                                    involved.add(prev)
                            else:
                                first_seen[key] = dd + 1
                                next_level.append((cur, nb, dd + 1))
                    level = next_level
                    d += 1
            return involved
        
        # Use numpy for faster penalty computation
        def _penalty_np(adj, n, g_target):
            """Fast penalty using adjacency matrix."""
            A = np.zeros((n, n), dtype=np.int32)
            for u in range(n):
                for v in adj[u]:
                    A[u, v] = 1
            # Count closed non-backtracking walks of length < g_target
            # For small g, we can compute this via matrix operations
            penalty = 0
            # Walks of length 3: triangles
            # Number of triangles = trace(A^3) / 6
            A2 = A @ A
            A3 = A2 @ A
            penalty += np.trace(A3) // 6 * 3  # Each triangle counted 6 times, but we want 3 per triangle (3 edges)
            # Actually, let's just count cycles properly
            # For girth check, we need cycles of length 3 to g_target-1
            # This is complex, so let's use a simpler approach
            return penalty
        
        while time.time() < t_end:
            # Get vertices in short cycles for targeted moves
            if rng.random() < 0.3:
                short_verts = _get_short_cycle_vertices(adj, n, g)
                if short_verts:
                    u = rng.choice(list(short_verts))
                else:
                    u = rng.randrange(n)
            else:
                u = rng.randrange(n)
            
            v = rng.choice(adj[u])
            x = rng.randrange(n)
            y = rng.choice(adj[x])
            
            if u in (x, y) or v in (x, y):
                continue
                
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            new_pen = _count_short_cycles(adj, n, g)
            
            if new_pen < cur_pen:
                cur_pen = new_pen
                if new_pen < best_pen:
                    best_pen = new_pen
                    best_adj = [lst[:] for lst in adj]
                    cur_temp = 1.0
            else:
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
                
                delta = new_pen - cur_pen
                accept_prob = 0.1 / (1.0 + delta * 0.5)
                if rng.random() < accept_prob:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)
                    cur_pen = new_pen
            
            cur_temp *= 0.999

        return best_adj

    # Driver
    try:
        time_budget = min(110.0, max(5.0, (deadline - time.time()) - 20.0))
        
        n_target = 320
        adj = _search(n_target, k, g, time_budget, seed=42)
        
        if adj is not None and _fast_girth_check(adj, n_target, g):
            edges = []
            for v in range(n_target):
                for w in adj[v]:
                    if v < w:
                        edges.append((v, w))
            return n_target, edges
    except Exception:
        pass
    
    try:
        return _lw_fallback()
    except Exception:
        return _lw_fallback()
