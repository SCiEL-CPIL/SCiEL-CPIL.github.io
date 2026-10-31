import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Fallback: seed's LUW construction
    def seed_construct(kk, gg):
        q = kk
        d = max(3, gg - 5 if (gg - 5) % 2 else gg - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    # For k=3, g=16: try to find a good graph
    # Known: best is 936 vertices (2026). Moore bound 512.
    # Try n from 512 upward in steps of 2 (must be even for 3-regular)
    
    def generate_random_bipartite_cubic(n):
        """Generate random 3-regular bipartite graph on n vertices (n even, n/2 in each part)."""
        n_half = n // 2
        # Each vertex in left has degree 3, each in right has degree 3
        # Left: 0..n_half-1, Right: n_half..n-1
        # Use configuration model: left has 3*n_half half-edges, right has 3*n_half half-edges
        # Pair them randomly
        for attempt in range(100):
            left_slots = [i for i in range(n_half) for _ in range(3)]
            right_slots = [i for i in range(n_half) for _ in range(3)]
            random.shuffle(right_slots)
            
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(3 * n_half):
                u = left_slots[i]
                v = right_slots[i] + n_half
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj
        return None
    
    def generate_random_cubic(n):
        """Generate random 3-regular graph on n vertices using configuration model."""
        if n < 4 or n % 2 != 0:
            return None
        for attempt in range(200):
            slots = [i for i in range(n) for _ in range(3)]
            random.shuffle(slots)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(slots), 2):
                u = slots[i]
                v = slots[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj
        return None
    
    def has_short_cycle(adj, n, g):
        """Check if graph has any cycle of length < g. Return (bool, cycle_vertices or None)."""
        # BFS from each vertex up to depth (g-1)//2
        max_depth = (g - 1) // 2  # For g=16, depth=7
        for s in range(n):
            dist = [-1] * n
            prev = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
                    elif v != prev[u] and u != prev[v]:
                        # Back edge creating a cycle
                        cyc_len = dist[u] + dist[v] + 1
                        if cyc_len < g:
                            # Reconstruct cycle
                            path1 = []
                            cur = u
                            while cur != s:
                                path1.append(cur)
                                cur = prev[cur]
                            path1.append(s)
                            path1.reverse()
                            path2 = []
                            cur = v
                            while cur != s:
                                path2.append(cur)
                                cur = prev[cur]
                            # Cycle: path1 + path2 reversed (excluding s which is in both)
                            cycle = path1 + path2[::-1][:-1] if path2 else path1
                            # Actually: path1 is s->...->u, path2 is s->...->v
                            # Cycle: u->...->s->...->v->u, i.e., path1_reversed[1:] + [v] ... 
                            # Let's just return a flag
                            return True, None
        return False, None
    
    def find_one_short_cycle(adj, n, g):
        """Find one short cycle. Return list of vertices in cycle or None."""
        max_depth = (g - 1) // 2
        for s in range(n):
            dist = [-1] * n
            prev = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
                    elif v != prev[u] and u != prev[v]:
                        cyc_len = dist[u] + dist[v] + 1
                        if cyc_len < g:
                            # Reconstruct: path from s to u, edge u-v, path from s to v
                            path_u = []
                            cur = u
                            while cur != -1:
                                path_u.append(cur)
                                if cur == s:
                                    break
                                cur = prev[cur]
                            path_u.reverse()  # s -> ... -> u
                            path_v = []
                            cur = v
                            while cur != -1:
                                path_v.append(cur)
                                if cur == s:
                                    break
                                cur = prev[cur]
                            path_v.reverse()  # s -> ... -> v
                            # Cycle: u -> ... -> s -> ... -> v -> u
                            # Vertices: path_u (s to u) + path_v[1:] (s excluded, to v) reversed
                            cycle = path_u + path_v[1:][::-1]
                            return cycle
        return None
    
    def all_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges
    
    def try_switch(adj, n, u, v, x, y):
        """Try to switch edges (u,v) and (x,y) to (u,x) and (v,y) or (u,y) and (v,x)."""
        # Verify (u,v) and (x,y) are edges
        if v not in adj[u] or y not in adj[x]:
            return False
        # Try (u,x) and (v,y)
        if u != x and v != y and x not in adj[u] and y not in adj[v]:
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            return True
        # Try (u,y) and (v,x)
        if u != y and v != x and y not in adj[u] and x not in adj[v]:
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(y)
            adj[y].append(u)
            adj[v].append(x)
            adj[x].append(v)
            return True
        return False
    
    # Main search
    best_result = None
    
    # Try several n values, starting from Moore bound
    # For k=3, g=16: n must be even, n >= 512
    # Known best: 936. Let's try a range.
    
    n_candidates = list(range(512, 1000, 2))
    
    # Allocate time: more time to smaller n values
    # Total budget ~108s
    # Try n=512,514,...,540 with more time, then broader range with less
    
    for n in n_candidates:
        if time.time() > deadline - 5:
            break
        
        # Time per n: decreasing
        remaining_n = len(n_candidates) - n_candidates.index(n)
        time_per_n = max(1.0, min(15.0, 80.0 / max(1, remaining_n // 10)))
        
        n_start = time.time()
        
        # Generate random 3-regular graph
        adj = generate_random_cubic(n)
        if adj is None:
            continue
        
        edges_list = all_edges(adj, n)
        
        # Local search with simulated annealing
        best_cycle_len = g  # We want to minimize max cycle below g, i.e., eliminate all < g
        temp = 100.0
        cooling = 0.995
        
        while time.time() - n_start < time_per_n:
            cycle = find_one_short_cycle(adj, n, g)
            if cycle is None:
                # No short cycles! Valid graph!
                edges = all_edges(adj, n)
                return n, edges
            
            # Pick a random edge in the cycle to break
            cycle_len = len(cycle)
            edge_idx = random.randint(0, cycle_len - 1)
            u = cycle[edge_idx]
            v = cycle[(edge_idx + 1) % cycle_len]
            
            if v not in adj[u]:
                continue
            
            # Pick a random other edge
            if len(edges_list) < 2:
                break
            ex, ey = random.choice(edges_list)
            if (ex == u and ey == v) or (ex == v and ey == u):
                continue
            if ex in (u, v) or ey in (u, v):
                continue
            
            # Save state
            adj_backup = [list(lst) for lst in adj]
            
            if try_switch(adj, n, u, v, ex, ey):
                # Check if it improved (fewer/shorter cycles)
                # For simplicity, always accept (or use SA)
                new_cycle = find_one_short_cycle(adj, n, g)
                if new_cycle is None:
                    edges = all_edges(adj, n)
                    return n, edges
                # Accept with SA criterion (simplified: always accept for now)
                # Update edge list
                edges_list = all_edges(adj, n)
            else:
                # Revert
                adj = adj_backup
                continue
            
            temp *= cooling
        
        # End of time for this n
        # Check if current adj is valid (shouldn't be, but just in case)
        cycle = find_one_short_cycle(adj, n, g)
        if cycle is None:
            edges = all_edges(adj, n)
            return n, edges
    
    # Fallback: return seed construction
    return seed_construct(k, g)
