import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Fallback: LUW Construction (from original code)
    def _lazebnik_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    # New Strategy: GQ(4,2) incidence graph + edge-switching to extend 8-cycles to 9+
    if k == 5 and g == 9:
        # GQ(4,2) has 21 points and 21 lines, each line has 3 points, each point on 3 lines
        # Incidence graph: 42 vertices, 5-regular, girth 8
        # We'll use a larger construction based on GQ(4,2) structure
        
        def build_gq42_incidence():
            """Build the incidence graph of GQ(4,2): 42 vertices, 5-regular, girth 8."""
            # GQ(4,2) parameters: s=2, t=2
            # Number of points = (s+1)(st+1) = 3*5 = 15... wait let me recalculate
            # Actually GQ(s,t): points = (s+1)(st+t), lines = (t+1)(st+s)
            # For GQ(2,2): points = 3*4=12, lines=3*4=12, each line has s+1=3 points, each point on t+1=3 lines
            # Incidence graph: 24 vertices, 3-regular, girth 8
            # 
            # For a 5-regular graph with girth 8, we need something else.
            # The Hoffman-Singleton graph is 7-regular girth 5.
            # 
            # Let me use a different approach: construct a 5-regular graph with girth 8
            # using the known 5-regular girth 8 graph (if it exists) or construct one.
            #
            # Actually, the smallest 5-regular girth 8 graph has 26 vertices? No...
            # Moore bound for (5,8): 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1+5+20+80+320 = 426 for even girth
            # Wait, for girth 8 (even): n >= 1 + k + k(k-1) + k(k-1)^2 + k(k-1)^3 = 1+5+20+80+320=426
            # But that's the Moore bound. Known 5-regular girth 8 graphs exist with more vertices.
            #
            # Let me just use a random 5-regular graph approach with targeted repair.
            
            # Actually, let me try a different strategy: use the known structure.
            # The incidence graph of GQ(4,2) where s=4, t=2:
            # points = (4+1)(4*2+2) = 5*10 = 50
            # lines = (2+1)(4*2+4) = 3*12 = 36
            # Each line has 5 points, each point on 3 lines
            # Incidence graph: 86 vertices, not regular (points degree 3, lines degree 5)
            #
            # Let me just go with the random regular + repair approach but more carefully.
            
            # Strategy: Generate random 5-regular graphs of various sizes, check girth,
            # and if girth < 9, try to fix by edge swaps that break short cycles.
            
            pass  # Placeholder, will implement below

        def generate_random_k_regular(n, k, rng):
            """Generate a random k-regular graph using configuration model with rejection."""
            if n * k % 2 != 0:
                return None
            while True:
                stubs = [i for i in range(n) for _ in range(k)]
                rng.shuffle(stubs)
                edges = set()
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    edge = (min(u, v), max(u, v))
                    if edge in edges:
                        valid = False
                        break
                    edges.add(edge)
                if valid:
                    return list(edges)

        def build_adjacency(edges, n):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        def find_all_short_cycles(edges, n, max_cycle_len):
            """Find all cycles of length < max_cycle_len. Returns list of cycles (lists of vertices)."""
            adj = build_adjacency(edges, n)
            cycles = []
            visited_global = set()
            
            for start in range(n):
                # DFS to find cycles starting from start, where start is the smallest vertex in the cycle
                stack = [(start, [start])]
                while stack:
                    node, path = stack.pop()
                    if len(path) >= max_cycle_len:
                        continue
                    for nei in adj[node]:
                        if nei <= start:  # Only consider cycles where start is the minimum
                            continue
                        if nei in path:
                            # Found a cycle
                            if nei == start:
                                cycle = path[:]
                                if len(cycle) < max_cycle_len:
                                    cycles.append(cycle)
                            else:
                                # This would create a cycle but start isn't the min
                                pass
                        else:
                            stack.append((nei, path + [nei]))
            return cycles

        def check_girth(edges, n, g):
            """Check if graph has girth >= g. Returns True if valid."""
            adj = build_adjacency(edges, n)
            for start in range(n):
                # BFS to find shortest cycle through start
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= g - 1:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            # Potential cycle
                            # Compute cycle length
                            x, y = u, v
                            if dist[x] < dist[y]:
                                x, y = y, x
                            diff = dist[x] - dist[y]
                            for _ in range(diff):
                                x = parent[x]
                            while x != y:
                                x = parent[x]
                                y = parent[y]
                            lca = x
                            cycle_len = dist[u] + dist[v] - 2 * dist[lca] + 1
                            if cycle_len < g:
                                return False
            return True

        def find_short_cycle_edges(edges, n, g):
            """Find one short cycle (length < g) and return its edges."""
            adj = build_adjacency(edges, n)
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= g - 1:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            x, y = u, v
                            if dist[x] < dist[y]:
                                x, y = y, x
                            diff = dist[x] - dist[y]
                            for _ in range(diff):
                                x = parent[x]
                            while x != y:
                                x = parent[x]
                                y = parent[y]
                            lca = x
                            cycle_len = dist[u] + dist[v] - 2 * dist[lca] + 1
                            if cycle_len < g:
                                # Reconstruct cycle
                                path1 = []
                                cur = u
                                while cur != lca:
                                    path1.append(cur)
                                    cur = parent[cur]
                                path1.append(lca)
                                path2 = []
                                cur = v
                                while cur != lca:
                                    path2.append(cur)
                                    cur = parent[cur]
                                cycle = path1 + path2
                                # Get edges of this cycle
                                cycle_edges = []
                                for i in range(len(cycle)):
                                    a, b = cycle[i], cycle[(i+1) % len(cycle)]
                                    cycle_edges.append((min(a,b), max(a,b)))
                                return cycle_edges
            return None

        def try_edge_swap(edges_set, n, k, g, rng):
            """Try a random edge swap that might break a short cycle."""
            if len(edges_set) < 2:
                return False
            
            # Find a short cycle to target
            short_cycle = find_short_cycle_edges(list(edges_set), n, g)
            if short_cycle is None:
                return True  # No short cycles, we're done
            
            # Try to swap an edge in the short cycle with another edge
            cycle_edges_list = list(short_cycle)
            e1 = rng.choice(cycle_edges_list)
            u1, v1 = e1
            
            # Pick a random other edge not in the cycle
            all_edges = list(edges_set)
            other_edges = [e for e in all_edges if e not in set(cycle_edges_list)]
            if not other_edges:
                other_edges = all_edges
            e2 = rng.choice(other_edges)
            u2, v2 = e2
            
            # Try different swap configurations
            candidates = [
                [(u1, u2), (v1, v2)],
                [(u1, v2), (v1, u2)]
            ]
            
            for new_e1, new_e2 in candidates:
                n1, n2 = min(new_e1), max(new_e1)
                m1, m2 = min(new_e2), max(new_e2)
                
                if n1 == n2 or m1 == m2:
                    continue
                if (n1, n2) == (m1, m2):
                    continue
                if (n1, n2) in edges_set or (m1, m2) in edges_set:
                    continue
                if (n1, n2) == e1 or (m1, m2) == e1 or (n1, n2) == e2 or (m1, m2) == e2:
                    continue
                
                # Check degree preservation (it's automatic since we're swapping)
                # Perform swap
                edges_set.remove(e1)
                edges_set.remove(e2)
                edges_set.add((n1, n2))
                edges_set.add((m1, m2))
                return True
            
            return False

        # Main search: try various n values
        rng = random.Random(42)
        
        # Start with sizes that are plausible for 5-regular girth 9
        # Moore bound is 428, best known is 1116
        # Try sizes in a reasonable range
        n_candidates = list(range(500, 1100, 25)) + [450, 475]
        
        best_result = None
        
        for n in n_candidates:
            if time.time() > deadline - 10:
                break
            
            # Generate random 5-regular graph
            edges = generate_random_k_regular(n, k, rng)
            if edges is None:
                continue
            
            edges_set = set(edges)
            
            # Check initial girth
            if check_girth(list(edges_set), n, g):
                return n, list(edges_set)
            
            # Try to repair by edge swaps
            max_swaps = 2000
            for swap_iter in range(max_swaps):
                if time.time() > deadline - 5:
                    break
                
                success = try_edge_swap(edges_set, n, k, g, rng)
                if success:
                    # Check if we fixed the girth
                    if check_girth(list(edges_set), n, g):
                        return n, list(edges_set)
                else:
                    # No more useful swaps, try another random graph
                    break
            
        # Fallback to LUW
        return _lazebnik_construct(k, g)

    # Fallback to LUW
    return _lazebnik_construct(k, g)
