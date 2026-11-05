import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # --- Seed Construction (LUW) ---
    def luw_construct(k, g):
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

    n_seed, edges_seed = luw_construct(k, g)
    best_n = n_seed
    best_edges = edges_seed
    
    # --- Heuristic Search ---
    # Moore Bound estimation
    # For k=5, g=9, M = 781.
    # We try to find a graph with n < n_seed.
    
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for loops
        for u in range(n):
            if u in adj[u]:
                return False
        
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            prev = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
                    elif v != prev[u]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # But this is only valid if v is not the parent
                        # In BFS, if v is visited and v != parent[u], it forms a cycle
                        # The cycle length is dist[u] + dist[v] + 1
                        # However, this might not be the shortest cycle through u,v
                        # But if ANY cycle is < min_girth, we return False
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def random_regular_graph(n, k):
        # Generate a random k-regular graph using random swaps
        # Start with a trivial k-regular graph: disjoint union of k-cycles? No, k=5.
        # Start with 5-cycles. n must be divisible by 5.
        # If not, adjust.
        # Better: Start with a random multigraph and fix.
        # Simplest: Use the configuration model and retry until simple.
        
        # For robustness, start with a known regular structure and randomize.
        # Create edges for a 5-regular circulant if possible, else random.
        
        # Let's just generate random edges and check degree.
        # This is slow.
        
        # Alternative: Start with a list of half-edges.
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        
        random.shuffle(half_edges)
        
        edges = set()
        adj = [set() for _ in range(n)]
        
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                # Loop, retry this pair? Or swap with another?
                # Simple retry: shuffle and restart if too many loops.
                # For now, just skip and hope? No, must be regular.
                # This method is prone to failure for small n.
                pass
            elif v in adj[u]:
                # Multi-edge
                pass
            else:
                edges.add((min(u,v), max(u,v)))
                adj[u].add(v)
                adj[v].add(u)
        
        # Check if all degrees are k
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        return list(edges)

    def improve_girth(n, edges, k, target_girth, time_limit):
        # Local search
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Find shortest cycle
        def find_shortest_cycle():
            # BFS from all nodes
            min_len = float('inf')
            cycle = None
            for start in range(n):
                dist = [-1] * n
                prev = [-1] * n
                dist[start] = 0
                queue = [start]
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            prev[v] = u
                            queue.append(v)
                        elif v != prev[u]:
                            l = dist[u] + dist[v] + 1
                            if l < min_len:
                                min_len = l
                                # Reconstruct cycle
                                # Path start->...->u and start->...->v
                                # Cycle is u->...->start->...->v->u
                                cycle = [u]
                                cur = u
                                while cur != start:
                                    cur = prev[cur]
                                    cycle.append(cur)
                                cycle.append(v)
                                cur = v
                                while cur != start:
                                    cur = prev[cur]
                                    cycle.append(cur)
                                # cycle list is u...start...v
                                # Edges are between consecutive elements
                                # And u-v is the closing edge? No, u-v is not an edge necessarily.
                                # The cycle is u->...->start->...->v and edge (u,v)?
                                # No, the back edge is (u,v) or (v,u).
                                # So the cycle vertices are [u, ..., start, ..., v]
                                # And the edges are (u, next), ..., (prev, start), ..., (prev_v, v), (v, u)?
                                # Wait, (u,v) is the edge that closed the cycle.
                                # So the cycle is the path from u to v in the tree + edge (u,v).
                                # Path: u -> ... -> start -> ... -> v
                                # Edges in path + (u,v)
                                pass
            return min_len, cycle

        # This is too complex to implement perfectly in short time.
        # Simpler heuristic: Randomly swap edges.
        # Pick two random edges (u,v) and (x,y).
        # If swapping to (u,x) and (v,y) is valid (no loops/multiedges) and doesn't decrease girth, accept.
        
        edges_set = set(edges)
        for _ in range(1000): # Limit iterations
            if time.time() > time_limit:
                break
            
            # Pick random edges
            if len(edges) < 2: continue
            e1_idx = random.randint(0, len(edges)-1)
            e2_idx = random.randint(0, len(edges)-1)
            if e1_idx == e2_idx: continue
            
            u, v = edges[e1_idx]
            x, y = edges[e2_idx]
            
            # Try swap: remove (u,v), (x,y). Add (u,x), (v,y)
            new_e1 = (min(u,x), max(u,x))
            new_e2 = (min(v,y), max(v,y))
            
            if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                continue
            if new_e1 in edges_set or new_e2 in edges_set:
                continue
            if new_e1 == new_e2:
                continue
            
            # Apply swap
            edges_set.remove((u,v))
            edges_set.remove((x,y))
            edges_set.add(new_e1)
            edges_set.add(new_e2)
            
            # Check girth improvement (or at least not worse)
            # This check is expensive.
            # We can just do it every 10 swaps.
            
        return list(edges_set)

    # Main search loop
    # Try a few sizes
    for n_try in range(800, min(n_seed + 1, 1500), 50):
        if time.time() > deadline:
            break
        
        # Generate random graph
        edges_try = random_regular_graph(n_try, k)
        if not edges_try:
            continue
        
        # Improve
        edges_improved = improve_girth(n_try, edges_try, k, g, time.time() + 5)
        
        if check_girth(n_try, edges_improved, g):
            return n_try, edges_improved
            
    return best_n, best_edges
