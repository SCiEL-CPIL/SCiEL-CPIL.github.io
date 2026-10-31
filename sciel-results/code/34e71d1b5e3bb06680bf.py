import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Heuristic Search for k=3, g=16
    if k == 3 and g == 16:
        best_n, best_edges = luw_construct(k, g)
        
        def find_shortest_cycle(adj, max_len):
            n = len(adj)
            min_len = max_len + 1
            for start in range(n):
                # BFS
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= max_len // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u] and dist[v] >= dist[u]:
                            # Found a cycle
                            # Reconstruct to find length
                            # This is a simplified check. 
                            # For efficiency, we just check if we hit a visited node with dist <= dist[u]
                            # Actually, standard BFS finds shortest cycle if we track parents.
                            # Let's do a proper check.
                            pass
            return min_len, []

        # A more efficient girth check
        def get_girth(adj):
            n = len(adj)
            girth = float('inf')
            for i in range(n):
                # BFS from i
                dist = [-1] * n
                parent = [-1] * n
                dist[i] = 0
                q = deque([i])
                while q:
                    u = q.popleft()
                    if dist[u] >= girth // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u] and dist[v] >= dist[u]:
                            # Cycle found
                            # Length is dist[u] + dist[v] + 1?
                            # No, if v is visited and not parent, it's a back edge or cross edge.
                            # In BFS tree, a cross edge (u,v) creates a cycle of length dist[u] + dist[v] + 1.
                            # But we need to ensure it's the shortest.
                            # Actually, just checking dist[u] + dist[v] + 1 is an upper bound.
                            # To be exact, we should check all edges.
                            # For speed, we'll just return a conservative estimate or use a known algorithm.
                            # Let's assume we find the cycle length.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth:
                                girth = cycle_len
            return girth

        # Random 3-regular graph generator
        def random_3_reg(n):
            if n < 4 or n % 2 != 0:
                return None
            # Configuration model
            half = [i for i in range(n) for _ in range(3)]
            random.shuffle(half)
            edges = []
            for i in range(0, len(half), 2):
                u, v = half[i], half[i+1]
                if u != v:
                    edges.append((u, v))
            # Check for multi-edges and loops
            # If multi-edges, retry
            adj = [[] for _ in range(n)]
            for u, v in edges:
                if u == v:
                    return None
                if v in adj[u]:
                    return None
                adj[u].append(v)
                adj[v].append(u)
            return adj

        # Local Search
        def local_search(adj, target_girth, time_limit):
            n = len(adj)
            current_girth = get_girth(adj)
            if current_girth >= target_girth:
                return adj, current_girth
            
            T = 1000.0
            cooling = 0.999
            best_adj = [list(x) for x in adj]
            best_girth = current_girth
            
            while time.time() < time_limit and T > 1.0:
                # Find a short cycle
                # Simplified: pick a random edge, see if it's in a short cycle
                # For now, just try random swaps
                u = random.randint(0, n-1)
                v = random.choice(adj[u])
                # Find another edge to swap with
                # Pick a random neighbor of v
                if not adj[v]:
                    continue
                w = random.choice(adj[v])
                if w == u:
                    continue
                # Swap (u,v) and (v,w) with (u, x) and (w, y)?
                # Standard 2-opt: swap (u,v) and (x,y) with (u,x) and (v,y)
                # Let's pick x from adj[u] and y from adj[v]
                if len(adj[u]) < 2 or len(adj[v]) < 2:
                    continue
                x = random.choice(adj[u])
                y = random.choice(adj[v])
                if x == v or y == u:
                    continue
                if x == y:
                    continue
                if x in adj[y]:
                    continue
                if y in adj[x]:
                    continue
                
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].append(x)
                adj[x].append(u)
                adj[v].append(y)
                adj[y].append(v)
                
                new_girth = get_girth(adj)
                
                if new_girth >= best_girth:
                    best_girth = new_girth
                    best_adj = [list(row) for row in adj]
                    if new_girth >= target_girth:
                        return best_adj, new_girth
                else:
                    # Accept with probability exp((new-old)/T)
                    delta = new_girth - current_girth
                    if random.random() < math.exp(delta / T):
                        pass
                    else:
                        # Revert
                        adj[u].remove(x)
                        adj[x].remove(u)
                        adj[v].remove(y)
                        adj[y].remove(v)
                        
                        adj[u].append(v)
                        adj[v].append(u)
                        adj[x].append(y)
                        adj[y].append(x)
                
                current_girth = new_girth
                T *= cooling
                
            return best_adj, best_girth

        # Try to find a smaller graph
        for n in [512, 550, 600, 700, 800, 900]:
            if time.time() > deadline:
                break
            adj = random_3_reg(n)
            if adj is None:
                continue
            # Run local search
            time_left = deadline - time.time()
            if time_left < 1:
                break
            adj_out, girth_out = local_search(adj, 16, time.time() + time_left * 0.8)
            if girth_out >= 16:
                if n < best_n:
                    # Convert adj to edges
                    edges = []
                    for i in range(n):
                        for j in adj_out[i]:
                            if i < j:
                                edges.append((i, j))
                    best_n = n
                    best_edges = edges
                    # We found a valid graph. Can we do better?
                    # Yes, try smaller n? No, we started at 512.
                    # So if we found one at 512, it's optimal (lower bound).
                    # If we found one at 900, we should try 512 again?
                    # The loop goes 512 -> 900.
                    # So if we succeed at 512, we break.
                    if n == 512:
                        return best_n, best_edges
            # If we didn't find a valid one at this n, try next n?
            # Actually, if 512 is impossible, we should skip to larger n.
            # But the loop does that.

        return best_n, best_edges

    return luw_construct(k, g)
# EVOLVE-BLOCK-END
