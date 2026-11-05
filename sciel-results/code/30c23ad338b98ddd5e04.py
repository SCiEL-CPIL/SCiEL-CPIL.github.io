import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    def check_girth_bfs(adj, n, g):
        """Check girth using BFS from each vertex - no cycle shorter than g."""
        max_depth = g // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                if dist[x] >= max_depth:
                    continue
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        queue.append(y)
                    elif y != parent[x] and parent[y] != x:
                        if dist[x] + dist[y] + 1 < g:
                            return False
        return True

    def count_4_cycles(adj, n):
        """Count number of 4-cycles. Used as heuristic."""
        count = 0
        for u in range(n):
            neighbors = adj[u]
            for i in range(len(neighbors)):
                v = neighbors[i]
                for j in range(i + 1, len(neighbors)):
                    w = neighbors[j]
                    if v < w:
                        # Check if v and w have a common neighbor other than u
                        # This counts each 4-cycle 4 times (once per vertex)
                        # Actually, this counts pairs of neighbors of u that are connected.
                        # A 4-cycle u-v-x-w-u is counted when u is processed and v,w are neighbors.
                        # We need to check if v and w are connected.
                        if w in adj[v]:
                            count += 1
        return count // 4 # Each 4-cycle is counted 4 times

    def random_k_regular(n, k):
        """Generate a random k-regular graph on n vertices."""
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        
        random.shuffle(half_edges)
        
        adj = [set() for _ in range(n)]
        used = [False] * len(half_edges)
        
        i = 0
        while i < len(half_edges):
            if used[i]:
                i += 1
                continue
            j = i + 1
            while j < len(half_edges) and (used[j] or half_edges[j] == half_edges[i]):
                j += 1
            if j >= len(half_edges):
                return None
            u, v = half_edges[i], half_edges[j]
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                used[i] = True
                used[j] = True
            i += 1

        for i in range(n):
            if len(adj[i]) != k:
                return None

        # Convert to list for faster access
        adj_list = [list(s) for s in adj]
        return adj_list

    def try_sa(n, k, g, time_budget):
        """Simulated Annealing to find a graph with girth >= g."""
        start = time.time()
        adj = random_k_regular(n, k)
        if adj is None:
            return None
        
        # Initial check
        if check_girth_bfs(adj, n, g):
            return (n, [(i, j) for i in range(n) for j in adj[i] if i < j])
        
        current_energy = count_4_cycles(adj, n)
        T = 1.0
        cooling = 0.99
        
        while time.time() - start < time_budget:
            # Pick random edge swap
            # 1. Pick random vertex u
            u = random.randint(0, n - 1)
            # 2. Pick random neighbor v of u
            if not adj[u]: continue
            v = random.choice(adj[u])
            # 3. Pick random non-neighbor w of u
            # To ensure regularity, we swap (u,v) with (x,y)
            # This is complex. 
            # Simpler: Pick two random edges (u,v) and (x,y) that are not adjacent.
            # Remove them, add (u,x) and (v,y) or (u,y) and (v,x).
            
            # Let's pick random edge (u, v)
            # We need another edge (x, y) such that u,v,x,y are distinct.
            # And (u,x) and (v,y) are not existing edges.
            
            # Find a candidate edge (x, y)
            # Iterate over a few random edges
            found_swap = False
            for _ in range(50):
                x = random.randint(0, n - 1)
                if x == u or x == v: continue
                if not adj[x]: continue
                y = random.choice(adj[x])
                if y == u or y == v: continue
                if x == y: continue
                
                # Check if (u,x) and (v,y) are valid
                if x in adj[u] or y in adj[v]:
                    continue
                if x in adj[v] or y in adj[u]:
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
                
                # Check girth
                if check_girth_bfs(adj, n, g):
                    return (n, [(i, j) for i in range(n) for j in adj[i] if i < j])
                
                new_energy = count_4_cycles(adj, n)
                delta = new_energy - current_energy
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    found_swap = True
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
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        return None

    # Seed LUW construction (fallback)
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

    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)

    # Try SA for smaller n
    # Start from max(n_min, 500) up to best_n - 1
    # Step size 10
    search_start = max(n_min, 500)
    search_end = min(best_n - 1, 1100)
    
    time_per_n = 5.0
    
    for n_try in range(search_start, search_end + 1, 10):
        if time.time() > deadline - 10:
            break
        if n_try % 2 != 0 and k * n_try % 2 != 0:
            continue
            
        result = try_sa(n_try, k, g, time_budget=time_per_n)
        if result is not None:
            best_n = n_try
            best_result = result
            # If we found one, we can try to go lower, but we are limited by time.
            # We'll break and return this valid graph.
            # Or we can continue to see if we find a smaller one.
            # Given time limits, breaking is safer.
            break

    return best_result

# EVOLVE-BLOCK-END
