"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    # Moore bound for lower bound on n
    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    best_n = None
    best_edges = None

    def verify_girth(n, edges, min_girth):
        """Verify no cycle shorter than min_girth using BFS from each vertex."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def generate_random_regular(n, k, seed=None):
        if seed is not None:
            random.seed(seed)
        if n * k % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        edge_set = set()
        valid = True
        for i in range(n * k // 2):
            u, v = stubs[2 * i], stubs[2 * i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        if not valid:
            return None
        edges = list(edge_set)
        return n, edges

    def try_explicit_construction():
        q = k
        if (g - 5) % 2 == 1:
            d = g - 5
        else:
            d = g - 4
        if d < 3:
            d = 3

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    def local_search_girth(n, edges, target_girth, max_time):
        """
        Attempts to increase girth using 2-switches.
        Returns (n, edges) if girth >= target_girth is achieved, else None.
        """
        # Build adjacency structure
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u, v), max(u, v)))
        
        # Function to count short cycles
        def count_short_cycles(adj, limit):
            count = 0
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                q = deque([start])
                dist[start] = 0
                while q:
                    u = q.popleft()
                    if dist[u] >= limit // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < limit:
                                count += 1
            return count
        
        current_cycles = count_short_cycles(adj, target_girth)
        
        if current_cycles == 0:
            return n, edges
        
        # Simulated Annealing
        temperature = 100.0
        cooling_rate = 0.99
        max_iter = 50000
        
        for i in range(max_iter):
            if time.time() > max_time:
                break
            if current_cycles == 0:
                return n, edges
                
            # Pick two random edges
            # To be efficient, we can pick random vertices and their neighbors
            u1 = random.randint(0, n - 1)
            v1 = random.choice(adj[u1])
            
            u2 = random.randint(0, n - 1)
            v2 = random.choice(adj[u2])
            
            # Ensure edges are distinct and not sharing vertices (for simple 2-switch)
            # Actually, 2-switches can share vertices, but let's stick to disjoint edges first for simplicity
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            if len({u1, v1, u2, v2}) < 4:
                continue
                
            # Try swapping (u1,v1) and (u2,v2) with (u1,u2) and (v1,v2)
            # Check if new edges exist or are loops
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
                
            key1 = (min(u1, u2), max(u1, u2))
            key2 = (min(v1, v2), max(v1, v2))
            
            if key1 in edge_set or key2 in edge_set:
                continue
                
            # Perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            edge_set.remove((min(u1, v1), max(u1, v1)))
            edge_set.remove((min(u2, v2), max(u2, v2)))
            
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            
            edge_set.add(key1)
            edge_set.add(key2)
            
            new_cycles = count_short_cycles(adj, target_girth)
            
            if new_cycles <= current_cycles:
                current_cycles = new_cycles
            else:
                # Revert
                adj[u1].remove(u2)
                adj[u2].remove(u1)
                adj[v1].remove(v2)
                adj[v2].remove(v1)
                
                edge_set.remove(key1)
                edge_set.remove(key2)
                
                adj[u1].append(v1)
                adj[v1].append(u1)
                adj[u2].append(v2)
                adj[v2].append(u2)
                
                edge_set.add((min(u1, v1), max(u1, v1)))
                edge_set.add((min(u2, v2), max(u2, v2)))
                
                # Accept with probability exp(-delta/T)
                delta = new_cycles - current_cycles
                if random.random() < math.exp(-delta / temperature):
                    # Re-apply
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                    edge_set.remove((min(u1, v1), max(u1, v1)))
                    edge_set.remove((min(u2, v2), max(u2, v2)))
                    
                    adj[u1].append(u2)
                    adj[u2].append(u1)
                    adj[v1].append(v2)
                    adj[v2].append(v1)
                    
                    edge_set.add(key1)
                    edge_set.add(key2)
                    current_cycles = new_cycles
            
            temperature *= cooling_rate
        
        if current_cycles == 0:
            final_edges = list(edge_set)
            return n, final_edges
        return None, None

    # Step 1: Try explicit construction
    if time_left() > 10:
        try:
            n_exp, edges_exp = try_explicit_construction()
            if verify_girth(n_exp, edges_exp, g):
                best_n = n_exp
                best_edges = edges_exp
        except Exception:
            pass

    # Step 2: Local search from random graphs
    if best_n is None or time_left() > 5:
        # Start from Moore bound, try increasing n
        # We want to find the smallest n.
        # Try n from moore to best_n (or some upper bound)
        
        upper_n = best_n if best_n else max(moore * 3, 1000)
        
        # To save time, we can jump in steps
        # But we want the minimum n.
        # Let's try a few values.
        
        # Heuristic: Random regular graphs have girth ~ log(n).
        # For g=9, log_4(n) ~ 9 => n ~ 4^9 ~ 262k. This is too high.
        # But local search can improve girth.
        # Let's try n around 1000-2000.
        
        start_n = max(moore, 100)
        
        # If we have a best_n, we can try to go lower.
        # But going lower is harder.
        # Let's just try to find *any* graph smaller than the explicit one.
        
        for n_try in range(start_n, min(upper_n + 1, 3000), 100):
            if time_left() < 10:
                break
            
            # Try a few random seeds
            for seed in range(10):
                if time_left() < 5:
                    break
                result = generate_random_regular(n_try, k, seed=seed)
                if result is None:
                    continue
                n_r, edges_r = result
                
                # Run local search
                ls_result = local_search_girth(n_r, edges_r, g, time.time() + 5)
                if ls_result[0] is not None:
                    n_ls, edges_ls = ls_result
                    if best_n is None or n_ls < best_n:
                        best_n = n_ls
                        best_edges = edges_ls
                    break # Found one for this n, move to next n?
                    # Actually, we might find a smaller n later, so we should continue searching lower n.
                    # But we are iterating upwards.
                    # If we found a valid graph at n_try, we can break the seed loop and try smaller n?
                    # No, we started from Moore bound. So if we found one at 1000, we can stop if 1000 is close to Moore.
                    # But we want the MINIMUM n.
                    # So we should search n from Moore bound upwards, and stop at the first valid n.
                    # This assumes that if n is valid, smaller n might not be.
                    # But we are searching upwards. The first valid n we find is not necessarily the minimum.
                    # However, it's a good heuristic.
                    
    if best_n is None:
        # Fallback
        n_exp, edges_exp = try_explicit_construction()
        return n_exp, edges_exp

    return best_n, best_edges
# EVOLVE-BLOCK-END
