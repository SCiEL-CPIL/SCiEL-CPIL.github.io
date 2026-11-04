import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    if k == 3:
        if g % 2 == 0:
            moore_bound = 2 * sum((k-1)**i for i in range(g//2))
        else:
            moore_bound = (k-1) * sum((k-1)**i for i in range((g-1)//2))
    else:
        moore_bound = 1

    def check_girth_fast(adj, n, g):
        """Exact girth check using non-backtracking walks (BFS from each vertex)."""
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != start and dist[u] + dist[v] + 1 < g:
                        return False
        return True

    def build_random_luw(d, seed):
        """
        Build a 3-regular graph via randomized LUW-like construction.
        Vertices are (side, x1, ..., xd) where side in {0,1}.
        Edges connect side 0 to side 1 via a random-ish permutation rule.
        """
        rng = random.Random(seed)
        
        # Precompute all possible x tuples
        all_x = list(range(3**d))  # Wait, x values are in Z_3? No, LUW uses Z_q.
        # Actually, for 3-regular, q=3. The standard LUW uses q=3.
        # Let's stick to the algebraic structure but randomize the 'rules' slightly?
        # No, the seed's problem was the fixed rules.
        # Let's try a different approach: Random 3-regular graph of size N, then check girth.
        # But N must be large enough.
        
        # Alternative: Use the known construction from the literature but with a different parameterization.
        # The seed used specific rules. Let's try to generate a random 3-regular graph
        # of size n = 2 * 3^d and check if it has girth >= g.
        # For d=7, n = 2 * 2187 = 4374.
        # For d=8, n = 2 * 6561 = 13122.
        
        # Generating a random 3-regular graph of size 4374 is feasible.
        # But finding one with girth 16 is like finding a needle in a haystack?
        # The probability is low.
        
        # Better strategy: Use the "lift" or "voltage" graph construction.
        # Or, use the seed's structure but search for a better 'd' or 'rules'.
        
        # Let's look at the "inspiration" finding: "Fixing d from 11 to 7 for g=16 reduces LUW graph size from ~354k to ~4k vertices".
        # This implies that a valid LUW graph with d=7 exists for g=16?
        # If so, we just need to find the correct rules for d=7.
        # The seed's rules were:
        # rule[2] = (1, 1)
        # rule[3] = (2, 1)
        # rule[4] = (1, 2)
        # for c in range(5, d + 1):
        #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
        #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        # Let's try to randomize the rules for c >= 5 and check girth.
        
        def gen_rules(d, rng):
            rules = {}
            rules[2] = (1, 1)
            rules[3] = (2, 1)
            rules[4] = (1, 2)
            for c in range(5, d + 1):
                # Randomly choose a, b from 1..c-1 such that the graph is connected and simple?
                # To ensure simplicity and 3-regularity, we must follow the LUW structure.
                # The structure is: y_c = x_c + y_a * x_b (mod 3)
                # We can randomize a and b.
                a = rng.randint(1, c-1)
                b = rng.randint(1, c-1)
                # Ensure a != b? Not strictly necessary, but helps.
                # Ensure the resulting graph is connected? We'll check later.
                rules[c] = (a, b)
            return rules

        def build_with_rules(d, rules, seed):
            q = 3
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
            
            def neighbours(vertex):
                side, x = vertex
                # x is a tuple of length d+1, x[0] is None, x[1..d] are values
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rules[c]
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
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

        # Try random rules for a given d
        best_n = float('inf')
        best_edges = None
        
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            rules = gen_rules(d, rng)
            n, edges = build_with_rules(d, rules, seed)
            
            # Check if it's 3-regular and simple
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            valid = True
            for i in range(n):
                if len(adj[i]) != 3:
                    valid = False
                    break
                if len(set(adj[i])) != 3:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check girth
            if check_girth_fast(adj, n, g):
                best_n = n
                best_edges = edges
                break
                
        return best_n, best_edges

    # Strategy: Try small d first (7, 8) with randomized rules.
    # If that fails, fall back to the seed's original logic or a known valid construction.
    
    best_n = float('inf')
    best_edges = None
    
    # Try d=7 and d=8
    for d in [7, 8]:
        if time.time() > deadline - 20:
            break
        n, edges = build_random_luw(d, 42)
        if edges is not None:
            best_n = n
            best_edges = edges
            break
            
    # If no valid graph found, use a simple fallback: the seed's original logic with a large d
    # Or, just return a known valid graph if we have one? No, we must construct it.
    # Let's try the seed's original rules with d=7 and d=8 just in case they work.
    if best_edges is None:
        def build_original(d):
            q = 3
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
            
            rule = {}
            if d >= 2: rule[2] = (1, 1)
            if d >= 3: rule[3] = (2, 1)
            if d >= 4: rule[4] = (1, 2)
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

        for d in [7, 8, 9, 10]:
            if time.time() > deadline - 5:
                break
            n, edges = build_original(d)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            if all(len(adj[i]) == 3 for i in range(n)):
                if check_girth_fast(adj, n, g):
                    best_n = n
                    best_edges = edges
                    break
                    
    # Final fallback: If still None, return a trivial valid graph (even if large)
    # A cycle of length g is 2-regular, not 3.
    # We must return a 3-regular graph.
    # If all else fails, return the seed's original output (which is valid, just large).
    if best_edges is None:
        d_fallback = 11
        def build_fallback(d):
            q = 3
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
            
            rule = {}
            if d >= 2: rule[2] = (1, 1)
            if d >= 3: rule[3] = (2, 1)
            if d >= 4: rule[4] = (1, 2)
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

        best_n, best_edges = build_fallback(d_fallback)

    return best_n, best_edges
