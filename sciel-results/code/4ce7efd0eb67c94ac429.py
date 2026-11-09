import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Fallback: LUW Graph
    def get_luw(k, g):
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

    # Try to find a Circulant Graph with girth g
    def check_circulant_girth(n, steps, g):
        # steps is a set of 4 integers
        # Check for cycles of length < g
        # BFS from 0 up to depth g-1
        # If we reach 0 at depth < g, there is a cycle.
        # Note: BFS finds shortest paths. If dist(0,0) < g, girth < g.
        # But dist(0,0) is 0. We need to check if there is a *cycle* of length < g.
        # This is equivalent to checking if there is a path from 0 to 0 of length L < g, L>0.
        # In a regular graph, this is checking if the girth is < g.
        
        # Efficient check:
        # For each step s in steps, and each subsequent step, track sums.
        # We want to know if sum(x1...xL) = 0 mod n for L < g.
        # And x1...xL is a valid walk (no immediate backtracking? No, girth allows backtracking? 
        # No, a cycle cannot backtrack immediately. But BFS handles shortest paths.
        # If the shortest cycle is < g, then girth < g.
        
        # Let's just do a BFS up to depth g-1 and see if we hit 0.
        # But we must avoid the trivial path 0.
        # We start with neighbors of 0.
        
        visited = {0: 0}
        frontier = [0]
        for depth in range(1, g):
            new_frontier = []
            for u in frontier:
                for s in steps:
                    v = (u + s) % n
                    if v not in visited:
                        visited[v] = depth
                        new_frontier.append(v)
                    # If v == 0 and depth < g, we found a cycle of length depth.
                    if v == 0:
                        return False
            frontier = new_frontier
            if not frontier:
                break
        return True

    def try_circulant(n, k, g):
        if k != 4:
            return None
        # Steps must be distinct, non-zero, and s != -s (unless 2s=0)
        # For n even, s != n/2.
        valid_steps = []
        for i in range(1, n):
            if i == n - i: # i = n/2
                continue
            valid_steps.append(i)
        
        # We need to pick 4 steps.
        # To ensure 4-regular, we pick 4 distinct steps.
        # The graph is undirected, so s and -s are the same edge.
        # So we should pick steps such that no two are negatives of each other.
        # And we include both s and -s in the "steps" for the BFS?
        # No, the step set for the graph is {s1, -s1, s2, -s2, s3, -s3, s4, -s4}?
        # No, a 4-regular circulant graph is defined by 4 steps {s1, s2, s3, s4} such that the neighbors of v are v+s1, v-s1, v+s2, v-s2, etc?
        # No, usually defined by a set S of size k such that S is symmetric (if s in S, -s in S).
        # If k=4, S = {s1, -s1, s2, -s2}.
        # So we just pick 2 pairs {s1, -s1}, {s2, -s2}.
        # Or 4 steps if n is odd? No, k=4 is even.
        # So we pick 2 steps s1, s2 such that s1 != s2, s1 != -s2, s1 != 0, s2 != 0.
        # And the step set is {s1, -s1, s2, -s2}.
        
        random.shuffle(valid_steps)
        
        # Try a few random pairs
        for _ in range(1000):
            if time.time() > deadline:
                return None
            # Pick 2 distinct steps from valid_steps
            # Ensure s1 != s2 and s1 != -s2
            s1 = valid_steps[random.randint(0, len(valid_steps)-1)]
            s2 = valid_steps[random.randint(0, len(valid_steps)-1)]
            if s1 == s2 or s1 == n - s2:
                continue
            
            steps = [s1, n - s1, s2, n - s2]
            # Remove duplicates if any (e.g. if s1 = n-s1, which we excluded)
            steps = list(set(steps))
            if len(steps) != 4:
                continue
                
            if check_circulant_girth(n, steps, g):
                # Construct edges
                edges = []
                for u in range(n):
                    for s in steps:
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
        return None

    # Search for n
    # Start with known good n or slightly above lower bound
    # Lower bound 243. Best known 320.
    # Try 320 first.
    for n in [320, 300, 280, 260, 250]:
        if time.time() > deadline:
            break
        res = try_circulant(n, k, g)
        if res:
            return res
            
    # If circulant fails, try a random 4-regular graph with local search?
    # Or just return LUW.
    # Given the difficulty of local search in 120s, LUW is safe.
    # But maybe try n=400, 500 for circulant?
    for n in [400, 500, 600]:
        if time.time() > deadline:
            break
        res = try_circulant(n, k, g)
        if res:
            return res

    return get_luw(k, g)
