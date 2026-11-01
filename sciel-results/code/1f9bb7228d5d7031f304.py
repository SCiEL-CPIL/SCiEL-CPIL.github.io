import time
import random
import sys
import math

def construct(k, g):
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

    # Moore bound calculation
    D = (g - 2) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(D))
    
    deadline = time.time() + 110
    
    # Helper to check girth of a circulant graph C_n(a, b)
    def is_circulant_girth(n, a, b, g):
        # Check for cycles of length < g
        # A cycle corresponds to a sum of +/- a, +/- b equal to 0 mod n
        # with no consecutive inverses (non-backtracking)
        # We can use BFS on the state space (current_pos, last_step)
        # But for n=320, we can just check all paths of length up to g-1
        
        # Faster: Check if any short cycle exists using BFS from 0
        # dist[v] = min steps to reach v
        # But we need to detect cycles.
        
        # Simple check: 
        # For each length L from 3 to g-1:
        #   Check if there is a non-backtracking walk of length L from 0 to 0.
        
        # Optimization: 
        # A cycle of length L exists if there is a sequence of L steps from {a, -a, b, -b}
        # summing to 0 mod n, such that no step is the inverse of the previous.
        
        # For g=10, we check L=3..9.
        # L=3: a+a+a, a+a-b, etc.
        
        # We can do a BFS on the directed graph of non-backtracking walks.
        # Nodes: (pos, last_dir) where last_dir is 1 (a), -1 (-a), 2 (b), -2 (-b)
        # Start: (0, None)
        
        # Actually, simpler:
        # Just check if the graph has any cycle of length < g using standard BFS from 0
        # on the undirected graph, but tracking the path?
        # No, BFS from 0 finds shortest path to each node.
        # If we find a node v at distance d, and it has an edge to a node u at distance d'
        # such that d + d' + 1 < g, and the path to u and v are disjoint (except 0)?
        # This is tricky.
        
        # Let's use the property: 
        # Girth is the smallest L such that there is a non-backtracking walk of length L from 0 to 0.
        
        def has_short_cycle(n, a, b, max_len):
            # State: (pos, prev_step)
            # prev_step: 0: none, 1: a, -1: -a, 2: b, -2: -b
            # We want to reach (0, any) with length < g
            
            # BFS
            visited = set()
            # (length, pos, prev_step)
            queue = [(0, 0, 0)]
            visited.add((0, 0, 0))
            
            steps = [(0, 1), (0, -1), (0, 2), (0, -2)] # dummy
            # Actual steps:
            # 1: +a, -1: -a, 2: +b, -2: -b
            
            def next_steps(prev):
                if prev == 1: # was +a, cannot -a
                    return [(2, b), (-2, -b)]
                elif prev == -1: # was -a, cannot +a
                    return [(1, a), (2, b), (-2, -b)] # Wait, -1 is -a. Inverse is +a. So cannot +a.
                    # Correction:
                    # If prev is +a (1), next cannot be -a (-1).
                    # If prev is -a (-1), next cannot be +a (1).
                    # If prev is +b (2), next cannot be -b (-2).
                    # If prev is -b (-2), next cannot be +b (2).
                    pass
                
            # Let's just enumerate all sequences of length up to 9
            # There are 4^9 = 262k sequences. Very fast.
            
            def check_sequence(seq):
                s = 0
                for i, step in enumerate(seq):
                    if step == 1: s += a
                    elif step == -1: s -= a
                    elif step == 2: s += b
                    elif step == -2: s -= b
                    if i > 0:
                        if (step == 1 and seq[i-1] == -1) or \
                           (step == -1 and seq[i-1] == 1) or \
                           (step == 2 and seq[i-1] == -2) or \
                           (step == -2 and seq[i-1] == 2):
                            return False # Backtracking
                return s % n == 0

            # Generate all sequences of length 3 to max_len
            # Using recursion or itertools.product
            import itertools
            for L in range(3, max_len + 1):
                for seq in itertools.product([1, -1, 2, -2], repeat=L):
                    if check_sequence(seq):
                        return True
            return False

        return not has_short_cycle(n, a, b, g - 1)

    # Search for circulant graph for k=4, g=10, n=320
    if k == 4 and g == 10:
        n = 320
        # Search a, b
        # a, b should be coprime to n? Not necessarily, but usually.
        # We can restrict a, b to 1..n/2
        found = False
        for a in range(1, n // 2 + 1):
            for b in range(a + 1, n // 2 + 1):
                if time.time() > deadline:
                    break
                if is_circulant_girth(n, a, b, g):
                    # Found!
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i + b) % n))
                    return n, edges
            if time.time() > deadline:
                break
        
        if not found:
            # Fall back to LUW or heuristic
            pass

    # Fallback to LUW
    return luw_construct(k, g)
