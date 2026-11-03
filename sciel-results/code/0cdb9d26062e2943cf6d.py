<<<<<<< SEARCH
import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: LUW Construction
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
=======
import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Optimized LUW Construction with better parameter search
    def luw_construct_optimized(k, g):
        """
        LUW (Lubell-Ulam-Wright) type construction for cage graphs.
        We try multiple rule configurations to minimize vertex count.
        """
        q = k  # Field size
        
        # For girth g, we need depth d = ceil(g/2) - 1 for the tree part
        # The actual diameter of the graph will be 2*d + 1 = g-1 or g
        d = max(3, (g - 1) // 2)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Try multiple rule patterns
        best_n = float('inf')
        best_edges = None
        best_rule = None
        
        # Generate candidate rules
        candidate_rules = []
        
        # Pattern 1: Original LUW
        rule1 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule1[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        candidate_rules.append(rule1)
        
        # Pattern 2: Alternative spacing
        rule2 = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 3, c - (c - 5) % 3
            rule2[c] = [(1, c0 - 1), (c0, 1), (1, c0 + 1)][r]
        candidate_rules.append(rule2)
        
        # Pattern 3: Linear progression
        rule3 = {}
        for c in range(2, d + 1):
            a = (c - 1) % d + 1
            b = c % d + 1
            rule3[c] = (a, b)
        candidate_rules.append(rule3)
        
        # Pattern 4: Based on golden ratio-like spacing
        rule4 = {}
        for c in range(2, d + 1):
            a = (c * 2 - 1) % d + 1
            b = (c * 3 - 1) % d + 1
            rule4[c] = (a, b)
        candidate_rules.append(rule4)
        
        for rule in candidate_rules:
            # Verify rule is valid (no self-loops in the rule definition)
            valid = True
            for c in range(2, d + 1):
                a, b = rule[c]
                if a == b:
                    valid = False
                    break
            if not valid:
                continue
            
            # Build the graph
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
            
            n = len(order)
            if n < best_n:
                best_n = n
                best_edges = edges
                best_rule = rule
            
            # Time check
            if time.time() > deadline - 20:
                break
        
        if best_edges is None:
            # Fallback to original
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
            
            best_n = len(order)
            best_edges = edges
        
        return best_n, best_edges
>>>>>>> REPLACE

