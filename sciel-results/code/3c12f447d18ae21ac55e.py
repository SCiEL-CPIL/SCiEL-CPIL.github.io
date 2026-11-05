import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using known constructions.
    For k=4, g=10, returns the (4,10)-cage (320 vertices) if known, or uses
    the Lazebnik-Ustimenko-Woldar construction as fallback.
    """
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # For k=4, g=10: try to use known small cages or constructions
    if k == 4 and g == 10:
        # Try to construct a 4-regular graph with girth 10
        # The Moore bound for (4,10) is 2*(1 + 4 + 4^2 + ... + 4^4) = 2*(1+4+16+64+256) = 650
        # Actually for bipartite: n >= 2 * (k^(g/2) - 1)/(k - 1) = 2 * (4^5 - 1)/3 = 2*341/3 ≈ 227.33, so n >= 228
        # But wait, girth 10 means no cycles of length < 10, so girth >= 10
        # Moore bound for girth g (even): n >= 2 * (k^(g/2) - 1)/(k - 1)
        # For k=4, g=10: n >= 2 * (4^5 - 1)/(4-1) = 2 * 1023/3 = 682
        # Hmm, that's the Moore bound. The actual cage is 320 vertices per the task description.
        # Let me just use the LUW construction or try to find a good one.
        
        # Use a known 4-regular (4,10)-cage construction
        # The (4,10)-cage has 320 vertices. We can try to construct it using
        # the generalized polygon approach or known explicit constructions.
        
        # For now, use the Lazebnik-Ustimenko-Woldar graph which gives a valid
        # 4-regular graph with girth at least 10 (it may have more vertices than optimal)
        q = 4
        # For girth g=10, we need d such that girth >= d+5 >= 10, so d >= 5
        d = 5
        
        # GF(4) arithmetic
        # GF(4) = {0, 1, alpha, alpha^2} where alpha^2 = alpha + 1
        # Represent elements as 0, 1, 2, 3 where 2=alpha, 3=alpha^2
        # Addition is XOR
        # Multiplication:
        # 0*x=0, 1*x=x, 2*2=3, 2*3=1, 3*3=2
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        
        def gf_add(x, y):
            return x ^ y
        
        def gf_sub(x, y):
            return x ^ y  # In char 2, subtraction = addition
        
        def gf_mul(x, y):
            return mul_table[x][y]
        
        # Rule for LUW graph
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
                        y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                    else:
                        y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
                yield (1 - side, tuple(y))
        
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
    
    elif k == 4 and g == 8:
        # Hoffman-Singleton type or known (4,8)-cage
        # Use LUW with d=3 (girth >= 8)
        q = 4
        d = 3
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def gf_add(x, y):
            return x ^ y
        def gf_sub(x, y):
            return x ^ y
        def gf_mul(x, y):
            return mul_table[x][y]
        
        rule = {2: (1, 1), 3: (2, 1)}
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                    else:
                        y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
                yield (1 - side, tuple(y))
        
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
    
    else:
        # General fallback: use LUW construction
        q = k
        if q == 4:
            mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            def gf_add(x, y):
                return x ^ y
            def gf_sub(x, y):
                return x ^ y
            def gf_mul(x, y):
                return mul_table[x][y]
        else:
            def gf_add(x, y):
                return (x + y) % q
            def gf_sub(x, y):
                return (x - y) % q
            def gf_mul(x, y):
                return x * y % q
        
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
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
                        y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                    else:
                        y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
                yield (1 - side, tuple(y))
        
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
# EVOLVE-BLOCK-END
