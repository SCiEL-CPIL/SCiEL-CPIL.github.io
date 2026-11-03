"""Construct 3-regular graph of girth 16 with minimal vertices."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110  # safety margin before 150s limit
    
    # For k=3, g=16: try to build a small 3-regular graph with girth >= 16
    # Use a deterministic construction based on known cages or algebraic methods
    
    # First, try LUW construction as fallback (guaranteed valid)
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
    
    # Try to build a smaller graph using a known construction or search
    # For k=3, g=16, use a deterministic algebraic construction
    
    # Use the (3,16)-cage construction if possible, otherwise LUW
    if k == 3 and g == 16:
        # Try to construct a smaller graph
        # Use a known 3-regular graph with girth 16
        # The (3,16)-cage has 936 vertices
        # Try to build it using a deterministic method
        
        # Approach: Use a tree-like construction with careful edge matching
        # Start with a 3-ary tree of depth 7 (girth 16 requires no cycles < 16)
        # A 3-regular graph with girth 16 has at least 2*(1+3+9+27+81+243+729) = 5120 vertices
        # Wait, that's the Moore bound: 2*(3^7 - 1)/2 = 3^7 - 1 = 2186... 
        # Actually Moore bound for 3-regular, girth 16: n >= 2 * sum_{i=0}^{7} 3^i = 2*(3^8-1)/2 = 3^8 - 1 = 6560
        # Hmm, let me recalculate: for even girth g=2r, Moore bound is 2 * sum_{i=0}^{r-1} (k-1)^i
        # For k=3, r=8: 2 * sum_{i=0}^{7} 2^i = 2 * (2^8 - 1) = 2 * 255 = 510
        # So minimum is 510, best known is 936
        
        # Try to construct a graph with ~936 vertices
        # Use a deterministic construction based on circulant or algebraic methods
        
        # For now, use LUW as it's guaranteed to work
        pass
    
    # Use LUW construction
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
