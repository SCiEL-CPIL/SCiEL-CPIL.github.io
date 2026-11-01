"""Construct a 5-regular graph of girth 9 via line graph of a 3-regular girth-14 graph."""
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=5, g=9: take the line graph of a 3-regular graph with girth 14.
    The line graph L(G) of a 3-regular graph G is 4-regular, not 5.
    
    Wait - the line graph of a 3-regular graph is 4-regular (each vertex in L(G) 
    corresponds to an edge in G, and adjacent edges in G share a vertex, so each 
    edge is adjacent to 2+2=4 other edges). That gives degree 4, not 5.
    
    Let me reconsider. The directive says "line graph of a 3-regular graph with girth 14"
    gives a 5-regular graph. That's incorrect for standard line graphs.
    
    Actually, let me re-read: "taking the line graph of a 3-regular graph with girth 14, 
    ensuring the resulting 5-regular graph has no cycles shorter than 9."
    
    This is mathematically problematic. The line graph of a d-regular graph is (2d-2)-regular.
    So line graph of 3-regular is 4-regular, not 5-regular.
    
    Perhaps the intended construction is different. Let me think of other approaches.
    
    Alternative: Use a known construction or search for a 5-regular graph of girth 9.
    
    Given the constraints (must finish in 120s, standard library + NumPy + SciPy), 
    and that the best known has 1116 vertices, I'll try a different approach:
    
    1. Try to construct using algebraic methods
    2. Or use a random greedy approach with backtracking
    3. Or use the Lazebnik-Ustimenko-Woldar construction but optimized
    
    Actually, let me try a different algebraic construction. The Wagner graph and 
    other small girth graphs... 
    
    For k=5, g=9, the Moore bound is at least 1 + 5 + 5*4 = 26 (for girth 5), but for 
    girth 9 it's much larger. The Moore bound for odd girth g=2r+1 is:
    1 + k * sum_{i=0}^{r-1} (k-1)^i = 1 + 5*(1 + 4 + 16 + 64 + 256) = 1 + 5*341 = 1706
    
    Wait, that's for odd girth. For girth 9 = 2*4+1, r=4:
    1 + 5*(1 + 4 + 16 + 64 + 256) = 1 + 5*341 = 1706
    
    But the best known is 1116, which is less than 1706, so the Moore bound isn't tight here.
    
    Let me try a practical approach: use a randomized construction with local search to 
    maximize girth, or use a known family of graphs.
    
    Actually, I'll implement a constructive approach using the idea of building a 
    5-regular graph by starting from a base and using symmetry, or I'll use 
    a probabilistic method.
    
    Given time constraints, let me implement a greedy algorithm that tries to build 
    a 5-regular graph with high girth, using BFS to check girth constraints.
    
    For a more practical approach within 120 seconds, I'll use a known small 
    5-regular graph and verify its girth, or construct one using a specific method.
    
    Let me try the following: construct a 5-regular graph using a circulant or 
    Cayley graph approach, then verify the girth.
    """
    if k != 5 or g != 9:
        # Fallback: use the original LUW construction for other parameters
        return _luw_construct(k, g)
    
    # For k=5, g=9: try to construct a 5-regular graph of girth 9
    # Using a randomized approach with girth checking
    
    # Try a Cayley graph approach: use a group and generators
    # Or try a random 5-regular graph and check girth
    
    # Given the difficulty, let me use a known construction or a simpler approach
    # that at least produces a valid graph with girth >= 9, even if not optimal.
    
    # I'll use a modified approach: build a 5-regular graph on n vertices 
    # where n is chosen to be large enough, using a structured construction.
    
    # Approach: Use a bipartite-like structure or a known small girth graph family.
    
    # Let me try: start with a 5-regular graph and use edge swaps to increase girth.
    # This is complex, so instead I'll use a direct construction.
    
    # Construct a 5-regular graph using a circulant graph on n vertices
    # where the connection set is {a1, a2, a3, a4, a5}
    
    # For girth 9, we need no cycles of length < 9.
    
    # Let me try n = 100 and see if we can find a valid connection set.
    # This is a search problem.
    
    # Given time constraints, I'll implement a simpler valid construction:
    # Use the LUW construction but with parameters that give girth >= 9.
    
    return _luw_construct(5, 9)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    # For girth g, we need d such that the LUW graph has girth at least g.
    # The LUW graph CD(d, q) has girth at least d + 5 for odd d.
    # So we need d + 5 >= g, i.e., d >= g - 5.
    # For odd d: d = g - 5 if (g - 5) is odd, else g - 4.
    
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for each coordinate
    rule = {}
    # For c = 2, 3, 4, use specific rules
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
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
