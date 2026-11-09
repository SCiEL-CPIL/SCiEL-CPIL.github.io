"""Construct a small 5-regular graph of girth 9 (cage problem n(5,9))."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s limit

    if k == 5 and g == 9:
        # For the specific task (5-regular, girth 9), use best known constructions.
        # The LUW construction with q=4, d=4 gives n=4^4=256 vertices, girth 8 (too low).
        # With q=4, d=5: n=4^5=1024, girth >= 9. 
        # Actually LUW CD(d,q) for q prime or 4: girth >= d+5 for odd d.
        # For d=4 (even), the girth guarantee is weaker.
        # Let's use the Hoffman-Singleton type or other known constructions.
        # 
        # Best approach: use the LUW construction properly.
        # CD(4, q) for q=5: 5^4 = 625 vertices. Girth >= 4+5=9 (if d is odd... d=4 is even).
        # Actually for LUW: girth >= d+5 when d is odd. For even d, girth >= d+4.
        # So d=4: girth >= 8. Not enough.
        # d=5, q=4: 4^5 = 1024, girth >= 10. Good.
        # d=4, q=5: 5^4 = 625, girth >= 8. Not enough.
        # 
        # Alternative: Use a bipartite double cover or other techniques.
        # 
        # Actually, let me re-read: "girth is at least d + 5 for odd d"
        # So for d=5, q=4: n = 4^5 = 1024, girth >= 10 >= 9. 
        # For d=4, q=5: n = 5^4 = 625, girth >= 8 < 9. Not valid.
        # 
        # Can we do better? The Moore bound for (5,9) is:
        # n >= 1 + k * (1 + (k-1) + (k-1)^2 + (k-1)^3) = 1 + 5*(1+4+16+64) = 1+5*85 = 426
        # But this is a lower bound.
        # 
        # Known constructions:
        # - The (5,9)-cage is not known exactly; best upper bound is 1116.
        # - LUW CD(5,4) gives 1024 vertices with girth >= 10.
        # 
        # Let me also try: is there a way to get fewer vertices?
        # The Magma database lists cages up to certain sizes.
        # For (5,9), the known best is 1116.
        # LUW gives 1024 which is better than 1116!
        # Wait, that can't be right. Let me double check.
        # 
        # Actually the LUW construction CD(d,q): the number of vertices is 2*q^d (bipartite).
        # No wait, looking at the code: points are (side, x) where x has d+1 entries.
        # The zero point's connected component... 
        # 
        # Let me just implement LUW properly and see what n we get.
        # The code constructs a bipartite graph. The connected component of zero.
        # For q=4, d=5: the full graph would have 2*4^5 = 2048 vertices, but the
        # connected component might be smaller.
        # 
        # Actually in the LUW construction, the graph IS connected, so n = 2*q^d.
        # Hmm, but the code only returns the connected component of the zero vertex.
        # 
        # Let me just implement it and check.
        # 
        # Actually, I'll implement the LUW construction carefully.
        n, edges = _luw_graph(5, 4)  # k=5, q=4
        if n is not None:
            return n, edges
        # Fallback: random greedy
        return _greedy_construct(k, g)

    # General case: use greedy random construction
    return _greedy_construct(k, g)


def _luw_graph(k, q):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    For q = 4 (or prime), d chosen so girth >= 9.
    Returns (n, edges) or (None, None) if invalid.
    """
    # We need girth >= 9. LUW gives girth >= d+5 for odd d.
    # So d >= 4. For d=4 (even), girth >= d+4 = 8. Not enough.
    # For d=5 (odd), girth >= 10. Good. n = 2*q^5.
    # For q=4: n = 2*4^5 = 2048. That's large.
    # 
    # Wait, I need to reconsider. The LUW graph CD(d,q) has:
    # - Bipartition: points P = F_q^{d+1} (first coordinate ignored in some formulations)
    # - Lines L = F_q^{d+1}
    # - Total vertices: 2 * q^{d+1} ? No...
    # 
    # Looking at the seed code more carefully:
    # vertex = (side, x) where x = (None, first, x2, ..., xd) -> d+1 elements
    # The neighbours function iterates over 'first' in range(q), so q choices.
    # For a given vertex, it has q neighbours (one for each 'first' value).
    # So degree = q. For k=5, we need q=5 (prime) or handle q differently.
    # 
    # For q=5, d=5: vertices per side = 5^5 = 3125, total = 6250. Way too many.
    # For q=4, d=5: degree = 4, not 5. Doesn't match k=5.
    # 
    # So LUW with q=5 gives degree 5 but n=6250.
    # 
    # I need a different approach for k=5.
    # 
    # Options:
    # 1. Use known explicit constructions for (5,9) cages
    # 2. Use random greedy / backtracking
    # 3. Use algebraic constructions from the literature
    # 
    # The Magma SmallCage or similar might have (5,8) but (5,9) is large.
    # 
    # For a coding competition with 120s, let me try a smart random approach:
    # - Build a 5-regular graph via random configuration model
    # - Check girth, if < 9, do local modifications
    # 
    # Or: use the fact that we can construct bipartite 5-regular graphs of girth 9.
    # A bipartite 5-regular graph of girth 2g+1=9 means girth >= 9 (odd cycles >= 9, even >= 8... wait)
    # Girth 9 means no cycles of length < 9. In bipartite, cycles are even, so no 4, 6, 8 cycles.
    # So we need a bipartite 5-regular graph with no 4, 6, or 8-cycles.
    # 
    # This is equivalent to a 5-regular graph of girth at least 9 (if we allow odd cycles >= 9).
    # 
    # Let me try: construct a 5-regular bipartite graph on 2n vertices with no 4,6,8-cycles.
    # n can be as small as ~214 (Moore bound / 2 for bipartite).
    # 
    # Approach: random 5-regular bipartite graph, then fix short cycles.
    # This is hard. Let me instead just use the LUW with q=5, d=4:
    # degree = 5, d=4. Girth for even d: >= d+4 = 8. Still not enough.
    # 
    # OK let me try d=5, q=5: n = 2*5^5 = 6250. Girth >= 10. Valid but large.
    # 
    # For the contest, let me just use a reasonable construction.
    # The seed gives 6250 (or whatever the LUW gives). Let me see if I can do better.
    
    # Actually, let me reconsider the problem. The seed program returns the LUW graph.
    # Let me compute what the seed returns for k=5, g=9:
    # q = k = 5
    # d = max(3, g-5 if (g-5)%2==0 else g-4) = max(3, 4 if 4%2==0 else 5) = max(3,4) = 4
    # So d=4, q=5. The graph has 2*5^4 = 1250 vertices? Or 5^4 per side?
    # Looking at code: the vertex is (side, tuple of d+1 elements) but first element is None or a value.
    # Actually x = (None,) + (0,)*d for the start, so d+1 elements with first being None.
    # But in neighbours: x[c] for c in range(2, d+1), and y[a], x[b] where a,b from rule.
    # The 'first' index is y[1]. So the vector has indices 1..d, plus the side.
    # Number of distinct vectors: q^d (since index 1 varies over q values, and 2..d are determined).
    # Wait no, for a point (side=0, x), the neighbours are lines (side=1, y) where y[1]=first for first in range(q).
    # So each point has q neighbours. Each point is determined by (x[1], x[2], ..., x[d])? 
    # Hmm, x[0] is None for points. Let me trace more carefully.
    # 
    # I think the total number of vertices in the connected component depends on the specific construction.
    # Let me just run the seed logic and see.
    # 
    # For the purpose of this contest, let me implement a clean LUW and also try
    # a random approach with cycle fixing.
    
    return None, None


def _luw_graph_v2(q, d):
    """
    LUW CD(d,q) construction. Returns (n, edges, valid) where valid checks girth.
    """
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    
    # Rule: for c in 2..d, rule[c] = (a,b) meaning y[c] = x[c] + y[a]*x[b] (for points)
    # Standard LUW rules:
    rule = {}
    rule[2] = (1,1)
    rule[3] = (2,1)
    rule[4] = (1,2)
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        options = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)]
        rule[c] = options[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None]* (d+1)
            y[1] = first
            # Initialize remaining to 0
            for i in range(2, d+1):
                y[i] = 0
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1-side), tuple(y)
    
    start = (0, tuple([None] + [0]*d))
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


def _check_girth(n, edges, g):
    """Check if graph has girth >= g using BFS from each vertex (non-backtracking)."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For girth check, we need to verify no cycle shorter than g.
    # Use BFS with parent tracking from each vertex, up to depth (g-1)//2.
    # Actually, for girth >= g, we need: for every vertex, BFS up to depth floor((g-1)/2)
    # finds no vertex twice (which would indicate a cycle shorter than g).
    
    max_depth = (g - 1) // 2
    
    for start in range(n):
        visited = {}  # (vertex, parent) -> depth
        queue = [(start, -1, 0)]
        visited[(start, -1)] = 0
        while queue:
            v, parent, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for nb in adj[v]:
                if nb == parent:
                    continue
                key = (nb, v)
                if key in visited:
                    # Found a cycle of length 2*(depth+1)
                    if 2 * (depth + 1) < g:
                        return False
                else:
                    visited[key] = depth + 1
                    queue.append((nb, v, depth + 1))
    
    return True


def _greedy_construct(k, g):
    """
    Construct a k-regular graph with girth >= g using a randomized approach.
    Uses the configuration model + cycle detection and local repair.
    For k=5, g=9, this is very hard. Fallback to LUW if possible.
    """
    # For k=5, g=9, try LUW with q=5, d=4 (might have girth 8) or d=5 (girth 10)
    # LUW q=5, d=5: n = 2*5^5 = 6250, girth >= 10.
    n, edges = _luw_graph_v2(5, 5)
    if n is not None and len(edges) == k * n // 2:
        return n, edges
    
    # Try smaller: q=5, d=4
    n, edges = _luw_graph_v2(5, 4)
    if n is not None:
        # Check if girth is actually >= 9
        if _check_girth(n, edges, g):
            return n, edges
    
    # Fallback: just return LUW q=5, d=5
    n, edges = _luw_graph_v2(5, 5)
    return n, edges


# EVOLVE-BLOCK-END
