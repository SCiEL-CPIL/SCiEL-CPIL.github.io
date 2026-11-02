import time
import random
import sys
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; keep margin before 150s kill

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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

    def check_girth(n, edges, g):
        """Check girth using BFS from each vertex (non-backtracking)."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check no cycles shorter than g using BFS with parent tracking
        for start in range(n):
            # BFS tracking (node, parent) to avoid backtracking
            visited = set()
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            
            found_short = False
            while queue:
                node, parent, dist = queue.popleft()
                if dist >= g:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if (nb, node) in visited:
                        if dist + 1 < g:
                            return False
                    else:
                        if dist + 1 < g:
                            visited.add((nb, node))
                            queue.append((nb, node, dist + 1))
        return True

    def gp_search(k, g, max_time):
        """
        Search for a small 3-regular graph with girth >= g using
        a GP (Generalized Polygon) tree completion approach.
        
        For girth g, the graph must contain a GP(g/2, 3) subtree around
        each edge. We grow the tree and try to close cycles at length g.
        """
        if k != 3:
            return None
        
        # For girth g, we need g to be even
        if g % 2 != 0:
            return None
        
        half_g = g // 2  # radius of the GP tree
        
        # The GP(3,3) tree has 2 + 3*(1+3+9+...) = 2 + 3*(40) = 122 vertices for radius 4
        # For g=16, half_g=8, the tree is much larger
        # Actually for girth 16, we need a GP(8,3) which has:
        # Level 0: 1, Level 1: 3, Level 2: 3*2=6, Level 3: 6*2=12, ...
        # Total = 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = 766 vertices in the tree
        # But the actual graph can be smaller due to identifications at the boundary
        
        # Start with a smaller search: try to build a graph with n vertices
        # where n is between the Moore bound and some upper bound
        
        # Moore bound for (3,16): 1 + 3*sum_{i=0}^{7} 2^i = 1 + 3*255 = 766
        # So minimum n is 766. Known best is 936.
        
        # Let's try a different approach: use the known structure of small cages
        # The (3,16) cage has 936 vertices. Let's try to construct it directly.
        
        # Alternative: Use a random restart search with local improvement
        # Start with a random 3-regular graph of size n, then try to increase girth
        
        # For now, let's try a structured approach:
        # Build a tree of depth half_g-1 from a root edge, then try to close
        # the boundary vertices to form cycles of length exactly g
        
        # This is complex. Let's instead try a simpler heuristic:
        # Generate random 3-regular graphs of increasing size and check girth
        # This is slow but might work for small n
        
        # Better: Use the fact that the (3,16) cage is known to exist with 936 vertices
        # and try to construct it using a known method
        
        # For this implementation, let's try a few specific constructions:
        
        # 1. Try to find a graph using a Cayley graph of a small group
        # 2. Try to use a known cage database entry
        
        # Given time constraints, let's return the LUW graph as fallback
        # but first try a quick random search for small graphs
        
        # Quick random search: try n from 766 to 936 in steps
        # This is too slow in pure Python
        
        # Instead, let's use a smarter approach:
        # The (3,16) cage can be constructed as a covering graph
        # of a smaller graph
        
        # For now, return None to indicate this method failed
        return None

    # Try GP search first
    result = gp_search(k, g, deadline - (time.time() - t0) - 5.0)
    if result is not None:
        return result
    
    # Fallback to LUW
    return luw_construct(k, g)

# EVOLVE-BLOCK-END
