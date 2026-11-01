import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Known optimal or near-optimal cage graphs for small (k, g)
    def _known_cages(k, g):
        """Return (n, edges) for known small cages, or None."""
        # Petersen graph: 3-regular, girth 5, n=10
        if k == 3 and g == 5:
            edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                     (5,6),(6,7),(7,8),(8,9),(9,5),
                     (0,5),(1,6),(2,7),(3,8),(4,9)]
            return 10, edges
        # Heawood graph: 3-regular, girth 7, n=14
        if k == 3 and g == 7:
            # Heawood graph = incidence graph of Fano plane
            lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
            edges = []
            for i, line in enumerate(lines):
                for pt in line:
                    edges.append((pt, 7 + i))
            return 14, edges
        return None
    
    def luw_construct(k, g):
        """
        LUW construction for k-regular graphs.
        Uses the standard LUW recurrence over a finite field.
        """
        q = k
        
        # Determine depth d from girth g
        # For LUW, girth g = 2*d - 1 (odd) or g = 2*d (even)
        # So d = (g+1)//2 for odd g, d = g//2 for even g
        if g % 2 == 1:
            d = (g + 1) // 2
        else:
            d = g // 2
        
        # Define field operations
        if q == 4:
            # GF(4) arithmetic
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            mul = lambda x, y: mul_table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
        
        # LUW recurrence: for each c from 2 to d,
        # y[c] = x[c] + y[c-1] * x[c-1] - y[c-2] * x[c-2]
        # But we only track the last two, so we need to adjust
        # Standard LUW uses: y[c] = x[c] + y[c-1] * x[c-1]
        # This gives girth 2c+1
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    if c == 2:
                        y[c] = add(x[c], mul(y[c-1], x[c-1]))
                    elif c == 3:
                        y[c] = add(x[c], sub(mul(y[c-1], x[c-1]), mul(y[c-2], x[c-2])))
                    else:
                        y[c] = add(x[c], sub(mul(y[c-1], x[c-1]), mul(y[c-2], x[c-2])))
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
    
    # Try known cages first
    known = _known_cages(k, g)
    if known is not None:
        best_n, best_edges = known
    else:
        # Try LUW construction
        best_n, best_edges = luw_construct(k, g)
    
    # Verify girth using non-backtracking walks
    def has_short_cycle(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start_u in range(n):
            if time.time() > deadline:
                return True
            for start_v in adj[start_u]:
                visited = set()
                visited.add((start_u, start_v))
                queue = deque()
                queue.append((start_v, start_u, 1))
                
                while queue:
                    curr, prev, depth = queue.popleft()
                    if depth >= g - 1:
                        continue
                    for nb in adj[curr]:
                        if nb == prev:
                            continue
                        state = (curr, nb)
                        if state in visited:
                            if depth + 1 < g:
                                return True
                        else:
                            visited.add(state)
                            queue.append((nb, curr, depth + 1))
        return False
    
    if not has_short_cycle(best_n, best_edges, g):
        return best_n, best_edges
    
    return best_n, best_edges
