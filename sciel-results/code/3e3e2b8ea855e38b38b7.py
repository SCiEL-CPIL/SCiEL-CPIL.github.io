import time
import itertools
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # For the specific case k=4, g=10, try to find a Cayley graph on Z_16 x Z_20 (320 vertices)
    if k == 4 and g == 10:
        result = _try_cayley_320(deadline)
        if result is not None:
            return result

    # Fallback: LUW Construction
    return _luw_construct(k, g)

def _try_cayley_320(deadline):
    """
    Try to find a 4-regular Cayley graph on Z_16 x Z_20 with girth >= 10.
    Generators: a, -a, b, -b.
    We fix a = (1, 0).
    We search for b = (x, y).
    """
    m, n = 16, 20
    total = m * n
    
    # Group elements are (i, j) with 0 <= i < m, 0 <= j < n
    # a = (1, 0)
    # b = (x, y)
    
    # We need to check girth >= 10.
    # This means no cycles of length 3, 4, 5, 6, 7, 8, 9.
    
    # To check girth, we can run BFS from the identity (0,0).
    # If we find a cycle of length < 10, we reject.
    
    # Optimization: 
    # A cycle of length L corresponds to a word of length L evaluating to 0.
    # We can check if the identity is reachable by a non-backtracking walk of length < 10?
    # Actually, BFS is easiest.
    
    def check_girth(x, y):
        # Generators: (1,0), (-1,0), (x,y), (-x,-y)
        # Modulo m, n
        gens = [
            (1, 0),
            (m-1, 0),
            (x, y),
            ((-x)%m, (-y)%n)
        ]
        
        # BFS from (0,0)
        visited = {}
        visited[(0,0)] = 0
        queue = deque([(0,0)])
        
        while queue:
            u, v = queue.popleft()
            depth = visited[(u,v)]
            if depth >= 4: # If depth >= 4, any new cycle would be length >= 2*4+1=9? 
                # Actually, if we are at depth d, and we find an edge to a vertex at depth d'
                # Cycle length is d + 1 + d'.
                # We need d + 1 + d' >= 10.
                # So if d + d' >= 9, it's safe.
                # If d=4, d' can be 4, 5. 4+1+4=9 (bad). 4+1+5=10 (ok).
                # So we must check edges from depth 4 to depth 4.
                # And depth 3 to depth 4? 3+1+4=8 (bad).
                # So we need to check all edges.
                pass
            
            for g1, g2 in gens:
                nu = (u + g1) % m
                nv = (v + g2) % n
                if (nu, nv) not in visited:
                    visited[(nu, nv)] = depth + 1
                    queue.append((nu, nv))
                else:
                    # Found a cycle
                    # Cycle length = depth + 1 + visited[(nu, nv)]
                    # Wait, this is only if the edge is not a tree edge.
                    # In BFS, if (nu, nv) is already visited, the cycle length is depth + 1 + visited[(nu, nv)]
                    # ONLY IF the path from (nu, nv) to root does not go through (u, v).
                    # In a tree, the path is unique.
                    # So the cycle is formed by path(u) + edge + path(nu)^-1.
                    # Length = depth(u) + 1 + depth(nu).
                    cycle_len = depth + 1 + visited[(nu, nv)]
                    if cycle_len < 10:
                        return False
        
        # If we finish BFS, check all edges for short cycles?
        # The BFS check above catches cycles where the two endpoints are in the BFS tree.
        # But what about edges between two vertices at the same depth?
        # The BFS check above handles it: if (nu, nv) is visited, cycle_len = depth + 1 + depth.
        # If depth=4, cycle_len = 9. Rejected.
        # If depth=5, cycle_len = 11. Accepted.
        # So the BFS check is sufficient for undirected graphs?
        # Yes, any cycle can be decomposed into a path in the BFS tree and one edge.
        # The length of that cycle is depth(u) + 1 + depth(v).
        # If this is < 10, we reject.
        # Is it possible that the shortest cycle is NOT captured by this?
        # No, the BFS tree property ensures that the fundamental cycles have length depth(u) + 1 + depth(v).
        # The girth is the minimum of these fundamental cycles.
        
        return True

    # Search for b
    # We can restrict x, y to reduce symmetry?
    # a = (1, 0).
    # We can assume x is in a canonical form?
    # Let's just iterate.
    
    for x in range(m):
        if time.time() > deadline:
            return None
        for y in range(n):
            if time.time() > deadline:
                return None
            
            # Basic checks
            if x == 0 and y == 0:
                continue
            if x == 1 and y == 0: # a
                continue
            if x == m-1 and y == 0: # -a
                continue
            if x == 0 and y == n//2 and m % 2 == 0: # Potential -a if b has order 2?
                # -b = (m-x, n-y). If b = -b, then 2b = 0.
                # If 2b = 0, then b is an involution.
                # Then generators are a, -a, b. Only 3 distinct generators?
                # If b = -b, then the set is {a, -a, b}. Size 3.
                # We need 4 distinct generators for 4-regular simple graph.
                # So b must not be an involution.
                # 2b = 0 mod (m,n) => 2x=0 mod m, 2y=0 mod n.
                if (2*x) % m == 0 and (2*y) % n == 0:
                    continue
            
            if check_girth(x, y):
                # Construct edges
                edges = []
                for u in range(m):
                    for v in range(n):
                        # Add edges to a and b (to avoid duplicates, only add if u,v < neighbor)
                        # Or just add all and deduplicate?
                        # Better: add edge (u,v) to (u+1, v) and (u+x, v+y)
                        # And ensure we don't add duplicates.
                        # Since graph is undirected, add (u,v) to (u+1, v) only if (u,v) < (u+1, v) in some ordering?
                        # Simpler: collect all, then deduplicate.
                        # But for 320 vertices, 640 edges, it's small.
                        
                        # Edge to a
                        nu = (u + 1) % m
                        nv = v
                        if (u, v) < (nu, nv):
                            edges.append(((u*m+v, nu*m+nv)))
                        
                        # Edge to b
                        nu = (u + x) % m
                        nv = (v + y) % n
                        if (u, v) < (nu, nv):
                            edges.append(((u*m+v, nu*m+nv)))
                
                # Verify
                # Check degree
                deg = [0]*total
                for e1, e2 in edges:
                    deg[e1] += 1
                    deg[e2] += 1
                if all(d == 4 for d in deg):
                    return total, edges
                else:
                    # Should not happen if generators are distinct and non-zero
                    pass

    return None

def _luw_construct(k, g):
    """
    LUW construction for k-regular graphs with large girth.
    """
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
