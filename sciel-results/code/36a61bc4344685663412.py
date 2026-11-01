import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Helper: Check Girth using non-backtracking walks
    def check_girth(adjacency, n, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adjacency[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Helper: Verify Graph
    def verify_graph(n, edges, k, g):
        if n < 1:
            return False
        adjacency = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if u >= n or v >= n:
                return False
            if (min(u, v), max(u, v)) in edge_set:
                return False
            edge_set.add((min(u, v), max(u, v)))
            adjacency[u].append(v)
            adjacency[v].append(u)
        
        for i in range(n):
            if len(adjacency[i]) != k:
                return False
        
        if not check_girth(adjacency, n, g):
            return False
            
        return True

    # GQ(4,4) incidence graph construction
    def gq44_construct():
        # GF(4) = {0, 1, 2, 3} with 2=omega, 3=omega^2, omega^2=omega+1
        # Addition is XOR
        # Multiplication: 2*2=3, 2*3=1, 3*3=2
        
        def gf_add(a, b):
            return a ^ b
        
        def gf_mul(a, b):
            if a == 0 or b == 0:
                return 0
            if a == 1:
                return b
            if b == 1:
                return a
            # a,b in {2,3}
            if a == 2 and b == 2:
                return 3
            if a == 2 and b == 3:
                return 1
            if a == 3 and b == 2:
                return 1
            if a == 3 and b == 3:
                return 2
        
        # GQ(4,4) has 85 points and 85 lines
        # Construction: Points are (i, j) where i in {0,1,2,3,4}, j in GF(4)^2
        # Plus 5 "infinity" points
        # Total: 5*16 + 5 = 85
        
        # Lines are similarly indexed
        # Incidence: point (i, (a,b)) is on line (i, (c,d)) if a=c and b=d (trivial)
        # Plus additional incidences based on the GQ structure
        
        # Actually, let me use a different, more standard construction.
        # GQ(4,4) can be constructed as follows:
        # Points: (x, y, z) in GF(4)^3 with x^2 + y^2 + z^2 = 0 (Hermitian)
        # This gives (4+1)^2 - 1 = 15 points on the Hermitian curve, not 85.
        
        # Let me use the "canonical" GQ(4,4) construction from the literature.
        # The points are:
        # - 16 points of the form (x, y) with x,y in GF(4)
        # - 64 points of the form (x, y, z) with x,y,z in GF(4) and x+y+z=0
        # - 5 points at infinity
        # Total: 16 + 64 + 5 = 85
        
        # Wait, x+y+z=0 in GF(4) means z = x+y (since char 2).
        # So there are 4*4 = 16 such points, not 64.
        # 16 + 16 + 5 = 37. Not 85.
        
        # I think I need a different construction. Let me use the following:
        # GQ(4,4) points: all (u, v) where u in GF(4)^2 and v in GF(4), plus 5 special points.
        # 4^2 * 4 + 5 = 64 + 5 = 69. Still not 85.
        
        # The correct count: (s+1)(st+1) = 5*17 = 85.
        # So there are 85 points.
        
        # Let me use a concrete enumeration. I'll define points as tuples.
        # Points 0-15: (0, a, b) for a,b in GF(4)
        # Points 16-31: (1, a, b) for a,b in GF(4)
        # Points 32-47: (2, a, b) for a,b in GF(4)
        # Points 48-63: (3, a, b) for a,b in GF(4)
        # Points 64-79: (4, a, b) for a,b in GF(4)
        # Points 80-84: infinity points
        # Total: 80 + 5 = 85
        
        # Lines: similarly 85 lines
        # Line (i, a, b) contains points based on the GQ incidence structure
        
        # This is getting too complex without a reference. Let me abort this approach
        # and use LUW instead.
        return None

    # LUW Construction
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

    # Try GQ(4,4) for k=5, g<=10
    if k == 5 and g <= 10:
        gq_result = gq44_construct()
        if gq_result is not None:
            n, edges = gq_result
            if verify_graph(n, edges, k, g):
                return n, edges

    # Fallback to LUW
    luw_n, luw_edges = luw_construct(k, g)
    
    # Verify LUW
    if verify_graph(luw_n, luw_edges, k, g):
        return luw_n, luw_edges
        
    # If LUW fails (should not), return empty
    return 0, []
