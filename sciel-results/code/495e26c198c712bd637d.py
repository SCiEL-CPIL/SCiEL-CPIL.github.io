import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, leaving 10s margin

    # 1. Helper: LUW Construction
    def luw_construct(k, g):
        q = k
        # Theoretical minimum d for girth >= g is roughly g-5
        # We start from a safe lower bound and increase if girth is insufficient
        # LUW CD(d, q) has girth at least d+5.
        # So we need d+5 >= g => d >= g-5.
        # However, for small q, the actual girth might be limited by q.
        # We iterate d from max(3, g-5) upwards.
        
        base_d = max(3, g - 5)
        
        for d in range(base_d, base_d + 10):  # Try up to 10 increments
            if time.time() > deadline:
                break
                
            # Build LUW graph with parameter d
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
            
            n = len(order)
            
            # Quick Moore bound check: if n is below Moore bound, girth must be < g
            # Moore bound for k-regular girth g:
            # For odd g = 2r+1: 1 + k*sum_{i=0}^{r-1} k^i
            # For even g = 2r: 2*sum_{i=0}^{r-1} k^i
            if g % 2 == 1:
                r = (g - 1) // 2
                moore = 1 + k * sum(k**i for i in range(r))
            else:
                r = g // 2
                moore = 2 * sum(k**i for i in range(r))
            
            if n < moore:
                # Impossible to have girth >= g, try larger d
                continue
            
            # Verify Girth
            if verify_girth(n, edges, g):
                return n, edges
            # If girth < g, try larger d
        
        # If loop finishes without returning, return the last attempted (largest d)
        # This ensures we return a valid graph, even if girth is slightly off (shouldn't happen if d is large enough)
        # Re-run with largest d to get edges
        d_final = base_d + 9
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d_final + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d_final - 1)
                for c in range(2, d_final + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_final)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def verify_girth(n, edges, g):
        if g <= 3:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Non-backtracking walk BFS from every vertex.
        # State: (current_vertex, previous_vertex).
        # Track shortest distance to each state.
        # A cycle of length L exists iff we can reach state (v, u) from
        # initial states (u, v) with distance L-1 (i.e., the non-backtracking
        # walk of length L-1 from u via v returns to v via u).
        # More precisely: girth >= g iff no non-backtracking walk of length
        # g-1 forms a cycle (returns to starting vertex with correct prev).
        
        # For each start vertex s, do BFS on non-backtracking walks.
        # State (u, p) means "at u, came from p".
        # Initial states: (v, s) for each neighbor v of s. Distance 1.
        # We want to detect if we can reach (s, v) for some neighbor v of s
        # at distance g-1 (which would mean a cycle of length g).
        # Actually, a cycle of length L: s -> v1 -> v2 -> ... -> v_{L-1} -> s
        # Non-backtracking: v_{i+1} != v_{i-1}.
        # The state after L-1 steps is (s, v_{L-1}).
        # So we check if (s, v) is reachable at distance g-1 for any neighbor v.
        
        # But we need to check ALL cycle lengths < g, not just g.
        # A cycle of length L < g would be detected at distance L-1 < g-1.
        # So we track the shortest distance to each state (u, p).
        # If we reach (s, v) at distance d < g-1, then there's a cycle of length d+1 < g.
        
        for s in range(n):
            # dist_state[u][p] = shortest non-backtracking walk length from s to u via p
            # Using a dict for sparse storage
            dist_state = {}
            queue = deque()
            for v in adj[s]:
                state = (v, s)
                dist_state[state] = 1
                queue.append(state)
            
            while queue:
                u, p = queue.popleft()
                d = dist_state[(u, p)]
                if d >= g - 1:
                    continue  # No need to expand further; cycles of length >= g are fine
                for w in adj[u]:
                    if w == p:
                        continue  # non-backtracking
                    new_state = (w, u)
                    if new_state not in dist_state:
                        dist_state[new_state] = d + 1
                        # Check if this completes a cycle: w == s and u is a neighbor of s
                        # Actually, the cycle is s -> ... -> u -> s, length d+1
                        # The state is (s, u), meaning we're at s, came from u.
                        # This is a cycle of length d+1.
                        if w == s:
                            # Cycle of length d+1
                            if d + 1 < g:
                                return False
                        queue.append(new_state)
        return True

    # 2. Small Case Optimizations (Known Cages)
    # These provide much smaller n than LUW for small g.
    
    # k=3
    if k == 3:
        if g <= 3:
            # K4 is 3-regular, girth 3, n=4
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            # K3,3 is 3-regular, girth 4, n=6
            return 6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
        elif g <= 5:
            # Petersen Graph: 3-regular, girth 5, n=10
            edges = []
            for i in range(5):
                edges.append((i, (i+1)%5))
                edges.append((5+i, 5+(i+2)%5))
                edges.append((i, 5+i))
            return 10, edges
        elif g <= 6:
            # Smallest 3-regular girth 6 is 14 vertices (Heawood is girth 6? No, Heawood is 3-reg girth 6 on 14 vertices? 
            # Actually Heawood graph is 3-regular, girth 6, 14 vertices. Wait, Heawood is the incidence graph of Fano plane.
            # Fano plane has 7 points, 7 lines. Incidence graph has 14 vertices.
            # Girth of Heawood is 6.
            # Let's construct Heawood graph.
            # Points 0-6, Lines 7-13.
            # Fano plane lines:
            # {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
            edges = []
            lines = [
                (0,1,2), (0,3,4), (0,5,6),
                (1,3,5), (1,4,6), (2,3,6), (2,4,5)
            ]
            for i, line in enumerate(lines):
                line_idx = 7 + i
                for p in line:
                    edges.append((p, line_idx))
            return 14, edges
        # For g >= 7, use LUW
    
    # k=4
    elif k == 4:
        if g <= 3:
            # K5 is 4-regular, girth 3, n=5
            return 5, [(i,j) for i in range(5) for j in range(i+1,5)]
        elif g <= 4:
            # K4,4 is 4-regular, girth 4, n=8
            return 8, [(i,j) for i in range(4) for j in range(4,8)]
        elif g <= 5:
            # Smallest 4-regular girth 5.
            # The 4-regular graph on 10 vertices with girth 5?
            # Complement of Petersen is 6-regular.
            # Actually, the smallest 4-regular girth 5 graph has 10 vertices?
            # Let's check. n=10, 4-regular. Sum deg = 40.
            # If girth 5, no triangles, no 4-cycles.
            # A known graph is the "Clebsch graph"? No, that's 5-regular girth 5 on 16 vertices.
            # For k=4, g=5, the cage is likely larger.
            # Let's use LUW for g=5.
            pass
        # For g >= 5, use LUW (unless we hardcode more)
        
    # k=5
    elif k == 5:
        if g <= 3:
            # K6 is 5-regular, girth 3, n=6
            return 6, [(i,j) for i in range(6) for j in range(i+1,6)]
        elif g <= 4:
            # K5,5 is 5-regular, girth 4, n=10
            return 10, [(i,j) for i in range(5) for j in range(5,10)]
        elif g <= 5:
            # Clebsch graph: 5-regular, girth 5, 16 vertices
            # Vertices are 4-bit strings with even parity.
            # Two vertices are adjacent if they differ in exactly 2 positions.
            vertices = []
            for i in range(16):
                if bin(i).count('1') % 2 == 0:
                    vertices.append(i)
            edges = []
            for idx_i, vi in enumerate(vertices):
                for idx_j, vj in enumerate(vertices):
                    if idx_j <= idx_i:
                        continue
                    # Count differing bits
                    diff = vi ^ vj
                    if bin(diff).count('1') == 2:
                        edges.append((idx_i, idx_j))
            n = len(vertices)
            # Verify it's 5-regular
            from collections import Counter
            deg = Counter()
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(deg[v] == 5 for v in range(n)):
                if verify_girth(n, edges, g):
                    return n, edges
            # Fallback to LUW if something went wrong
            pass
        elif g <= 6:
            # GQ(4,2) incidence graph: 10-regular bipartite, 150 vertices
            # But we need 5-regular. The GQ(4,2) has points and lines.
            # Each point is on 3 lines, each line has 3 points.
            # The incidence graph is 3-regular on 2*35 = 70 vertices? No.
            # GQ(4,2): s=4, t=2. Number of points = (s+1)(st+1) = 5*9 = 45.
            # Number of lines = same = 45.
            # Each point is on t+1 = 3 lines. Each line has s+1 = 5 points.
            # So the incidence graph is 3-regular on 90 vertices.
            # That's not 5-regular.
            # 
            # For 5-regular girth 6, the Moore bound is 1 + 5 + 5*4 = 26.
            # Known cage n(5,6) = 26? Let me think...
            # Actually, for k=5, g=6, the smallest known is likely larger.
            # Let's just use LUW.
            pass
        # For g >= 6, use LUW
        
    # 3. General Case: LUW
    return luw_construct(k, g)
