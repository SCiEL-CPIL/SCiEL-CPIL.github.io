"""Combined approach: LUW family with deadline guard and fast circulant search."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 118.0  # safety margin before 150s kill

    def check_deadline():
        return time.time() > deadline

    # --- Fast girth check for circulant graphs ---
    # S is the full symmetric set of 2k steps. A cycle of length l < g exists iff
    # there is a sequence of l distinct steps (each used at most once, no immediate backtracking
    # needed for simple cycle detection) summing to 0 mod n, or a vertex at BFS depth d < g-1
    # is itself in S (giving a cycle of length d+1).
    # We use BFS from 0; if any vertex v in S is first reached at depth d where d+1 < g,
    # we have a short cycle. Also check: if a vertex at depth d has an edge to a vertex at
    # depth d' where d' < d (i.e., a back-edge), the cycle length is d + d' + 1.
    # For efficiency, we only need to check up to depth g-2.
    
    def circulant_girth_ok(n, S, g):
        """Check if circulant graph with step set S (symmetric, |S|=2k) on Z_n has girth >= g."""
        S_set = frozenset(S)
        max_depth = g - 2  # depths 0..g-2; a cycle of length l < g means l <= g-1
        
        depth = [-1] * n
        depth[0] = 0
        q = deque([0])
        
        while q:
            u = q.popleft()
            du = depth[u]
            if du >= max_depth:
                continue
            for s in S:
                v = (u + s) % n
                dv = depth[v]
                if dv == -1:
                    depth[v] = du + 1
                    # If v is in S (i.e., v is a neighbor of 0) and du+1+1 < g, cycle of length du+2
                    # Actually: path 0->...->u->v has length du+1, and v->0 is an edge (v in S),
                    # so cycle length = du+2. We need du+2 < g, i.e., du < g-2.
                    if v in S_set and du + 2 < g:
                        return False
                    q.append(v)
                else:
                    # Back-edge or cross-edge: cycle length = du + dv + 1
                    # Only count if it's not the tree edge (dv != du + 1)
                    # In BFS, dv <= du+1 for neighbors. If dv < du, it's a back-edge.
                    # If dv == du, it's a cross-edge. Cycle length = du + dv + 1.
                    if dv < du:
                        if du + dv + 1 < g:
                            return False
        
        return True

    # --- Family 1: Lazebnik-Ustimenko-Woldar (from seed) ---
    def luw_family():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Get LUW result as fallback
    n_luw, e_luw = luw_family()
    best_n = n_luw
    best_e = e_luw

    # --- Search for Circulant Graphs ---
    if k == 5 and g == 9:
        # Moore bound for k=5, g=9 is 426.
        # n must be even for k=5 (odd degree)
        # We'll search n from 426 upward
        # For k=5, S must have 10 elements (5 pairs {a, -a}).
        # General form: S = {a1, -a1, a2, -a2, a3, -a3, a4, -a4, a5, -a5}
        # where ai are distinct mod n (and ai != n/2 unless n even and we use n/2 as one step)
        # If n is even, one step can be n/2 (self-inverse).
        
        # Strategy: try n from 426 up, with various 5-element step sets.
        # For efficiency, parameterize as:
        #   Type A: {n/2, a, n-a, b, n-b, c, n-c} — but that's 7 elements, not 10.
        #   Wait: k=5 means degree 5, so S has 5 elements if we consider directed steps,
        #   but for undirected graph, S must be symmetric: S = {-S} and |S| = 2k = 10.
        #   So we need 5 pairs {a, -a}. If n is even, one can be {n/2} (self-inverse).
        
        # So for even n: S = {n/2, a, n-a, b, n-b, c, n-c, d, n-d} — that's 1+8=9... no.
        # {n/2} is one element (since n/2 = -n/2 mod n). So S = {n/2, a, n-a, b, n-b, c, n-c, d, n-d}
        # has 1 + 2+2+2+2 = 9 elements. But we need 10. So we can't use n/2 and have 5 pairs.
        # Actually: if we use n/2, it's one element. Then we need 4 more pairs = 8 elements. Total = 9.
        # That gives degree 9/2... no. The degree is |S|/2 if S is symmetric and doesn't contain n/2,
        # or (|S|-1)/2 + 1 if it does. Wait, no: the degree is just |S| since each s in S gives an edge.
        # For undirected graph, each edge {u,v} is counted once. If S is symmetric, the number of
        # edges from vertex u is |S|, but each edge is shared by two vertices, so degree = |S|.
        # Wait no: if S = {a, -a, b, -b, ...}, then from vertex 0, neighbors are a, -a, b, -b, ...
        # That's |S| neighbors, so degree = |S|. But for k=5, we need degree 5, so |S| = 5.
        # But S must be symmetric: if a in S, then -a in S. So |S| must be even unless n/2 in S.
        # For k=5 (odd), we MUST have n/2 in S (for even n), giving 1 + 4 = 5 elements:
        # S = {n/2, a, n-a, b, n-b} — that's 5 elements! Yes!
        
        # So for k=5, even n: S = {n/2, a, n-a, b, n-b} with a, b distinct, a != b, a != n/2, etc.
        
        for n in range(426, 1400, 2):
            if check_deadline():
                break
            
            half_n = n // 2
            # Try values of a and b from 1 to n-1, excluding 0, n/2, and each other
            # To be efficient, try a in 1..min(40, n-1), b in a+1..min(40, n-1),
            # also try larger spread values.
            
            found = False
            # First pass: small a, b
            max_ab = min(40, n - 1)
            for a in range(1, max_ab + 1):
                if a == half_n:
                    continue
                if check_deadline():
                    break
                for b in range(a + 1, max_ab + 1):
                    if b == half_n:
                        continue
                    S = [half_n, a, n - a, b, n - b]
                    if len(set(S)) != 5:
                        continue
                    
                    if circulant_girth_ok(n, S, g):
                        edges = []
                        for i in range(n):
                            for s in S:
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                        
                        best_n = n
                        best_e = edges
                        found = True
                        break
                if found:
                    break
            
            # Second pass: more spread out a, b
            if not found and not check_deadline():
                # Try a near n/3, b near 2n/3, etc.
                candidates_a = list(set([n//3, n//4, n//5, 2*n//5, n//6, 5*n//6,
                                         n//7, 2*n//7, 3*n//7, 4*n//7]))
                candidates_b = list(set([n//3, n//4, n//5, 2*n//5, n//6, 5*n//6,
                                         n//7, 2*n//7, 3*n//7, 4*n//7,
                                         2*n//3, 3*n//4, 4*n//5, n//2+1, n//2-1]))
                candidates_a = [x for x in candidates_a if 1 <= x <= n-1 and x != half_n]
                candidates_b = [x for x in candidates_b if 1 <= x <= n-1 and x != half_n]
                
                for a in candidates_a:
                    if check_deadline():
                        break
                    for b in candidates_b:
                        if b == a or b == n - a:
                            continue
                        S = [half_n, a, n - a, b, n - b]
                        if len(set(S)) != 5:
                            continue
                        
                        if circulant_girth_ok(n, S, g):
                            edges = []
                            for i in range(n):
                                for s in S:
                                    j = (i + s) % n
                                    if i < j:
                                        edges.append((i, j))
                            
                            best_n = n
                            best_e = edges
                            found = True
                            break
                    if found:
                        break
            
            if found:
                break

    return best_n, best_e
# EVOLVE-BLOCK-END
