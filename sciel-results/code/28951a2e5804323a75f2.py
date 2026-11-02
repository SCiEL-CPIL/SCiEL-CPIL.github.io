import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 118.0  # safety margin before 150s wall-clock kill

    def elapsed():
        return time.time() - t0

    # ---------- Fallback: Lazebnik-Ustimenko-Woldar graph ----------
    def luw_construct(k, g):
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

    # ---------- Fast girth check for circulant graph ----------
    def circulant_girth(n, conn, limit):
        """Return minimum cycle length in circulant C_n(conn), or limit if none < limit."""
        best = limit
        # Build adjacency
        adj = [[] for _ in range(n)]
        for d in conn:
            dd = d % n
            if dd == 0:
                continue
            for i in range(n):
                adj[i].append((i+dd)%n)
                adj[(i+dd)%n].append(i)
        # Remove duplicates
        for i in range(n):
            adj[i] = list(set(adj[i]))
        # Check 3-cycles quickly
        adjset = [set(adj[i]) for i in range(n)]
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    for l in adj[i]:
                        if l != j and l in adjset[j]:
                            best = min(best, 3)
        if best == 3:
            return 3
        # BFS from each vertex for longer cycles
        for s in range(n):
            # Non-backtracking BFS
            for t0 in adj[s]:
                seen = {}
                q = deque()
                q.append((t0, s, 1))
                seen[(t0, s)] = 1
                while q:
                    cur, prev, depth = q.popleft()
                    if depth >= best:
                        continue
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        if nb == s:
                            if depth + 1 >= 3:
                                best = min(best, depth+1)
                            continue
                        st = (nb, cur)
                        if st not in seen:
                            seen[st] = depth+1
                            q.append((nb, cur, depth+1))
        return best

    # ---------- Circulant construction via simulated annealing ----------
    def search_circulant(k, g, n, budget):
        """Search for a k-regular circulant graph on n vertices with girth >= g."""
        # For k=3, conn has 3 elements (symmetric: each d and n-d pair)
        # For general k, conn has k elements
        # We need conn such that all elements are distinct mod n and no element is 0
        # and the set is symmetric: if d in conn then (n-d) mod n is also represented
        
        # For k=3: pick 3 distinct nonzero residues, ensuring symmetry
        # A 3-regular circulant: conn = {a, b, n-a} or {a, b, c} where the graph is undirected
        # Actually for undirected: if d in conn, we add both d and -d. So for odd k, one element must be n/2 (if n even)
        # For k=3, n must be even. conn = {a, b, n-a} where a != b, a != n-a, etc.
        # Or conn = {a, b, c} with the understanding that edges go both ways.
        # Simplest: conn is a set of k residues, and adjacency is i~j iff (j-i) mod n in conn OR (i-j) mod n in conn.
        # To make it exactly k-regular, we need conn to be symmetric: d in conn iff (n-d)%n in conn,
        # except when d = n/2 (self-inverse).
        
        if k == 3:
            # n must be even
            # conn = {a, b, n-a} with a != b, a != n-a, b != n-a, all nonzero
            # Or more generally: pick 3 distinct nonzero residues where the set is "symmetric"
            # For a 3-regular circulant on even n: conn = {d1, d2, n/2} doesn't work (n/2 is self-inverse, gives degree 2 from it + ... )
            # Actually: if conn contains n/2, that's one edge per vertex from that direction.
            # Let's just use: conn has exactly k elements, and we define adj[i] = {(i+d)%n for d in conn}
            # For this to be k-regular undirected, we need: for every d in conn, (n-d)%n is also in conn,
            # OR d = (n-d)%n i.e. 2d = 0 mod n i.e. d = n/2.
            # For k=3: either conn = {a, n-a, b} where b = n/2 (n even), or conn = {a, b, c} all self-inverse (only works if n=2).
            # So for k=3, n even: conn = {a, n-a, n/2} with a != n/2, a != 0, n-a != 0, a != n-a (i.e. 2a != 0 mod n).
            # Wait, that gives degree 3: edges to i+a, i-(a), i+n/2. Yes!
            
            if n % 2 != 0:
                return None, 0
            
            n2 = n // 2
            # conn = {a, n-a, n/2} for some a in 1..n/2-1, a != n/2 (which is n2)
            # a ranges over 1..n2-1, excluding n2 (but n2 is the third element)
            # So a in 1..n2-1, and we need a != n-a mod n, i.e., 2a != 0 mod n, i.e., a != n/2.
            # Since a < n/2, 2a < n, so 2a != 0 automatically. Good.
            
            best_g = 0
            best_a = None
            
            # Try all a values
            for a in range(1, n2):
                conn = [a, n-a, n2]
                gg = circulant_girth(n, conn, g)
                if gg >= g:
                    # Build edges
                    edges = []
                    for i in range(n):
                        for d in conn:
                            j = (i+d) % n
                            if i < j:
                                edges.append((i, j))
                    return edges, gg
                if gg > best_g:
                    best_g = gg
                    best_a = a
            
            # If no exact match, try to find best
            if best_a is not None:
                conn = [best_a, n-best_a, n2]
                edges = []
                for i in range(n):
                    for d in conn:
                        j = (i+d) % n
                        if i < j:
                            edges.append((i, j))
                return edges, best_g
            return None, 0
        else:
            # For general k, use LUW fallback
            return None, 0

    # ---------- Main ----------
    if k == 3:
        # Moore bound for (3, g) with even g: 2*(2^((g-2)/2) - 1)
        if g % 2 == 0:
            n_moore = 2 * (2 ** ((g-2)//2) - 1)
        else:
            n_moore = 3 * (2 ** ((g-1)//2) - 1)
        
        # Try circulant search at various sizes starting from Moore bound
        start_n = n_moore
        if start_n % 2 != 0:
            start_n += 1
        
        best_result = None
        
        # Try a range of sizes
        max_try = min(start_n + 200, start_n + 400)  # limit number of sizes to try
        
        for n in range(start_n, max_try + 1, 2):  # step by 2 (must be even for k=3)
            if elapsed() > deadline - 5:
                break
            budget = deadline - 3 - elapsed()
            if budget < 0.5:
                break
            
            edges, gg = search_circulant(k, g, n, budget)
            if edges is not None and gg >= g:
                best_result = (n, edges)
                break  # Found valid graph at this size, stop
        
        if best_result is not None:
            return best_result
        
        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
