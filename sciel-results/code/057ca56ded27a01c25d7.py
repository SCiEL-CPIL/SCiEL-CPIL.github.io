import random
import time
import sys
from collections import deque

def construct(k, g):
    # Fallback to LUW construction if search fails
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

    start_time = time.time()
    timeout = 110.0  # Leave 10s for overhead

    def get_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        min_girth = 10**9
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    def is_valid(n, edges, g, k):
        if len(edges) != n * k // 2:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            if (min(u, v), max(u, v)) in seen:
                return False
            seen.add((min(u, v), max(u, v)))
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            return False
        girth = get_girth(n, edges)
        return girth >= g

    # Try projective plane based construction
    # For k=5, q=5 (smallest prime power >= 5)
    # Projective plane of order q has q^2+q+1 points and lines
    # Each point is on q+1 lines, each line has q+1 points
    # The incidence graph is (q+1)-regular bipartite
    
    def try_projective_plane(k, g):
        # Find smallest prime power >= k
        def is_prime_power(n):
            if n <= 1:
                return 1, True
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
                if p > n:
                    break
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return p, True
            return 0, False
        
        # Find smallest prime power >= k
        q = k
        while not is_prime_power(q)[1]:
            q += 1
        base_q, is_pp = is_prime_power(q)
        if not is_pp:
            return None, None
        q = base_q  # Use the actual prime power
        
        # For q=5, projective plane has 31 points and 31 lines
        # Incidence graph is 6-regular on 62 vertices
        # We need to reduce to 5-regular
        
        # Build projective plane of order q
        # Points: [0,1,x] for x in GF(q), [0,1,0], [1,0,0], [0,1,0] etc.
        # Actually, standard construction:
        # Points: (x,y) for x,y in GF(q), plus points at infinity
        # Let's use the standard construction
        
        if q == 2:
            # Fano plane
            points = list(range(7))
            lines = [[0,1,2], [0,3,4], [0,5,6], [1,3,5], [1,4,6], [2,3,6], [2,4,5]]
        elif q == 3:
            # Projective plane of order 3: 13 points, 13 lines
            points = list(range(13))
            lines = []
            # Standard lines for PG(2,3)
            for i in range(4):
                for j in range(4):
                    pass
            # Use known incidence
            lines = [
                [0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,0,1],
                [12,3,4], [12,6,7], [12,9,10],
                [0,3,6], [1,4,7], [2,5,8], [12,11],
                [9,10,11]
            ]
            # This is getting complicated. Let's just use the known fact that
            # for q=5, the projective plane incidence graph has 62 vertices, 6-regular.
            # We'll try to remove a matching to get 5-regular.
            return None, None
        elif q == 5:
            # Projective plane of order 5: 31 points, 31 lines
            # Incidence graph: 62 vertices, 6-regular bipartite
            # Points: P0..P30, Lines: L0..L30
            n_pp = q * q + q + 1  # 31
            # Build incidence structure
            # Use the standard construction with GF(5)
            GF = list(range(q))
            
            # Points: (x, y) for x,y in GF, plus points at infinity
            # Actually, let's use the homogeneous coordinates approach
            # Points are equivalence classes of [a:b:c] not all zero
            # Two points are equivalent if one is a scalar multiple of the other
            
            points_list = []
            seen_points = set()
            for a in range(q):
                for b in range(q):
                    for c in range(1, q):  # c != 0, normalize so c=1... no
                        pass
            
            # Simpler: enumerate all non-zero triples up to scalar
            triples = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a == 0 and b == 0 and c == 0:
                            continue
                        triples.append((a, b, c))
            
            # Group by equivalence
            point_reps = []
            used = set()
            for t in triples:
                if t in used:
                    continue
                # Find all scalar multiples
                for s in range(1, q):
                    mult = ((t[0] * s) % q, (t[1] * s) % q, (t[2] * s) % q)
                    used.add(mult)
                point_reps.append(t)
            
            n_points = len(point_reps)  # Should be q^2+q+1 = 31
            
            # Lines: each line is the set of points [a:b:c] satisfying ax+by+cz=0 for some [a:b:c]
            # Same enumeration for line coefficients
            line_coeffs = []
            used_l = set()
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a == 0 and b == 0 and c == 0:
                            continue
                        t = (a, b, c)
                        if t in used_l:
                            continue
                        for s in range(1, q):
                            mult = ((a * s) % q, (b * s) % q, (c * s) % q)
                            used_l.add(mult)
                        line_coeffs.append(t)
            
            n_lines = len(line_coeffs)  # Should be 31
            
            # Build incidence: point i is on line j if dot product is 0
            incidence = [[False] * n_lines for _ in range(n_points)]
            for i, pt in enumerate(point_reps):
                for j, lc in enumerate(line_coeffs):
                    if (pt[0] * lc[0] + pt[1] * lc[1] + pt[2] * lc[2]) % q == 0:
                        incidence[i][j] = True
            
            # Build the incidence graph
            # Vertices: 0..n_points-1 are points, n_points..n_points+n_lines-1 are lines
            n_inc = n_points + n_lines  # 62
            deg_inc = [0] * n_inc
            edges_inc = []
            for i in range(n_points):
                for j in range(n_lines):
                    if incidence[i][j]:
                        v1 = i
                        v2 = n_points + j
                        edges_inc.append((v1, v2))
                        deg_inc[v1] += 1
                        deg_inc[v2] += 1
            
            # Should be (q+1)-regular = 6-regular
            # Now we need to reduce to k=5-regular
            # Remove a perfect matching from the bipartite graph
            # Find a perfect matching in the incidence graph
            
            adj_inc = [[] for _ in range(n_inc)]
            for u, v in edges_inc:
                adj_inc[u].append(v)
                adj_inc[v].append(u)
            
            # Find maximum matching using greedy + augmenting paths
            match = [-1] * n_inc
            def try_augment(u, visited):
                for v in adj_inc[u]:
                    if visited[v]:
                        continue
                    visited[v] = True
                    if match[v] == -1 or try_augment(match[v], visited):
                        match[u] = v
                        match[v] = u
                        return True
            return False
            
            for u in range(n_inc):
                if match[u] == -1:
                    visited = [False] * n_inc
                    try_augment(u, visited)
            
            # Check if perfect matching
            matched_edges = set()
            for u in range(n_inc):
                if match[u] != -1 and match[u] > u:
                    matched_edges.add((min(u, match[u]), max(u, match[u])))
            
            if len(matched_edges) == n_inc // 2:
                # Remove matching edges
                new_edges = [e for e in edges_inc if e not in matched_edges]
                # Verify
                n_new = n_inc
                if is_valid(n_new, new_edges, g, k):
                    return n_new, new_edges
            
            # Try different matchings with randomization
            for attempt in range(50):
                match = [-1] * n_inc
                for u in range(n_inc):
                    if match[u] == -1:
                        visited = [False] * n_inc
                        random.shuffle(adj_inc[u][:])
                        try_augment(u, visited)
                matched_edges = set()
                for u in range(n_inc):
                    if match[u] != -1 and match[u] > u:
                        matched_edges.add((min(u, match[u]), max(u, match[u])))
                if len(matched_edges) == n_inc // 2:
                    new_edges = [e for e in edges_inc if e not in matched_edges]
                    if is_valid(n_new, new_edges, g, k):
                        return n_new, new_edges
            return None, None
        else:
            return None, None
    
    # Try the projective plane approach
    n_pp, edges_pp = try_projective_plane(k, g)
    if n_pp is not None:
        return n_pp, edges_pp
    
    # Fallback to random search + LUW
    best_n = float('inf')
    best_edges = None
    
    def get_girth_fast(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        min_girth = 10**9
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            dq = deque([i])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        dq.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth
    
    # Try random regular graphs for a few sizes
    for n_try in [500, 600, 700, 800, 900, 1000, 1100]:
        if time.time() - start_time > timeout:
            break
        for attempt in range(3):
            if time.time() - start_time > timeout:
                break
            stubs = [i for i in range(n_try) for _ in range(k)]
            random.shuffle(stubs)
            adj = [[] for _ in range(n_try)]
            edges = []
            valid = True
            for i in range(0, n_try * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
                edges.append((u, v))
            if not valid:
                continue
            if len(edges) != n_try * k // 2:
                continue
            girth = get_girth_fast(n_try, edges)
            if girth >= g:
                return n_try, edges
    
    return luw_construct(k, g)
