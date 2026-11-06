import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def check_valid(adj, n, min_deg):
        for i in range(n):
            if bin(adj[i]).count('1') < min_deg:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                if bin(adj[i] & adj[j]).count('1') > 1:
                    return False
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def polarity_graph(q):
        """Construct polarity graph of projective plane of order q."""
        if q == 2:
            n = 7
            adj = [0] * n
            edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,6),(3,6),(4,5),(4,6),(5,0),(2,4),(3,5)]
            for u,v in edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            return adj, n
        elif q == 3:
            n = 13
            adj = [0] * n
            # Fano plane polarity graph for q=3
            # Using standard construction
            points = list(range(13))
            # Define polarity via quadratic form
            # For PG(2,3), use the standard polarity
            # This is complex, so use known C4-free graph
            # Actually, let's use a different approach for q=3
            # The polarity graph of PG(2,3) has 13 vertices, degree 4
            # Let me construct it properly
            # Points of PG(2,3): (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1),
            # (1,2,0), (1,3,0), (2,1,0), (3,1,0), (1,2,1), (1,3,1), (2,1,1)
            # Actually, let's just use a known C4-free graph on 13 vertices with degree 4
            # Or better, use the incidence graph approach
            # For now, let's skip q=3 and use q=5
            return None, None
        elif q == 5:
            n = 31
            adj = [0] * n
            # Polarity graph of PG(2,5)
            # Points: all 1D subspaces of F_5^3
            # A point (x,y,z) is adjacent to point (x',y',z') iff xx'+yy'+zz' == 0 (mod 5)
            # and the point is not self-adjacent (not on its polar line)
            
            # Generate all points of PG(2,5)
            points = []
            seen = set()
            for x in range(6):
                for y in range(6):
                    for z in range(6):
                        if x == 0 and y == 0 and z == 0:
                            continue
                        # Normalize: first non-zero coordinate is 1
                        # Actually, for F_5, we need to be careful
                        # Let's generate all non-zero vectors and quotient by scaling
                        pass
            
            # Better approach: enumerate all 1D subspaces
            # There are (5^3 - 1)/(5-1) = 124/4 = 31 points
            # Represent each point by a canonical form
            pts = []
            for a in range(6):
                for b in range(6):
                    for c in range(6):
                        if a == 0 and b == 0 and c == 0:
                            continue
                        # Find first non-zero
                        if a != 0:
                            # Normalize so a=1
                            a_inv = pow(a, 3, 5)  # a^3 = a^-1 mod 5
                            b_norm = (b * a_inv) % 5
                            c_norm = (c * a_inv) % 5
                            key = (1, b_norm, c_norm)
                        elif b != 0:
                            b_inv = pow(b, 3, 5)
                            c_norm = (c * b_inv) % 5
                            key = (0, 1, c_norm)
                        else:
                            key = (0, 0, 1)
                        
                        if key not in seen:
                            seen.add(key)
                            pts.append(key)
            
            # Now, for each pair of points, check if they're adjacent
            # Point p = (a,b,c) is adjacent to point q = (a',b',c') iff aa'+bb'+cc' == 0 mod 5
            # AND p is not on its own polar line (i.e., pp != 0)
            
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    p = pts[i]
                    q = pts[j]
                    dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
                    if dot == 0:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Check degrees
            for i in range(n):
                deg = bin(adj[i]).count('1')
                if deg < 4:
                    # This shouldn't happen for a proper polarity graph
                    pass
            
            return adj, n
        else:
            return None, None

    def fast_local_search(adj, n, target_min_deg, max_iters=50000):
        """Fast local search that only checks affected pairs."""
        deg = [bin(adj[i]).count('1') for i in range(n)]
        best_adj = adj[:]
        best_min_deg = min(deg)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find vertex with minimum degree
            u = min(range(n), key=lambda i: deg[i])
            if deg[u] >= target_min_deg:
                # Try to improve further
                low_verts = [i for i in range(n) if deg[i] < target_min_deg + 1]
                if not low_verts:
                    break
                u = random.choice(low_verts)
            
            # Find candidates to connect to
            candidates = []
            for v in range(n):
                if v == u:
                    continue
                if not (adj[u] & (1 << v)):
                    # Check if adding (u,v) creates C4
                    # Need common_neighbors(u, v) == 0
                    common = adj[u] & adj[v]
                    if common == 0:
                        candidates.append(v)
            
            if not candidates:
                # Try removing an edge from a high-degree vertex to free up space
                high_verts = [i for i in range(n) if deg[i] > target_min_deg + 2]
                if high_verts:
                    h = random.choice(high_verts)
                    # Find a neighbor of h that we can disconnect
                    neighbors = []
                    mask = adj[h]
                    k = 0
                    while mask:
                        if mask & 1:
                            if k != h:
                                neighbors.append(k)
                        mask >>= 1
                        k += 1
                    if neighbors:
                        v = random.choice(neighbors)
                        # Remove edge (h, v)
                        adj[h] &= ~(1 << v)
                        adj[v] &= ~(1 << h)
                        deg[h] -= 1
                        deg[v] -= 1
                continue
            
            # Try adding a random edge
            v = random.choice(candidates)
            
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
            # Check C4 only for affected pairs
            c4_free = True
            # Check pairs involving u or v
            for i in range(n):
                if i == u or i == v:
                    for j in range(i+1, n):
                        if j == u or j == v:
                            continue
                        common = adj[i] & adj[j]
                        if bin(common).count('1') > 1:
                            c4_free = False
                            break
                if not c4_free:
                    break
            
            if c4_free:
                new_min = min(deg)
                if new_min > best_min_deg:
                    best_adj = adj[:]
                    best_min_deg = new_min
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                deg[u] -= 1
                deg[v] -= 1
        
        return best_adj, best_min_deg

    def solve_for_n(n, min_deg):
        """Try to find C4-free graph on n vertices with min degree >= min_deg."""
        
        # Strategy 1: Start with polarity graph if n matches
        if n == 31:
            adj, nn = polarity_graph(5)
            if adj is not None and nn == n:
                # Check if it meets min_deg requirement
                actual_min = min(bin(adj[i]).count('1') for i in range(n))
                if actual_min >= min_deg:
                    if check_valid(adj, n, min_deg):
                        return adj
                # Try to improve with local search
                adj, achieved = fast_local_search(adj, n, min_deg, max_iters=20000)
                if achieved >= min_deg and check_valid(adj, n, min_deg):
                    return adj
        
        # Strategy 2: Random initialization + repair
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + 13)
            adj = [0] * n
            deg = [0] * n
            
            # Initialize with random edges
            p = min(min_deg / n * 1.2, 0.6)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        deg[i] += 1
                        deg[j] += 1
            
            # Repair phase
            for _ in range(300):
                if time.time() > deadline:
                    break
                
                # Check for C4s
                violating = []
                for i in range(n):
                    for j in range(i+1, n):
                        if bin(adj[i] & adj[j]).count('1') > 1:
                            violating.append((i, j))
                
                if violating:
                    i, j = random.choice(violating)
                    common = adj[i] & adj[j]
                    common_list = []
                    mask = common
                    k = 0
                    while mask:
                        if mask & 1:
                            common_list.append(k)
                        mask >>= 1
                        k += 1
                    
                    for w in common_list[1:]:
                        if deg[i] >= deg[j]:
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                            deg[i] -= 1
                            deg[w] -= 1
                        else:
                            adj[j] &= ~(1 << w)
                            adj[w] &= ~(1 << j)
                            deg[j] -= 1
                            deg[w] -= 1
                else:
                    min_actual = min(deg)
                    if min_actual >= min_deg:
                        if check_valid(adj, n, min_deg):
                            return adj
                        break
                    else:
                        low = [i for i in range(n) if deg[i] < min_deg]
                        if not low:
                            break
                        u = random.choice(low)
                        candidates = []
                        for v in range(n):
                            if v != u and not (adj[u] & (1 << v)):
                                if (adj[u] & adj[v]) == 0:
                                    candidates.append(v)
                        if candidates:
                            v = random.choice(candidates)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            deg[u] += 1
                            deg[v] += 1
        
        # Strategy 3: Greedy construction
        adj = [0] * n
        deg = [0] * n
        for u in range(n):
            candidates = []
            for v in range(u):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append(v)
            random.shuffle(candidates)
            for v in candidates:
                if deg[u] >= min_deg:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
        
        if min(deg) >= min_deg and check_valid(adj, n, min_deg):
            return adj
        
        # Strategy 4: Local search from greedy
        if min(deg) >= min_deg - 2:
            adj, achieved = fast_local_search(adj, n, min_deg, max_iters=30000)
            if achieved >= min_deg and check_valid(adj, n, min_deg):
                return adj
        
        return None

    # Try n from 47 down to 30
    for n in range(47, 29, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        result = solve_for_n(n, min_deg)
        if result is not None:
            if check_valid(result, n, min_deg):
                edges = get_edges(result, n)
                return {'n': n, 'edges': edges}
    
    # Fallback: simple cycle on 40 vertices
    n = 40
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i+1) % n))
        adj[(i+1) % n] |= (1 << i)
    
    if check_valid(adj, n, 1):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
