"""De Bruijn-type finite-state construction for regular graphs of large girth."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 120.0  # seconds; evaluator kills at 150s, keep margin

    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Helper: Generate a simple random 3-regular graph on n vertices
    def generate_random_3_reg_simple(n):
        if n < 4 or n % 2 != 0:
            return None
        
        # Configuration model
        stubs = [i for i in range(n) for _ in range(3)]
        
        for _ in range(100): # Try 100 times to get a simple graph
            random.shuffle(stubs)
            edges = set()
            valid = True
            # Pair up stubs
            for i in range(len(stubs) // 2):
                u, v = stubs[2*i], stubs[2*i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
            
            if not valid:
                continue
            
            # Check degree (should be 3 if no loops/multi-edges and simple pairing)
            # Actually, simple pairing ensures degree 3 if no loops.
            # Multi-edges are handled by the set.
            # So if valid is True, we have a simple 3-regular graph.
            return list(edges)
        
        return None

    # Helper: Check girth
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # New: Probabilistic algebraic construction via simulated annealing
    def algebraic_construct(k, g):
        """
        Vertices are pairs (u, v) in Z_m x Z_m where m = n.
        Edges connect (u, v) to (u + a, v + f(a)) for a in A.
        For k=3, we need 3 generators. We use A = {1, -1, s} where s is another element.
        Actually for 3-regular, we need exactly 3 neighbors per vertex.
        Let's use a different approach: 
        Vertices: Z_m x Z_m
        For each vertex (u,v), neighbors are:
        (u+1, v+f1), (u-1, v+f2), (u, v+s) where s is a fixed shift
        This gives degree 3 if all three are distinct.
        We optimize f1, f2, s to maximize girth.
        """
        # For k=3, g=16, try m such that m*m is even and >= Moore bound
        # Moore bound for 3,16 is 512. So m >= ceil(sqrt(512)) = 23, m*m >= 512
        # m must be such that m*m is even, so m must be even.
        # Try m values: 24 (576), 26 (676), 28 (784), 30 (900), 32 (1024)
        
        best_result = None
        best_n = float('inf')
        
        for m in range(24, 35, 2):
            if time.time() - t0 > deadline - 30:
                break
            n = m * m
            if n < 512:
                continue
            
            # Simulated annealing on f1, f2, s
            # f1, f2: Z_m -> Z_m, s: Z_m
            # Represent f1, f2 as arrays of length m
            
            def compute_girth(f1, f2, s, m, target_girth):
                """Compute girth of the algebraic graph."""
                # Build adjacency
                # Vertex (u, v) -> index u*m + v
                # Neighbors of (u, v):
                # 1: (u+1 mod m, v + f1[u] mod m)
                # 2: (u-1 mod m, v + f2[u] mod m)  
                # 3: (u, v + s mod m)
                
                # For speed, we'll do BFS from a few sample vertices
                # and check for short cycles
                
                n = m * m
                adj = [[] for _ in range(n)]
                
                for u in range(m):
                    for v in range(m):
                        idx = u * m + v
                        # Neighbor 1
                        u1 = (u + 1) % m
                        v1 = (v + f1[u]) % m
                        adj[idx].append(u1 * m + v1)
                        # Neighbor 2
                        u2 = (u - 1) % m
                        v2 = (v + f2[u]) % m
                        adj[idx].append(u2 * m + v2)
                        # Neighbor 3
                        v3 = (v + s) % m
                        adj[idx].append(u * m + v3)
                
                # Check girth by BFS from each vertex up to target_girth//2
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    qi = 0
                    while qi < len(queue):
                        u = queue[qi]
                        qi += 1
                        if dist[u] >= target_girth // 2:
                            continue
                        for v in adj[u]:
                            if v == parent[u]:
                                continue
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            else:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < target_girth:
                                    return cycle_len
                return target_girth  # No shorter cycle found
            
            # Initialize with random values
            f1 = [random.randint(0, m-1) for _ in range(m)]
            f2 = [random.randint(0, m-1) for _ in range(m)]
            s = random.randint(1, m-1)
            
            # Simulated annealing
            current_girth = compute_girth(f1, f2, s, m, g)
            best_girth = current_girth
            best_f1 = f1[:]
            best_f2 = f2[:]
            best_s = s
            
            T = 1000.0
            cooling = 0.95
            max_iters = 500
            iters_per_check = 10
            
            for it in range(max_iters):
                if time.time() - t0 > deadline - 30:
                    break
                
                # Propose a move
                move_type = random.random()
                if move_type < 0.4:
                    # Change one entry of f1
                    idx = random.randint(0, m-1)
                    f1[idx] = (f1[idx] + random.randint(-1, 1)) % m
                elif move_type < 0.8:
                    # Change one entry of f2
                    idx = random.randint(0, m-1)
                    f2[idx] = (f2[idx] + random.randint(-1, 1)) % m
                else:
                    # Change s
                    s = (s + random.randint(-1, 1)) % m
                    if s == 0:
                        s = 1
                
                new_girth = compute_girth(f1, f2, s, m, g)
                
                if new_girth >= current_girth:
                    current_girth = new_girth
                    if new_girth > best_girth:
                        best_girth = new_girth
                        best_f1 = f1[:]
                        best_f2 = f2[:]
                        best_s = s
                else:
                    # Accept with probability exp((new - current)/T)
                    if random.random() < pow(2.71828, (new_girth - current_girth) / T):
                        current_girth = new_girth
                    else:
                        # Revert
                        f1 = best_f1[:] if new_girth < current_girth else f1[:]
                        # Actually, we need to revert properly. Let's simplify:
                        # Just keep the current state and try again
                        pass
                
                T *= cooling
                
                if best_girth >= g:
                    # Found a graph with girth >= g
                    break
            
            if best_girth >= g:
                # Construct the edge list
                f1 = best_f1
                f2 = best_f2
                s = best_s
                edges = []
                seen = set()
                for u in range(m):
                    for v in range(m):
                        idx = u * m + v
                        # Neighbor 1
                        u1 = (u + 1) % m
                        v1 = (v + f1[u]) % m
                        nidx = u1 * m + v1
                        key = (min(idx, nidx), max(idx, nidx))
                        if key not in seen:
                            seen.add(key)
                            edges.append((idx, nidx))
                        # Neighbor 2
                        u2 = (u - 1) % m
                        v2 = (v + f2[u]) % m
                        nidx = u2 * m + v2
                        key = (min(idx, nidx), max(idx, nidx))
                        if key not in seen:
                            seen.add(key)
                            edges.append((idx, nidx))
                        # Neighbor 3
                        v3 = (v + s) % m
                        nidx = u * m + v3
                        key = (min(idx, nidx), max(idx, nidx))
                        if key not in seen:
                            seen.add(key)
                            edges.append((idx, nidx))
                
                # Verify it's 3-regular
                deg = [0] * n
                for a, b in edges:
                    deg[a] += 1
                    deg[b] += 1
                if all(d == 3 for d in deg):
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
        
        return best_result

    # Main logic
    if k == 3 and g == 16:
        # Try algebraic construction first
        alg_result = algebraic_construct(k, g)
        if alg_result is not None:
            return alg_result
        
        # Search for smaller graph using random construction
        # Start from Moore bound 512
        # Step by 2 to ensure even n
        for n in range(512, 1200, 2):
            if time.time() - t0 > deadline - 5:
                break
            
            # Try a few random graphs
            attempts = 20
            for _ in range(attempts):
                if time.time() - t0 > deadline - 5:
                    break
                edges = generate_random_3_reg_simple(n)
                if edges is None:
                    continue
                if check_girth(n, edges, g):
                    return n, edges
        
        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
# EVOLVE-BLOCK-END
