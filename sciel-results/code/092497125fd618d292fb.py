import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- Group-based Ramanujan lift construction ----------------
    def group_lift():
        """
        Construct a 3-regular lift of the Petersen graph using a group action.
        We use a cyclic group Z_m and assign voltages to the 15 edges of the Petersen.
        The key insight: if we can find voltages such that no closed non-backtracking
        walk of length < g has total voltage 0 mod m, then the lift has girth >= g.
        
        We use a structured family of voltage assignments based on polynomial functions.
        """
        if k != 3 or g != 16:
            return None
        
        # Petersen graph edges
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        edge_index = {}
        for i, (u, v) in enumerate(base_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        adj = [[] for _ in range(10)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Generate all closed non-backtracking walks of length [5, 15]
        # Each walk gives a vector in Z^15 (edge incidence with signs)
        max_L = g - 1  # 15
        walk_vectors = []
        
        for start_v in range(10):
            stack = [(start_v, -1, 0, [], [])]
            while stack:
                current, prev, length, p_edges, p_dirs = stack.pop()
                
                if length >= 5 and current == start_v:
                    vec = [0] * 15
                    for e_idx, d in zip(p_edges, p_dirs):
                        vec[e_idx] += d
                    # Only add if this is a new vector (use tuple for hashability)
                    walk_vectors.append(tuple(vec))
                    
                if length == max_L:
                    continue
                    
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = base_edges[e_idx]
                    d = 1 if current == u else -1
                    stack.append((next_v, current, length + 1, p_edges + [e_idx], p_dirs + [d]))
        
        if not walk_vectors:
            return None
        
        # Deduplicate
        unique_walks = list(set(walk_vectors))
        C = np.array(unique_walks, dtype=np.int64)
        n_walks = len(C)
        
        # Moore bound: n = 10*m >= 512 => m >= 52
        min_m = 52
        
        best_m = None
        best_x = None
        
        # Strategy: Try polynomial voltage assignments v_i = f(i) mod m
        # where f is a low-degree polynomial. This creates a structured family
        # that's more likely to avoid zero sums than random assignments.
        
        for m in range(min_m, 200):
            if remaining() < 20:
                break
            
            # Try various polynomial families
            found = False
            
            # Family 1: v_i = (a*i^2 + b*i + c) mod m for small a, b, c
            for a in range(1, 6):
                for b in range(0, 6):
                    for c in range(0, 6):
                        x = np.array([(a*i*i + b*i + c) % m for i in range(15)], dtype=np.int64)
                        S = (C @ x) % m
                        if not np.any(S == 0):
                            best_m = m
                            best_x = x
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
            
            # Family 2: v_i = alpha^(i) mod m for primitive root alpha
            # Try small primitive roots
            for alpha in range(2, min(10, m)):
                x = [1]
                for i in range(1, 15):
                    x.append((x[-1] * alpha) % m)
                x = np.array(x, dtype=np.int64)
                S = (C @ x) % m
                if not np.any(S == 0):
                    best_m = m
                    best_x = x
                    found = True
                    break
            if found:
                break
            
            # Family 3: v_i = (i * a + b) mod m (linear)
            for a in range(1, min(10, m)):
                for b in range(0, 10):
                    x = np.array([(i * a + b) % m for i in range(15)], dtype=np.int64)
                    S = (C @ x) % m
                    if not np.any(S == 0):
                        best_m = m
                        best_x = x
                        found = True
                        break
                if found:
                    break
            if found:
                break
            
            # Family 4: v_i = i^3 mod m
            x = np.array([pow(i, 3, m) for i in range(15)], dtype=np.int64)
            S = (C @ x) % m
            if not np.any(S == 0):
                best_m = m
                best_x = x
                found = True
                break
            
            # Family 5: Alternating pattern v_i = (i % 2) * a + (i // 2) * b
            for a in range(1, 6):
                for b in range(1, 6):
                    x = np.array([((i % 2) * a + (i // 2) * b) % m for i in range(15)], dtype=np.int64)
                    S = (C @ x) % m
                    if not np.any(S == 0):
                        best_m = m
                        best_x = x
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if best_m is None:
            # Fallback: more systematic search with larger m
            for m in range(200, 400):
                if remaining() < 10:
                    break
                # Try quadratic with wider coefficient range
                for a in range(1, 15):
                    for b in range(0, 15):
                        x = np.array([(a*i*i + b*i) % m for i in range(15)], dtype=np.int64)
                        S = (C @ x) % m
                        if not np.any(S == 0):
                            best_m = m
                            best_x = x
                            break
                    if best_m is not None:
                        break
                if best_m is not None:
                    break
        
        if best_m is None:
            return None

        # Construct the lifted graph
        n = 10 * best_m
        edges_set = set()
        for u, v in base_edges:
            e_idx = edge_index[(u, v)]
            for i in range(best_m):
                u_lift = u * best_m + i
                v_lift = v * best_m + (i + best_x[e_idx]) % best_m
                if u_lift < v_lift:
                    edges_set.add((u_lift, v_lift))
                else:
                    edges_set.add((v_lift, u_lift))
        
        final_edges = list(edges_set)
        
        # Verify girth
        gc = check_girth(n, final_edges, g, deadline - 5)
        if gc == -1:
            return None
        if gc >= g:
            return (n, final_edges)
        else:
            return None

    # ---------------- girth verification (fast, early exit) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Voltage lifting on Petersen graph (deterministic + random) ----------------
    def voltage_lift():
        if k != 3 or g != 16:
            return None
        
        # Petersen graph definition
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        edge_index = {}
        for i, (u, v) in enumerate(base_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        adj = [[] for _ in range(10)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)

        max_L = g - 1  # 15
        
        # Generate all closed non-backtracking walks of length L in [5, g-1]
        all_walk_vectors = set()
        
        for start_v in range(10):
            stack = [(start_v, -1, 0, [], [])]
            while stack:
                current, prev, length, p_edges, p_dirs = stack.pop()
                
                if length >= 5 and current == start_v:
                    vec = [0] * 15
                    for e_idx, d in zip(p_edges, p_dirs):
                        vec[e_idx] += d
                    all_walk_vectors.add(tuple(vec))
                    
                if length == max_L:
                    continue
                    
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = base_edges[e_idx]
                    if current == u:
                        d = 1
                    else:
                        d = -1
                    stack.append((next_v, current, length + 1, p_edges + [e_idx], p_dirs + [d]))
                    
        if not all_walk_vectors:
            return None
            
        C = np.array(list(all_walk_vectors))
        
        # Moore bound for k=3, g=16 is ~512. n=10m. m >= 52.
        # But Petersen has girth 5, so we need to lift carefully.
        # Actually, the Moore bound for (3,16) is 512.
        # n = 10 * m. So 10m >= 512 => m >= 52.
        min_m = 52
        
        best_m = None
        best_x = None
        
        # Strategy 1: Deterministic linear algebra over Z_m
        # We want C @ x % m != 0 for all rows.
        # This is equivalent to: for each row c in C, c @ x is not 0 mod m.
        
        # Try small primes first, then composites.
        # We'll iterate m upwards.
        
        # Precompute some small primes and composites
        candidates = list(range(min_m, 300))
        
        for m in candidates:
            if remaining() < 15:
                break
                
            # Try deterministic vectors first for this m
            # 1. All ones
            x_ones = np.ones(15, dtype=int) % m
            S = (C @ x_ones) % m
            if not np.any(S == 0):
                best_m = m
                best_x = x_ones
                break
                
            # 2. Powers of 2
            x_powers = np.array([2**i % m for i in range(15)])
            S = (C @ x_powers) % m
            if not np.any(S == 0):
                best_m = m
                best_x = x_powers
                break
                
            # 3. Fibonacci-like
            fib = [1, 1]
            for i in range(13):
                fib.append((fib[-1] + fib[-2]) % m)
            x_fib = np.array(fib)
            S = (C @ x_fib) % m
            if not np.any(S == 0):
                best_m = m
                best_x = x_fib
                break
                
            # 4. Random search (limited)
            np.random.seed(m)  # Deterministic seed for reproducibility
            found = False
            for _ in range(200):
                x = np.random.randint(0, m, size=15)
                S = (C @ x) % m
                if not np.any(S == 0):
                    best_m = m
                    best_x = x
                    found = True
                    break
                    
            if found:
                break
                
            # 5. Try specific patterns
            # x[i] = i * m // 15 (linear spacing)
            x_lin = np.array([i * m // 15 for i in range(15)])
            S = (C @ x_lin) % m
            if not np.any(S == 0):
                best_m = m
                best_x = x_lin
                break
        
        if best_m is None:
            # Fallback: try larger m with more random trials
            for m in range(300, 600):
                if remaining() < 10:
                    break
                np.random.seed(m * 7 + 3)
                for _ in range(500):
                    x = np.random.randint(0, m, size=15)
                    S = (C @ x) % m
                    if not np.any(S == 0):
                        best_m = m
                        best_x = x
                        break
                if best_m is not None:
                    break
        
        if best_m is None:
            return None

        # Construct the lifted graph
        n = 10 * best_m
        edges_set = set()
        for u, v in base_edges:
            e_idx = edge_index[(u, v)]
            for i in range(best_m):
                u_lift = u * best_m + i
                v_lift = v * best_m + (i + best_x[e_idx]) % best_m
                if u_lift < v_lift:
                    edges_set.add((u_lift, v_lift))
                else:
                    edges_set.add((v_lift, u_lift))
        
        final_edges = list(edges_set)
        
        # Verify girth
        gc = check_girth(n, final_edges, g, deadline - 5)
        if gc == -1:
            return None
        if gc >= g:
            return (n, final_edges)
        else:
            return None

    # ---------------- LUW construction (Fallback) ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0, offset=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                base[c] = (1, c - 2) if (c + offset) % 2 == 0 else (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if (c + offset) % 3 == 0:
                    base[c] = (1, c - 3)
                elif (c + offset) % 3 == 1:
                    base[c] = (c - 1, 1)
                else:
                    base[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                base[c] = (1, c - (1 + (c + offset) % 2))
        elif variant == 4:
            for c in range(5, d + 1):
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][(c + offset) % 4]
        elif variant == 5:
            for c in range(5, d + 1):
                base[c] = (c - 2, 1) if (c + offset) % 2 == 0 else (1, c - 1)
        elif variant == 6:
            for c in range(5, d + 1):
                base[c] = (1, c - 1) if (c + offset) % 3 == 0 else (c - 2, 1)
        else:
            for c in range(5, d + 1):
                base[c] = (c - 1, 1) if (c + offset) % 4 == 0 else (1, c - 2)
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

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
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best = None
        best_n = float('inf')

        for variant in range(8):
            for offset in range(6):
                if remaining() < 12:
                    break
                rule = generate_rules(d, variant, offset)
                res = build_luw(q, d, rule, deadline - 12)
                if res is None:
                    continue
                n, edges = res
                if n >= best_n:
                    continue
                gc = check_girth(n, edges, g, deadline - 8)
                if gc == -1:
                    continue
                if gc >= g:
                    best, best_n = (n, edges), n
                    break

        if best is None:
            for d_try in range(d + 1, d + 10):
                if remaining() < 10:
                    break
                for variant in range(3):
                    for offset in range(3):
                        if remaining() < 5:
                            break
                        rule = generate_rules(d_try, variant, offset)
                        res = build_luw(q, d_try, rule, deadline - 10)
                        if res is None:
                            continue
                        n, edges = res
                        gc = check_girth(n, edges, g, deadline - 5)
                        if gc == -1:
                            continue
                        if gc >= g:
                            best = (n, edges)
                            break
                    if best is not None:
                        break
                if best is not None:
                    break

        if best is None:
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
        return best

    # ---------------- main ----------------
    if k == 3 and g == 16:
        res = group_lift()
        if res is not None:
            return res
        res = voltage_lift()
        if res is not None:
            return res

    res = luw_construct(k, g)
    return res if res is not None else (0, [])
