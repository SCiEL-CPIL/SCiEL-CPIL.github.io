import time
import random
import numpy as np
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- Cayley graph construction on small non-abelian groups ----------------
    def cayley_construct():
        if k != 3 or g != 16:
            return None
        
        # Try dihedral group D_n for various n
        # D_n has 2n elements: rotations r^0..r^(n-1) and reflections s, sr, ..., sr^(n-1)
        # Generators: r (rotation), s (reflection), and we need exactly 3 generators
        # For 3-regular: use {r, r^a, s} or similar
        # Actually for 3-regular Cayley: need 3 involutions or 1 rotation + 2 reflections
        
        # Let's try: generators {r, r^2, s} where r^2 is also a rotation
        # This gives degree 3 if r, r^2, s are all distinct and none is identity
        
        # Better: use {r^a, r^b, s} with a+b < n to avoid duplicates
        # Or use {r, s, sr^c} which are all involutions
        
        # For D_n with generators {s, sr^a, sr^b}:
        # Each generator is an involution (order 2)
        # Degree = 3
        # Girth depends on n and the relations
        
        best_result = None
        best_n = float('inf')
        
        # Try D_n for n from 15 to 200
        for n in range(15, 200):
            if remaining() < 15:
                break
            
            # Represent D_n elements as (i, j) where j=0 is rotation r^i, j=1 is reflection sr^i
            # Multiplication: (i,0)*(j,0) = (i+j,0)
            #               (i,0)*(j,1) = (i+j,1)
            #               (i,1)*(j,0) = (i-j,1)
            #               (i,1)*(j,1) = (i-j,0)
            
            # Generators: s=(0,1), sr^a=(a,1), sr^b=(b,1)
            # All are involutions: (a,1)^2 = (a-a,0) = (0,0) = identity
            
            for a in range(1, n // 2 + 1):
                for b in range(a + 1, n // 2 + 1):
                    if remaining() < 5:
                        break
                    
                    # Build Cayley graph
                    # Vertices: (i, j) for i in 0..n-1, j in 0,1
                    # Total: 2n vertices
                    
                    def elem_mul(e1, e2):
                        i1, j1 = e1
                        i2, j2 = e2
                        if j1 == 0:
                            return ((i1 + i2) % n, j2)
                        else:
                            return ((i1 - i2) % n, 1 - j2)
                    
                    identity = (0, 0)
                    gens = [(0, 1), (a, 1), (b, 1)]
                    
                    # Build adjacency
                    adj = {}
                    for i in range(n):
                        for j in range(2):
                            v = (i, j)
                            adj[v] = []
                            for gen in gens:
                                w = elem_mul(v, gen)
                                adj[v].append(w)
                    
                    # Check it's 3-regular (should be by construction)
                    # Check girth via BFS from each vertex (non-backtracking)
                    
                    # Convert to integer indices
                    idx = {}
                    for i in range(n):
                        for j in range(2):
                            idx[(i, j)] = i * 2 + j
                    
                    edges_set = set()
                    for i in range(n):
                        for j in range(2):
                            v = (i, j)
                            for w in adj[v]:
                                vi = idx[v]
                                wi = idx[w]
                                if vi < wi:
                                    edges_set.add((vi, wi))
                                else:
                                    edges_set.add((wi, vi))
                    
                    n_verts = 2 * n
                    edges = list(edges_set)
                    
                    # Quick girth check: BFS from first few vertices
                    # Full check is expensive, so do a quick check first
                    min_cycle = float('inf')
                    checked = 0
                    for s_idx in range(min(n_verts, 50)):
                        if remaining() < 10:
                            break
                        dist = [-1] * n_verts
                        parent = [-1] * n_verts
                        dist[s_idx] = 0
                        queue = [s_idx]
                        qi = 0
                        while qi < len(queue):
                            node = queue[qi]
                            qi += 1
                            for nb in adj[(node // 2, node % 2)]:
                                nb_idx = idx[nb]
                                if nb_idx == parent[node]:
                                    continue
                                if dist[nb_idx] != -1:
                                    cycle_len = dist[node] + dist[nb_idx] + 1
                                    if cycle_len < min_cycle:
                                        min_cycle = cycle_len
                                    if cycle_len < g:
                                        min_cycle = cycle_len
                                        break
                                else:
                                    dist[nb_idx] = dist[node] + 1
                                    parent[nb_idx] = node
                                    queue.append(nb_idx)
                        if min_cycle < g:
                            break
                    
                    if min_cycle >= g:
                        # Full verification
                        gc = check_girth(n_verts, edges, g, deadline - 5)
                        if gc == -1:
                            continue
                        if gc >= g:
                            if n_verts < best_n:
                                best_n = n_verts
                                best_result = (n_verts, edges)
                            break  # Found a valid one for this n, move to next n
                    else:
                        # If min_cycle < g, this (a,b) pair doesn't work, try next
                        pass
        
        return best_result

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

    # ---------------- Voltage lifting on Petersen graph ----------------
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

        max_L = g - 1
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
        
        min_m = 77
        best_m = None
        best_x = None
        
        # Systematic scan of small primes and composites
        candidates = list(range(min_m, 250))
        # Also try some larger values
        candidates += [251, 257, 263, 269, 271, 277, 281, 283, 293, 307]
        
        for m in candidates:
            if remaining() < 10:
                break
            
            found = False
            # For small m, try more systematic assignments
            if m <= 100:
                # Try all 1-element and 2-element patterns first
                for c in range(1, m):
                    x = np.full(15, c, dtype=int)
                    S = (C @ x) % m
                    if not np.any(S == 0):
                        best_m = m
                        best_x = x
                        found = True
                        break
                if found:
                    break
                
                # Try 2-element patterns
                for c1 in range(1, m):
                    for c2 in range(1, m):
                        if remaining() < 5:
                            break
                        x = np.array([c1 if i % 2 == 0 else c2 for i in range(15)])
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
                
                # Random tries
                for _ in range(200):
                    if remaining() < 5:
                        break
                    x = np.random.randint(1, m, size=15)
                    S = (C @ x) % m
                    if not np.any(S == 0):
                        best_m = m
                        best_x = x
                        found = True
                        break
            else:
                # For larger m, random is more likely to work
                for _ in range(50):
                    if remaining() < 5:
                        break
                    x = np.random.randint(1, m, size=15)
                    S = (C @ x) % m
                    if not np.any(S == 0):
                        best_m = m
                        best_x = x
                        found = True
                        break
            
            if found:
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
        elif variant == 7:
            for c in range(5, d + 1):
                base[c] = (c - 1, 1) if (c + offset) % 4 == 0 else (1, c - 2)
        else:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
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
        # Try voltage lifting first
        res = voltage_lift()
        if res is not None:
            return res
        
        # Try Cayley graph on dihedral group
        res = cayley_dihedral()
        if res is not None:
            return res

    # Fallback to LUW
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
