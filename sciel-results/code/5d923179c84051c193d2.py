import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    def luw_fallback():
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

    # --- PSL2(q) Cayley graph approach for k=5, g=9 ---
    if k == 5 and g == 9:
        try:
            q = 13
            # Build PSL2(13) elements: 2x2 matrices mod 13 with det=1, modulo ±I
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        # det = ad - bc = 1 mod q
                        rhs = (1 + b * c) % q
                        if a == 0:
                            if rhs == 0:
                                # -bc = 1 => bc = -1
                                if (b * c) % q == (q - 1) % q:
                                    for d in range(q):
                                        sl_elements.append((0, b, c, d))
                        else:
                            d = rhs * pow(a, -1, q) % q
                            sl_elements.append((a, b, c, d))

            # Quotient by ±I
            psl_elements = []
            seen = set()
            for e in sl_elements:
                a, b, c, d = e
                key = (a, b, c, d)
                neg_key = (-a % q, -b % q, -c % q, -d % q)
                if key < neg_key:
                    if key not in seen:
                        seen.add(key)
                        psl_elements.append(e)
                else:
                    if neg_key not in seen:
                        seen.add(neg_key)
                        psl_elements.append(neg_key)

            n_vertices = len(psl_elements)
            idx_map = {e: i for i, e in enumerate(psl_elements)}
            identity = (1, 0, 0, 1)
            identity_idx = idx_map[identity]

            # Precompute group multiplication table
            # Using NumPy for speed
            # Convert to numpy arrays for vectorized multiplication?
            # Actually, a simple loop with NumPy array storage is fine for 1092 elements.
            mul_table = np.zeros((n_vertices, n_vertices), dtype=np.int32)
            elems = np.array(psl_elements)
            
            # Vectorized multiplication of all pairs?
            # a1*a2 + b1*c2, etc.
            # This might be memory heavy if done for all pairs at once (1092x1092x4).
            # 1092^2 * 4 * 4 bytes ~ 19 MB. Fine.
            
            a1 = elems[:, 0].reshape(-1, 1)
            b1 = elems[:, 1].reshape(-1, 1)
            c1 = elems[:, 2].reshape(-1, 1)
            d1 = elems[:, 3].reshape(-1, 1)
            
            a2 = elems[:, 0].reshape(1, -1)
            b2 = elems[:, 1].reshape(1, -1)
            c2 = elems[:, 2].reshape(1, -1)
            d2 = elems[:, 3].reshape(1, -1)
            
            prod_a = (a1 * a2 + b1 * c2) % q
            prod_b = (a1 * b2 + b1 * d2) % q
            prod_c = (c1 * a2 + d1 * c2) % q
            prod_d = (c1 * b2 + d1 * d2) % q
            
            # Map products back to indices
            # Create a lookup for 4-tuples
            # Since q is small, we can flatten the 4-tuple into a single integer key
            # key = a + b*q + c*q^2 + d*q^3
            keys = (prod_a + prod_b * q + prod_c * (q * q) + prod_d * (q * q * q)).astype(np.int64)
            
            # Precompute key to index map for all elements
            all_keys = (elems[:, 0] + elems[:, 1] * q + elems[:, 2] * (q * q) + elems[:, 3] * (q * q * q)).astype(np.int64)
            key_to_idx = np.zeros(q**4, dtype=np.int32)
            key_to_idx[all_keys] = np.arange(n_vertices)
            
            mul_table = key_to_idx[keys].reshape(n_vertices, n_vertices)

            # Find involutions
            involution_indices = []
            for i in range(n_vertices):
                if i == identity_idx:
                    continue
                if mul_table[i, i] == identity_idx:
                    involution_indices.append(i)

            involution_indices = np.array(involution_indices, dtype=np.int32)
            n_inv = len(involution_indices)

            def check_girth_cayley(S_indices):
                """Check if Cayley graph with generators S_indices has girth >= g.
                BFS from Identity."""
                # Build neighbor table for these generators
                # neighbor[v, s] = mul_table[v, S_indices[s]]
                neighbor = mul_table[:, S_indices] # Shape (n_vertices, 5)

                dist = np.full(n_vertices, -1, dtype=np.int32)
                parent = np.full(n_vertices, -1, dtype=np.int32)
                dist[identity_idx] = 0
                parent[identity_idx] = -2 # Sentinel for start
                
                queue = deque()
                queue.append(identity_idx)

                max_depth = g - 1
                while queue:
                    if time.time() > deadline:
                        return False
                    u = queue.popleft()
                    du = dist[u]
                    if du >= max_depth:
                        continue
                    
                    # Get neighbors
                    neighbors = neighbor[u]
                    for v in neighbors:
                        v = int(v)
                        if v == identity_idx:
                            # Found a cycle back to start
                            if du + 1 < g:
                                return False
                        elif v == parent[u]:
                            continue
                        elif dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            queue.append(v)
                return True

            # Search for a valid set of 5 involutions
            random.seed(42)
            tried = set()
            max_tries = 2000
            for attempt in range(max_tries):
                if time.time() > deadline:
                    break
                # Pick 5 distinct involutions
                S = np.random.choice(n_inv, size=5, replace=False)
                S = np.sort(S)
                key = tuple(S)
                if key in tried:
                    continue
                tried.add(key)

                if check_girth_cayley(S):
                    # Valid! Build the edge list
                    edges = []
                    # To avoid duplicates, only add edge if v < u
                    for v in range(n_vertices):
                        neighbors = mul_table[v, S]
                        for u in neighbors:
                            u = int(u)
                            if v < u:
                                edges.append((v, u))
                    return n_vertices, edges

        except Exception:
            pass

    return luw_fallback()
