import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW graph from seed
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

    # Try PSL(2, 13)
    if k == 5 and g <= 12: # Girth 9, 10, 11, 12
        try:
            q = 13
            # Generate PSL(2, q)
            # Elements are (a, b, c, d) with ad - bc = 1
            elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        # d = (1 + bc) * a^-1 if a != 0
                        if a != 0:
                            inv_a = pow(a, -1, q)
                            d = (1 + b * c) * inv_a % q
                            elements.append((a, b, c, d))
                        else:
                            # a=0 => -bc = 1 => bc = -1
                            # b != 0, c = -1 * b^-1
                            for b in range(1, q):
                                c = (-1 * pow(b, -1, q)) % q
                                d = 0 # wait, if a=0, ad=0. bc=1? No, det=1.
                                # If a=0, det = -bc = 1 => bc = -1.
                                # d can be anything? No, matrix is [0 b; c d].
                                # det = 0*d - b*c = -bc = 1.
                                # So d is free?
                                # In PSL(2,q), we identify M and -M.
                                # Standard rep: a+d=0 or a+d!=0?
                                # Let's just generate all and dedupe.
                                pass
            
            # Better generation:
            # Iterate a, b, c. If a != 0, d is fixed.
            # If a == 0, b != 0, c = -1/b. d is free?
            # No, in SL(2,q), if a=0, then -bc=1. d is free.
            # So for each b in 1..q-1, c is fixed, d in 0..q-1.
            
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a != 0:
                            d = (1 + b * c) * pow(a, -1, q) % q
                            sl_elements.append((a, b, c, d))
                        else:
                            if b == 0:
                                continue # det = 0
                            # -bc = 1 => c = -1/b
                            c_val = (-1 * pow(b, -1, q)) % q
                            if c_val != c:
                                continue
                            for d in range(q):
                                sl_elements.append((a, b, c, d))
            
            # Deduplicate PSL(2,q) = SL(2,q) / {I, -I}
            # Representative: a+d > 0 or (a+d==0 and a>0) etc.
            # Simpler: use a set of tuples, and for each, if (-a,-b,-c,-d) is in set, skip.
            # Or just normalize: ensure (a,d) is in a canonical half.
            
            psl_elements = []
            seen = set()
            for e in sl_elements:
                a, b, c, d = e
                # Canonical: (a, b, c, d) such that (a, d) is lexicographically smaller than (-a, -d)
                # Or just add both to a set, then pick one.
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
            # n_vertices should be q(q^2-1)/2 = 13*168/2 = 1092
            
            # Map to indices
            idx_map = {e: i for i, e in enumerate(psl_elements)}
            
            # Find involutions (order 2)
            # A^2 = I => trace = 0 (in PSL, A^2=I implies A=-I or A=I? No, in PSL A^2=I means A is involution or identity)
            # In SL, A^2 = I => A = I or A = -I or trace 0.
            # In PSL, I and -I are same. So identity.
            # So involutions are those with trace 0 (and not identity).
            involutions = []
            for e in psl_elements:
                a, b, c, d = e
                if (a + d) % q == 0 and (b, c) != (0, 0):
                    involutions.append(e)
            
            # Random search for S
            best_n = n_vertices
            best_edges = None
            
            for _ in range(500):
                if time.time() > deadline:
                    break
                if len(involutions) < 5:
                    break
                S = random.sample(involutions, 5)
                # Check if S generates the group
                # BFS from identity
                identity = (1, 0, 0, 1)
                if identity not in idx_map:
                    identity = (1, 0, 0, 1) # Should be there
                
                visited = {identity: 0}
                queue = [identity]
                edges = []
                
                while queue:
                    if time.time() > deadline:
                        break
                    u = queue.pop(0)
                    for s in S:
                        # Multiply u * s
                        u_a, u_b, u_c, u_d = u
                        s_a, s_b, s_c, s_d = s
                        # Matrix mult mod q
                        p_a = (u_a * s_a + u_b * s_c) % q
                        p_b = (u_a * s_b + u_b * s_d) % q
                        p_c = (u_c * s_a + u_d * s_c) % q
                        p_d = (u_c * s_b + u_d * s_d) % q
                        v = (p_a, p_b, p_c, p_d)
                        if v not in visited:
                            visited[v] = len(visited)
                            queue.append(v)
                            edges.append((idx_map[u], idx_map[v]))
                
                if len(visited) != n_vertices:
                    continue
                
                # Check girth
                # BFS from each vertex up to depth 4
                # If we find a cycle < 9, fail.
                # Actually, just check if any vertex has a neighbor at dist < 4 that is already visited in a way that forms a short cycle?
                # Easier: Run BFS from each vertex, record dist and parent.
                # If we encounter a visited node v from u, and dist[u] + 1 - dist[v] is small?
                # Standard girth check:
                # For each start node, BFS.
                # If we reach a node already visited by the SAME BFS, and the path length + 1 < 9?
                # No, we need to check for ANY cycle.
                
                # Optimization: Only check girth if we have time.
                # 1092 vertices. BFS depth 4 is fast.
                girth_ok = True
                for start in range(n_vertices):
                    # BFS
                    dist = [-1] * n_vertices
                    dist[start] = 0
                    q_bfs = [start]
                    head = 0
                    while head < len(q_bfs):
                        if time.time() > deadline:
                            girth_ok = False
                            break
                        u = q_bfs[head]
                        head += 1
                        if dist[u] >= 4:
                            continue
                        # Neighbors of u
                        u_elem = psl_elements[u]
                        for s in S:
                            u_a, u_b, u_c, u_d = u_elem
                            s_a, s_b, s_c, s_d = s
                            p_a = (u_a * s_a + u_b * s_c) % q
                            p_b = (u_a * s_b + u_b * s_d) % q
                            p_c = (u_c * s_a + u_d * s_c) % q
                            p_d = (u_c * s_b + u_d * s_d) % q
                            v = idx_map[(p_a, p_b, p_c, p_d)]
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                q_bfs.append(v)
                            else:
                                # Found a cycle
                                # Cycle length = dist[u] + 1 + dist[v]? No.
                                # If v is visited, the cycle length is dist[u] + 1 + dist[v] is not correct.
                                # The cycle is start -> ... -> u -> v -> ... -> start.
                                # Length = dist[u] + 1 + dist[v] is only if v is on the path?
                                # No, BFS tree.
                                # If v is already visited, the cycle length is dist[u] + 1 + dist[v] is NOT the cycle length.
                                # The cycle length is dist[u] + 1 + (dist from v to start via tree path).
                                # Actually, if we just want to know if girth < 9:
                                # If we find a back edge (u, v) where dist[u] and dist[v] are known,
                                # the cycle length is dist[u] + 1 + dist[v] is wrong.
                                # Correct: The cycle length is dist[u] + 1 + dist[v] is only if we consider the path to root.
                                # The length of the cycle formed by the edge (u,v) and the tree paths to root is dist[u] + 1 + dist[v].
                                # Wait, the path from u to root is dist[u]. Path from v to root is dist[v].
                                # The cycle is u -> v -> ... -> root -> ... -> u.
                                # Length = 1 + dist[v] + dist[u].
                                # If this length < 9, then girth < 9.
                                if dist[u] + 1 + dist[v] < 9:
                                    girth_ok = False
                                    break
                        if not girth_ok:
                            break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    best_edges = edges
                    break
            
            if best_edges is not None:
                return n_vertices, best_edges
                
        except Exception:
            pass
            
    return luw_fallback()
