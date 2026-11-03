import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

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

    if k == 5 and g <= 12:
        try:
            q = 13
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a != 0:
                            d = (1 + b * c) * pow(a, -1, q) % q
                            sl_elements.append((a, b, c, d))
                        else:
                            if b == 0:
                                continue
                            c_val = (-1 * pow(b, -1, q)) % q
                            if c_val != c:
                                continue
                            for d in range(q):
                                sl_elements.append((a, b, c, d))

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

            # Precompute group multiplication table as numpy array for fast neighbor lookup
            # mul_table[i, j] = index of psl_elements[i] * psl_elements[j]
            mul_table = np.zeros((n_vertices, n_vertices), dtype=np.int32)
            for i in range(n_vertices):
                a1, b1, c1, d1 = psl_elements[i]
                for j in range(n_vertices):
                    a2, b2, c2, d2 = psl_elements[j]
                    prod = ((a1 * a2 + b1 * c2) % q,
                            (a1 * b2 + b1 * d2) % q,
                            (c1 * a2 + d1 * c2) % q,
                            (c1 * b2 + d1 * d2) % q)
                    mul_table[i, j] = idx_map[prod]

            # Find involution indices
            involution_indices = []
            for i, e in enumerate(psl_elements):
                a, b, c, d = e
                if (a + d) % q == 0 and (b, c) != (0, 0):
                    involution_indices.append(i)

            involution_indices = np.array(involution_indices, dtype=np.int32)
            n_inv = len(involution_indices)

            best_edges = None

            # Precompute: for each pair of involution indices, the product index
            # This allows fast cycle detection
            # inv_prod[i, j] = mul_table[involution_indices[i], involution_indices[j]]
            inv_prod = mul_table[np.ix_(involution_indices, involution_indices)]

            def check_girth_numpy(S_indices):
                """Check girth >= g using numpy-vectorized BFS.
                S_indices: array of 5 involution indices (into psl_elements)."""
                # Precompute neighbors for each vertex: neighbor[v, s_idx] = neighbor vertex
                # neighbor[v, s] = mul_table[v, S_indices[s]]
                # Shape: (n_vertices, 5)
                S_arr = S_indices  # shape (5,)
                # For each vertex v and each generator s, compute v * s
                # We can do this as: for each s, mul_table[:, S_arr[s]]
                # neighbors[v, s] = mul_table[v, S_arr[s]]
                # Build full neighbor table: shape (n_vertices, 5)
                # neighbors = np.empty((n_vertices, 5), dtype=np.int32)
                # for s in range(5):
                #     neighbors[:, s] = mul_table[:, S_arr[s]]
                # That's 5 column assignments, each O(n)
                
                # Actually, let's just do BFS level by level with numpy
                # dist: array of length n_vertices, -1 for unvisited
                # Use a queue-based BFS but vectorize the neighbor expansion
                
                dist = np.full(n_vertices, -1, dtype=np.int32)
                dist[0] = 0  # identity is at index 0
                current_level = np.array([0], dtype=np.int32)
                
                for depth in range(4):  # We need to check up to depth 4 (cycles of length < 9)
                    if len(current_level) == 0:
                        break
                    # For all vertices in current_level, find all their neighbors
                    # neighbors_of_level = mul_table[current_level[:, None], S_arr[None, :]]
                    # Shape: (len(current_level), 5)
                    neighbors_of_level = mul_table[current_level[:, None], S_arr[None, :]]
                    # Flatten
                    flat_neighbors = neighbors_of_level.ravel()
                    
                    # Check if any neighbor has been visited at a distance that would create a short cycle
                    # For a neighbor v of u at depth d, if dist[v] >= 0 and dist[u] + 1 + dist[v] < g, fail
                    # dist[u] = depth for all u in current_level
                    # We need: depth + 1 + dist[v] < g  =>  dist[v] < g - depth - 1
                    threshold = g - depth - 1
                    
                    # Get distances of flat_neighbors
                    neighbor_dists = dist[flat_neighbors]
                    # Check if any neighbor is visited and dist < threshold
                    # But we also need to make sure we're not just going back to parent (non-backtracking)
                    # For simplicity in this check, we'll use a slightly conservative approach:
                    # If dist[v] >= 0 and dist[v] + depth + 1 < g, it's a short cycle
                    # However, this might catch the case where v is the parent of u (backtracking)
                    # For girth >= 9, we need no cycles of length 4, 6, 8
                    # A non-backtracking walk of length L from start back to start gives cycle length L
                    # But BFS finds shortest paths, so if dist[v] + depth + 1 < g and v != parent, it's a cycle
                    # For now, let's be conservative: if dist[v] >= 0 and dist[v] + depth + 1 < g, fail
                    # This may have false positives (backtracking) but for girth checking, 
                    # a proper non-backtracking check is needed. Let me refine.
                    
                    # Actually, the standard BFS girth check: 
                    # When expanding u at depth d, for each neighbor v:
                    #   if v is unvisited: mark dist[v] = d+1
                    #   elif v is not the parent of u and d+1+dist[v] < g: girth < g
                    # The "not parent" condition is important. In a tree, the only visited neighbor 
                    # should be the parent. If there's another visited neighbor, it's a cycle.
                    # But in BFS, we don't easily track parents in vectorized form.
                    # 
                    # Alternative: use the fact that in a graph with girth >= g, the BFS tree from 
                    # any vertex has no edges between vertices at the same level or between levels 
                    # that would create a short cycle. 
                    # 
                    # Let me use a different approach: check for cycles of length 4, 6, 8 directly
                    # using the precomputed inv_prod table.
                    
                    # For now, let's just check: if any neighbor has dist >= 0 and dist + depth + 1 < g
                    # This is a necessary condition for girth >= g (sufficient with proper non-backtracking)
                    # For a quick filter, this should work well.
                    
                    short_cycle_mask = (neighbor_dists >= 0) & (neighbor_dists < threshold)
                    if np.any(short_cycle_mask):
                        return False
                    
                    # Find new vertices to add to next level
                    unvisited_mask = neighbor_dists == -1
                    new_vertices = flat_neighbors[unvisited_mask]
                    if len(new_vertices) > 0:
                        # Mark them as visited
                        # Need to be careful: multiple entries might point to same vertex
                        unique_new = np.unique(new_vertices)
                        dist[unique_new] = depth + 1
                        current_level = unique_new
                    else:
                        break
                        
                return True

            # Better girth check: use non-backtracking properly
            # For a Cayley graph with generators S (all involutions), 
            # a cycle of length L exists iff there exist s1,...,sL in S such that s1*s2*...*sL = I
            # and no consecutive si = s{i+1} (non-backtracking)
            # For girth >= 9, we need no such cycles of length 4, 6, 8
            
            # Precompute: for each pair (i,j) of involution indices, the product
            # inv_prod[i,j] = psl_elements[involution_indices[i]] * psl_elements[involution_indices[j]]
            # A 4-cycle: s1*s2*s1*s2 = I, i.e., (s1*s2)^2 = I, i.e., s1*s2 is an involution
            # A 6-cycle: s1*s2*s3*s1*s2*s3 = I, i.e., (s1*s2*s3)^2 = I
            # An 8-cycle: s1*s2*s3*s4*s1*s2*s3*s4 = I, i.e., (s1*s2*s3*s4)^2 = I
            
            # Precompute is_involution lookup for fast girth checking
            is_involution = np.zeros(n_vertices, dtype=bool)
            for i, e in enumerate(psl_elements):
                a, b, c, d = e
                if (a + d) % q == 0 and (b, c) != (0, 0):
                    is_involution[i] = True
            
            def check_girth_direct(S_indices):
                """Directly check for cycles of length 4, 6, 8 using precomputed products."""
                # S_indices: 5-element array of indices into involution_indices
                # Get the actual psl element indices
                psl_S = involution_indices[S_indices]  # shape (5,)
                
                # 4-cycle: s1*s2*s1*s2 = (s1*s2)^2 = I iff s1*s2 is an involution
                # Non-backtracking: s1 != s2
                for i in range(5):
                    for j in range(i+1, 5):
                        prod_idx = mul_table[psl_S[i], psl_S[j]]
                        if is_involution[prod_idx]:
                            return False
                
                # 6-cycle: s1*s2*s3*s1*s2*s3 = I iff (s1*s2*s3)^2 = I
                # Non-backtracking: s1!=s2, s2!=s3, s3!=s1 (for the wrap-around s3*s1)
                # Also s1!=s2, s2!=s3 already enforced; need s3!=s1
                for i in range(5):
                    for j in range(5):
                        if i == j:
                            continue
                        for l in range(5):
                            if l == i or l == j:
                                continue
                            tmp = mul_table[psl_S[i], psl_S[j]]
                            prod_idx = mul_table[tmp, psl_S[l]]
                            if is_involution[prod_idx]:
                                return False
                
                # 8-cycle: s1*s2*s3*s4*s1*s2*s3*s4 = I iff (s1*s2*s3*s4)^2 = I
                # Non-backtracking: s1!=s2, s2!=s3, s3!=s4, s4!=s1
                for i in range(5):
                    for j in range(5):
                        if i == j:
                            continue
                        for l in range(5):
                            if l == i or l == j:
                                continue
                            for m in range(5):
                                if m == i or m == l or m == j:
                                    continue
                                tmp1 = mul_table[psl_S[i], psl_S[j]]
                                tmp2 = mul_table[tmp1, psl_S[l]]
                                prod_idx = mul_table[tmp2, psl_S[m]]
                                if is_involution[prod_idx]:
                                    return False
                
                return True

            best_edges = None
            np.random.seed(42)
            
            # Precompute all 5-combinations of involution indices for systematic search
            # But first try random sampling for speed
            inv_list = involution_indices.tolist()
            
            # Try random sampling first
            for attempt in range(50000):
                if time.time() > deadline - 10:  # Leave 10s for final verification
                    break
                if n_inv < 5:
                    break
                    
                # Sample 5 distinct involution indices
                S_idx = np.random.choice(n_inv, size=5, replace=False)
                
                # Quick girth check
                if not check_girth_direct(S_idx):
                    continue
                
                psl_S = involution_indices[S_idx]
                
                # Check if S generates the group using BFS
                visited = np.zeros(n_vertices, dtype=bool)
                visited[0] = True
                queue = [0]
                head = 0
                edges = []
                
                while head < len(queue):
                    if time.time() > deadline - 10:
                        break
                    u = queue[head]
                    head += 1
                    for s in range(5):
                        v = mul_table[u, psl_S[s]]
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                            edges.append((u, v))
                
                if not visited.all():
                    continue
                
                # For Cayley graphs, girth is uniform by translation invariance
                # check_girth_direct already verified no cycles of length 4, 6, 8
                # Do a final non-backtracking BFS from identity to be safe
                dist = np.full(n_vertices, -1, dtype=np.int32)
                dist[0] = 0
                queue_nb = [(0, -1)]
                head_nb = 0
                girth_ok = True
                while head_nb < len(queue_nb):
                    if time.time() > deadline - 5:
                        girth_ok = False
                        break
                    u, prev_s = queue_nb[head_nb]
                    head_nb += 1
                    du = dist[u]
                    if du >= 4:
                        continue
                    for s in range(5):
                        if s == prev_s:
                            continue
                        v = mul_table[u, psl_S[s]]
                        dv = dist[v]
                        if dv == -1:
                            dist[v] = du + 1
                            queue_nb.append((v, s))
                        else:
                            if du + 1 + dv < g:
                                girth_ok = False
                                break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    best_edges = edges
                    break

            if best_edges is None and time.time() < deadline - 15:
                # Fallback: systematic search over combinations
                # Precompute a lookup for fast 8-cycle checking
                # Use itertools for combinations
                from itertools import combinations
                for combo in combinations(range(n_inv), 5):
                    if time.time() > deadline - 5:
                        break
                    S_idx = np.array(combo)
                    if not check_girth_direct(S_idx):
                        continue
                    psl_S = involution_indices[S_idx]
                    visited = np.zeros(n_vertices, dtype=bool)
                    visited[0] = True
                    queue = [0]
                    head = 0
                    edges = []
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        for s in range(5):
                            v = mul_table[u, psl_S[s]]
                            if not visited[v]:
                                visited[v] = True
                                queue.append(v)
                                edges.append((u, v))
                    if visited.all():
                        # Verify girth
                        dist = np.full(n_vertices, -1, dtype=np.int32)
                        dist[0] = 0
                        queue_nb = [(0, -1)]
                        head_nb = 0
                        girth_ok = True
                        while head_nb < len(queue_nb):
                            u, prev_s = queue_nb[head_nb]
                            head_nb += 1
                            du = dist[u]
                            if du >= 4:
                                continue
                            for s in range(5):
                                if s == prev_s:
                                    continue
                                v = mul_table[u, psl_S[s]]
                                dv = dist[v]
                                if dv == -1:
                                    dist[v] = du + 1
                                    queue_nb.append((v, s))
                                else:
                                    if du + 1 + dv < g:
                                        girth_ok = False
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
