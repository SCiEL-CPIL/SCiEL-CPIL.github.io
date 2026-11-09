import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin under the 150s evaluator limit

    # Fallback: Andrásfai graph And(9) on 26 vertices (triangle-free, alpha = 9)
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed)
                  if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def has_independent_set_10(adj, n):
        """Return True iff the graph has an independent set of size 10."""
        all_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            non_neighbors[i] = (all_mask ^ (1 << i)) & ~adj[i]

        def dfs(mask, size):
            if size == 10:
                return True
            if mask == 0:
                return False
            if mask.bit_count() < 10 - size:
                return False

            min_count = n + 1
            best_v = -1
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                cnt = (non_neighbors[v] & mask).bit_count()
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                    if cnt == 0:
                        break
                temp_mask ^= lsb

            if best_v == -1:
                return False

            v_bit = 1 << best_v

            if dfs(non_neighbors[best_v] & mask, size + 1):
                return True

            if dfs(mask & ~v_bit, size):
                return True

            return False

        return dfs(all_mask, 0)

    # --- Strategy 1: Hoffman-Singleton Graph Subgraph (n=40) ---
    # The Hoffman-Singleton graph is a 5-regular triangle-free graph on 50 vertices
    # with independence number 7. Any subgraph is also triangle-free and has
    # independence number <= 7, which is < 10.
    # We construct the full HS graph and take the first 40 vertices.
    
    try:
        n_hs = 50
        adj_hs = [0] * n_hs
        
        # Map (i, j) to index i*10 + j
        # i in 0..4, j in 0..9
        
        for i in range(5):
            for j in range(10):
                u = i * 10 + j
                # Neighbors:
                # 1. (i, j+1)
                v1 = i * 10 + ((j + 1) % 10)
                # 2. (i, j-1)
                v2 = i * 10 + ((j - 1) % 10)
                # 3. (i+1, j+2)
                v3 = ((i + 1) % 5) * 10 + ((j + 2) % 10)
                # 4. (i+1, j+3)
                v4 = ((i + 1) % 5) * 10 + ((j + 3) % 10)
                # 5. (i-1, j+8)  (which is j-2)
                v5 = ((i - 1) % 5) * 10 + ((j + 8) % 10)
                
                for v in [v1, v2, v3, v4, v5]:
                    if u < v:
                        adj_hs[u] |= (1 << v)
                        adj_hs[v] |= (1 << u)
        
        # Take first 40 vertices (indices 0 to 39)
        n_target = 40
        adj_sub = [0] * n_target
        for u in range(n_target):
            # Only keep edges to vertices within 0..39
            mask = adj_hs[u] & ((1 << n_target) - 1)
            adj_sub[u] = mask
        
        # Verify triangle-free and alpha < 10 (should be true, but check for safety)
        # Triangle-free check:
        is_tf = True
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if adj_sub[i] & (1 << j):
                    common = adj_sub[i] & adj_sub[j]
                    if common:
                        is_tf = False
                        break
            if not is_tf:
                break
        
        if is_tf:
            if not has_independent_set_10(adj_sub, n_target):
                best_result = {'n': n_target, 'edges': edges_from_adj(adj_sub, n_target)}
                # Since n=40 is the max possible, we can return immediately
                return best_result
    except Exception:
        pass

    # --- Strategy 2: Deterministic Polarity Graph PG(2,5) (n=31) ---
    def construct_polarity_graph(q):
        n = q * q + q + 1
        if n > 40 or n < 2:
            return None, None
        points = []
        seen = set()
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv = pow(x, q - 2, q)
                        rep = (1, (y * inv) % q, (z * inv) % q)
                    elif y != 0:
                        inv = pow(y, q - 2, q)
                        rep = (0, 1, (z * inv) % q)
                    else:
                        rep = (0, 0, 1)
                    if rep not in seen:
                        seen.add(rep)
                        points.append(rep)
        n_actual = len(points)
        adj = [0] * n_actual
        for i in range(n_actual):
            for j in range(i + 1, n_actual):
                dot = (points[i][0] * points[j][0] +
                       points[i][1] * points[j][1] +
                       points[i][2] * points[j][2]) % q
                if dot == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return n_actual, adj

    for q in (5, 4, 3, 2):
        if time.time() > deadline - 5:
            break
        n_pol, adj_pol = construct_polarity_graph(q)
        if n_pol is not None and n_pol > best_result['n']:
            if not has_independent_set_10(adj_pol, n_pol):
                best_result = {'n': n_pol, 'edges': edges_from_adj(adj_pol, n_pol)}

    # --- Strategy 3: Random search for n=32..39 if HS failed ---
    for n in range(39, best_result['n'], -1):
        if time.time() > deadline - 10:
            break
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        restarts = 0
        while time.time() < deadline - 5:
            restarts += 1
            if restarts > 2000:
                break

            adj = [0] * n
            order = pairs[:]
            random.shuffle(order)

            for u, v in order:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            for u, v in order:
                if not (adj[u] & (1 << v)) and (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            if not has_independent_set_10(adj, n):
                best_result = {'n': n, 'edges': edges_from_adj(adj, n)}
                break

    return best_result
