import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

    # ---------- Exact checker (bitmask, early exit) ----------
    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            row = adj[u]
            for v in range(n):
                if row[v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def has_k4(n, masks):
        for u in range(n):
            mu = masks[u]
            for v in range(u + 1, n):
                if not (mu & (1 << v)):
                    continue
                common = mu & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            return True
        return False

    def has_ind6(n, masks):
        full_mask = (1 << n) - 1
        comp = [0] * n
        for u in range(n):
            comp[u] = full_mask & ~(masks[u] | (1 << u))
        for u in range(n):
            cu = comp[u]
            for v in range(u + 1, n):
                if not (cu & (1 << v)):
                    continue
                common = cu & comp[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp[wi] & (1 << xi):
                            common2 = cu & comp[v] & comp[wi] & comp[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp[yi] & (1 << zi):
                                        return True
        return False

    def is_valid(n, adj):
        masks = build_masks(n, adj)
        if has_k4(n, masks):
            return False
        if has_ind6(n, masks):
            return False
        return True

    def edges_from_adj(n, adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    # ---------- Cost: exact count of K4 + Ind6 (soft surrogate) ----------
    def count_k4(n, masks):
        c = 0
        for u in range(n):
            mu = masks[u]
            for v in range(u + 1, n):
                if not (mu & (1 << v)):
                    continue
                common = mu & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            c += 1
        return c

    def count_ind6(n, masks):
        full_mask = (1 << n) - 1
        comp = [0] * n
        for u in range(n):
            comp[u] = full_mask & ~(masks[u] | (1 << u))
        c = 0
        for u in range(n):
            cu = comp[u]
            for v in range(u + 1, n):
                if not (cu & (1 << v)):
                    continue
                common = cu & comp[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp[wi] & (1 << xi):
                            common2 = cu & comp[v] & comp[wi] & comp[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp[yi] & (1 << zi):
                                        c += 1
        return c

    def cost(n, adj):
        masks = build_masks(n, adj)
        return count_k4(n, masks) + count_ind6(n, masks)

    # ---------- Seeds ----------
    def paley17():
        sq = {x * x % 17 for x in range(1, 17)}
        adj = np.zeros((17, 17), dtype=bool)
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in sq:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    def spectral_seed(n, seed):
        rng = np.random.default_rng(seed)
        # Structured random symmetric matrix (Hadamard/Paley-like mix)
        A = rng.normal(size=(n, n))
        A = (A + A.T) / 2.0
        # Add a bit of structure: low-rank + diagonal
        A += 0.3 * np.eye(n)
        # Eigendecomposition
        w, V = np.linalg.eigh(A)
        # Take top-k eigenvectors, form projection P = V_k V_k^T
        k = max(1, int(0.5 * n))
        Vk = V[:, -k:]
        P = Vk @ Vk.T
        # Binarize: edge if P_ij > threshold (median)
        thr = np.median(P)
        adj = (P > thr)
        np.fill_diagonal(adj, False)
        # Symmetrize
        adj = (adj | adj.T)
        return adj

    def structured_seed(n, seed):
        """Combine Paley blocks + random cross edges."""
        rng = np.random.default_rng(seed)
        p17 = paley17()
        adj = np.zeros((n, n), dtype=bool)
        # Place Paley blocks
        pos = 0
        while pos + 17 <= n:
            adj[pos:pos+17, pos:pos+17] = p17
            pos += 17
        # Random cross edges (tuned density)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    continue
                # Same-block handled; cross-block random
                if i // 17 != j // 17:
                    if rng.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
        return adj

    # ---------- Simulated annealing with targeted moves ----------
    def local_search(n, adj0, time_budget, seed):
        rng = random.Random(seed)
        end_time = time.time() + time_budget
        adj = adj0.copy()
        cur_cost = cost(n, adj)
        if cur_cost == 0:
            return edges_from_adj(n, adj)

        best_cost = cur_cost
        best_adj = adj.copy()
        T = 5.0
        T_min = 0.01
        T_decay = 0.99995
        no_improve = 0
        max_no_improve = 2000
        step = 0

        # Precompute neighbor lists for faster edge sampling
        while time.time() < end_time and no_improve < max_no_improve:
            step += 1
            # Choose move type: single edge flip or pair flip
            if rng.random() < 0.7:
                # Single edge flip
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                old_u = adj[u, v]
                adj[u, v] = not old_u
                adj[v, u] = not old_u
                new_cost = cost(n, adj)
                if new_cost < cur_cost:
                    cur_cost = new_cost
                    no_improve = 0
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_adj = adj.copy()
                        if new_cost == 0:
                            return edges_from_adj(n, adj)
                else:
                    delta = new_cost - cur_cost
                    if delta > 0 and rng.random() < np.exp(-delta / T):
                        cur_cost = new_cost
                        no_improve += 1
                    else:
                        adj[u, v] = old_u
                        adj[v, u] = old_u
                        no_improve += 1
            else:
                # Pair of edge flips (2-opt style)
                u1 = rng.randint(0, n - 1)
                v1 = rng.randint(0, n - 1)
                u2 = rng.randint(0, n - 1)
                v2 = rng.randint(0, n - 1)
                if u1 == v1 or u2 == v2 or (u1 == u2 and v1 == v2):
                    continue
                old1 = adj[u1, v1]
                old2 = adj[u2, v2]
                adj[u1, v1] = not old1
                adj[v1, u1] = not old1
                adj[u2, v2] = not old2
                adj[v2, u2] = not old2
                new_cost = cost(n, adj)
                if new_cost < cur_cost:
                    cur_cost = new_cost
                    no_improve = 0
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_adj = adj.copy()
                        if new_cost == 0:
                            return edges_from_adj(n, adj)
                else:
                    delta = new_cost - cur_cost
                    if delta > 0 and rng.random() < np.exp(-delta / T):
                        cur_cost = new_cost
                        no_improve += 1
                    else:
                        adj[u1, v1] = old1
                        adj[v1, u1] = old1
                        adj[u2, v2] = old2
                        adj[v2, u2] = old2
                        no_improve += 1
            T = max(T_min, T * T_decay)

        if is_valid(n, best_adj):
            return edges_from_adj(n, best_adj)
        return None

    def try_find(n, time_budget, seed):
        end_time = time.time() + time_budget
        rng_master = random.Random(seed)
        
        # Generate diverse starting points
        def make_random_adj(density):
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng_master.random() < density:
                        adj[i, j] = True
                        adj[j, i] = True
            return adj
        
        def make_paley_like_adj():
            """Create a Paley-like structure using quadratic residues mod p where p≈n"""
            adj = np.zeros((n, n), dtype=bool)
            # Find prime p >= n
            def is_prime(x):
                if x < 2:
                    return False
                for i in range(2, int(x**0.5)+1):
                    if x % i == 0:
                        return False
                return True
            p = n
            while not is_prime(p):
                p += 1
            sq = {x*x % p for x in range(1, p)}
            for i in range(n):
                for j in range(i+1, n):
                    if (j-i) % p in sq or (i-j) % p in sq:
                        adj[i, j] = True
                        adj[j, i] = True
            return adj
        
        def make_complement_paley_adj():
            adj = make_paley_like_adj()
            # Complement (but keep simple graph)
            for i in range(n):
                for j in range(i+1, n):
                    adj[i, j] = not adj[i, j]
                    adj[j, i] = not adj[i, j]
            return adj
        
        # Multiple seed strategies
        seeds = []
        seeds.append(spectral_seed(n, seed))
        seeds.append(structured_seed(n, seed))
        seeds.append(make_random_adj(0.5))
        seeds.append(make_random_adj(0.35))
        seeds.append(make_random_adj(0.65))
        seeds.append(make_paley_like_adj())
        seeds.append(make_complement_paley_adj())
        
        # Evaluate all seeds, pick best few
        scored = []
        for s in seeds:
            c = cost(n, s)
            if c == 0:
                return edges_from_adj(n, s)
            scored.append((c, s))
        scored.sort(key=lambda x: x[0])
        
        # Allocate time to top seeds with multi-restart
        top_k = min(4, len(scored))
        time_per_seed = time_budget / top_k
        
        for idx in range(top_k):
            if time.time() >= end_time:
                break
            c, s = scored[idx]
            budget = min(time_per_seed, end_time - time.time())
            if budget < 0.5:
                break
            
            # Multiple restarts within this seed
            n_restarts = max(1, int(budget / 2.0))
            for r in range(n_restarts):
                if time.time() >= end_time:
                    break
                rem = end_time - time.time()
                if rem < 0.5:
                    break
                sb = min(rem, budget / n_restarts)
                # Jitter the seed slightly for diversity
                jittered = s.copy()
                for _ in range(rng_master.randint(0, 5)):
                    u = rng_master.randint(0, n-1)
                    v = rng_master.randint(0, n-1)
                    if u != v:
                        jittered[u, v] = not jittered[u, v]
                        jittered[v, u] = not jittered[v, u]
                result = local_search(n, jittered, sb, seed + idx * 100 + r)
                if result is not None:
                    return result
        
        return None

    # ---------- Main: grow n ----------
    best_n = 17
    best_edges = edges_from_adj(17, paley17())

    # Verify seed
    if is_valid(17, paley17()):
        best_n = 17
    else:
        # Fallback: random search for small n
        for n in range(10, 17):
            for attempt in range(50):
                adj = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i, j] = True
                            adj[j, i] = True
                if is_valid(n, adj):
                    best_n = n
                    best_edges = edges_from_adj(n, adj)
                    break
            if best_n >= n:
                break

    # Strategy: first try to jump to high n (30-35) with aggressive search,
    # then fill in gaps if needed. This is better than linear growth.
    
    # Phase 1: Try high n values first (35 down to 30)
    for n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        budget = min(remaining * 0.3, 8.0)
        if n <= best_n:
            continue
        edges = try_find(n, budget, seed=n * 101 + 7)
        if edges is not None:
            best_n = n
            best_edges = edges
    
    # Phase 2: Fill in from best_n+1 upward to 35
    for n in range(best_n + 1, 36):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        if n <= 22:
            budget = min(remaining * 0.3, 5.0)
        elif n <= 28:
            budget = min(remaining * 0.2, 4.0)
        else:
            budget = min(remaining * 0.15, 3.0)
        edges = try_find(n, budget, seed=n * 101 + 7)
        if edges is not None:
            best_n = n
            best_edges = edges
            if n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}
