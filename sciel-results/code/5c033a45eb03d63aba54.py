import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)  # 120

    def creates_q3(u, v, adj):
        """Check if adding edge (u,v) to graph with adjacency 'adj' creates a Q3."""
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)

        if not mask_u or not mask_v:
            return False

        # Precompute common neighbors for all pairs to speed up inner loops
        common = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                common[i][j] = adj[i] & adj[j]
                common[j][i] = common[i][j]

        # Iterate a in N(u) \ {v}
        mask_a = mask_u
        while mask_a:
            lsb_a = mask_a & -mask_a
            a = lsb_a.bit_length() - 1
            mask_a ^= lsb_a

            # Iterate b in N(u) \ {v, a}
            mask_b = mask_u & ~lsb_a
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b ^= lsb_b

                # c in N(v) \ {u} ∩ N(a)
                mask_c = mask_v & adj[a]
                if not mask_c:
                    continue

                # d in N(v) \ {u} ∩ N(b)
                mask_d = mask_v & adj[b]
                if not mask_d:
                    continue

                # Exclude a and b from c and d candidates
                mask_c &= ~(lsb_a | lsb_b)
                mask_d &= ~(lsb_a | lsb_b)

                if not mask_c or not mask_d:
                    continue

                # Use precomputed common neighbors
                ab_mask = common[a][b]

                # Iterate c in mask_c
                mask_c_iter = mask_c
                while mask_c_iter:
                    lsb_c = mask_c_iter & -mask_c_iter
                    c = lsb_c.bit_length() - 1
                    mask_c_iter ^= lsb_c

                    # d in mask_d, d != c
                    mask_d_iter = mask_d & ~lsb_c
                    if not mask_d_iter:
                        continue

                    while mask_d_iter:
                        lsb_d = mask_d_iter & -mask_d_iter
                        d = lsb_d.bit_length() - 1
                        mask_d_iter ^= lsb_d

                        # e in N(a) ∩ N(b) \ {u,v,a,b,c,d}
                        forbidden = (1 << u) | (1 << v) | lsb_a | lsb_b | lsb_c | lsb_d
                        mask_e = ab_mask & ~forbidden
                        if not mask_e:
                            continue

                        # f in N(c) ∩ N(d) \ {u,v,a,b,c,d}
                        mask_f = common[c][d] & ~forbidden
                        if not mask_f:
                            continue

                        # Check if ∃ e in mask_e, f in mask_f s.t. e != f and (e,f) edge
                        if mask_e.bit_count() <= mask_f.bit_count():
                            mask_e_iter = mask_e
                            while mask_e_iter:
                                lsb_e = mask_e_iter & -mask_e_iter
                                e = lsb_e.bit_length() - 1
                                mask_e_iter ^= lsb_e
                                if mask_f & adj[e] & ~lsb_e:
                                    return True
                        else:
                            mask_f_iter = mask_f
                            while mask_f_iter:
                                lsb_f = mask_f_iter & -mask_f_iter
                                f = lsb_f.bit_length() - 1
                                mask_f_iter ^= lsb_f
                                if mask_e & adj[f] & ~lsb_f:
                                    return True
        return False

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def greedy_construct():
        adj = [0] * n
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        for u, v in shuffled_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def greedy_construct_weighted():
        """Greedy with preference for edges that connect high-degree vertices."""
        adj = [0] * n
        remaining = all_edges[:]
        while remaining and time.time() < deadline - 10:
            # Score each edge by sum of degrees of endpoints
            best_idx = 0
            best_score = -1
            for i, (u, v) in enumerate(remaining):
                score = adj[u].bit_count() + adj[v].bit_count()
                if score > best_score:
                    best_score = score
                    best_idx = i
            u, v = remaining.pop(best_idx)
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search_add(adj):
        """Try to add any missing edge that doesn't create Q3. Repeat until no improvement."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                       if not (adj[i] >> j) & 1]
            random.shuffle(missing)
            for u, v in missing:
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        return adj

    def local_search_add_exhaustive(adj):
        """Exhaustive single-pass add: try every missing edge exactly once."""
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    if not creates_q3(i, j, adj):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        improved = True
        return improved

    def local_search_swap(adj, max_swaps=200):
        """Try 2-opt swaps: remove one edge, add another. Returns improved adj if found."""
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] >> j) & 1]
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] >> j) & 1]
        if not edges or not missing:
            return adj
        
        random.shuffle(edges)
        random.shuffle(missing)
        
        for _ in range(max_swaps):
            if time.time() > deadline - 2:
                break
            e_idx = random.randint(0, len(edges) - 1)
            m_idx = random.randint(0, len(missing) - 1)
            u1, v1 = edges[e_idx]
            u2, v2 = missing[m_idx]
            
            # Remove edge (u1,v1)
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            
            # Try to add edge (u2,v2)
            if not creates_q3(u2, v2, adj):
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                edges[e_idx] = (u2, v2)
                missing[m_idx] = (u1, v1)
            else:
                # Restore
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
        return adj

    def kick_and_refill(adj, kick_size):
        """Remove kick_size random edges, then refill via local search."""
        current_adj = adj[:]
        # Get current edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (current_adj[i] >> j) & 1]
        for _ in range(kick_size):
            if not edges:
                break
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            edges.pop(idx)
        current_adj = local_search_add(current_adj)
        # Try exhaustive add as well
        while local_search_add_exhaustive(current_adj) and time.time() < deadline - 5:
            pass
        return current_adj

    def kick_structural(adj, kick_size):
        """Remove edges from vertices with highest degree to create more structure."""
        current_adj = adj[:]
        # Find vertices with highest degree
        degs = [(adj[i].bit_count(), i) for i in range(n)]
        degs.sort(reverse=True)
        high_deg_vertices = [v for _, v in degs[:kick_size]]
        
        removed_count = 0
        for v in high_deg_vertices:
            if removed_count >= kick_size:
                break
            # Get neighbors of v
            neighbors = [(v, u) for u in range(n) if (current_adj[v] >> u) & 1]
            random.shuffle(neighbors)
            for u, w in neighbors:
                if removed_count >= kick_size:
                    break
                if not creates_q3(u, w, current_adj):
                    # Can't remove if it would break Q3-freeness? No, removing always helps
                    current_adj[u] &= ~(1 << w)
                    current_adj[w] &= ~(1 << u)
                    removed_count += 1
                    break
        current_adj = local_search_add(current_adj)
        while local_search_add_exhaustive(current_adj) and time.time() < deadline - 5:
            pass
        return current_adj

    def tabu_kick(adj, kick_size, tabu_size=3):
        """Remove kick_size edges, avoid re-adding tabu_size recently removed edges."""
        current_adj = adj[:]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (current_adj[i] >> j) & 1]
        removed = []
        for _ in range(kick_size):
            if not edges:
                break
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            edges.pop(idx)
            removed.append((u, v))

        # Refill: try to add edges, but skip those in 'removed' for the first pass
        # then allow them
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                       if not (current_adj[i] >> j) & 1]
            random.shuffle(missing)
            for u, v in missing:
                if (u, v) in removed or (v, u) in removed:
                    continue  # tabu
                if not creates_q3(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    improved = True
        # Second pass: allow tabu edges
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                       if not (current_adj[i] >> j) & 1]
            random.shuffle(missing)
            for u, v in missing:
                if not creates_q3(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    improved = True
        return current_adj

    best_adj = [0] * n
    best_count = 0

    # Try multiple initialization strategies
    init_strategies = ['random', 'weighted']
    
    while time.time() < deadline - 10:
        # Alternate between different initialization strategies
        strategy = random.choice(init_strategies)
        if strategy == 'random':
            adj = greedy_construct()
        else:
            adj = greedy_construct_weighted()
        
        adj = local_search_add(adj)
        # Exhaustive add pass
        while local_search_add_exhaustive(adj) and time.time() < deadline - 5:
            pass
        count = edge_count(adj)

        if count > best_count:
            best_count = count
            best_adj = adj[:]

        # Try several kicks from this local optimum
        if time.time() < deadline - 5:
            for kick_size in [2, 3, 4, 5, 7, 10]:
                if time.time() > deadline - 5:
                    break
                # Alternate between different kick types
                if random.random() < 0.3:
                    new_adj = kick_structural(adj, kick_size)
                else:
                    new_adj = kick_and_refill(adj, kick_size)
                new_adj = local_search_swap(new_adj)
                new_adj = local_search_add(new_adj)
                while local_search_add_exhaustive(new_adj) and time.time() < deadline - 5:
                    pass
                new_count = edge_count(new_adj)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = new_adj[:]
                    adj = new_adj  # continue kicking from improved solution

            # Try tabu kick
            if time.time() < deadline - 5:
                for kick_size in [3, 5, 7]:
                    if time.time() > deadline - 5:
                        break
                    new_adj = tabu_kick(adj, kick_size)
                    new_adj = local_search_swap(new_adj)
                    new_adj = local_search_add(new_adj)
                    while local_search_add_exhaustive(new_adj) and time.time() < deadline - 5:
                        pass
                    new_count = edge_count(new_adj)
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = new_adj[:]

    # Final polish: multiple passes of add + swap
    for _ in range(5):
        if time.time() > deadline - 2:
            break
        best_adj = local_search_add(best_adj)
        while local_search_add_exhaustive(best_adj) and time.time() < deadline - 2:
            pass
        best_adj = local_search_swap(best_adj, max_swaps=300)
    
    # Final exhaustive add
    best_adj = local_search_add(best_adj)
    while local_search_add_exhaustive(best_adj) and time.time() < deadline - 1:
        pass
    best_count = edge_count(best_adj)

    # Return edge list
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    return result
