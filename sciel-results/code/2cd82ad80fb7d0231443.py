def construct():
    import random
    import sys
    sys.setrecursionlimit(10000)

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []

        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        conflict = [0] * m
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                else:
                    if (adj_mask[u1] & (1 << u2)) or (adj_mask[u1] & (1 << v2)) or \
                       (adj_mask[v1] & (1 << u2)) or (adj_mask[v1] & (1 << v2)):
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)

        # Fast greedy MIS size for pruning
        def greedy_bound(cands):
            size = 0
            while cands:
                v_bit = cands & -cands
                v = v_bit.bit_length() - 1
                size += 1
                cands &= ~(conflict[v] | v_bit)
            return size

        def find_mis(cands):
            best_size = 0
            best_mask = 0
            
            def bb(cands, curr_mask, curr_size):
                nonlocal best_size, best_mask
                
                if curr_size + cands.bit_count() <= best_size:
                    return
                
                # Use greedy bound
                if curr_size + greedy_bound(cands) <= best_size:
                    return
                
                if cands == 0:
                    if curr_size > best_size:
                        best_size = curr_size
                        best_mask = curr_mask
                    return
                
                # Pivot: Max degree in cands
                max_deg = -1
                pivot_v = -1
                temp = cands
                while temp:
                    bit = temp & -temp
                    v = bit.bit_length() - 1
                    deg = (conflict[v] & cands).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        pivot_v = v
                    temp ^= bit
                
                if pivot_v == -1:
                    return

                pivot_bit = 1 << pivot_v
                
                # Include
                bb(cands & ~(conflict[pivot_v] | pivot_bit), curr_mask | pivot_bit, curr_size + 1)
                
                # Exclude
                bb(cands & ~pivot_bit, curr_mask, curr_size)

            bb(cands, 0, 0)
            return best_mask

        colors = [0] * m
        available = (1 << m) - 1
        color_idx = 0
        
        while available:
            mis_mask = find_mis(available)
            temp = mis_mask
            while temp:
                bit = temp & -temp
                idx = bit.bit_length() - 1
                colors[idx] = color_idx
                available &= ~bit
                temp ^= bit
            color_idx += 1
            
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        res = solve_instance(n, d, seed)
        out.append(res)
        
    return out
