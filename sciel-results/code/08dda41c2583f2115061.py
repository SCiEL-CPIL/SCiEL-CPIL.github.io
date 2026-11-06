import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    # --- Greenwood-Gleason 32-vertex base ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def base_32():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    base_colors = base_32()

    # --- Bitset helpers ---
    def build_adj(colors, n):
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def count_mono_triangles(adj, n):
        count = 0
        for c in range(4):
            for i in range(n):
                neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
                j_mask = neigh
                while j_mask:
                    j_bit = j_mask & (-j_mask)
                    j = j_bit.bit_length() - 1
                    j_mask ^= j_bit
                    common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                    count += common.bit_count()
        return count

    def find_one_mono_triangle(adj, n):
        for c in range(4):
            for i in range(n):
                neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
                j_mask = neigh
                while j_mask:
                    j_bit = j_mask & (-j_mask)
                    j = j_bit.bit_length() - 1
                    j_mask ^= j_bit
                    common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                    if common:
                        k_bit = common & (-common)
                        k = k_bit.bit_length() - 1
                        return (i, j, k, c)
        return None

    def greedy_init(n):
        m = n * (n - 1) // 2
        colors = [0] * m
        adj = [[0] * n for _ in range(4)]
        for v in range(1, n):
            for u in range(v):
                best_c = 0
                min_cost = float('inf')
                for c in range(4):
                    cost = (adj[c][u] & ((1 << u) - 1)).bit_count()
                    if cost < min_cost:
                        min_cost = cost
                        best_c = c
                idx = u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
                colors[idx] = best_c
                adj[best_c][u] |= (1 << v)
                adj[best_c][v] |= (1 << u)
        return colors

    def local_search(n, time_budget):
        m = n * (n - 1) // 2
        start = time.time()
        best_result = None
        best_tri = float('inf')

        for restart in range(200):
            if time.time() - start > time_budget:
                break
            colors = greedy_init(n)
            adj = build_adj(colors, n)
            tri_count = count_mono_triangles(adj, n)
            if tri_count < best_tri:
                best_tri = tri_count
                best_result = colors[:]
            if tri_count == 0:
                return colors

            no_improve = 0
            max_no_improve = 50
            while no_improve < max_no_improve:
                if time.time() - start > time_budget:
                    break
                tri = find_one_mono_triangle(adj, n)
                if tri is None:
                    return colors
                i, j, k, c = tri
                edges = [(i, j), (i, k), (j, k)]
                best_delta = float('inf')
                best_edge = None
                best_new_c = -1
                for u, v in edges:
                    for new_c in range(4):
                        if new_c == c:
                            continue
                        removed = (adj[c][u] & adj[c][v]).bit_count()
                        added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                        delta = added - removed
                        if delta < best_delta:
                            best_delta = delta
                            best_edge = (u, v)
                            best_new_c = new_c
                if best_edge is not None and best_delta < 0:
                    u, v = best_edge
                    old_c = c
                    new_c = best_new_c
                    colors[u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)] = new_c
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    tri_count += best_delta
                    no_improve = 0
                    if tri_count < best_tri:
                        best_tri = tri_count
                        best_result = colors[:]
                    if tri_count == 0:
                        return colors
                else:
                    no_improve += 1
        return best_result if best_tri == 0 else None

    def simulated_annealing(n, time_budget):
        m = n * (n - 1) // 2
        start = time.time()
        idx_to_edge = [None] * m
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_to_edge[idx] = (i, j)
                idx += 1

        best_result = None
        best_tri = float('inf')

        for restart in range(100):
            if time.time() - start > time_budget:
                break
            colors = greedy_init(n)
            adj = build_adj(colors, n)
            tri_count = count_mono_triangles(adj, n)
            if tri_count < best_tri:
                best_tri = tri_count
                best_result = colors[:]
            if tri_count == 0:
                return colors

            T = 5.0
            T_min = 0.01
            cooling = 0.9995
            accepted = 0
            total_steps = 0

            while True:
                if time.time() - start > time_budget:
                    break
                ri = random.randint(0, m - 1)
                old_c = colors[ri]
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                u, v = idx_to_edge[ri]
                removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
                added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                delta = added - removed
                total_steps += 1
                if delta <= 0:
                    colors[ri] = new_c
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    tri_count += delta
                    accepted += 1
                    if tri_count < best_tri:
                        best_tri = tri_count
                        best_result = colors[:]
                        if tri_count == 0:
                            return colors
                else:
                    if T > T_min:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            colors[ri] = new_c
                            adj[old_c][u] &= ~(1 << v)
                            adj[old_c][v] &= ~(1 << u)
                            adj[new_c][u] |= (1 << v)
                            adj[new_c][v] |= (1 << u)
                            tri_count += delta
                            accepted += 1
                T *= cooling
                if total_steps > 1000 and accepted < total_steps * 0.01:
                    break
        return best_result if best_tri == 0 else None

    def backtrack_add_vertex(adj, n, time_limit):
        start = time.time()
        assign = [-1] * n

        def is_valid(v, c):
            for u in range(v):
                if assign[u] == c and adj[u][v] == c:
                    return False
            return True

        def forward_check(v, c):
            for w in range(v + 1, n):
                has_color = False
                for cw in range(4):
                    valid = True
                    for u in range(w):
                        if u == v:
                            if c == cw and adj[v][w] == cw:
                                valid = False
                                break
                        elif assign[u] == cw and adj[u][w] == cw:
                            valid = False
                            break
                    if valid:
                        has_color = True
                        break
                if not has_color:
                    return False
            return True

        def bt(v):
            if time.time() - start > time_limit:
                return False
            if v == n:
                return True
            colors_to_try = []
            for c in range(4):
                if not is_valid(v, c):
                    continue
                block = 0
                for w in range(v + 1, n):
                    if adj[v][w] == c:
                        w_has_other = False
                        for cw in range(4):
                            if cw == c:
                                continue
                            valid = True
                            for u in range(w):
                                if u == v:
                                    continue
                                if assign[u] == cw and adj[u][w] == cw:
                                    valid = False
                                    break
                            if valid:
                                w_has_other = True
                                break
                        if not w_has_other:
                            block += 1
                colors_to_try.append((block, c))
            colors_to_try.sort()
            for _, c in colors_to_try:
                assign[v] = c
                if forward_check(v, c):
                    if bt(v + 1):
                        return True
                assign[v] = -1
            return False

        for attempt in range(5):
            if time.time() - start > time_limit:
                break
            random.seed(42 + n + attempt)
            assign = [-1] * n
            if bt(0):
                return assign
        return None

    # --- Main strategy ---
    # Phase 1: Try to extend from 32 using backtracking
    adj = build_adj(base_colors, 32)
    best_n = 32
    best_colors = base_colors[:]

    for target_n in range(33, 51):
        if time_left() < 30:
            break
        n = len(adj)
        time_limit = min(10.0, max(3.0, time_left() - 30))
        assignment = backtrack_add_vertex(adj, n, time_limit)
        if assignment is not None:
            new_n = n + 1
            new_adj = [[0] * new_n for _ in range(4)]
            for c in range(4):
                for i in range(n):
                    new_adj[c][i] = adj[c][i]
            for i in range(n):
                c = assignment[i]
                new_adj[c][i] |= (1 << n)
                new_adj[c][n] |= (1 << i)
            adj = new_adj
            best_n = new_n
            # Rebuild colors list
            m = best_n * (best_n - 1) // 2
            best_colors = [0] * m
            idx = 0
            for i in range(best_n):
                for j in range(i + 1, best_n):
                    for c in range(4):
                        if (adj[c][i] >> j) & 1:
                            best_colors[idx] = c
                            break
                    idx += 1
        else:
            break

    # Phase 2: If stuck, try local search / SA on larger n
    if best_n < 50 and time_left() > 40:
        for target in [40, 45, 50]:
            if time_left() < 40:
                break
            if target <= best_n:
                continue
            time_limit = min(25.0, max(10.0, time_left() - 40))
            result = local_search(target, time_limit)
            if result is not None:
                best_n = target
                best_colors = result
                break
            result = simulated_annealing(target, time_limit)
            if result is not None:
                best_n = target
                best_colors = result
                break

    # Final verification
    adj_final = build_adj(best_colors, best_n)
    if count_mono_triangles(adj_final, best_n) > 0:
        return {'n': 32, 'colors': base_colors}

    return {'n': best_n, 'colors': best_colors}
