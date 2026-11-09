import random
import time
import math

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = 135.0  # Safety margin

    adj_mask = [0] * n
    edges = set()

    def add_edge(u, v):
        if u > v:
            u, v = v, u
        if (u, v) not in edges:
            edges.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

    def remove_edge(u, v):
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            edges.remove((u, v))
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)

    def get_degrees():
        deg = [0] * n
        for i in range(n):
            deg[i] = adj_mask[i].bit_count()
        return deg

    def get_triangles_and_score():
        deg = get_degrees()
        max_sum = 0
        tri_list = []
        for i in range(n):
            mask_i = adj_mask[i]
            for j in range(i + 1, n):
                if not (mask_i & (1 << j)):
                    continue
                common = mask_i & adj_mask[j]
                common &= ~((1 << (j + 1)) - 1)
                while common:
                    k = (common & (-common)).bit_length() - 1
                    common &= ~(1 << k)
                    s = deg[i] + deg[j] + deg[k]
                    tri_list.append((i, j, k, s))
                    if s > max_sum:
                        max_sum = s
        return tri_list, max_sum

    def sum_sq_penalty():
        deg = get_degrees()
        penalty = 0
        for i in range(n):
            mask_i = adj_mask[i]
            for j in range(i + 1, n):
                if not (mask_i & (1 << j)):
                    continue
                common = mask_i & adj_mask[j]
                common &= ~((1 << (j + 1)) - 1)
                while common:
                    k = (common & (-common)).bit_length() - 1
                    common &= ~(1 << k)
                    s = deg[i] + deg[j] + deg[k]
                    penalty += s * s
        return penalty

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_pairs_set = set(all_pairs)

    def reset_graph(edge_list):
        nonlocal adj_mask, edges
        adj_mask = [0] * n
        edges = set()
        for u, v in edge_list:
            add_edge(u, v)

    def base_edges():
        """K_{8,8} + one intra-part edge."""
        e = []
        for u in range(8):
            for v in range(8, 16):
                e.append((u, v))
        e.append((0, 1))
        return e

    def perturb(base_e, k_swaps=3):
        """Apply k random edge swaps to base."""
        e = set(base_e)
        for _ in range(k_swaps):
            rem = random.choice(list(e))
            non_edges = [p for p in all_pairs if p not in e]
            if not non_edges:
                break
            add = random.choice(non_edges)
            e.discard(rem)
            e.add(add)
        return list(e)

    best_score = float('inf')
    best_edges_list = None
    best_adj = None

    def try_sa(initial_edges, max_time):
        nonlocal best_score, best_edges_list, best_adj
        reset_graph(initial_edges)
        cur_tris, cur_score = get_triangles_and_score()
        T = 8.0
        cooling = 0.9997
        T_min = 0.05

        local_best_score = cur_score
        local_best_adj = adj_mask[:]
        local_best_edges = set(edges)

        def make_move():
            """Generate a single edge swap move, biased toward problematic areas."""
            tris, _ = get_triangles_and_score()
            if not tris:
                # No triangles: just random swap
                if not edges:
                    return None, None
                rem = random.choice(list(edges))
                non_edges = [p for p in all_pairs if p not in edges]
                if not non_edges:
                    return None, None
                add = random.choice(non_edges)
                return rem, add

            # Find worst triangles
            tris_sorted = sorted(tris, key=lambda x: -x[3])
            top_k = max(1, len(tris_sorted) // 5)
            worst = random.choice(tris_sorted[:top_k])
            i, j, k, s = worst

            # Option A: remove an edge from the worst triangle
            tri_edges = [(i, j), (i, k), (j, k)]
            rem = random.choice(tri_edges)
            if rem[0] > rem[1]:
                rem = (rem[1], rem[0])

            # Option B: remove an edge incident to a high-degree vertex in the triangle
            deg = get_degrees()
            hi = max([i, j, k], key=lambda v: deg[v])
            incident = [(hi, x) for x in range(n) if x != hi and (min(hi,x), max(hi,x)) in edges]
            if incident and random.random() < 0.4:
                rem = random.choice(incident)
                if rem[0] > rem[1]:
                    rem = (rem[1], rem[0])

            # Choose add edge: prefer low-degree, non-triangle-creating
            deg = get_degrees()
            candidates = []
            for u, v in all_pairs:
                if (u, v) in edges:
                    continue
                if (u, v) == rem:
                    continue
                # Estimate triangle creation risk
                common = adj_mask[u] & adj_mask[v]
                tri_risk = common.bit_count()
                score = deg[u] + deg[v] + 2 * tri_risk
                candidates.append((score, u, v))

            if not candidates:
                return None, None
            candidates.sort()
            top_add = max(1, len(candidates) // 4)
            _, add_u, add_v = random.choice(candidates[:top_add])
            return rem, (add_u, add_v)

        t_start = time.time()
        iters = 0
        while time.time() < t_start + max_time and time.time() < deadline - 3:
            iters += 1
            rem, add = make_move()
            if rem is None or add is None:
                continue

            remove_edge(*rem)
            add_edge(*add)
            _, new_score = get_triangles_and_score()

            if new_score < cur_score:
                cur_score = new_score
                if new_score < local_best_score:
                    local_best_score = new_score
                    local_best_adj = adj_mask[:]
                    local_best_edges = set(edges)
            elif new_score == cur_score:
                if random.random() < 0.6:
                    pass
                else:
                    add_edge(*rem)
                    remove_edge(*add)
            else:
                delta = new_score - cur_score
                if random.random() < math.exp(-delta / max(T, T_min)):
                    cur_score = new_score
                else:
                    add_edge(*rem)
                    remove_edge(*add)

            T = max(T * cooling, T_min)

        # Update global best
        if local_best_score < best_score:
            best_score = local_best_score
            best_edges_list = list(local_best_edges)
            best_adj = local_best_adj[:]

        return local_best_score

    # Phase 1: Multiple restarts with SA
    restarts = [
        base_edges(),
        perturb(base_edges(), 1),
        perturb(base_edges(), 2),
        perturb(base_edges(), 3),
        perturb(base_edges(), 5),
        perturb(base_edges(), 8),
    ]

    # Also try asymmetric splits
    for split in [(7, 9), (6, 10)]:
        a, b = split
        e = []
        for u in range(a):
            for v in range(a, a + b):
                e.append((u, v))
        # Need 65 edges; a*b may not be 64
        while len(e) < 64:
            # Add random intra-part edges
            for _ in range(5):
                u = random.randint(0, a - 1)
                v = random.randint(0, a - 1)
                if u != v and (min(u,v), max(u,v)) not in e:
                    e.append((min(u,v), max(u,v)))
                    break
        # Add one more edge
        non_edges = [p for p in all_pairs if p not in e]
        if non_edges:
            e.append(random.choice(non_edges))
        e = e[:65]
        restarts.append(e)

    # Shuffle restarts
    random.shuffle(restarts)

    # Allocate time: ~15s per restart for first few, less for later
    for idx, init_e in enumerate(restarts):
        if time.time() > deadline - 10:
            break
        remaining_restarts = len(restarts) - idx
        time_per = min(15.0, (deadline - 10 - time.time()) / max(1, remaining_restarts))
        if time_per < 2.0:
            time_per = 2.0
        try_sa(init_e, time_per)

    # Phase 2: Final exhaustive polish on best solution
    if best_edges_list is not None and time.time() < deadline - 5:
        reset_graph(best_edges_list)
        improved = True
        polish_rounds = 0
        while improved and time.time() < deadline - 3 and polish_rounds < 30:
            improved = False
            polish_rounds += 1
            for rem in list(edges):
                if time.time() > deadline - 3:
                    break
                for add in all_pairs:
                    if add in edges or add == rem:
                        continue
                    remove_edge(*rem)
                    add_edge(*add)
                    _, new_score = get_triangles_and_score()
                    if new_score < best_score:
                        best_score = new_score
                        best_edges_list = list(edges)
                        best_adj = adj_mask[:]
                        improved = True
                    else:
                        add_edge(*rem)
                        remove_edge(*add)

    # Build result
    if best_edges_list is None:
        best_edges_list = base_edges()

    res_edges = []
    for u, v in best_edges_list:
        if u < v:
            res_edges.append([u, v])
        else:
            res_edges.append([v, u])

    return {'n': n, 'edges': res_edges}
