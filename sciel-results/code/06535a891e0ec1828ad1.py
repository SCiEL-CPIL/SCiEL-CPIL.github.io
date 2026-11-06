"""Rewritten program with stronger local search and more diverse initial solutions."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math
    import sys

    def bron_kerbosch_bitmask(n, adj_mask, cliques):
        """Bron-Kerbosch with pivoting, using bitmasks. Collects cliques of size >= 2."""
        def bk(R, P, X):
            if P == 0 and X == 0:
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            PX = P | X
            if PX == 0:
                return
            best_v = -1
            best_count = -1
            temp = PX
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                cnt = (P & adj_mask[v]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
                temp ^= lsb
            if best_v >= 0:
                candidates = P & ~adj_mask[best_v]
            else:
                candidates = P
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                bk(R | lsb, P & adj_mask[v], X & adj_mask[v])
                P ^= lsb
                X |= lsb
                temp ^= lsb

        bk(0, (1 << n) - 1, 0)

    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def greedy_transversal(cliques_masks, n, mode="count", rng=None):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques.
        mode: 'count' = number of cliques, 'size' = sum of clique sizes, 'inverse' = prefer smaller cliques,
              'random' = random tie-breaking, 'weighted' = product of clique sizes (log-sum-exp surrogate)"""
        if not cliques_masks:
            return 0

        num_cliques = len(cliques_masks)
        vertex_cliques = [[] for _ in range(n)]
        for ci, cm in enumerate(cliques_masks):
            temp = cm
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                vertex_cliques[v].append(ci)
                temp ^= lsb

        uncovered = [True] * num_cliques
        selected = 0
        uncovered_count = num_cliques

        while uncovered_count > 0:
            best_v = -1
            best_score = -1.0
            best_v2 = -1
            for v in range(n):
                if selected & (1 << v):
                    continue
                score = 0.0
                for ci in vertex_cliques[v]:
                    if uncovered[ci]:
                        if mode == "count":
                            score += 1.0
                        elif mode == "size":
                            score += cliques_masks[ci].bit_count()
                        elif mode == "inverse":
                            score += 1.0 / max(1, cliques_masks[ci].bit_count())
                        elif mode == "weighted":
                            score += math.log(max(1, cliques_masks[ci].bit_count()))
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_v2 = -1
                elif score == best_score and best_score > 0:
                    if best_v2 < 0:
                        best_v2 = v
            if best_v < 0 or best_score <= 0:
                break
            if mode == "random" and best_v2 >= 0 and rng is not None:
                if rng.random() < 0.5:
                    best_v = best_v2
            selected |= (1 << best_v)
            for ci in vertex_cliques[best_v]:
                if uncovered[ci]:
                    uncovered[ci] = False
                    uncovered_count -= 1

        return selected

    def local_search_reduce(selected_mask, cliques_masks, n, max_iters=1000):
        """Try to remove vertices; if it breaks coverage, stop."""
        if not cliques_masks:
            return selected_mask

        num_cliques = len(cliques_masks)
        cover_count = [0] * num_cliques
        for ci, cm in enumerate(cliques_masks):
            cnt = (cm & selected_mask).bit_count()
            cover_count[ci] = cnt

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(n):
                if not (selected_mask & (1 << v)):
                    continue
                v_bit = 1 << v
                removable = True
                for ci, cm in enumerate(cliques_masks):
                    if cover_count[ci] == 1 and (cm & v_bit):
                        removable = False
                        break
                if removable:
                    selected_mask ^= v_bit
                    for ci, cm in enumerate(cliques_masks):
                        if cm & v_bit:
                            cover_count[ci] -= 1
                    improved = True
                    break
        return selected_mask

    def local_search_swap(selected_mask, cliques_masks, n, max_iters=500):
        """Try 1-swap: remove vertex u, add vertex v if net size doesn't increase and coverage is maintained."""
        if not cliques_masks:
            return selected_mask

        selected_vertices = [v for v in range(n) if selected_mask & (1 << v)]
        best_mask = selected_mask
        best_size = selected_mask.bit_count()

        for it in range(max_iters):
            improved = False
            for u in selected_vertices:
                trial_without_u = best_mask ^ (1 << u)
                missing = []
                for cm in cliques_masks:
                    if not (cm & trial_without_u):
                        missing.append(cm)
                if not missing:
                    best_mask = trial_without_u
                    best_size -= 1
                    selected_vertices.remove(u)
                    improved = True
                    break
                # Find a vertex v that covers all missing cliques
                for v in range(n):
                    if best_mask & (1 << v):
                        continue
                    covers_all = True
                    v_bit = 1 << v
                    for cm in missing:
                        if not (cm & v_bit):
                            covers_all = False
                            break
                    if covers_all:
                        best_mask = (best_mask ^ (1 << u)) | (1 << v)
                        selected_vertices[selected_vertices.index(u)] = v
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best_mask

    def local_search_2swap(selected_mask, cliques_masks, n, max_iters=200):
        """Try 2-swap: remove two vertices u,v, add two vertices x,y if size decreases and coverage maintained."""
        if not cliques_masks or selected_mask.bit_count() < 2:
            return selected_mask

        selected_vertices = [v for v in range(n) if selected_mask & (1 << v)]
        best_mask = selected_mask
        best_size = selected_mask.bit_count()
        num_cliques = len(cliques_masks)

        for it in range(max_iters):
            improved = False
            for i in range(len(selected_vertices)):
                for j in range(i + 1, len(selected_vertices)):
                    u = selected_vertices[i]
                    v = selected_vertices[j]
                    trial = best_mask ^ (1 << u) ^ (1 << v)
                    missing = []
                    for cm in cliques_masks:
                        if not (cm & trial):
                            missing.append(cm)
                    if len(missing) == 0:
                        # Can remove both
                        best_mask = trial
                        best_size -= 2
                        # Remove from selected_vertices
                        for idx in sorted([i, j], reverse=True):
                            selected_vertices.pop(idx)
                        improved = True
                        break
                    # Try to find two vertices x,y that cover all missing
                    if len(missing) <= 4:  # Only try if few cliques to cover
                        found_pair = False
                        for x in range(n):
                            if best_mask & (1 << x):
                                continue
                            x_bit = 1 << x
                            covers_x = all(cm & x_bit for cm in missing)
                            if covers_x:
                                # x covers all, but we need to add only one (net -1)
                                best_mask = trial | (1 << x)
                                best_size -= 1
                                selected_vertices[i] = x
                                selected_vertices.pop(j)
                                improved = True
                                found_pair = True
                                break
                            # Find y that covers remaining
                            remaining = [cm for cm in missing if not (cm & x_bit)]
                            for y in range(n):
                                if best_mask & (1 << y) or y == x:
                                    continue
                                y_bit = 1 << y
                                if all(cm & y_bit for cm in remaining):
                                    best_mask = trial | (1 << x) | (1 << y)
                                    best_size -= 1
                                    selected_vertices[i] = x
                                    selected_vertices[j] = y
                                    improved = True
                                    found_pair = True
                                    break
                            if found_pair:
                                break
                        if improved:
                            break
                if improved:
                    break
            if not improved:
                break
        return best_mask

    def solve_instance(n, p, seed, deadline):
        adj_mask = build_graph(n, p, seed)
        cliques = []
        bron_kerbosch_bitmask(n, adj_mask, cliques)

        if not cliques:
            return []

        best_mask = None
        best_size = n + 1

        rng = random.Random(seed + 1000)

        # Try multiple greedy strategies with multiple random seeds for random mode
        modes = ["count", "size", "inverse", "weighted"]
        for mode in modes:
            mask = greedy_transversal(cliques, n, mode)
            mask = local_search_reduce(mask, cliques, n)
            mask = local_search_swap(mask, cliques, n)
            mask = local_search_reduce(mask, cliques, n)
            mask = local_search_2swap(mask, cliques, n)
            mask = local_search_reduce(mask, cliques, n)
            s = mask.bit_count()
            if s < best_size:
                best_size = s
                best_mask = mask

            if time.time() > deadline:
                break

        # Try random-tie-breaking greedy multiple times
        for attempt in range(5):
            if time.time() > deadline:
                break
            mask = greedy_transversal(cliques, n, "random", rng)
            mask = local_search_reduce(mask, cliques, n)
            mask = local_search_swap(mask, cliques, n)
            mask = local_search_reduce(mask, cliques, n)
            mask = local_search_2swap(mask, cliques, n)
            mask = local_search_reduce(mask, cliques, n)
            s = mask.bit_count()
            if s < best_size:
                best_size = s
                best_mask = mask

        # Additional local search passes on best solution
        if best_mask is not None:
            for _ in range(5):
                if time.time() > deadline:
                    break
                new_mask = local_search_reduce(best_mask, cliques, n)
                new_mask = local_search_swap(new_mask, cliques, n)
                new_mask = local_search_reduce(new_mask, cliques, n)
                new_mask = local_search_2swap(new_mask, cliques, n)
                new_mask = local_search_reduce(new_mask, cliques, n)
                s = new_mask.bit_count()
                if s < best_size:
                    best_size = s
                    best_mask = new_mask
                elif s == best_size:
                    # Keep the one with more internal redundancy for future swaps
                    # (heuristic: prefer solution where removing any single vertex breaks fewer cliques)
                    pass
                if new_mask == best_mask:
                    break

        return [i for i in range(n) if best_mask & (1 << i)]

    start_time = time.time()
    deadline = start_time + 135

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            out.append(list(range(n)))
            continue

        result = solve_instance(n, p, seed, deadline)
        out.append(result)

    return out
# EVOLVE-BLOCK-END
