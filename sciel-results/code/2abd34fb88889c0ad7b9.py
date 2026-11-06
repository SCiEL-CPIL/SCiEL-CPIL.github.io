import time
import sys
import math
from random import Random


def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def dsatur_color(m, neigh, deg, rng=None, rand_order=False):
        color = [-1] * m
        sat = [set() for _ in range(m)]
        remaining = list(range(m))
        for _ in range(m):
            best = -1
            best_key = None
            for v in remaining:
                if rand_order and rng is not None:
                    key = (len(sat[v]), deg[v], rng.random())
                else:
                    key = (len(sat[v]), deg[v])
                if best_key is None or key > best_key:
                    best_key = key
                    best = v
            c = 0
            while c in sat[best]:
                c += 1
            color[best] = c
            remaining.remove(best)
            for u in neigh[best]:
                if color[u] == -1:
                    sat[u].add(c)
        return color, (max(color) + 1 if m else 0)

    def greedy_clique(m, neigh):
        neighset = [set(s) for s in neigh]
        cand = list(range(m))
        clique = 0
        while cand:
            # pick vertex with max degree inside cand
            best = cand[0]
            best_d = -1
            bs = neighset[best]
            for x in cand:
                s = neighset[x]
                dd = sum(1 for y in cand if y in s)
                if dd > best_d:
                    best_d = dd
                    best = x
                    bs = s
            clique += 1
            keep = bs
            cand = [x for x in cand if x != best and x in keep]
        return clique

    def recompute_conflict(color, neigh, m):
        conflict = [0] * m
        for v in range(m):
            cv = color[v]
            s = 0
            for u in neigh[v]:
                if color[u] == cv:
                    s += 1
            conflict[v] = s
        return conflict, sum(conflict) // 2

    def local_search(m, neigh, color_init, K, rng, time_limit):
        if K <= 0:
            return color_init, False
        color = [c if c < K else rng.randrange(K) for c in color_init]
        conflict, total = recompute_conflict(color, neigh, m)
        if total == 0:
            return color, True
        best_color = color[:]
        best_total = total
        no_improve = 0
        max_no_improve = 5000
        it = 0
        while time.time() < time_limit:
            if total == 0:
                return color, True
            # choose vertex: with prob 0.7 the most-conflicted, else random conflicting
            if total > 0:
                if rng.random() < 0.7:
                    v = -1
                    bv = -1
                    for x in range(m):
                        if conflict[x] > bv:
                            bv = conflict[x]
                            v = x
                else:
                    cand = [x for x in range(m) if conflict[x] > 0]
                    v = cand[rng.randrange(len(cand))]
            else:
                v = rng.randrange(m)
            old = color[v]
            # find best new colour (minimise conflict delta)
            best_c = old
            best_delta = 0
            for c in range(K):
                if c == old:
                    continue
                cnt_c = 0
                cnt_old = 0
                for u in neigh[v]:
                    if color[u] == c:
                        cnt_c += 1
                    elif color[u] == old:
                        cnt_old += 1
                delta = cnt_c - cnt_old
                if delta < best_delta or (delta == best_delta and rng.random() < 0.5):
                    best_delta = delta
                    best_c = c
            # occasional random kick to escape local minima
            if rng.random() < 0.12:
                best_c = rng.randrange(K)
                cnt_c = 0
                cnt_old = 0
                for u in neigh[v]:
                    if color[u] == best_c:
                        cnt_c += 1
                    elif color[u] == old:
                        cnt_old += 1
                best_delta = cnt_c - cnt_old
            if best_c == old:
                # Try a 2-opt swap: pick another vertex w and swap colors with v
                if total > 0 and rng.random() < 0.3:
                    # Find a good swap partner
                    best_w = -1
                    best_swap_delta = 0
                    for w in range(m):
                        if w == v or color[w] == old:
                            continue
                        cw = color[w]
                        # Delta if we swap colors of v and w
                        # v goes from old to cw, w goes from cw to old
                        delta_v = 0
                        delta_w = 0
                        for u in neigh[v]:
                            if color[u] == cw and u != w:
                                delta_v += 1
                            elif color[u] == old and u != w:
                                delta_v -= 1
                        for u in neigh[w]:
                            if color[u] == old and u != v:
                                delta_w += 1
                            elif color[u] == cw and u != v:
                                delta_w -= 1
                        # Check if v and w are neighbors
                        if v in neigh[w]:
                            # They are neighbors, so swapping doesn't help their mutual conflict
                            pass
                        total_swap_delta = delta_v + delta_w
                        if total_swap_delta < best_swap_delta:
                            best_swap_delta = total_swap_delta
                            best_w = w
                    if best_w >= 0 and best_swap_delta < 0:
                        # Perform the swap
                        cv, cw = color[v], color[best_w]
                        for u in neigh[v]:
                            if color[u] == cv and u != best_w:
                                conflict[u] -= 1
                                conflict[v] -= 1
                            if color[u] == cw and u != best_w:
                                conflict[u] += 1
                                conflict[v] += 1
                        for u in neigh[best_w]:
                            if color[u] == cw and u != v:
                                conflict[u] -= 1
                                conflict[best_w] -= 1
                            if color[u] == cv and u != v:
                                conflict[u] += 1
                                conflict[best_w] += 1
                        # Handle the v-w edge if they are neighbors
                        if best_w in neigh[v]:
                            conflict[v] -= 1
                            conflict[best_w] -= 1
                        color[v], color[best_w] = cw, cv
                        total += best_swap_delta
                        if total < best_total:
                            best_total = total
                            best_color = color[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                        it += 1
                        continue
                continue
            for u in neigh[v]:
                if color[u] == old:
                    conflict[u] -= 1
                    conflict[v] -= 1
                if color[u] == best_c:
                    conflict[u] += 1
                    conflict[v] += 1
            color[v] = best_c
            total += best_delta
            if total < best_total:
                best_total = total
                best_color = color[:]
                no_improve = 0
            else:
                no_improve += 1
            it += 1
            if no_improve > max_no_improve:
                # snap back to best and continue
                color = best_color[:]
                conflict, total = recompute_conflict(color, neigh, m)
                no_improve = 0
        return best_color, best_total == 0

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)

        deg = [len(neigh[i]) for i in range(m)]
        rng = Random(seed * 7919 + 42)

        # Best valid initial colouring: try several DSATUR variants, keep fewest colours
        best_color, best_K = dsatur_color(m, neigh, deg)
        for _ in range(6):
            col, K = dsatur_color(m, neigh, deg, rng=rng, rand_order=True)
            if K < best_K:
                best_color, best_K = col, K

        LB = greedy_clique(m, neigh)
        LB = max(LB, 1)

        # Descend K from best_K-1 down to LB, using min-conflicts local search
        current_color = best_color
        current_K = best_K
        for target_K in range(current_K - 1, LB - 1, -1):
            if time.time() > time_limit - 0.3:
                break
            # per-K time cap (a few seconds) but bounded by remaining
            cap = min(3.5, max(0.6, time_limit - time.time()))
            k_deadline = time.time() + cap
            found = False
            restart = 0
            while time.time() < k_deadline:
                if restart % 2 == 0:
                    init = current_color[:]
                else:
                    init = [rng.randrange(target_K) for _ in range(m)]
                init = [c if c < target_K else rng.randrange(target_K) for c in init]
                res, ok = local_search(m, neigh, init, target_K, rng,
                                       min(k_deadline, time.time() + 1.0))
                if ok:
                    current_color = res
                    current_K = target_K
                    found = True
                    break
                restart += 1
            if not found:
                break
        
        # Final polish: try to reduce conflicts further at the achieved K
        if time.time() < time_limit - 0.2 and current_K > LB:
            polish_deadline = min(time_limit - 0.1, time.time() + 1.5)
            for _ in range(3):
                if time.time() > polish_deadline:
                    break
                init = current_color[:]
                res, ok = local_search(m, neigh, init, current_K, rng,
                                       min(polish_deadline, time.time() + 0.5))
                if ok:
                    current_color = res
        
        return compress(current_color)

    def fallback():
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            adj_orig = [set() for _ in range(n)]
            for u, v in edges:
                adj_orig[u].add(v)
                adj_orig[v].add(u)
            neigh = [[] for _ in range(m)]
            for i in range(m):
                a, b = edges[i]
                for j in range(i + 1, m):
                    c, dd = edges[j]
                    if a == c or a == dd or b == c or b == dd:
                        neigh[i].append(j); neigh[j].append(i)
                    elif (c in adj_orig[a] or dd in adj_orig[a] or
                          c in adj_orig[b] or dd in adj_orig[b]):
                        neigh[i].append(j); neigh[j].append(i)
            col, _ = dsatur_color(m, neigh, [len(neigh[i]) for i in range(m)])
            out.append(col)
        return out

    try:
        start = time.time()
        master_deadline = start + 118.0
        # allocate time by difficulty weight d^2
        weights = [d * d for (_, d, _) in specs]
        total_w = sum(weights)
        out = []
        cursor = start
        for idx, (n, d, seed) in enumerate(specs):
            frac = weights[idx] / total_w
            budget = (master_deadline - cursor) * frac
            budget = max(budget, 3.0)
            inst_deadline = min(cursor + budget, master_deadline)
            out.append(solve_instance(n, d, seed, inst_deadline))
            cursor = max(cursor, time.time())
        return out
    except Exception:
        return fallback()
