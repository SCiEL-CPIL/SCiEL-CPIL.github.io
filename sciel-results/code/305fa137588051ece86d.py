"""Direct constructive approach for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0   # safety margin vs 90s kill

    n = 14
    m = 35
    max_val = 10000
    all_edges = list(combinations(range(n), 2))

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        ns = len(sums)
        np_ = len(prods)
        return ns, np_, max(ns, np_)

    def soft_score(values, edges, beta=3.0):
        """Smooth surrogate for max(distinct_sums, distinct_prods) via log-sum-exp."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        ns = len(sums)
        np_ = len(prods)
        a, b = float(ns), float(np_)
        mx = max(a, b)
        if mx == 0:
            return 0.0
        return mx + math.log(math.exp(a - mx) + math.exp(b - mx)) / beta

    # ---- Edge selection: greedy with a distinct-count pruning cap (from inspiration) ----
    def find_edges_pruned(values, rng, time_limit, cap):
        sums_seen = set()
        prods_seen = set()
        selected = []
        remaining = list(all_edges)
        rng.shuffle(remaining)

        for _ in range(m):
            if time.time() > time_limit:
                break
            best_edge = None
            best_score = float('inf')
            for edge in remaining:
                u, v = edge
                s = values[u] + values[v]
                p = values[u] * values[v]
                sum_new = s not in sums_seen
                prod_new = p not in prods_seen
                if sum_new and len(sums_seen) + 1 > cap:
                    continue
                if prod_new and len(prods_seen) + 1 > cap:
                    continue
                score = 0.0
                if sum_new:
                    score += 2.0
                if prod_new:
                    score += 2.0
                score += (s + p) / 1000000.0
                if score < best_score:
                    best_score = score
                    best_edge = edge
            if best_edge is None:
                # Relax: pick the least-intrusive remaining edge
                for edge in remaining:
                    u, v = edge
                    s = values[u] + values[v]
                    p = values[u] * values[v]
                    score = 0.0
                    if s not in sums_seen:
                        score += 2.0
                    if p not in prods_seen:
                        score += 2.0
                    score += (s + p) / 1000000.0
                    if score < best_score:
                        best_score = score
                        best_edge = edge
            if best_edge is None:
                break
            selected.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])

        if len(selected) < m:
            used = set(tuple(e) for e in selected)
            for e in all_edges:
                if len(selected) >= m:
                    break
                if tuple(e) not in used:
                    selected.append(e)
                    used.add(tuple(e))
        return selected

    def fill_edges(values, edges):
        used = set(tuple(e) for e in edges)
        for e in all_edges:
            if len(edges) >= m:
                break
            if tuple(e) not in used:
                edges.append(e)
                used.add(tuple(e))
        return edges

    # ---- Local search on edges ----
    def local_search_edges(values, edges, rng, time_limit, use_soft=False):
        edges_set = set(tuple(e) for e in edges)
        _, _, current_score = evaluate(values, edges)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(300):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges) - 1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                old_edge = edges[i]
                edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                ns, np_, new_score = evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
        return edges, current_score

    # ---- Local search on values (multiplicative-structure candidate set) ----
    def local_search_values(values, edges, rng, time_limit):
        _, _, current_score = evaluate(values, edges)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(50):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n - 1)
                old_val = values[idx]
                candidates = set()
                for delta in range(-12, 13):
                    v = old_val + delta
                    if 1 <= v <= max_val:
                        candidates.add(v)
                for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 48, 60]:
                    for mult in range(1, min(50, max_val // base + 1)):
                        candidates.add(base * mult)
                for base in [2, 3, 4, 5]:
                    v = 1
                    while v <= max_val:
                        candidates.add(v)
                        v *= base
                for _ in range(15):
                    candidates.add(rng.randint(1, max_val))

                best_new_val = None
                best_new_score = current_score
                for new_val in sorted(candidates):
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in values:
                        continue
                    values[idx] = new_val
                    ns, np_, new_score = evaluate(values, edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    values[idx] = old_val
                if best_new_val is not None:
                    values[idx] = best_new_val
                    current_score = best_new_score
                    improved = True
        return values, current_score

    # ---- Smooth-surrogate value polish (soft max via log-sum-exp) ----
    def smooth_value_polish(values, edges, rng, time_limit, beta=3.0):
        cur = soft_score(values, edges, beta)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(40):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n - 1)
                old_val = values[idx]
                candidates = set()
                for delta in range(-8, 9):
                    v = old_val + delta
                    if 1 <= v <= max_val:
                        candidates.add(v)
                for base in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 48, 60]:
                    for mult in range(1, min(40, max_val // base + 1)):
                        candidates.add(base * mult)
                best_new_val = None
                best_new_soft = cur
                for new_val in sorted(candidates):
                    if new_val < 1 or new_val > max_val or new_val in values:
                        continue
                    values[idx] = new_val
                    s = soft_score(values, edges, beta)
                    if s < best_new_soft - 1e-9:
                        best_new_soft = s
                        best_new_val = new_val
                    values[idx] = old_val
                if best_new_val is not None:
                    values[idx] = best_new_val
                    cur = best_new_soft
                    improved = True
        return values

    # ---- Final high-precision polish: exhaustive best-move over all value/edge pairs ----
    def final_polish(values, edges, rng, time_limit):
        edges_set = set(tuple(e) for e in edges)
        _, _, current_score = evaluate(values, edges)
        # value polish
        for _ in range(3):
            if (time.time() - start_time) > time_limit:
                break
            improved = False
            for idx in range(n):
                old_val = values[idx]
                cands = set()
                for delta in range(-6, 7):
                    v = old_val + delta
                    if 1 <= v <= max_val:
                        cands.add(v)
                for base in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 48, 60]:
                    for mult in range(1, min(30, max_val // base + 1)):
                        cands.add(base * mult)
                best_v = None
                best_s = current_score
                for nv in sorted(cands):
                    if nv < 1 or nv > max_val or nv in values:
                        continue
                    values[idx] = nv
                    ns, np_, s = evaluate(values, edges)
                    if s < best_s:
                        best_s = s
                        best_v = nv
                    values[idx] = old_val
                if best_v is not None:
                    values[idx] = best_v
                    current_score = best_s
                    improved = True
            if not improved:
                break
        # edge polish
        for _ in range(3):
            if (time.time() - start_time) > time_limit:
                break
            improved = False
            for i in range(len(edges)):
                old_edge = edges[i]
                best_e = None
                best_s = current_score
                for ne in all_edges:
                    if tuple(ne) in edges_set:
                        continue
                    edges[i] = ne
                    ns, np_, s = evaluate(values, edges)
                    if s < best_s:
                        best_s = s
                        best_e = ne
                    edges[i] = old_edge
                if best_e is not None:
                    edges_set.discard(tuple(old_edge))
                    edges[i] = best_e
                    edges_set.add(tuple(best_e))
                    current_score = best_s
                    improved = True
            if not improved:
                break
        return values, edges, current_score

    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(12345)

    # Value strategies: many share common factors -> products/sums collide.
    value_strategies = [
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168],
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 30],
        [2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112],
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    ]

    for init_values in value_strategies:
        if (time.time() - start_time) > deadline - 8.0:
            break
        init_values = list(init_values)

        # Edge selection across several pruning caps (lower cap -> fewer distinct)
        best_edges_for_values = None
        best_edges_score = float('inf')
        for cap in [12, 14, 16, 18, 20, 24, 30]:
            if (time.time() - start_time) > deadline - 8.0:
                break
            e = find_edges_pruned(init_values, rng, deadline - 8.0, cap)
            fill_edges(init_values, e)
            _, _, s = evaluate(init_values, e)
            if s < best_edges_score:
                best_edges_score = s
                best_edges_for_values = [list(x) for x in e]

        if best_edges_for_values is None:
            best_edges_for_values = [list(x) for x in all_edges[:m]]

        if best_edges_score < best_overall_score:
            best_overall_score = best_edges_score
            best_result = {'values': list(init_values), 'edges': best_edges_for_values}

        # Edge local search
        e, s_e = local_search_edges(init_values, best_edges_for_values, rng, deadline - 8.0)
        if s_e < best_overall_score:
            best_overall_score = s_e
            best_result = {'values': list(init_values), 'edges': [list(x) for x in e]}

        # Value local search (on the best edges)
        vals = list(init_values)
        vals, s_v = local_search_values(vals, best_edges_for_values, rng, deadline - 6.0)
        if s_v < best_overall_score:
            best_overall_score = s_v
            best_result = {'values': list(vals), 'edges': [list(x) for x in best_edges_for_values]}

        # Smooth-surrogate polish, then re-optimize edges
        vals2 = smooth_value_polish(vals, best_edges_for_values, rng, deadline - 4.0)
        e2, s_e2 = local_search_edges(vals2, best_edges_for_values, rng, deadline - 3.0)
        if s_e2 < best_overall_score:
            best_overall_score = s_e2
            best_result = {'values': list(vals2), 'edges': [list(x) for x in e2]}

    # Phase 2: refine the global best with alternating value/edge + smooth polish
    if best_result is not None:
        values = list(best_result['values'])
        edges = [list(x) for x in best_result['edges']]
        for _ in range(6):
            if (time.time() - start_time) > deadline - 3.0:
                break
            values, s1 = local_search_values(values, edges, rng, deadline - 3.0)
            values = smooth_value_polish(values, edges, rng, deadline - 3.0)
            edges, s2 = local_search_edges(values, edges, rng, deadline - 3.0)
            if s2 < best_overall_score:
                best_overall_score = s2
                best_result = {'values': list(values), 'edges': [list(x) for x in edges]}

    # Phase 3: final high-precision polish pass
    if best_result is not None:
        values = list(best_result['values'])
        edges = [list(x) for x in best_result['edges']]
        values, edges, s3 = final_polish(values, edges, rng, deadline - 1.0)
        if s3 < best_overall_score:
            best_overall_score = s3
            best_result = {'values': list(values), 'edges': [list(x) for x in edges]}

    if best_result is None:
        best_result = {'values': list(range(1, n + 1)), 'edges': [list(x) for x in all_edges[:m]]}

    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
