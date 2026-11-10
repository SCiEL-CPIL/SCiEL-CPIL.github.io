# EVOLVE-BLOCK-START
import math
import time
import random
from fractions import Fraction

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 255  # numbers 2..256
    nums = list(range(2, 257))

    # Precompute conflict bitmasks
    conflict_mask = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if math.lcm(nums[i], nums[j]) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Precompute reciprocal values
    recip = [1.0 / nums[i] for i in range(N)]

    # Pre-sort vertices by reciprocal descending
    sorted_by_recip = sorted(range(N), key=lambda x: -recip[x])

    # Greedy initialization
    def greedy(order):
        selected_mask = 0
        for v in order:
            if (selected_mask & conflict_mask[v]) == 0:
                selected_mask |= (1 << v)
        return selected_mask

    # Local Search with 1-for-1 and 2-for-1 exchanges
    def local_search(selected_mask, time_limit=5.0):
        sel_mask = selected_mask
        best_val = 0.0
        for i in range(N):
            if (sel_mask >> i) & 1:
                best_val += recip[i]

        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < time_limit:
            improved = False
            sel_list = [i for i in range(N) if (sel_mask >> i) & 1]

            # 1-for-1 Exchange: Remove v, add u
            for v in sel_list:
                for u in range(N):
                    if (sel_mask >> u) & 1:
                        continue
                    if (conflict_mask[u] & (sel_mask ^ (1 << v))) == 0:
                        delta = recip[u] - recip[v]
                        if delta > 1e-12:
                            sel_mask = (sel_mask ^ (1 << v)) | (1 << u)
                            best_val += delta
                            improved = True
                            break
                if improved:
                    break

            # 2-for-1 Exchange: Remove v1, v2, add u
            if not improved:
                sel_list = [i for i in range(N) if (sel_mask >> i) & 1]
                for i_idx in range(len(sel_list)):
                    if time.time() - t_start > time_limit:
                        break
                    v1 = sel_list[i_idx]
                    for j_idx in range(i_idx + 1, len(sel_list)):
                        v2 = sel_list[j_idx]
                        base_mask = sel_mask ^ (1 << v1) ^ (1 << v2)
                        found = False
                        for u in sorted_by_recip:
                            if (sel_mask >> u) & 1:
                                continue
                            if (conflict_mask[u] & base_mask) != 0:
                                continue
                            if recip[u] > recip[v1] + recip[v2] + 1e-12:
                                sel_mask = base_mask | (1 << u)
                                best_val += recip[u] - recip[v1] - recip[v2]
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if improved:
                        break

        return sel_mask, best_val

    # Exhaustive Polish: ensure no single add/swap improvement is missed
    def exhaustive_polish(selected_mask):
        sel_mask = selected_mask
        best_val = 0.0
        for i in range(N):
            if (sel_mask >> i) & 1:
                best_val += recip[i]

        # Try adding any valid u (0-for-1 move)
        for u in range(N):
            if (sel_mask >> u) & 1:
                continue
            if (conflict_mask[u] & sel_mask) == 0:
                sel_mask |= (1 << u)
                best_val += recip[u]

        # Try 1-for-1 swaps
        sel_list = [i for i in range(N) if (sel_mask >> i) & 1]
        changed = True
        while changed:
            changed = False
            sel_list = [i for i in range(N) if (sel_mask >> i) & 1]
            for v in sel_list:
                best_u = -1
                best_delta = 0.0
                for u in range(N):
                    if (sel_mask >> u) & 1:
                        continue
                    if (conflict_mask[u] & (sel_mask ^ (1 << v))) == 0:
                        delta = recip[u] - recip[v]
                        if delta > best_delta + 1e-12:
                            best_delta = delta
                            best_u = u
                if best_u >= 0:
                    sel_mask = (sel_mask ^ (1 << v)) | (1 << best_u)
                    best_val += best_delta
                    changed = True

        # Try 2-for-1
        changed = True
        while changed:
            changed = False
            sel_list = [i for i in range(N) if (sel_mask >> i) & 1]
            for i_idx in range(len(sel_list)):
                v1 = sel_list[i_idx]
                for j_idx in range(i_idx + 1, len(sel_list)):
                    v2 = sel_list[j_idx]
                    base_mask = sel_mask ^ (1 << v1) ^ (1 << v2)
                    best_u = -1
                    best_delta = 0.0
                    for u in range(N):
                        if (sel_mask >> u) & 1:
                            continue
                        if (conflict_mask[u] & base_mask) != 0:
                            continue
                        delta = recip[u] - recip[v1] - recip[v2]
                        if delta > best_delta + 1e-12:
                            best_delta = delta
                            best_u = u
                    if best_u >= 0:
                        sel_mask = base_mask | (1 << best_u)
                        best_val += best_delta
                        changed = True

        return sel_mask, best_val

    # Multi-start local search
    best_result = 0
    best_score = 0.0

    # Start 1: natural order
    sel1 = greedy(list(range(N)))
    sel1, val1 = local_search(sel1, time_limit=8.0)
    if val1 > best_score:
        best_score = val1
        best_result = sel1

    # Start 2: reverse order
    sel2 = greedy(list(range(N - 1, -1, -1)))
    sel2, val2 = local_search(sel2, time_limit=8.0)
    if val2 > best_score:
        best_score = val2
        best_result = sel2

    # Start 3: reciprocal-sorted order
    sel3 = greedy(sorted_by_recip)
    sel3, val3 = local_search(sel3, time_limit=8.0)
    if val3 > best_score:
        best_score = val3
        best_result = sel3

    # Start 4-8: random orderings
    rng = random.Random(42)
    for _ in range(5):
        if time.time() - t0 > DEADLINE - 20:
            break
        order = list(range(N))
        rng.shuffle(order)
        sel = greedy(order)
        sel, val = local_search(sel, time_limit=5.0)
        if val > best_score:
            best_score = val
            best_result = sel

    # Final exhaustive polish on best result
    if time.time() - t0 < DEADLINE - 3:
        best_result, best_score = exhaustive_polish(best_result)

    # Convert to list of actual numbers
    result = sorted(nums[i] for i in range(N) if (best_result >> i) & 1)
    return result if result else [2]
# EVOLVE-BLOCK-END
