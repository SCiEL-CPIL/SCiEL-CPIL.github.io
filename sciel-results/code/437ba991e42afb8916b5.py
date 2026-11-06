import time
import random
import math
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 118.0

    # Precompute all 4-term APs
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(aps)
    ap_indices = np.array(aps, dtype=np.int32)

    def count_mono_np(x):
        vals = x[ap_indices]
        prod1 = np.prod(vals, axis=1)
        prod0 = np.prod(1 - vals, axis=1)
        return int(np.sum(prod1 + prod0))

    # Link info for delta_flip
    link_info = [[] for _ in range(N)]
    for ap in aps:
        for pos in range(4):
            idx = ap[pos]
            others = (ap[(pos+1)%4], ap[(pos+2)%4], ap[(pos+3)%4])
            link_info[idx].append(others)

    # For pair flip evaluation, store AP index lists
    bit_aps = [[] for _ in range(N)]
    for ai, ap in enumerate(aps):
        for idx in ap:
            bit_aps[idx].append(ai)

    # Common APs for pairs
    common_aps = {}
    for ai, ap in enumerate(aps):
        for k in range(4):
            for l in range(k+1, 4):
                i, j = ap[k], ap[l]
                if i > j:
                    i, j = j, i
                key = (i, j)
                if key not in common_aps:
                    common_aps[key] = []
                common_aps[key].append(ai)

    def delta_flip(colors, k):
        old_c = colors[k]
        new_c = 1 - old_c
        delta = 0
        for others in link_info[k]:
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c1 == c2 == c3 == new_c:
                delta += 1
            if c1 == c2 == c3 == old_c:
                delta -= 1
        return delta

    def eval_pair_flip(colors, i, j):
        if i > j:
            i, j = j, i
        old_i = colors[i]
        old_j = colors[j]
        new_i = 1 - old_i
        new_j = 1 - old_j
        delta = 0

        common_set = set(common_aps.get((i, j), []))

        for ap_idx in bit_aps[i]:
            if ap_idx in common_set:
                continue
            ap = aps[ap_idx]
            others = [x for x in ap if x != i]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c1 == c2 == c3 == old_i:
                delta -= 1
            elif c1 == c2 == c3 == new_i:
                delta += 1

        for ap_idx in bit_aps[j]:
            if ap_idx in common_set:
                continue
            ap = aps[ap_idx]
            others = [x for x in ap if x != j]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c1 == c2 == c3 == old_j:
                delta -= 1
            elif c1 == c2 == c3 == new_j:
                delta += 1

        for ap_idx in common_set:
            ap = aps[ap_idx]
            others = [x for x in ap if x != i and x != j]
            c_k, c_l = colors[others[0]], colors[others[1]]
            if old_i == old_j == c_k == c_l:
                delta -= 1
            elif new_i == new_j == c_k == c_l:
                delta += 1

        return delta

    def local_search(colors, cost, time_limit, start_time):
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            for i in range(N):
                d = delta_flip(colors, i)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    cost += d
                    improved = True
            if improved:
                continue
            # Pair flips
            for _ in range(3000):
                if time.time() > start_time + time_limit:
                    break
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    continue
                d = eval_pair_flip(colors, i, j)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    colors[j] = 1 - colors[j]
                    cost += d
                    improved = True
                    break
        return colors, cost

    def thue_morse(n):
        return bin(n).count('1') % 2

    # Generate initial solutions
    inits = []
    inits.append([thue_morse(i) for i in range(N)])
    random.seed(123)
    for _ in range(8):
        inits.append([random.randint(0, 1) for _ in range(N)])
    for bs in [4, 8, 16, 32, 64]:
        inits.append([(i // bs) % 2 for i in range(N)])
    # Alternating blocks with offset
    for offset in range(4):
        inits.append([((i + offset) // 4) % 2 for i in range(N)])

    best_init = min(inits, key=lambda c: count_mono_np(np.array(c, dtype=np.int32)))
    global_best = best_init[:]
    global_best_cost = count_mono_np(np.array(global_best, dtype=np.int32))

    # Phase 1: ILS on best init
    t1 = time.time()
    ils_budget = (deadline - t1) * 0.35
    current = best_init[:]
    current_cost = global_best_cost
    while time.time() < t1 + ils_budget:
        current, current_cost = local_search(current, current_cost, ils_budget, t1)
        if current_cost < global_best_cost:
            global_best_cost = current_cost
            global_best = current[:]
        # Perturb
        n_flips = random.randint(4, 12)
        for idx in random.sample(range(N), n_flips):
            current[idx] = 1 - current[idx]
        current_cost = count_mono_np(np.array(current, dtype=np.int32))

    # Phase 2: SA with variable temperature and multi-flip moves
    t2 = time.time()
    sa_budget = (deadline - t2) * 0.55
    sa_deadline = t2 + sa_budget

    curr = np.array(global_best, dtype=np.int32)
    curr_cost = global_best_cost
    best_sa = global_best[:]
    best_sa_cost = global_best_cost

    T = 1.5
    T_min = 0.005
    cooling = 0.99997

    while time.time() < sa_deadline:
        ms = random.choice([1, 1, 1, 1, 2, 2, 3])
        if ms == 1:
            k = random.randint(0, N - 1)
            d = delta_flip(curr.tolist(), k)
            if d < 0 or (T > T_min and random.random() < math.exp(-d / T)):
                curr[k] = 1 - curr[k]
                curr_cost += d
        elif ms == 2:
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            cl = curr.tolist()
            d = eval_pair_flip(cl, i, j)
            if d < 0 or (T > T_min and random.random() < math.exp(-d / T)):
                curr[i] = 1 - curr[i]
                curr[j] = 1 - curr[j]
                curr_cost += d
        else:
            idxs = random.sample(range(N), 3)
            cl = curr.tolist()
            for idx in idxs:
                cl[idx] = 1 - cl[idx]
            nc = count_mono_np(np.array(cl, dtype=np.int32))
            d = nc - curr_cost
            if d < 0 or (T > T_min and random.random() < math.exp(-d / T)):
                curr = np.array(cl, dtype=np.int32)
                curr_cost = nc
            else:
                curr = np.array(cl, dtype=np.int32)  # already reverted in cl

        if curr_cost < best_sa_cost:
            best_sa_cost = curr_cost
            best_sa = curr.tolist()

        T *= cooling
        if T < T_min:
            T = T_min

        if T <= T_min and random.random() < 0.001:
            # Restart with perturbation
            for idx in random.sample(range(N), random.randint(8, 16)):
                curr[idx] = 1 - curr[idx]
            curr_cost = count_mono_np(curr)
            T = 1.0

    if best_sa_cost < global_best_cost:
        global_best_cost = best_sa_cost
        global_best = best_sa[:]

    # Phase 3: Final polish
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for pos in range(N):
                d = delta_flip(global_best, pos)
                if d < 0:
                    global_best[pos] = 1 - global_best[pos]
                    global_best_cost += d
                    improved = True
            if not improved:
                for _ in range(2000):
                    if time.time() > deadline - 2.0:
                        break
                    i = random.randint(0, N - 1)
                    j = random.randint(0, N - 1)
                    if i == j:
                        continue
                    d = eval_pair_flip(global_best, i, j)
                    if d < 0:
                        global_best[i] = 1 - global_best[i]
                        global_best[j] = 1 - global_best[j]
                        global_best_cost += d
                        improved = True
                        break

    return global_best
