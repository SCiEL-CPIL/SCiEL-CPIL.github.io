import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 82.0

    # ---------- Precomputation ----------
    # All 3-APs as (a,b,c)
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            aps_3.append((a, a + d, a + 2 * d))
    m3 = len(aps_3)

    # For each element x, list of (ap_idx, pos) where x appears in ap
    ap_index_by_elem = [[] for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(aps_3):
        ap_index_by_elem[a].append((idx, 0))
        ap_index_by_elem[b].append((idx, 1))
        ap_index_by_elem[c].append((idx, 2))

    # All 4-APs
    four_aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for s in range(1, n - 3 * d + 1):
            four_aps.append((s, s + d, s + 2 * d, s + 3 * d))
    m4 = len(four_aps)

    # For each x, list of (ap_idx, pos) for 4-APs containing x
    four_aps_by_elem = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(four_aps):
        for pos, elem in enumerate(ap):
            four_aps_by_elem[elem].append((idx, pos))

    # Precompute mask4[x] = list of bitmasks of other 3 elements in each 4-AP containing x
    mask4 = [None] * (n + 1)
    for x in range(1, n + 1):
        lst = four_aps_by_elem[x]
        if lst:
            mask4[x] = []
            for ap_idx, pos in lst:
                ap = four_aps[ap_idx]
                m = 0
                for j in range(4):
                    if j != pos:
                        m |= (1 << (ap[j] - 1))
                mask4[x].append(m)
        else:
            mask4[x] = []

    # gain_pairs[x] = list of (bit1, bit2) where both must be in S for adding x to complete a 3-AP
    gain_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        pairs = []
        for idx, pos in ap_index_by_elem[x]:
            a, b, c = aps_3[idx]
            if pos == 0:
                pairs.append((1 << (b-1), 1 << (c-1)))
            elif pos == 1:
                pairs.append((1 << (a-1), 1 << (c-1)))
            else:
                pairs.append((1 << (a-1), 1 << (b-1)))
        gain_pairs[x] = pairs

    # ---------- Core operations on bitmask S (bit x-1 set if x in S) ----------
    def can_add(x, S):
        masks = mask4[x]
        for m in masks:
            if (S & m) == m:
                return False
        return True

    def gain_add(x, S):
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S & bit1) and (S & bit2):
                g += 1
        return g

    def loss_remove(x, S):
        S_wo_x = S & ~(1 << (x-1))
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S_wo_x & bit1) and (S_wo_x & bit2):
                g += 1
        return g

    def score(S):
        cnt = 0
        for a, b, c in aps_3:
            if (S & (1 << (a-1))) and (S & (1 << (b-1))) and (S & (1 << (c-1))):
                cnt += 1
        return cnt

    # ---------- Local search with add, remove-then-add, and swap moves ----------
    def local_search(S, current_score, max_iters=50, time_budget=None):
        S = int(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False
            iters += 1

            # Phase 1: Greedily add elements with positive gain
            for x in range(1, n + 1):
                if (S >> (x-1)) & 1:
                    continue
                if not can_add(x, S):
                    continue
                g = gain_add(x, S)
                if g > 0:
                    S |= (1 << (x-1))
                    current_score += g
                    improved = True
                    if time_budget is not None and time.time() > time_budget:
                        break

            if time_budget is not None and time.time() > time_budget:
                break

            # Phase 2: Remove-then-greedy-add (try each element in S)
            for y in range(1, n + 1):
                if time_budget is not None and time.time() > time_budget - 0.5:
                    break
                if not ((S >> (y-1)) & 1):
                    continue
                S_trial = S & ~(1 << (y-1))
                score_trial = current_score - loss_remove(y, S)
                # Greedily add back
                for x in range(1, n + 1):
                    if (S_trial >> (x-1)) & 1:
                        continue
                    if not can_add(x, S_trial):
                        continue
                    g = gain_add(x, S_trial)
                    if g > 0:
                        S_trial |= (1 << (x-1))
                        score_trial += g
                if score_trial > current_score:
                    S = S_trial
                    current_score = score_trial
                    improved = True
                    if time_budget is not None and time.time() > time_budget:
                        break

            if time_budget is not None and time.time() > time_budget:
                break

            # Phase 3: 1-swap moves
            for y in range(1, n + 1):
                if time_budget is not None and time.time() > time_budget - 0.5:
                    break
                if not ((S >> (y-1)) & 1):
                    continue
                S_wo_y = S & ~(1 << (y-1))
                score_wo_y = current_score - loss_remove(y, S)
                best_x = -1
                best_gain = 0
                for x in range(1, n + 1):
                    if (S_wo_y >> (x-1)) & 1:
                        continue
                    if not can_add(x, S_wo_y):
                        continue
                    g = gain_add(x, S_wo_y)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                if best_x > 0 and score_wo_y + best_gain > current_score:
                    S = S_wo_y | (1 << (best_x - 1))
                    current_score = score_wo_y + best_gain
                    improved = True
                    if time_budget is not None and time.time() > time_budget:
                        break

        return S, current_score

    # ---------- Simulated Annealing local search ----------
    def sa_local_search(S, current_score, max_iters=200, time_budget=None):
        S = int(S)
        T = 5.0
        T_min = 0.1
        cooling = 0.97
        iters = 0
        while iters < max_iters:
            if time_budget is not None and time.time() > time_budget:
                break
            iters += 1

            # Pick a random move
            move_type = random.random()
            if move_type < 0.4:
                # Add
                candidates = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1) and can_add(x, S)]
                if not candidates:
                    continue
                x = random.choice(candidates)
                S_new = S | (1 << (x-1))
                delta = gain_add(x, S)
            elif move_type < 0.7:
                # Remove
                in_set = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                if not in_set:
                    continue
                y = random.choice(in_set)
                S_new = S & ~(1 << (y-1))
                delta = -loss_remove(y, S)
            else:
                # Swap
                in_set = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                not_in_set = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1)]
                if not in_set or not not_in_set:
                    continue
                y = random.choice(in_set)
                S_wo_y = S & ~(1 << (y-1))
                valid_adds = [x for x in not_in_set if can_add(x, S_wo_y)]
                if not valid_adds:
                    continue
                x = random.choice(valid_adds)
                S_new = S_wo_y | (1 << (x-1))
                delta = gain_add(x, S_wo_y) - loss_remove(y, S)

            # Accept
            if delta > 0 or random.random() < np.exp(delta / T):
                S = S_new
                current_score += delta

            T = max(T_min, T * cooling)

        # Final greedy polish
        improved = True
        while improved:
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False
            for x in range(1, n + 1):
                if (S >> (x-1)) & 1:
                    continue
                if not can_add(x, S):
                    continue
                g = gain_add(x, S)
                if g > 0:
                    S |= (1 << (x-1))
                    current_score += g
                    improved = True
        return S, current_score

    # ---------- Construction strategies ----------
    def greedy_random(order, use_score=True, min_size=0):
        S = 0
        for x in order:
            if not can_add(x, S):
                continue
            if use_score:
                g = gain_add(x, S)
                if g > 0 or bin(S).count('1') < min_size:
                    S |= (1 << (x-1))
            else:
                S |= (1 << (x-1))
        return S

    def prune_from_full():
        S = (1 << n) - 1
        for ap in four_aps:
            if all((S >> (elem-1)) & 1 for elem in ap):
                min_loss = float('inf')
                min_elem = ap[0]
                for elem in ap:
                    l = loss_remove(elem, S)
                    if l < min_loss:
                        min_loss = l
                        min_elem = elem
                S &= ~(1 << (min_elem - 1))
        return S

    # ---------- Main search ----------
    best_S = 0
    best_score = 0

    random.seed(42)

    # Strategy 1: Natural order greedy
    if time.time() < deadline - 25:
        S1 = greedy_random(range(1, n + 1), use_score=False)
        sc1 = score(S1)
        S1, sc1 = local_search(S1, sc1, max_iters=30, time_budget=deadline - 20)
        if sc1 > best_score:
            best_score = sc1
            best_S = S1

    # Strategy 2: Many random restarts with local search
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True, min_size=15)
        sc_r = score(S_r)
        S_r, sc_r = local_search(S_r, sc_r, max_iters=20, time_budget=deadline - 15)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Strategy 3: Prune from full
    if time.time() < deadline - 20:
        S_p = prune_from_full()
        sc_p = score(S_p)
        S_p, sc_p = local_search(S_p, sc_p, max_iters=30, time_budget=deadline - 10)
        if sc_p > best_score:
            best_score = sc_p
            best_S = S_p

    # Strategy 4: Simulated annealing from best
    if time.time() < deadline - 15 and best_S:
        S_sa = int(best_S)
        sc_sa = best_score
        S_sa, sc_sa = sa_local_search(S_sa, sc_sa, max_iters=300, time_budget=deadline - 8)
        if sc_sa > best_score:
            best_score = sc_sa
            best_S = S_sa

    # Strategy 5: More random restarts if time permits
    for r in range(30):
        if time.time() > deadline - 8:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True, min_size=10)
        sc_r = score(S_r)
        S_r, sc_r = local_search(S_r, sc_r, max_iters=15, time_budget=deadline - 5)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Strategy 6: SA from another random start
    if time.time() < deadline - 10:
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True, min_size=10)
        sc_r = score(S_r)
        S_r, sc_r = sa_local_search(S_r, sc_r, max_iters=200, time_budget=deadline - 5)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Final local search on best
    if time.time() < deadline - 3 and best_S:
        best_S, best_score = local_search(best_S, best_score, max_iters=40, time_budget=deadline - 1)

    # Convert bitmask to sorted list
    result = [x for x in range(1, n + 1) if (best_S >> (x-1)) & 1]
    if not result:
        result = [1]
    return result
