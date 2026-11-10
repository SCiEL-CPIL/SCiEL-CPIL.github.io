import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute all 3-APs and 4-APs
    three_aps = []
    three_aps_by_elem = [[] for _ in range(n + 1)]
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            idx = len(three_aps)
            three_aps.append((a, b, c))
            three_aps_by_elem[a].append(idx)
            three_aps_by_elem[b].append(idx)
            three_aps_by_elem[c].append(idx)

    four_aps = []
    four_aps_by_elem = [[] for _ in range(n + 1)]
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            idx = len(four_aps)
            four_aps.append((a, b, c, dd))
            four_aps_by_elem[a].append(idx)
            four_aps_by_elem[b].append(idx)
            four_aps_by_elem[c].append(idx)
            four_aps_by_elem[dd].append(idx)

    # Precompute masks for 4-APs: for each element y and each 4-AP containing y,
    # the mask of the other 3 elements
    four_ap_other = {}
    for idx, (a, b, c, dd) in enumerate(four_aps):
        for y in (a, b, c, dd):
            others = [z for z in (a, b, c, dd) if z != y]
            m = 0
            for z in others:
                m |= (1 << (z - 1))
            four_ap_other[(y, idx)] = m

    # Precompute masks for 3-APs
    three_ap_other = {}
    for idx, (a, b, c) in enumerate(three_aps):
        for y in (a, b, c):
            others = [z for z in (a, b, c) if z != y]
            m = 0
            for z in others:
                m |= (1 << (z - 1))
            three_ap_other[(y, idx)] = m

    def mask_to_list(mask):
        return [i + 1 for i in range(n) if (mask >> i) & 1]

    def count_3aps(mask):
        count = 0
        for a, b, c in three_aps:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1:
                count += 1
        return count

    def has_4ap(mask):
        for a, b, c, dd in four_aps:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1 and (mask >> (dd - 1)) & 1:
                return True
        return False

    def adds_4ap(y, mask):
        bit = 1 << (y - 1)
        if mask & bit:
            return False
        for idx in four_aps_by_elem[y]:
            om = four_ap_other[(y, idx)]
            if (mask & om) == om:
                return True
        return False

    def gain_add(y, mask):
        bit = 1 << (y - 1)
        if mask & bit:
            return 0
        cnt = 0
        for idx in three_aps_by_elem[y]:
            om = three_ap_other[(y, idx)]
            if (mask & om) == om:
                cnt += 1
        return cnt

    def loss_rem(y, mask):
        bit = 1 << (y - 1)
        if not (mask & bit):
            return 0
        cnt = 0
        for idx in three_aps_by_elem[y]:
            om = three_ap_other[(y, idx)]
            if (mask & om) == om:
                cnt += 1
        return cnt

    def swap_gain(x, y, mask):
        bx = 1 << (x - 1)
        by = 1 << (y - 1)
        if (mask & bx) == 0 or (mask & by):
            return 0
        mask_wo_x = mask & ~bx
        if adds_4ap(y, mask_wo_x):
            return -10**9
        lx = 0
        for idx in three_aps_by_elem[x]:
            om = three_ap_other[(x, idx)]
            if (mask & om) == om:
                lx += 1
        gy = 0
        for idx in three_aps_by_elem[y]:
            om = three_ap_other[(y, idx)]
            if (mask_wo_x & om) == om:
                gy += 1
        return gy - lx

    best_score = -1
    best_mask = 0

    def update_best(mask, score):
        nonlocal best_score, best_mask
        if score > best_score:
            best_score = score
            best_mask = mask

    # ---- WalkSAT-style local search with clause weighting ----
    # We treat each 4-AP as a hard constraint (clause) and each 3-AP as a soft objective.
    # The "clause" for a 4-AP (a,b,c,d) is violated when all four are in the set.
    # We use a WalkSAT-like approach: flip bits to reduce violations and increase 3-AP count.

    def walksat_local_search(init_mask, max_iters=3000):
        mask = init_mask
        score = count_3aps(mask)
        update_best(mask, score)
        
        # Track violated 4-APs (hard constraints)
        violated_4aps = set()
        for idx, (a, b, c, dd) in enumerate(four_aps):
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1 and (mask >> (dd - 1)) & 1:
                violated_4aps.add(idx)
        
        T = 3.0
        T_min = 0.5
        cooling = 0.997
        p_walk = 0.3  # probability of random walk move
        best_local = score
        best_local_mask = mask
        no_imp = 0
        max_no_imp = 100
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            if no_imp >= max_no_imp:
                break
            
            if violated_4aps:
                # Pick a random violated clause and flip one of its elements
                vidx = random.choice(list(violated_4aps))
                a, b, c, dd = four_aps[vidx]
                elems = [a, b, c, dd]
                
                if random.random() < p_walk:
                    # Random walk: pick random element from violated clause
                    x = random.choice(elems)
                else:
                    # Greedy: pick element whose removal causes least 3-AP loss
                    best_x = None
                    best_loss = float('inf')
                    for x in elems:
                        if (mask >> (x - 1)) & 1:
                            loss = loss_rem(x, mask)
                            if loss < best_loss:
                                best_loss = loss
                                best_x = x
                    if best_x is None:
                        best_x = random.choice(elems)
                    x = best_x
                
                # Remove x
                mask &= ~(1 << (x - 1))
                # Update violated 4-APs
                new_violated = set()
                for idx2, (a2, b2, c2, d2) in enumerate(four_aps):
                    if (mask >> (a2 - 1)) & 1 and (mask >> (b2 - 1)) & 1 and (mask >> (c2 - 1)) & 1 and (mask >> (d2 - 1)) & 1:
                        new_violated.add(idx2)
                violated_4aps = new_violated
                score = count_3aps(mask)
            else:
                # No violated clauses: try to improve 3-AP count
                # Try add, remove, or swap
                move_type = random.random()
                if move_type < 0.4:
                    cands = [y for y in range(1, n + 1) if not (mask >> (y - 1)) & 1]
                    if not cands:
                        continue
                    if random.random() < p_walk:
                        y = random.choice(cands)
                    else:
                        best_y = None
                        best_g = -1
                        for y in cands:
                            if adds_4ap(y, mask):
                                continue
                            g = gain_add(y, mask)
                            if g > best_g:
                                best_g = g
                                best_y = y
                        if best_y is None:
                            continue
                        y = best_y
                    if adds_4ap(y, mask):
                        continue
                    g = gain_add(y, mask)
                    new_mask = mask | (1 << (y - 1))
                    new_score = score + g
                elif move_type < 0.6:
                    elems = mask_to_list(mask)
                    if not elems:
                        continue
                    if random.random() < p_walk:
                        x = random.choice(elems)
                    else:
                        best_x = None
                        best_g = -float('inf')
                        for x in elems:
                            g = -loss_rem(x, mask)
                            if g > best_g:
                                best_g = g
                                best_x = x
                        if best_x is None:
                            continue
                        x = best_x
                    g = -loss_rem(x, mask)
                    new_mask = mask & ~(1 << (x - 1))
                    new_score = score + g
                else:
                    elems = mask_to_list(mask)
                    cands = [y for y in range(1, n + 1) if not (mask >> (y - 1)) & 1]
                    if not elems or not cands:
                        continue
                    if random.random() < p_walk:
                        x = random.choice(elems)
                        y = random.choice(cands)
                    else:
                        best_x = None
                        best_y = None
                        best_g = -float('inf')
                        for _ in range(200):
                            x = random.choice(elems)
                            y = random.choice(cands)
                            g = swap_gain(x, y, mask)
                            if g != -10**9 and g > best_g:
                                best_g = g
                                best_x = x
                                best_y = y
                        if best_x is None:
                            continue
                        x = best_x
                        y = best_y
                    g = swap_gain(x, y, mask)
                    if g == -10**9:
                        continue
                    new_mask = (mask & ~(1 << (x - 1))) | (1 << (y - 1))
                    new_score = score + g
                
                # Accept based on simulated annealing
                delta = new_score - score
                if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                    mask = new_mask
                    score = new_score
                    # Update violated 4-APs
                    new_violated = set()
                    for idx2, (a2, b2, c2, d2) in enumerate(four_aps):
                        if (mask >> (a2 - 1)) & 1 and (mask >> (b2 - 1)) & 1 and (mask >> (c2 - 1)) & 1 and (mask >> (d2 - 1)) & 1:
                            new_violated.add(idx2)
                    violated_4aps = new_violated
                    no_imp = 0
                    if score > best_local:
                        best_local = score
                        best_local_mask = mask
                        update_best(mask, score)
                else:
                    no_imp += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_local_mask, best_local

    # ---- Generate diverse initial sets ----
    initial_sets = []

    # 1. Greedy by marginal gain
    def greedy_by_gain(order):
        mask = 0
        for x in order:
            if not adds_4ap(x, mask):
                g = gain_add(x, mask)
                if g >= 0:
                    mask |= (1 << (x - 1))
        return mask

    # 2. Simple greedy add
    def greedy_simple(order):
        mask = 0
        for x in order:
            if not adds_4ap(x, mask):
                mask |= (1 << (x - 1))
        return mask

    # Create various orderings
    orders = []
    orders.append(list(range(1, n + 1)))
    orders.append(list(range(n, 0, -1)))
    orders.append([x for x in range(1, n + 1) if x % 2 == 1] + [x for x in range(1, n + 1) if x % 2 == 0])
    orders.append([x for x in range(1, n + 1) if x % 2 == 0] + [x for x in range(1, n + 1) if x % 2 == 1])
    # Random orders
    for _ in range(10):
        o = list(range(1, n + 1))
        random.shuffle(o)
        orders.append(o)

    # Build initial sets
    for i, order in enumerate(orders):
        if time.time() > deadline - 40:
            break
        mask = greedy_simple(order)
        score = count_3aps(mask)
        update_best(mask, score)
        initial_sets.append((mask, score))

        mask2 = greedy_by_gain(order)
        score2 = count_3aps(mask2)
        update_best(mask2, score2)
        initial_sets.append((mask2, score2))

    # ---- Run WalkSAT local search on each initial set ----
    for init_mask, init_score in initial_sets:
        if time.time() > deadline - 5:
            break
        _, _ = walksat_local_search(init_mask, max_iters=2000)

    # ---- Final hill-climbing polish on best mask ----
    if best_mask:
        mask = best_mask
        score = count_3aps(mask)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            best_g = 0
            best_move = None
            for y in range(1, n + 1):
                if (mask >> (y - 1)) & 1:
                    continue
                if adds_4ap(y, mask):
                    continue
                g = gain_add(y, mask)
                if g > best_g:
                    best_g = g
                    best_move = ('add', y, None)
            for x in mask_to_list(mask):
                g = -loss_rem(x, mask)
                if g > best_g:
                    best_g = g
                    best_move = ('remove', None, x)
            elems = mask_to_list(mask)
            cands = [y for y in range(1, n + 1) if not (mask >> (y - 1)) & 1]
            if elems and cands:
                for _ in range(150):
                    x = random.choice(elems)
                    y = random.choice(cands)
                    g = swap_gain(x, y, mask)
                    if g != -10**9 and g > best_g:
                        best_g = g
                        best_move = ('swap', x, y)
            if best_move and best_g > 0:
                mt = best_move[0]
                if mt == 'add':
                    mask |= (1 << (best_move[1] - 1))
                elif mt == 'remove':
                    mask &= ~(1 << (best_move[2] - 1))
                else:
                    mask &= ~(1 << (best_move[1] - 1))
                    mask |= (1 << (best_move[2] - 1))
                score += best_g
                improved = True
                update_best(mask, score)
        best_mask = mask
        best_score = score

    # Final verification
    if best_mask and has_4ap(best_mask):
        mask = 0
        for x in range(1, n + 1):
            if not adds_4ap(x, mask):
                mask |= (1 << (x - 1))
        best_mask = mask
        best_score = count_3aps(mask)

    if best_mask == 0:
        best_mask = 1
        best_score = 0

    return mask_to_list(best_mask)
