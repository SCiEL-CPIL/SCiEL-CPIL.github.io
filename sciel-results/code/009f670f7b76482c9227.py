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

    # Precompute for each element and each 4-AP containing it, the mask of the other 3 elements
    four_ap_other = {}
    for idx, (a, b, c, dd) in enumerate(four_aps):
        for y in (a, b, c, dd):
            others = [z for z in (a, b, c, dd) if z != y]
            m = 0
            for z in others:
                m |= (1 << (z - 1))
            four_ap_other[(y, idx)] = m

    # Precompute 3-AP other masks for each (element, 3-AP index)
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
        """Gain from removing x and adding y (net change in 3-AP count)."""
        bx = 1 << (x - 1)
        by = 1 << (y - 1)
        if (mask & bx) == 0 or (mask & by):
            return 0
        # Check if adding y (without removing x yet) creates 4-AP
        if adds_4ap(y, mask & ~bx):
            return -10**9  # infeasible
        # 3-APs lost by removing x
        lx = 0
        for idx in three_aps_by_elem[x]:
            om = three_ap_other[(x, idx)]
            if (mask & om) == om:
                lx += 1
        # 3-APs gained by adding y (considering mask without x)
        mask_wo_x = mask & ~bx
        gy = 0
        for idx in three_aps_by_elem[y]:
            om = three_ap_other[(y, idx)]
            if (mask_wo_x & om) == om:
                gy += 1
        # Shared 3-APs containing both x and y: these are counted in both lx and gy
        # but only the net effect matters. A 3-AP containing both x and y:
        # - if it's in lx, it means the third element is in mask (with x present)
        # - if it's in gy, it means the third element is in mask without x
        # Since x is being removed, the 3-AP with x,y,z is lost (counted in lx)
        # and the 3-AP with y,z,... wait, a 3-AP has exactly 3 elements.
        # If a 3-AP contains both x and y, it's (x,y,z) or (y,x,z) etc.
        # Removing x: the 3-AP (x,y,z) is lost if z in mask -> counted in lx
        # Adding y: the 3-AP (x,y,z) is NOT gained because x is not in mask_wo_x
        # But a 3-AP (y,z,w) where z,w in mask_wo_x is gained -> counted in gy
        # The shared case: 3-AP containing both x and y, say (x, y, z).
        # In lx: (x,y,z) with z in mask -> yes, counted.
        # In gy: (x,y,z) with x in mask_wo_x? No, x removed. So NOT counted in gy.
        # So no double counting issue for 3-APs containing both x and y.
        # Actually wait: gy counts 3-APs where y is added and the other two are in mask_wo_x.
        # lx counts 3-APs where x is removed and the other two are in mask.
        # A 3-AP (x,y,z): in lx, other two are y,z; y in mask, z in mask -> counted.
        # In gy, other two are x,z; x NOT in mask_wo_x -> not counted.
        # So no overlap. Net gain = gy - lx.
        return gy - lx

    best_score = -1
    best_mask = 0

    def update_best(mask, score):
        nonlocal best_score, best_mask
        if score > best_score:
            best_score = score
            best_mask = mask

    # Strategy: multiple diverse greedy constructions + local search with simulated annealing

    # Generate diverse initial sets
    initial_sets = []

    # 1. Greedy by marginal gain (density-driven)
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
    for _ in range(8):
        o = list(range(1, n + 1))
        random.shuffle(o)
        orders.append(o)

    # Build initial sets
    for i, order in enumerate(orders):
        if time.time() > deadline - 30:
            break
        mask = greedy_simple(order)
        score = count_3aps(mask)
        update_best(mask, score)
        initial_sets.append((mask, score))

        # Also try gain-based
        mask2 = greedy_by_gain(order)
        score2 = count_3aps(mask2)
        update_best(mask2, score2)
        initial_sets.append((mask2, score2))

    # Local search with simulated annealing on each initial set
    for init_mask, init_score in initial_sets:
        if time.time() > deadline - 5:
            break

        mask = init_mask
        score = init_score
        update_best(mask, score)

        T = 5.0
        T_min = 0.1
        cooling = 0.995
        no_imp = 0
        max_no_imp = 50
        best_local = score
        best_local_mask = mask

        for it in range(5000):
            if time.time() > deadline - 3:
                break
            if no_imp >= max_no_imp:
                break

            move_type = random.random()
            if move_type < 0.4:
                # Add
                cands = [y for y in range(1, n + 1) if not (mask >> (y - 1)) & 1]
                if not cands:
                    continue
                y = random.choice(cands)
                if adds_4ap(y, mask):
                    continue
                g = gain_add(y, mask)
                new_mask = mask | (1 << (y - 1))
                new_score = score + g
            elif move_type < 0.6:
                # Remove
                elems = mask_to_list(mask)
                if not elems:
                    continue
                x = random.choice(elems)
                g = -loss_rem(x, mask)
                new_mask = mask & ~(1 << (x - 1))
                new_score = score + g
            else:
                # Swap
                elems = mask_to_list(mask)
                cands = [y for y in range(1, n + 1) if not (mask >> (y - 1)) & 1]
                if not elems or not cands:
                    continue
                x = random.choice(elems)
                y = random.choice(cands)
                g = swap_gain(x, y, mask)
                if g == -10**9:
                    continue
                new_mask = (mask & ~(1 << (x - 1))) | (1 << (y - 1))
                new_score = score + g

            delta = new_score - score
            if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                mask = new_mask
                score = new_score
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

        # After SA, do a quick hill-climbing polish
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            best_g = 0
            best_move = None
            # Try all adds
            for y in range(1, n + 1):
                if (mask >> (y - 1)) & 1:
                    continue
                if adds_4ap(y, mask):
                    continue
                g = gain_add(y, mask)
                if g > best_g:
                    best_g = g
                    best_move = ('add', y, None)
            # Try all removes
            for x in mask_to_list(mask):
                g = -loss_rem(x, mask)
                if g > best_g:
                    best_g = g
                    best_move = ('remove', None, x)
            # Try swaps (sampled)
            elems = mask_to_list(mask)
            cands = [y for y in range(1, n + 1) if not (mask >> (y - 1)) & 1]
            if elems and cands:
                for _ in range(100):
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

    # Final verification
    if best_mask and has_4ap(best_mask):
        # Rebuild safely
        mask = 0
        for x in range(1, n + 1):
            if not adds_4ap(x, mask):
                mask |= (1 << (x - 1))
        best_mask = mask
        best_score = count_3aps(mask)

    # Ensure nonempty
    if best_mask == 0:
        best_mask = 1  # {1}
        best_score = 0

    return mask_to_list(best_mask)
