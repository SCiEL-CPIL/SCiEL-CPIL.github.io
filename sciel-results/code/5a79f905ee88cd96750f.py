import time
import heapq
import numpy as np
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85

    denominators = list(range(2, 49))
    n = len(denominators)  # 47
    mid = 23
    A_dens = denominators[:mid]   # 2..24
    B_dens = denominators[mid:]   # 25..48

    def build_sums(dens):
        sums = np.zeros(1, dtype=np.float64)
        masks = [0]
        for d in dens:
            w = 1.0 / d
            sums = np.concatenate((sums, sums + w))
            masks = masks + [m | (1 << len(masks)) for m in masks]
        return sums, masks

    A_sums, A_masks = build_sums(A_dens)
    B_sums, B_masks = build_sums(B_dens)

    order = np.argsort(A_sums, kind='stable')
    A_sorted = A_sums[order]
    A_masks_sorted = [A_masks[i] for i in order]

    K = 10000
    heap = []
    chunk_size = 1 << 18
    nb = len(B_sums)

    for s in range(0, nb, chunk_size):
        if time.time() > deadline - 5:
            break
        Bc = B_sums[s:s+chunk_size]
        Bmc = B_masks[s:s+chunk_size]
        targets = 1.0 - Bc
        a_idxs = np.searchsorted(A_sorted, targets, side='right') - 1
        valid = a_idxs >= 0
        if not valid.any():
            continue
        a_sel = a_idxs[valid]
        b_sel = np.nonzero(valid)[0] + s
        totals = A_sorted[a_sel] + B_sums[b_sel]
        ok = totals < 1.0
        if not ok.any():
            continue
        a_sel = a_sel[ok]
        b_sel = b_sel[ok]
        totals = totals[ok]
        m = len(totals)
        if m <= K:
            top = np.argsort(totals)[::-1]
        else:
            top = np.argpartition(totals, -K)[-K:]
        for j in top:
            heapq.heappush(heap, (float(totals[j]), int(A_masks_sorted[a_sel[j]]), int(Bmc[b_sel[j] - s])))
            if len(heap) > K:
                heapq.heappop(heap)

    best_sum = 0.0
    best_am = 0
    best_bm = 0
    for t, am, bm in sorted(heap, key=lambda x: x[0], reverse=True):
        if t > best_sum:
            best_sum = t
            best_am = am
            best_bm = bm

    def mask_indices(mask, offset):
        res = []
        i = 0
        while mask:
            if mask & 1:
                res.append(offset + i)
            mask >>= 1
            i += 1
        return res

    sel = mask_indices(best_am, 0) + mask_indices(best_bm, mid)

    if not sel:
        sel = [0]

    unit_fracs = [Fraction(1, d) for d in denominators]
    exact_sum = sum(unit_fracs[i] for i in sel)
    if exact_sum >= 1:
        sel = [0]
        exact_sum = unit_fracs[0]

    deficit = 1 - exact_sum

    # Thorough local search with multiple passes
    if time.time() < deadline - 2:
        sel_set = set(sel)
        unselected = [i for i in range(n) if i not in sel_set]
        
        for pass_num in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            
            # Pass 1: Try all swaps
            for rem in list(sel):
                if time.time() > deadline - 2:
                    break
                new_base = exact_sum - unit_fracs[rem]
                for add in unselected:
                    cand = new_base + unit_fracs[add]
                    if cand < 1:
                        d = 1 - cand
                        if d < deficit:
                            deficit = d
                            exact_sum = cand
                            sel.remove(rem)
                            sel.append(add)
                            sel_set.discard(rem)
                            sel_set.add(add)
                            unselected.remove(add)
                            unselected.append(rem)
                            improved = True
                            break
            
            # Pass 2: Try all single additions
            if time.time() > deadline - 2:
                break
            for add in unselected:
                cand = exact_sum + unit_fracs[add]
                if cand < 1:
                    d = 1 - cand
                    if d < deficit:
                        deficit = d
                        exact_sum = cand
                        sel.append(add)
                        sel_set.add(add)
                        unselected.remove(add)
                        improved = True
            
            # Pass 3: Try all single removals (if it helps, i.e., if we're over)
            # But we're always under, so removals make deficit worse. Skip.
            
            if not improved:
                break

    final = sum(unit_fracs[i] for i in sel)
    if sel and final < 1:
        return sel
    return [0]
