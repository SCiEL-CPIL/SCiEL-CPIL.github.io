import time
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)  # 47
    
    # Phase 1: Meet-in-the-middle with float64 to get near-optimal candidate
    nA = 23
    nB = n - nA
    
    valsA = [1.0 / d for d in denominators[:nA]]
    valsB = [1.0 / d for d in denominators[nA:]]
    
    # Generate all subset sums < 1.0 for each half
    def gen_sums(vals):
        sums = [0.0]
        masks = [0]
        for i, v in enumerate(vals):
            bit = 1 << i
            cur_len = len(sums)
            for k in range(cur_len):
                ns = sums[k] + v
                if ns < 1.0:
                    sums.append(ns)
                    masks.append(masks[k] | bit)
            if time.time() - start_time > DEADLINE - 20:
                return None, None
        return np.array(sums), np.array(masks)
    
    sumsA, masksA = gen_sums(valsA)
    if sumsA is None:
        return _greedy(denominators)
    
    sumsB, masksB = gen_sums(valsB)
    if sumsB is None:
        return _greedy(denominators)
    
    orderB = np.argsort(sumsB)
    sumsB_s = sumsB[orderB]
    masksB_s = masksB[orderB]
    
    targets = 1.0 - sumsA
    pos = np.searchsorted(sumsB_s, targets, side='left')
    valid = pos > 0
    if not np.any(valid):
        return _greedy(denominators)
    
    p = pos[valid] - 1
    sA_c = sumsA[valid]
    mA_c = masksA[valid]
    sB_c = sumsB_s[p]
    mB_c = masksB_s[p]
    
    totals = sA_c + sB_c
    vt = totals < 1.0
    if not np.any(vt):
        return _greedy(denominators)
    
    tv = totals[vt]
    sAf = sA_c[vt]
    mAf = mA_c[vt]
    sBf = sB_c[vt]
    mBf = mB_c[vt]
    
    deficits = 1.0 - tv
    bi = np.argmin(deficits)
    
    best_maskA = int(mAf[bi])
    best_maskB = int(mBf[bi])
    
    selected = []
    for i in range(nA):
        if best_maskA & (1 << i):
            selected.append(i)
    for i in range(nB):
        if best_maskB & (1 << i):
            selected.append(nA + i)
    
    if not selected:
        return _greedy(denominators)
    
    # Verify with exact arithmetic
    total_frac = Fraction(0)
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])
    
    if total_frac >= 1:
        # Fix: remove elements until valid
        while total_frac >= 1 and selected:
            # Remove the largest reciprocal
            max_idx = max(selected, key=lambda i: Fraction(1, denominators[i]))
            total_frac -= Fraction(1, denominators[max_idx])
            selected.remove(max_idx)
        if not selected:
            return _greedy(denominators)
    
    # Phase 2: Exact local search to improve
    if time.time() - start_time < DEADLINE - 5:
        selected = _exact_local_search(denominators, selected, start_time, DEADLINE)
    
    return sorted(selected)


def _exact_local_search(denominators, initial_selected, start_time, deadline):
    n = len(denominators)
    selected = set(initial_selected)
    total = Fraction(0)
    for idx in selected:
        total += Fraction(1, denominators[idx])
    
    if total >= 1:
        while total >= 1 and selected:
            max_idx = max(selected, key=lambda i: Fraction(1, denominators[i]))
            total -= Fraction(1, denominators[max_idx])
            selected.remove(max_idx)
        if not selected:
            return _greedy(denominators)
    
    best_total = total
    best_selected = set(selected)
    available = set(range(n)) - selected
    
    improved = True
    while improved and time.time() - start_time < deadline - 2:
        improved = False
        
        # Try adding one element
        for idx in sorted(available, key=lambda i: -1.0/denominators[i]):
            nt = best_total + Fraction(1, denominators[idx])
            if nt < 1 and nt > best_total:
                best_total = nt
                best_selected = best_selected | {idx}
                available.discard(idx)
                improved = True
                break
        
        # Try removing one and adding one (swap)
        if not improved:
            for rem in sorted(best_selected, key=lambda i: 1.0/denominators[i]):
                rem_val = Fraction(1, denominators[rem])
                for add in sorted(available, key=lambda i: -1.0/denominators[i]):
                    add_val = Fraction(1, denominators[add])
                    nt = best_total - rem_val + add_val
                    if nt < 1 and nt > best_total:
                        best_total = nt
                        best_selected = (best_selected - {rem}) | {add}
                        available = (available - {add}) | {rem}
                        improved = True
                        break
                if improved:
                    break
        
        # Try removing one and adding two
        if not improved:
            for rem in sorted(best_selected, key=lambda i: 1.0/denominators[i]):
                rem_val = Fraction(1, denominators[rem])
                base = best_total - rem_val
                if base < 1:
                    # Try adding two from available
                    avail_list = sorted(available, key=lambda i: -1.0/denominators[i])
                    for i1 in range(len(avail_list)):
                        v1 = Fraction(1, denominators[avail_list[i1]])
                        if base + v1 >= 1:
                            continue
                        for i2 in range(i1+1, len(avail_list)):
                            v2 = Fraction(1, denominators[avail_list[i2]])
                            nt = base + v1 + v2
                            if nt < 1 and nt > best_total:
                                best_total = nt
                                best_selected = (best_selected - {rem}) | {avail_list[i1], avail_list[i2]}
                                available = (available - {avail_list[i1], avail_list[i2]}) | {rem}
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
    
    result = sorted(best_selected)
    if not result:
        return _greedy(denominators)
    
    final_total = Fraction(0)
    for idx in result:
        final_total += Fraction(1, denominators[idx])
    if final_total >= 1:
        return _greedy(denominators)
    
    return result


def _greedy(denominators):
    selected = []
    total = Fraction(0)
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
