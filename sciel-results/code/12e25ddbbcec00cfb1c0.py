import time
from math import lcm
from fractions import Fraction
import random

def construct():
    deadline = time.time() + 55
    N = 255
    nums = list(range(2, 257))
    recip = [1.0 / nums[i] for i in range(N)]
    recip_frac = [Fraction(1, nums[i]) for i in range(N)]

    # conflict_mask[i] = int bitset of j (j != i) with lcm(nums[i],nums[j]) <= 256
    conflict_mask = [0] * N
    for i in range(N):
        ni = nums[i]
        for j in range(i + 1, N):
            if lcm(ni, nums[j]) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # order by descending reciprocal
    order = list(range(N))
    order.sort(key=lambda i: -recip[i])
    
    # Precompute bit masks for order? No, just use order list.

    def greedy(start_mask):
        sel = []
        avail = start_mask
        for i in order:
            if (avail >> i) & 1:
                # compatible with all in sel
                ok = True
                for s in sel:
                    if (conflict_mask[s] >> i) & 1:
                        ok = False
                        break
                if ok:
                    sel.append(i)
                    # Remove i and its conflicts from avail
                    avail &= ~(1 << i)
                    avail &= ~conflict_mask[i]
        return sel

    def recip_sum(sel):
        return sum(recip[i] for i in sel)

    def recip_sum_frac(sel):
        s = Fraction(0)
        for i in sel:
            s += recip_frac[i]
        return s

    # Multi-start greedy + local search
    best_sel = []
    best_val = 0.0
    best_val_frac = Fraction(0)
    
    # Run greedy with full mask
    init_sel = greedy((1 << N) - 1)
    best_sel = init_sel
    best_val = recip_sum(best_sel)
    best_val_frac = recip_sum_frac(best_sel)
    
    # Local search
    def local_search(sel):
        nonlocal best_val, best_val_frac, best_sel
        sel = list(sel)
        improved = True
        while improved:
            improved = False
            cur = recip_sum(sel)
            # Add
            for i in order:
                if i in sel:
                    continue
                if all(not ((conflict_mask[s] >> i) & 1) for s in sel):
                    ns = sel + [i]
                    if recip_sum(ns) > cur:
                        sel = ns
                        improved = True
                        break
            # Swap
            if not improved:
                for idx, s in enumerate(sel):
                    for i in order:
                        if i in sel:
                            continue
                        if all(not ((conflict_mask[t] >> i) & 1) for t in sel if t != s):
                            ns = [t for k, t in enumerate(sel) if k != idx] + [i]
                            if recip_sum(ns) > cur:
                                sel = ns
                                improved = True
                                break
                    if improved:
                        break
        best_val = recip_sum(sel)
        best_val_frac = recip_sum_frac(sel)
        best_sel = list(sel)
        return sel, best_val

    best_sel, best_val = local_search(best_sel)
    
    full_mask = (1 << N) - 1

    # B&B
    def sum_avail_float(mask):
        s = 0.0
        m = mask
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            s += recip[i]
            m ^= lb
        return s

    def sum_avail_frac(mask):
        s = Fraction(0)
        m = mask
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            s += recip_frac[i]
            m ^= lb
        return s

    nodes = [0]

    def bnb(sel, avail, cur, cur_frac):
        nonlocal best_val, best_val_frac, best_sel
        if time.time() > deadline:
            return
        
        # Bound
        upper_float = cur + sum_avail_float(avail)
        if upper_float <= best_val - 1e-9:
            return
        
        # If close, use exact
        if upper_float <= best_val + 1e-6:
            upper_frac = cur_frac + sum_avail_frac(avail)
            if upper_frac <= best_val_frac:
                return
        
        # Find next candidate
        # Heuristic: pick the available node with highest weight?
        # The 'order' list is sorted by weight desc.
        # So we just find the first i in order that is in avail.
        next_i = -1
        for i in order:
            if (avail >> i) & 1:
                next_i = i
                break
        
        if next_i == -1:
            if cur_frac > best_val_frac:
                best_val = cur
                best_val_frac = cur_frac
                best_sel = list(sel)
            return
        
        nodes[0] += 1
        
        # Take branch
        sel.append(next_i)
        new_avail = avail & ~(conflict_mask[next_i]) & ~(1 << next_i)
        bnb(sel, new_avail, cur + recip[next_i], cur_frac + recip_frac[next_i])
        sel.pop()
        
        # Skip branch
        bnb(sel, avail & ~(1 << next_i), cur, cur_frac)

    bnb([], full_mask, 0.0, Fraction(0))

    if not best_sel:
        best_sel = [order[0]]
    return [nums[i] for i in best_sel]
