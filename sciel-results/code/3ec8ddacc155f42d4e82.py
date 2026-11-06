import time
from math import lcm
from fractions import Fraction

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
    order_set = set(order)

    # Precompute for each i, the list of elements compatible with i (for faster bound computation)
    # compat_with[i] = set of j != i such that not (conflict_mask[i] >> j) & 1
    # But we need compat with ALL selected, so we compute on the fly

    # ---- initial lower bound: greedy + local search ----
    def greedy(start_mask):
        sel = []
        avail = start_mask
        for i in order:
            if (avail >> i) & 1:
                ok = True
                for s in sel:
                    if (conflict_mask[s] >> i) & 1:
                        ok = False
                        break
                if ok:
                    sel.append(i)
        return sel

    def recip_sum(sel):
        return sum(recip[i] for i in sel)

    def recip_sum_frac(sel):
        s = Fraction(0)
        for i in sel:
            s += recip_frac[i]
        return s

    best_sel = greedy((1 << N) - 1)
    best_val = recip_sum(best_sel)
    best_val_frac = recip_sum_frac(best_sel)

    # local search: try adding any compatible element, swaps, and 2-for-1 swaps
    def local_search(sel):
        nonlocal best_val, best_val_frac, best_sel
        sel = list(sel)
        improved = True
        while improved:
            improved = False
            cur = recip_sum(sel)
            cur_set = set(sel)
            # add
            for i in order:
                if i in cur_set:
                    continue
                if all(not ((conflict_mask[s] >> i) & 1) for s in sel):
                    ns = sel + [i]
                    if recip_sum(ns) > cur:
                        sel = ns
                        cur_set.add(i)
                        improved = True
                        break
            # 1-for-1 swap
            if not improved:
                for idx, s in enumerate(sel):
                    for i in order:
                        if i in cur_set:
                            continue
                        if all(not ((conflict_mask[t] >> i) & 1) for t in sel if t != s):
                            ns = [t for k, t in enumerate(sel) if k != idx] + [i]
                            if recip_sum(ns) > cur:
                                sel = ns
                                cur_set.discard(s)
                                cur_set.add(i)
                                improved = True
                                break
                    if improved:
                        break
            # 2-for-1 swap: remove two, add one
            if not improved:
                n = len(sel)
                for a in range(n):
                    for b in range(a + 1, n):
                        removed = {sel[a], sel[b]}
                        rem_sum = recip[sel[a]] + recip[sel[b]]
                        for i in order:
                            if i in cur_set:
                                continue
                            if all(not ((conflict_mask[t] >> i) & 1) for t in sel if t not in removed):
                                if recip[i] > rem_sum:
                                    ns = [t for t in sel if t not in removed] + [i]
                                    sel = ns
                                    cur_set.difference_update(removed)
                                    cur_set.add(i)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        best_val = recip_sum(sel)
        best_val_frac = recip_sum_frac(sel)
        best_sel = list(sel)
        return sel, best_val

    best_sel, best_val = local_search(best_sel)

    full_mask = (1 << N) - 1

    # ---- branch and bound with tighter bound ----
    def sum_compat_avail(sel, avail):
        """Sum of recip[i] for i in avail that are compatible with all in sel."""
        s = 0.0
        m = avail
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            # Check if i is compatible with all in sel
            ok = True
            for t in sel:
                if (conflict_mask[t] >> i) & 1:
                    ok = False
                    break
            if ok:
                s += recip[i]
            m ^= lb
        return s

    def sum_compat_avail_frac(sel, avail):
        """Exact fraction version for tight bounds."""
        s = Fraction(0)
        m = avail
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            ok = True
            for t in sel:
                if (conflict_mask[t] >> i) & 1:
                    ok = False
                    break
            if ok:
                s += recip_frac[i]
            m ^= lb
        return s

    nodes = [0]

    def bnb(sel, avail, cur, cur_frac):
        nonlocal best_val, best_val_frac, best_sel
        if time.time() > deadline:
            return
        # Tight bound: only count elements compatible with current selection
        upper_float = cur + sum_compat_avail(sel, avail)
        if upper_float <= best_val - 1e-9:
            return
        # If close, use exact fraction bound
        if upper_float <= best_val + 1e-6:
            upper_frac = cur_frac + sum_compat_avail_frac(sel, avail)
            if upper_frac <= best_val_frac:
                return
        # Find next candidate in order that is in avail and compatible with sel
        for i in order:
            if (avail >> i) & 1:
                # Check compatibility with current selection
                ok = True
                for t in sel:
                    if (conflict_mask[t] >> i) & 1:
                        ok = False
                        break
                if ok:
                    break
        else:
            if cur_frac > best_val_frac:
                best_val = cur
                best_val_frac = cur_frac
                best_sel = list(sel)
            return
        nodes[0] += 1
        # Take branch
        sel.append(i)
        new_avail = avail & ~(conflict_mask[i]) & ~(1 << i)
        bnb(sel, new_avail, cur + recip[i], cur_frac + recip_frac[i])
        sel.pop()
        # Skip branch
        bnb(sel, avail & ~(1 << i), cur, cur_frac)

    bnb([], full_mask, 0.0, Fraction(0))

    # ensure nonempty
    if not best_sel:
        best_sel = [order[0]]
    return [nums[i] for i in best_sel]
