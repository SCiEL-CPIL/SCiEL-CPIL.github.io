import time
from math import lcm

def construct():
    deadline = time.time() + 55
    N = 255
    nums = list(range(2, 257))
    recip = [1.0 / nums[i] for i in range(N)]

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

    # ---- initial lower bound: greedy + local search ----
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
        return sel

    def recip_sum(sel):
        return sum(recip[i] for i in sel)

    best_sel = greedy((1 << N) - 1)
    best_val = recip_sum(best_sel)

    # local search: try adding any compatible element, and swaps
    def is_compatible_set(sel):
        for a in range(len(sel)):
            for b in range(a + 1, len(sel)):
                if (conflict_mask[sel[a]] >> sel[b]) & 1:
                    return False
        return True

    def local_search(sel):
        nonlocal best_val
        sel = list(sel)
        improved = True
        while improved:
            improved = False
            cur = recip_sum(sel)
            # add
            for i in order:
                if i in sel:
                    continue
                if all(not ((conflict_mask[s] >> i) & 1) for s in sel):
                    ns = sel + [i]
                    if recip_sum(ns) > cur:
                        sel = ns
                        improved = True
                        break
            # swap
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
        return sel, recip_sum(sel)

    best_sel, best_val = local_search(best_sel)

    full_mask = (1 << N) - 1

    # ---- branch and bound ----
    # precompute recip prefix not needed; compute sum over set bits
    def sum_avail(mask):
        s = 0.0
        m = mask
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            s += recip[i]
            m ^= lb
        return s

    nodes = [0]

    def bnb(sel, avail, cur):
        nonlocal best_val, best_sel
        if time.time() > deadline:
            return
        # bound
        if cur + sum_avail(avail) <= best_val + 1e-12:
            return
        # find next candidate in order that is in avail
        for i in order:
            if (avail >> i) & 1:
                break
        else:
            if cur > best_val:
                best_val = cur
                best_sel = list(sel)
            return
        nodes[0] += 1
        # take branch
        sel.append(i)
        new_avail = avail & ~(conflict_mask[i]) & ~(1 << i)
        bnb(sel, new_avail, cur + recip[i])
        sel.pop()
        # skip branch
        bnb(sel, avail & ~(1 << i), cur)

    bnb([], full_mask, 0.0)

    # ensure nonempty
    if not best_sel:
        best_sel = [order[0]]
    return [nums[i] for i in best_sel]
