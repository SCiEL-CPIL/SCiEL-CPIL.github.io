def construct():
    import time
    t0 = time.time()
    DEADLINE = 70.0  # safety margin under 90s

    def elapsed_ok():
        return (time.time() - t0) < DEADLINE

    from fractions import Fraction

    def exact_deficit(indices):
        s = Fraction(0)
        for i in indices:
            s += Fraction(1, i + 2)
        return s  # return sum

    # Fallback greedy
    def greedy():
        sel, tot = [], Fraction(0)
        for i in range(47):
            d = i + 2
            if tot + Fraction(1, d) < 1:
                sel.append(i)
                tot += Fraction(1, d)
        return sel

    best = None
    best_def = None

    def consider(indices):
        nonlocal best, best_def
        if not indices:
            return
        s = exact_deficit(indices)
        if s < 1:
            d = 1 - s
            if best_def is None or d < best_def:
                best_def = d
                best = list(indices)

    try:
        import numpy as np
        # group1: small weights, denoms 26..48 -> indices 24..46 (23 items)
        # group2: large weights, denoms 2..25 -> indices 0..23 (24 items)
        w1 = [1.0 / d for d in range(26, 49)]   # 23 items
        w2 = [1.0 / d for d in range(2, 26)]     # 24 items

        def gen_sums_masks(w):
            n = len(w)
            sums = np.array([0.0], dtype=np.float64)
            masks = np.array([0], dtype=np.int32)
            for k, ww in enumerate(w):
                bit = np.int32(1 << k)
                sums = np.concatenate([sums, sums + ww])
                masks = np.concatenate([masks, masks + bit])
            return sums, masks

        sums1, masks1 = gen_sums_masks(w1)
        if not elapsed_ok():
            raise TimeoutError
        sums2, masks2 = gen_sums_masks(w2)
        if not elapsed_ok():
            raise TimeoutError

        # filter group2 to sums < 1.0 (strict); keep
        m2 = sums2 < 1.0
        sums2 = sums2[m2]
        masks2 = masks2[m2]
        # filter group1 to sums < 1.0 (should be all)
        m1 = sums1 < 1.0
        sums1 = sums1[m1]
        masks1 = masks1[m1]

        # sort group2
        order = np.argsort(sums2, kind='stable')
        sums2s = sums2[order]
        masks2s = masks2[order]

        # for each group1 sum a, find largest b in sums2s with a+b < 1.0
        target = 1.0 - sums1
        idx = np.searchsorted(sums2s, target, side='right') - 1
        valid = idx >= 0
        if not np.any(valid):
            raise RuntimeError("no pair")
        total = np.full(len(sums1), -np.inf, dtype=np.float64)
        total[valid] = sums1[valid] + sums2s[idx[valid]]

        # top K
        K = min(100, int(np.count_nonzero(valid)))
        part = np.argpartition(total, -K)[-K:]
        # sort these by total desc
        part = part[np.argsort(total[part])[::-1]]

        for gi in part:
            if not elapsed_ok():
                break
            a_mask = int(masks1[gi])
            j = int(idx[gi])
            b_mask = int(masks2s[j])
            # global mask: group1 local bits 0..22 -> global 24..46 ; group2 local 0..23 -> global 0..23
            gmask = (a_mask << 24) | b_mask
            inds = [i for i in range(47) if (gmask >> i) & 1]
            consider(inds)
    except Exception:
        pass

    if best is None:
        best = greedy()
    return best
