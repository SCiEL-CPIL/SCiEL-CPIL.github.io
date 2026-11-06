import time
import random
import bisect

def construct():
    start = time.time()
    deadline = start + 85.0

    ALL = list(range(2, 129))
    N = len(ALL)
    rng = random.Random(20240601)

    def build_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def verify(seq):
        n = len(seq)
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def can_insert(seq, x, products, in_set):
        if x in in_set:
            return None
        k = bisect.bisect_left(seq, x)
        new = set()
        p = x
        new.add(p)
        for i in range(k - 1, -1, -1):
            p *= seq[i]
            if p in new:
                return None
            new.add(p)
        p = x
        for j in range(k + 1, len(seq)):
            p *= seq[j]
            if p in new:
                return None
            new.add(p)
        if new & products:
            return None
        return k, new

    def greedy(order):
        seq = []
        in_set = set()
        products = set()
        for x in order:
            res = can_insert(seq, x, products, in_set)
            if res is not None:
                k, np_ = res
                seq.insert(k, x)
                in_set.add(x)
                products.update(np_)
        return seq, in_set, products

    # deterministic safe seed
    safe_seq, _, _ = greedy(ALL[:])

    best = list(safe_seq)
    best_len = len(best)

    def consider(seq):
        nonlocal best, best_len
        if len(seq) > best_len:
            best_len = len(seq)
            best = list(seq)

    def add_sweep(seq, in_set, products, order):
        progressed = True
        while progressed:
            if time.time() > deadline:
                break
            progressed = False
            for x in order:
                if x in in_set:
                    continue
                res = can_insert(seq, x, products, in_set)
                if res is not None:
                    k, np_ = res
                    seq.insert(k, x)
                    in_set.add(x)
                    products.update(np_)
                    progressed = True
        return seq, in_set, products

    def smart_kick(seq, in_set, products):
        missing = [x for x in ALL if x not in in_set]
        if not missing:
            return None
        cand = list(range(len(seq)))
        rng.shuffle(cand)
        cand = cand[:12]
        best_i = None
        best_gain = 0
        for idx in cand:
            x = seq[idx]
            tmp = seq[:idx] + seq[idx + 1:]
            tmp_set = set(tmp)
            tmp_prods = build_products(tmp)
            gain = 0
            for m in missing:
                if can_insert(tmp, m, tmp_prods, tmp_set) is not None:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_i = idx
        return best_i

    for trial in range(25):
        if time.time() > deadline:
            break
        if trial == 0:
            order = ALL[:]
        elif trial == 1:
            order = ALL[::-1]
        else:
            order = ALL[:]
            rng.shuffle(order)
        seq, in_set, products = greedy(order)
        consider(seq)
        for it in range(150):
            if time.time() > deadline:
                break
            missing_order = [x for x in ALL if x not in in_set]
            rng.shuffle(missing_order)
            before = len(seq)
            add_sweep(seq, in_set, products, missing_order)
            if len(seq) > before:
                consider(seq)
            else:
                if len(seq) > 6:
                    if rng.random() < 0.2:
                        # double kick
                        i1 = rng.randrange(len(seq))
                        seq.pop(i1); in_set.discard(seq[i1] if False else None)
                        # careful: pop then recompute
                        products = build_products(seq)
                        if len(seq) > 6:
                            i2 = rng.randrange(len(seq))
                            seq.pop(i2)
                        in_set = set(seq)
                        products = build_products(seq)
                    else:
                        idx = smart_kick(seq, in_set, products)
                        if idx is None:
                            idx = rng.randrange(len(seq))
                        x = seq[idx]
                        seq.pop(idx)
                        in_set.discard(x)
                        products = build_products(seq)
        consider(seq)

    # final polish on best
    for _ in range(5):
        if time.time() > deadline:
            break
        in_set = set(best)
        products = build_products(best)
        add_sweep(best, in_set, products, ALL[:])
        consider(best)

    if not verify(best):
        best = list(safe_seq)
    return best
