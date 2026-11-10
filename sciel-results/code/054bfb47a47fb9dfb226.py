import time
import random

def construct():
    deadline = time.time() + 85
    n = 256

    # ---- Precompute conflicts as bitmasks (O(1) conflict checks) ----
    conflicts = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a] |= (1 << b)
                conflicts[b] |= (1 << a)

    degree = [conflicts[i].bit_count() for i in range(n + 1)]
    all_elements = list(range(1, n + 1))
    odds = set(range(1, n + 1, 2))
    evens = list(range(2, n + 1, 2))
    odds_mask = 0
    for o in odds:
        odds_mask |= (1 << o)

    # ---- Helpers ----
    def mask_size(m):
        return m.bit_count()

    def greedy_fill(s_mask, order):
        for e in order:
            bit = 1 << e
            if s_mask & bit:
                continue
            if not (conflicts[e] & s_mask):
                s_mask |= bit
        return s_mask

    def local_search(s_mask, max_iters=200):
        best_mask = s_mask
        best_size = s_mask.bit_count()
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # 1. Add any non-member that has no conflict.
            for b in range(1, n + 1):
                bit_b = 1 << b
                if s_mask & bit_b:
                    continue
                if not (conflicts[b] & s_mask):
                    s_mask |= bit_b
                    sz = s_mask.bit_count()
                    if sz > best_size:
                        best_size = sz
                        best_mask = s_mask
                    improved = True
                    break
            if improved:
                continue

            # 2. 1-swap: remove a, add b (b conflicts with exactly one member a).
            for b in range(1, n + 1):
                bit_b = 1 << b
                if s_mask & bit_b:
                    continue
                cwb = conflicts[b] & s_mask
                if cwb == 0:
                    continue
                if cwb & (cwb - 1) == 0:  # exactly one bit
                    a_bit = cwb
                    new_mask = (s_mask & ~a_bit) | bit_b
                    if not (conflicts[b] & (new_mask & ~bit_b)):
                        sz = new_mask.bit_count()
                        if sz > best_size:
                            s_mask = new_mask
                            best_size = sz
                            best_mask = new_mask
                            improved = True
                            break
                        elif sz == best_size and random.random() < 0.1:
                            s_mask = new_mask
                            improved = True
                            break
            if improved:
                continue

            # 3. 2-swap: remove a,c, add b (b conflicts with exactly {a,c}).
            for b in range(1, n + 1):
                bit_b = 1 << b
                if s_mask & bit_b:
                    continue
                cwb = conflicts[b] & s_mask
                if cwb == 0:
                    continue
                c1 = cwb & (cwb - 1)
                if c1 == 0:
                    continue
                if c1 & (c1 - 1) == 0:  # exactly two bits
                    a_bit = cwb & (-cwb)
                    c_bit = cwb ^ a_bit
                    new_mask = (s_mask & ~a_bit & ~c_bit) | bit_b
                    if not (conflicts[b] & (new_mask & ~bit_b)):
                        sz = new_mask.bit_count()
                        if sz > best_size:
                            s_mask = new_mask
                            best_size = sz
                            best_mask = new_mask
                            improved = True
                            break
            if improved:
                continue

            # 4. 2-remove / 1-add (from inspirations): drop a,c, add b that
            #    conflicts with at most {a,c}.  (Net size -1, but can unlock
            #    later growth and escape local optima.)
            if random.random() < 0.35:
                for b in range(1, n + 1):
                    bit_b = 1 << b
                    if s_mask & bit_b:
                        continue
                    cwb = conflicts[b] & s_mask
                    if cwb == 0:
                        continue
                    c1 = cwb & (cwb - 1)
                    if c1 == 0:
                        continue
                    if c1 & (c1 - 1) != 0:
                        continue  # conflicts with 3+; not a 2-remove candidate
                    a_bit = cwb & (-cwb)
                    c_bit = cwb ^ a_bit
                    new_mask = (s_mask & ~a_bit & ~c_bit) | bit_b
                    if not (conflicts[b] & (new_mask & ~bit_b)):
                        s_mask = new_mask
                        improved = True
                        break
            if improved:
                continue

            # 5. Random kick: remove a few, greedily refill in random order.
            if random.random() < 0.3:
                elems = [i for i in range(1, n + 1) if s_mask & (1 << i)]
                if elems:
                    k = random.randint(1, 3)
                    for r in random.sample(elems, min(k, len(elems))):
                        s_mask &= ~(1 << r)
                    remaining = [e for e in all_elements if not (s_mask & (1 << e))]
                    random.shuffle(remaining)
                    s_mask = greedy_fill(s_mask, remaining)
                    sz = s_mask.bit_count()
                    if sz > best_size:
                        best_size = sz
                        best_mask = s_mask
                    improved = True

            if not improved:
                break

        return best_mask, best_size

    def polish(mask):
        """Exhaustive 1-swap / 2-swap until no improving move remains."""
        s = mask
        best_size = s.bit_count()
        while time.time() < deadline - 2:
            improved = False
            for b in range(1, n + 1):
                bit_b = 1 << b
                if s & bit_b:
                    continue
                cwb = conflicts[b] & s
                if cwb == 0:
                    s |= bit_b
                    best_size += 1
                    improved = True
                    break
                if cwb & (cwb - 1) == 0:  # exactly one conflict
                    a_bit = cwb
                    nm = (s & ~a_bit) | bit_b
                    if not (conflicts[b] & (nm & ~bit_b)):
                        s = nm
                        improved = True
                        break
                else:
                    c1 = cwb & (cwb - 1)
                    if c1 & (c1 - 1) == 0:  # exactly two conflicts
                        a_bit = cwb & (-cwb)
                        c_bit = cwb ^ a_bit
                        nm = (s & ~a_bit & ~c_bit) | bit_b
                        if not (conflicts[b] & (nm & ~bit_b)):
                            s = nm
                            best_size -= 1
                            improved = True
                            break
            if not improved:
                break
        return s, s.bit_count()

    # ---- Phase 0: from-scratch starts (conflict-degree ordering) ----
    best_mask = 0
    best_size = 0

    order_deg = sorted(all_elements, key=lambda e: degree[e])
    s0 = greedy_fill(0, order_deg)
    s0, sz0 = local_search(s0, max_iters=120)
    if sz0 > best_size:
        best_size = sz0
        best_mask = s0

    random.seed(42)
    for trial in range(4):
        if time.time() > deadline - 6:
            break
        order = list(all_elements)
        random.shuffle(order)
        s0 = greedy_fill(0, order)
        s0, sz0 = local_search(s0, max_iters=60)
        if sz0 > best_size:
            best_size = sz0
            best_mask = s0

    # ---- Phase 1: structured odds + conflict-sorted evens ----
    evens_sorted = sorted(evens, key=lambda e: (conflicts[e] & odds_mask).bit_count())
    s1 = greedy_fill(odds_mask, evens_sorted)
    s1, sz1 = local_search(s1, max_iters=300)
    if sz1 > best_size:
        best_size = sz1
        best_mask = s1

    # ---- Phase 2: randomized greedy restarts from odds ----
    for trial in range(80):
        if time.time() > deadline - 6:
            break
        evens_shuffled = list(evens)
        random.shuffle(evens_shuffled)
        s2 = greedy_fill(odds_mask, evens_shuffled)
        s2, sz2 = local_search(s2, max_iters=100)
        if sz2 > best_size:
            best_size = sz2
            best_mask = s2

    # ---- Phase 3: random-subset seeds (inspiration 2) ----
    for trial in range(30):
        if time.time() > deadline - 6:
            break
        seed = random.sample(all_elements, random.randint(50, 100))
        valid = 0
        for e in seed:
            if not (conflicts[e] & valid):
                valid |= (1 << e)
        remaining = [e for e in all_elements if not (valid & (1 << e))]
        random.shuffle(remaining)
        s3 = greedy_fill(valid, remaining)
        s3, sz3 = local_search(s3, max_iters=100)
        if sz3 > best_size:
            best_size = sz3
            best_mask = s3

    # ---- Phase 4: kicks from best ----
    for trial in range(50):
        if time.time() > deadline - 6:
            break
        s4 = best_mask
        elems = [i for i in range(1, n + 1) if s4 & (1 << i)]
        K = random.randint(3, 12)
        for r in random.sample(elems, min(K, len(elems))):
            s4 &= ~(1 << r)
        remaining = [e for e in all_elements if not (s4 & (1 << e))]
        random.shuffle(remaining)
        s4 = greedy_fill(s4, remaining)
        s4, sz4 = local_search(s4, max_iters=150)
        if sz4 > best_size:
            best_size = sz4
            best_mask = s4

    # ---- Phase 5: final aggressive local search + exhaustive polish ----
    if time.time() < deadline - 4:
        s5, sz5 = local_search(best_mask, max_iters=500)
        if sz5 > best_size:
            best_size = sz5
            best_mask = s5
        s5, sz5 = polish(s5)
        if sz5 > best_size:
            best_size = sz5
            best_mask = s5

    result = [i for i in range(1, n + 1) if best_mask & (1 << i)]
    if not result:
        result = [1]
    return result
