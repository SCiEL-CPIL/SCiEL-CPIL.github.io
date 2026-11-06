"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from math import lcm
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s limit

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    unit_vals = [Fraction(1, d) for d in denominators]

    # ---------- Fast greedy + local search (fallback / refiner) ----------
    def greedy(order):
        selected = set()
        total = Fraction(0)
        for idx in order:
            v = unit_vals[idx]
            if total + v < 1:
                selected.add(idx)
                total += v
        return selected, total

    def local_search(start_set, max_iterations=20000):
        nonlocal best_indices, best_total, best_deficit
        if not start_set:
            return
        cur = set(start_set)
        cur_total = sum(unit_vals[i] for i in cur)
        cur_deficit = Fraction(1) - cur_total
        if cur_deficit <= 0:
            return

        not_in_set = set(range(n)) - cur
        temperature = Fraction(1, 1000)
        cooling = 0.999

        iters = 0
        while iters < max_iterations and time.time() < deadline:
            iters += 1
            move_type = random.random()
            best_new_def = cur_deficit
            best_new_set = None
            best_new_total = None

            if move_type < 0.3 and not_in_set:
                add = random.choice(tuple(not_in_set))
                nt = cur_total + unit_vals[add]
                if nt < 1:
                    nd = Fraction(1) - nt
                    if nd < best_new_def:
                        best_new_def = nd
                        best_new_set = cur | {add}
                        best_new_total = nt
            elif move_type < 0.6 and len(cur) > 1:
                rm = random.choice(tuple(cur))
                nt = cur_total - unit_vals[rm]
                nd = Fraction(1) - nt
                if nd < best_new_def:
                    best_new_def = nd
                    best_new_set = cur - {rm}
                    best_new_total = nt
            elif move_type < 0.9 and len(cur) > 0 and not_in_set:
                rm = random.choice(tuple(cur))
                add = random.choice(tuple(not_in_set))
                nt = cur_total - unit_vals[rm] + unit_vals[add]
                if nt < 1:
                    nd = Fraction(1) - nt
                    if nd < best_new_def:
                        best_new_def = nd
                        best_new_set = (cur - {rm}) | {add}
                        best_new_total = nt
            else:
                if len(cur) >= 2 and len(not_in_set) >= 2:
                    rm1, rm2 = random.sample(tuple(cur), 2)
                    add1, add2 = random.sample(tuple(not_in_set), 2)
                    nt = cur_total - unit_vals[rm1] - unit_vals[rm2] + unit_vals[add1] + unit_vals[add2]
                    if nt < 1:
                        nd = Fraction(1) - nt
                        if nd < best_new_def:
                            best_new_def = nd
                            best_new_set = (cur - {rm1, rm2}) | {add1, add2}
                            best_new_total = nt

            if best_new_set is not None:
                delta = best_new_def - cur_deficit
                if delta < 0:
                    cur = best_new_set
                    cur_total = best_new_total
                    cur_deficit = best_new_def
                    not_in_set = set(range(n)) - cur
                    if cur_deficit < best_deficit:
                        best_deficit = cur_deficit
                        best_total = cur_total
                        best_indices = sorted(cur)
                else:
                    import math
                    if random.random() < math.exp(-float(delta) / float(temperature)):
                        cur = best_new_set
                        cur_total = best_new_total
                        cur_deficit = best_new_def
                        not_in_set = set(range(n)) - cur
                    temperature *= cooling

    # ---------- Meet-in-the-middle exact search ----------
    def mitm_exact():
        # LCM of 2..48
        L = 1
        for d in denominators:
            L = lcm(L, d)
        weights = [L // d for d in denominators]

        mid = n // 2
        w1 = weights[:mid]
        w2 = weights[mid:]

        def gen_sums(wlist):
            sums = [(0, 0)]
            for i, w in enumerate(wlist):
                bit = 1 << i
                new = []
                for s, m in sums:
                    ns = s + w
                    if ns < L:
                        new.append((ns, m | bit))
                sums.extend(new)
                if time.time() > deadline - 30.0:
                    return None  # signal timeout
            return sums

        sums1 = gen_sums(w1)
        if sums1 is None:
            return None
        sums2 = gen_sums(w2)
        if sums2 is None:
            return None

        # Deduplicate sums2 by keeping max mask per sum (for variety) then sort
        sums2.sort(key=lambda x: x[0])
        vals2 = [s for s, _ in sums2]

        best_sum = -1
        best_mask = 0
        for s1, m1 in sums1:
            if time.time() > deadline - 20.0:
                return None
            target = L - s1 - 1
            idx = bisect.bisect_right(vals2, target)
            if idx > 0:
                s2, m2 = sums2[idx - 1]
                total = s1 + s2
                if total > best_sum:
                    best_sum = total
                    best_mask = m1 | (m2 << mid)

        if best_sum < 0:
            return None
        result = []
        for i in range(n):
            if best_mask & (1 << i):
                result.append(i)
        return result, best_sum, L

    # ---------- Main flow ----------
    best_indices = []
    best_total = Fraction(0)
    best_deficit = Fraction(1)

    # Phase 1: Quick greedy to have a valid baseline
    random.seed(123)
    orderings = [
        list(range(n)),
        list(reversed(range(n))),
    ]
    for _ in range(20):
        o = list(range(n))
        random.shuffle(o)
        orderings.append(o)

    for order in orderings:
        idxs, total = greedy(order)
        if idxs:
            deficit = 1 - total
            if deficit > 0 and deficit < best_deficit:
                best_deficit = deficit
                best_total = total
                best_indices = sorted(idxs)

    # Phase 2: Try meet-in-the-middle exact search (the structural win)
    if time.time() < deadline - 25.0:
        result = mitm_exact()
        if result is not None:
            mitm_indices, mitm_sum, L = result
            if mitm_indices:
                # Verify strictly below 1
                mitm_frac = sum(unit_vals[i] for i in mitm_indices)
                if mitm_frac < 1:
                    mitm_deficit = 1 - mitm_frac
                    if mitm_deficit < best_deficit:
                        best_deficit = mitm_deficit
                        best_total = mitm_frac
                        best_indices = sorted(mitm_indices)
        else:
            # Timed out; fall back to local search on current best
            pass

    # Phase 3: Local search refinement on the best found so far
    if best_indices and time.time() < deadline - 5.0:
        local_search(set(best_indices), max_iterations=30000)

        # A few random restarts if time remains
        for restart in range(100):
            if time.time() >= deadline - 3.0:
                break
            o = list(range(n))
            random.shuffle(o)
            idxs, total = greedy(o)
            if idxs:
                deficit = 1 - total
                if deficit > 0 and deficit < best_deficit:
                    best_deficit = deficit
                    best_total = total
                    best_indices = sorted(idxs)
                local_search(idxs, max_iterations=2000)

    # Ensure valid nonempty result
    if not best_indices:
        best_indices = [0]
        best_total = Fraction(1, 2)

    return best_indices
# EVOLVE-BLOCK-END
