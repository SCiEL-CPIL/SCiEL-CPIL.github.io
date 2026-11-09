"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    all_elements = list(range(1, 23))
    powers_of_2 = [1, 2, 4, 8, 16]
    primes = [2, 3, 5, 7, 11, 13, 17, 19]

    def lcm(a, b):
        return a // math.gcd(a, b) * b

    def build_from(state, order):
        """
        state: (L, sums, current_set)
        Greedily try to extend current_set with elements from order.
        Returns (L, sums, current_set) of the final valid set.
        sums is the set of all 2^n scaled reciprocal subset sums (integers).
        """
        L, sums, current = state
        cur_set = set(current)
        for x in order:
            if x in cur_set:
                continue
            new_L = lcm(L, x)
            if new_L > 10**18:
                # LCM too large for cheap integer arithmetic; skip this element
                continue
            factor = new_L // L
            # Scale existing sums to the new LCM
            if factor == 1:
                new_sums = sums
            else:
                new_sums = {s * factor for s in sums}
            w = new_L // x
            # Check for any collision: does s + w hit an existing sum?
            valid = True
            for s in new_sums:
                if (s + w) in new_sums:
                    valid = False
                    break
            if valid:
                sums = new_sums | {s + w for s in new_sums}
                L = new_L
                current = current + [x]
                cur_set.add(x)
        return L, sums, current

    def rebuild_state(subset):
        """Recompute (L, sums, list) from scratch for a given subset."""
        L = 1
        sums = {0}
        cur = []
        for x in subset:
            new_L = lcm(L, x)
            factor = new_L // L
            if factor == 1:
                new_sums = sums
            else:
                new_sums = {s * factor for s in sums}
            w = new_L // x
            sums = new_sums | {s + w for s in new_sums}
            L = new_L
            cur.append(x)
        return L, sums, cur

    best = []

    # 1) Deterministic orderings
    base_orders = [
        all_elements,
        all_elements[::-1],
        powers_of_2 + [x for x in all_elements if x not in powers_of_2],
        primes + [x for x in all_elements if x not in primes],
        sorted(all_elements, key=lambda x: 1.0 / x),
        sorted(all_elements, key=lambda x: -1.0 / x),
    ]

    for order in base_orders:
        if time.time() > deadline:
            break
        L, sums, result = build_from((1, {0}, []), order)
        if len(result) > len(best):
            best = result

    # 2) Randomized greedy (cheap thanks to incremental tracking)
    random.seed(42)
    while time.time() < deadline - 20.0:
        order = all_elements[:]
        random.shuffle(order)
        if random.random() < 0.25:
            rest = [x for x in order if x not in powers_of_2]
            order = powers_of_2 + rest
        elif random.random() < 0.25:
            rest = [x for x in order if x not in primes]
            order = primes + rest
        L, sums, result = build_from((1, {0}, []), order)
        if len(result) > len(best):
            best = result

    # 3) Local search: drop one element, then greedily regrow from the rest.
    #    This escapes greedy local optima that pure add-only passes get stuck in.
    if best and time.time() < deadline - 15.0:
        for x in list(best):
            if time.time() > deadline - 12.0:
                break
            reduced = [y for y in best if y != x]
            L, sums, cur = rebuild_state(reduced)
            # Regrow: try every remaining element, accepting any that fit.
            remaining = [y for y in all_elements if y not in set(reduced)]
            random.shuffle(remaining)
            L, sums, result = build_from((L, sums, cur), remaining)
            if len(result) > len(best):
                best = result

    # 4) Final squeeze: try to add one more element to best.
    if best and time.time() < deadline - 5.0:
        L, sums, cur = rebuild_state(best)
        in_set = set(best)
        for cand in all_elements:
            if cand in in_set:
                continue
            new_L = lcm(L, cand)
            if new_L > 10**18:
                continue
            factor = new_L // L
            new_sums = {s * factor for s in sums} if factor != 1 else sums
            w = new_L // cand
            if not any((s + w) in new_sums for s in new_sums):
                best = best + [cand]
                break

    if best:
        return best

    # Fallback: known valid small set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
