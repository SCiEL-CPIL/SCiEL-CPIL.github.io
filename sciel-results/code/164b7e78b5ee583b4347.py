import random
import time

N = 180

def _precompute_forbidden(N=180):
    """For each a, store the set of (b, c) pairs with b < c, b != c, b != a, c != a,
    1 <= b, c <= N, and (b-a)*(c-a) = a^2  (equivalent to 1/a = 1/b + 1/c)."""
    forbidden = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        a2 = a * a
        # iterate over positive divisors d of a^2
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                for dd in (d, a2 // d):
                    if dd == 0:
                        continue
                    # positive offsets
                    b1 = a + dd
                    c1 = a + a2 // dd
                    if b1 != a and c1 != a and b1 != c1 and 1 <= b1 <= N and 1 <= c1 <= N:
                        if b1 < c1:
                            forbidden[a].add((b1, c1))
                        else:
                            forbidden[a].add((c1, b1))
                    # negative offsets (b < a, c < a)
                    b2 = a - dd
                    c2 = a - a2 // dd
                    if b2 != a and c2 != a and b2 != c2 and 1 <= b2 <= N and 1 <= c2 <= N:
                        if b2 < c2:
                            forbidden[a].add((b2, c2))
                        else:
                            forbidden[a].add((c2, b2))
            d += 1
    return forbidden

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # safety margin under 90s kill

    rng = random.Random(12345)
    all_elems = set(range(1, N + 1))
    forbidden = _precompute_forbidden(N)

    def is_valid(S):
        S_set = set(S)
        for a in S:
            for b, c in forbidden[a]:
                if b in S_set and c in S_set:
                    return False
        return True

    def add_with_repair(current, elem):
        """Try to add elem to current. If invalid, try removing one involved element.
        Returns (new_set, success)."""
        candidate = set(current)
        candidate.add(elem)
        if is_valid(candidate):
            return candidate, True
        # find violating elements
        cand_set = candidate
        involved = set()
        for a in candidate:
            for b, c in forbidden[a]:
                if b in cand_set and c in cand_set:
                    involved.add(a)
                    involved.add(b)
                    involved.add(c)
        # try removing each involved element (not elem itself)
        for rem in involved:
            if rem == elem:
                continue
            candidate2 = set(candidate)
            candidate2.remove(rem)
            if is_valid(candidate2):
                return candidate2, True
        return current, False

    # ---- Phase 0: baseline top half ----
    best = set(range(91, N + 1))
    best_size = len(best)

    # ---- Phase 1: greedy add from below 91 with repair ----
    current = set(best)
    candidates = list(range(90, 0, -1))
    for elem in candidates:
        if time.time() > DEADLINE - 15:
            break
        new_set, ok = add_with_repair(current, elem)
        if ok and len(new_set) >= len(current):
            current = new_set
            if len(current) > best_size:
                best = set(current)
                best_size = len(best)

    # ---- Phase 2: local search (add + swap) ----
    for iteration in range(600):
        if time.time() > DEADLINE - 8:
            break

        improved = False
        missing = all_elems - best
        missing_list = list(missing)
        rng.shuffle(missing_list)

        # try to add a missing element (with repair)
        for add_elem in missing_list:
            if time.time() > DEADLINE - 8:
                break
            new_set, ok = add_with_repair(best, add_elem)
            if ok and len(new_set) > len(best):
                best = new_set
                best_size = len(best)
                improved = True
                break

        if not improved:
            # try a swap: remove one from best, add one from missing
            if best and missing:
                in_list = list(best)
                rng.shuffle(in_list)
                for rem in in_list[:20]:
                    if time.time() > DEADLINE - 8:
                        break
                    without = set(best)
                    without.remove(rem)
                    for add_elem in missing_list:
                        if time.time() > DEADLINE - 8:
                            break
                        cand = set(without)
                        cand.add(add_elem)
                        if is_valid(cand):
                            # same size; accept with low probability to explore
                            if random.random() < 0.05:
                                best = cand
                                improved = True
                                break
                    if improved:
                        break

        # periodic random restart
        if iteration % 30 == 0 and time.time() < DEADLINE - 20:
            new_set = set()
            elements = list(range(1, N + 1))
            rng.shuffle(elements)
            for elem in elements:
                if time.time() > DEADLINE - 20:
                    break
                cand = set(new_set)
                cand.add(elem)
                if is_valid(cand):
                    new_set = cand
            if len(new_set) > best_size:
                best = new_set
                best_size = len(best)

    # ---- Phase 3: final polish, try every missing element once more ----
    if time.time() < DEADLINE - 3:
        for elem in sorted(all_elems - best, reverse=True):
            if time.time() > DEADLINE - 3:
                break
            new_set, ok = add_with_repair(best, elem)
            if ok and len(new_set) > len(best):
                best = new_set
                best_size = len(best)

    return list(best)
