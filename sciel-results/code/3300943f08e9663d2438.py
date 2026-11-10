import time
import random

def construct():
    """
    Return a nonempty subset of [1..22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit

    # LCM of 1..22 for converting reciprocal sums to integers
    lcm_all = 232792560  # LCM(1,2,...,22)
    recip_int = {i: lcm_all // i for i in range(1, 23)}

    def check_dissociated(elems):
        """Check if the set has all distinct reciprocal subset sums using integer arithmetic."""
        k = len(elems)
        if k == 0:
            return True
        seen = {0}
        current_sums = [0]
        for e in elems:
            v = recip_int[e]
            new_sums = []
            for s in current_sums:
                ns = s + v
                if ns in seen:
                    return False
                seen.add(ns)
                new_sums.append(ns)
            current_sums.extend(new_sums)
        return True

    def build_sums(elems):
        """Build the set of all subset sums for elems using integer arithmetic."""
        cur_sums = {0}
        for e in elems:
            v = recip_int[e]
            cur_sums = cur_sums | {s + v for s in cur_sums}
        return cur_sums

    def can_add(e, cur_sums):
        """Check if element e can be added to the set with given subset sums."""
        v = recip_int[e]
        for s in cur_sums:
            if (s + v) in cur_sums:
                return False
        return True

    def greedy_build(ordering):
        """Greedily build a dissociated set following the given ordering."""
        S = []
        cur_sums = {0}
        for e in ordering:
            if time.time() > deadline:
                break
            v = recip_int[e]
            collision = False
            for s in cur_sums:
                if (s + v) in cur_sums:
                    collision = True
                    break
            if not collision:
                S.append(e)
                cur_sums = cur_sums | {s + v for s in cur_sums}
        return S

    def expand_set(best_set, deadline_margin=3.0):
        """Try to expand best_set by adding remaining elements."""
        remaining = [e for e in range(1, 23) if e not in best_set]
        remaining.sort(reverse=True)
        cur_sums = build_sums(best_set)
        S = list(best_set)
        changed = True
        while changed and time.time() < deadline - deadline_margin:
            changed = False
            for e in remaining[:]:
                if time.time() > deadline - deadline_margin:
                    break
                v = recip_int[e]
                collision = False
                for s in cur_sums:
                    if (s + v) in cur_sums:
                        collision = True
                        break
                if not collision:
                    S.append(e)
                    cur_sums = cur_sums | {s + v for s in cur_sums}
                    remaining.remove(e)
                    changed = True
        return S

    # Strategy 1: Greedy build with multiple orderings
    orderings = [
        list(range(22, 0, -1)),           # Descending (largest first) - best
        list(range(1, 23)),               # Ascending
        list(range(22, 1, -1)),           # Skip 1, descending
        list(range(2, 23)),               # Skip 1, ascending
        [e for e in range(22, 0, -1) if e >= 3],
        [e for e in range(22, 0, -1) if e >= 4],
        [e for e in range(22, 0, -1) if e >= 5],
        list(range(2, 23, 2)) + list(range(1, 23, 2)),  # Evens then odds
        list(range(1, 23, 2)) + list(range(2, 23, 2)),  # Odds then evens
    ]

    best_set = [1]
    for ordering in orderings:
        if time.time() > deadline - 30:
            break
        result = greedy_build(ordering)
        if result and len(result) > len(best_set):
            best_set = result

    # Expand the best set
    if time.time() < deadline - 25:
        expanded = expand_set(best_set, deadline_margin=25.0)
        if len(expanded) > len(best_set):
            best_set = expanded

    # Random greedy builds for diversity
    for _ in range(30):
        if time.time() > deadline - 20:
            break
        order = list(range(1, 23))
        random.shuffle(order)
        result = greedy_build(order)
        if result and len(result) > len(best_set):
            best_set = result
            # Expand immediately
            if time.time() < deadline - 20:
                expanded = expand_set(best_set, deadline_margin=20.0)
                if len(expanded) > len(best_set):
                    best_set = expanded

    # Strategy 2: Systematic single-swap search
    # Try replacing each element in best_set with each element not in best_set
    if time.time() < deadline - 10:
        remaining = [e for e in range(1, 23) if e not in best_set]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            remaining = [e for e in range(1, 23) if e not in best_set]
            for i, e_remove in enumerate(best_set[:]):
                if time.time() > deadline - 10:
                    break
                for e_add in remaining:
                    if time.time() > deadline - 10:
                        break
                    candidate = best_set[:i] + best_set[i+1:] + [e_add]
                    if check_dissociated(candidate):
                        # Try to expand this candidate
                        rem2 = [e for e in range(1, 23) if e not in candidate]
                        rem2.sort(reverse=True)
                        cur_sums2 = build_sums(candidate)
                        for f in rem2:
                            if time.time() > deadline - 10:
                                break
                            v = recip_int[f]
                            collision = False
                            for s in cur_sums2:
                                if (s + v) in cur_sums2:
                                    collision = True
                                    break
                            if not collision:
                                candidate.append(f)
                                cur_sums2 = cur_sums2 | {s + v for s in cur_sums2}
                        if len(candidate) > len(best_set):
                            best_set = candidate
                            improved = True
                            break
                if improved:
                    break

    # Strategy 3: Double-swap local search
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            remaining = [e for e in range(1, 23) if e not in best_set]
            n = len(best_set)
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break
                    for e_add1 in remaining:
                        if time.time() > deadline - 5:
                            break
                        for e_add2 in remaining:
                            if e_add2 <= e_add1:
                                continue
                            if time.time() > deadline - 5:
                                break
                            candidate = [e for idx, e in enumerate(best_set) if idx != i and idx != j] + [e_add1, e_add2]
                            if check_dissociated(candidate):
                                # Try to expand
                                rem3 = [e for e in range(1, 23) if e not in candidate]
                                rem3.sort(reverse=True)
                                cur_sums3 = build_sums(candidate)
                                for f in rem3:
                                    if time.time() > deadline - 5:
                                        break
                                    v = recip_int[f]
                                    collision = False
                                    for s in cur_sums3:
                                        if (s + v) in cur_sums3:
                                            collision = True
                                            break
                                    if not collision:
                                        candidate.append(f)
                                        cur_sums3 = cur_sums3 | {s + v for s in cur_sums3}
                                if len(candidate) > len(best_set):
                                    best_set = candidate
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # Strategy 4: Randomized local search with restarts
    # Use random perturbations of the current best set
    if time.time() < deadline - 3:
        for restart in range(50):
            if time.time() > deadline - 3:
                break
            # Start from a random subset of best_set plus some random elements
            if restart == 0:
                cur = list(best_set)
            else:
                # Perturb: remove 1-3 random elements, add 1-3 random others
                cur = list(best_set)
                n_remove = random.randint(1, min(3, len(cur)))
                for _ in range(n_remove):
                    if cur:
                        idx = random.randint(0, len(cur) - 1)
                        cur.pop(idx)
                # Add random elements not in cur
                avail = [e for e in range(1, 23) if e not in cur]
                n_add = random.randint(0, min(3, len(avail)))
                for _ in range(n_add):
                    if avail:
                        idx = random.randint(0, len(avail) - 1)
                        cur.append(avail[idx])
                        avail.pop(idx)
            
            if not cur:
                cur = [1]
            
            if check_dissociated(cur):
                # Expand
                remaining = [e for e in range(1, 23) if e not in cur]
                remaining.sort(reverse=True)
                cur_sums = build_sums(cur)
                for e in remaining:
                    if time.time() > deadline - 3:
                        break
                    v = recip_int[e]
                    collision = False
                    for s in cur_sums:
                        if (s + v) in cur_sums:
                            collision = True
                            break
                    if not collision:
                        cur.append(e)
                        cur_sums = cur_sums | {s + v for s in cur_sums}
                if len(cur) > len(best_set):
                    best_set = cur

    # Final expansion pass
    if time.time() < deadline - 1:
        expanded = expand_set(best_set, deadline_margin=1.0)
        if len(expanded) > len(best_set):
            best_set = expanded

    # Final safety: ensure best_set is nonempty and valid
    if not best_set:
        best_set = [1]

    if not check_dissociated(best_set):
        # Fallback to known valid sets
        fallbacks = [
            list(range(2, 18)),  # [2..17], size 16
            [2, 3, 5, 7, 11, 13, 17, 19, 22],
            [1],
        ]
        for fb in fallbacks:
            if check_dissociated(fb):
                best_set = fb
                break
        else:
            best_set = [1]

    return sorted(best_set)
