"""Find a maximum-cardinality dissociated reciprocal subset of [1,22]."""

def construct():
    import time
    import itertools
    from math import gcd

    deadline = time.time() + 75  # safety margin for 90s wall-clock

    # Precompute all subset sums for each candidate subset using the
    # following approach:
    # Two reciprocal subset sums are equal iff sum_{i in A} 1/a_i = sum_{j in B} 1/b_j
    # Multiply by LCM(1..22) to get integers.
    # LCM(1..22) = 232792560 (computed below)

    # LCM of 1..22
    def _lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, 23):
        L = _lcm(L, i)

    # Integer values: v_i = L // i
    vals = [L // i for i in range(1, 23)]  # vals[0] = L//1, vals[1] = L//2, etc.

    def _is_dissociated(indices):
        """Check if subset of indices (0-based into vals) is dissociated."""
        k = len(indices)
        if k == 0:
            return True
        # Use set-based subset sum enumeration with early termination
        sums = {0}
        for idx in indices:
            v = vals[idx]
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums |= new_sums
        return True

    # Start with the known good solution of size 9
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    # Try to extend to larger cardinality using backtracking search
    # Search from high cardinality down
    n = 22

    # Precompute: for quick pruning, if a set is not dissociated, no superset is
    # (actually that's not true - removing elements can make it dissociated,
    # but adding elements to a non-dissociated set keeps it non-dissociated)
    # So we can prune: if current partial set is not dissociated, skip.

    # Try all subsets of size 10, 11, ... in a smart order
    # For efficiency, try to build up from the known solution

    # First, check if we can get size 10+
    # Use a recursive backtracking that tries to add elements

    # Order: try to add elements not in best first
    # We'll do a search for size >= 10

    found_better = False

    # Try to find a size-10 dissociated set
    # Strategy: take the known size-9 set and try adding each of the remaining 13 elements
    remaining = [i for i in range(1, 23) if i not in best]

    for r in remaining:
        candidate = best + [r]
        if _is_dissociated(candidate):
            found_better = True
            best = sorted(candidate)
            break

    # If we found size 10, try to extend to size 11, etc.
    if found_better and time.time() < deadline:
        current = best
        for target_size in range(len(current) + 1, 23):
            if time.time() > deadline - 5:
                break
            # Try adding one more element
            remaining = [i for i in range(1, 23) if i not in current]
            extended = False
            for r in remaining:
                if time.time() > deadline - 3:
                    break
                candidate = current + [r]
                if _is_dissociated(candidate):
                    current = sorted(candidate)
                    extended = True
                    break
            if not extended:
                break
        if len(current) > len(best):
            best = current

    # If we couldn't extend the known solution, try a broader search for size 10
    if not found_better and time.time() < deadline:
        # Try all combinations of size 10 (there are C(22,10)=646646, which is feasible)
        # But we can prune: use the fact that if a subset is not dissociated,
        # we can skip supersets. Use backtracking.

        # Greedy + local search: try random subsets of size 10
        import random
        random.seed(42)

        # First try systematic: for each pair to remove from a size-12 candidate...
        # Actually, let's just do a backtracking search for size 10.

        # Backtracking: build a dissociated set, trying to reach size 10
        # Order elements by "hardness" - larger values (smaller reciprocals) first
        # since they're less likely to cause collisions

        # Try adding elements in decreasing order of value (i.e., 22, 21, 20, ...)
        # Actually, smaller reciprocals (larger numbers) are "safer" to include

        # Let's do: try all size-10 subsets using a pruned backtracking
        # This is complex; instead, try a randomized approach

        attempts = 0
        max_attempts = 200000

        while time.time() < deadline - 5 and attempts < max_attempts:
            # Random subset of size 10
            subset = sorted(random.sample(range(1, 23), 10))
            attempts += 1
            if _is_dissociated(subset):
                best = subset
                found_better = True
                break

        # Also try systematic: all 10-subsets containing the 9 known elements + one extra
        # (already done above)

        # Try 10-subsets with 8 from known + 2 extras
        if not found_better and time.time() < deadline - 5:
            known_set = set(best)
            extras = [i for i in range(1, 23) if i not in known_set]
            for i in range(len(extras)):
                for j in range(i + 1, len(extras)):
                    if time.time() > deadline - 3:
                        break
                    candidate = sorted(list(known_set) + [extras[i], extras[j]])
                    # Remove 2 random known elements? No, we need exactly 10
                    # Actually: 8 from known + 2 extras = 10
                    # Pick 8 from the 9 known
                    for skip in range(9):
                        if time.time() > deadline - 3:
                            break
                        cand8 = [best[k] for k in range(9) if k != skip]
                        candidate = sorted(cand8 + [extras[i], extras[j]])
                        if _is_dissociated(candidate):
                            best = candidate
                            found_better = True
                            break
                    if found_better:
                        break
                if found_better:
                    break

    return best
