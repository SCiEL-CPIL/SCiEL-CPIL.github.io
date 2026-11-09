"""Rewritten program: strong sum-free subset search with local improvement."""
import time

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def sum_free_check(S):
        """Return True if S is sum-free (no a+b=c with a,b,c in S, a=b allowed)."""
        s = set(S)
        for a in S:
            for b in S:
                if a + b in s:
                    return False
        return True

    def build_sum_free(arr):
        """Build a large sum-free subset of arr via greedy + local search."""
        # Start with elements sorted by |value| descending (large magnitudes
        # are least likely to form a+b=c).
        order = sorted(arr, key=lambda x: -abs(x))

        chosen = []
        chosen_set = set()
        # For fast violation testing: for each candidate c, it is invalid if
        # there exist a,b in chosen_set with a+b=c, i.e., a in chosen_set and
        # c-a in chosen_set.

        # Greedy add
        for c in order:
            ok = True
            for a in chosen_set:
                if c - a in chosen_set:
                    ok = False
                    break
            if ok:
                chosen_set.add(c)
                chosen.append(c)

        base = list(chosen_set)
        best = base

        # Local search: try to add any excluded element; if it creates violations,
        # try removing the 'least useful' element involved to resolve, if that helps.
        excluded = [x for x in arr if x not in chosen_set]

        # Precompute: for each candidate x, the set of pairs (a,b) in current
        # chosen_set with a+b=x. We'll recompute on the fly.

        def conflicts(x, S):
            """Return list of a in S such that x-a in S (a<=x-a to dedupe)."""
            res = []
            for a in S:
                b = x - a
                if b in S and a <= b:
                    res.append((a, b))
            return res

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try adding each excluded element
            for x in excluded[:]:
                if time.time() >= deadline - 2.0:
                    break
                S = set(best)
                conf = conflicts(x, S)
                if not conf:
                    # can add directly
                    best = best + [x]
                    excluded.remove(x)
                    improved = True
                    continue
                # Try: add x, remove one element from each conflicting pair to break
                # all conflicts. Greedy: remove the element with smallest |value|
                # that appears in conflicts, re-check.
                # Attempt a small repair: for each conflicting pair, pick to remove
                # the smaller-magnitude element; collect removals; apply; recheck.
                removals = set()
                for (a, b) in conf:
                    # remove the one with smaller abs (keep larger magnitude)
                    if abs(a) <= abs(b):
                        removals.add(a)
                    else:
                        removals.add(b)
                # Apply removals + add x, verify no remaining conflict
                newS = (S - removals) | {x}
                if not conflicts(x, newS) and sum_free_check(newS):
                    # also ensure we didn't break other sum-free-ness elsewhere:
                    # removing elements can't create new sums, adding x is checked.
                    if len(newS) > len(best):
                        best = list(newS)
                        excluded = [e for e in excluded if e not in newS]
                        improved = True
            # Also try: remove a low-value element and add a higher-value excluded one
            # (swap) to possibly grow.
            if time.time() >= deadline - 2.0:
                break
            for x in excluded[:]:
                if time.time() >= deadline - 2.0:
                    break
                S = set(best)
                conf = conflicts(x, S)
                if not conf:
                    continue
                # try swapping: for each removal candidate r in union of conflict elems,
                # newS = S - {r} + {x}; check sum-free
                cand_removals = set()
                for (a, b) in conf:
                    cand_removals.add(a)
                    cand_removals.add(b)
                for r in cand_removals:
                    if r == x:
                        continue
                    newS = (S - {r}) | {x}
                    if len(newS) <= len(best):
                        continue
                    if sum_free_check(newS):
                        best = list(newS)
                        excluded = [e for e in excluded if e not in newS]
                        improved = True
                        break

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() >= deadline - 2.0:
            # fallback: odd numbers (guaranteed valid)
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        sf = build_sum_free(vals)
        # final safety: verify; if somehow invalid, fall back to odds
        if not sum_free_check(sf):
            sf = [x for x in vals if x % 2]
        results.append(sf)

    return results
