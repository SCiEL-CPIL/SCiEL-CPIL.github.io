"""Improved program for sum-free subset maximization."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def make_valid(S, Sset):
        """Given a candidate list, greedily filter to a valid sum-free set."""
        valid = []
        vset = set()
        for x in S:
            if x in vset:
                continue
            ok = True
            for a in valid:
                if a + x in vset:
                    ok = False
                    break
                if x + x in vset:
                    ok = False
                    break
            if ok:
                valid.append(x)
                vset.add(x)
        return valid

    def can_add(S, Sset, x):
        """Check if x can be added to S without violating sum-freeness."""
        if x in Sset:
            return False
        for a in S:
            if a + x in Sset:
                return False
            if x + x in Sset:
                return False
        return True

    def local_search(S, all_set, time_budget):
        """Local search to improve S."""
        S = list(S)
        Sset = set(S)
        end = time.time() + time_budget
        improved = True
        while improved and time.time() < end:
            improved = False
            # Try adding each excluded element
            for x in all_set:
                if x in Sset:
                    continue
                if can_add(S, Sset, x):
                    S.append(x)
                    Sset.add(x)
                    improved = True
                    break
            if time.time() >= end:
                break
            # Try removing an element that blocks many additions
            # For each a in S, count excluded x where a+x in S (so x = s-a for s in S)
            best_remove = None
            best_gain = 0
            for a in S:
                gain = 0
                for s in S:
                    x = s - a
                    if x != 0 and x not in Sset and x in all_set:
                        # Check if x would be addable after removing a
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_remove = a
            if best_remove is not None and best_gain >= 2:
                S.remove(best_remove)
                Sset.discard(best_remove)
                improved = True
                # Try to add now
                for x in all_set:
                    if x in Sset:
                        continue
                    if can_add(S, Sset, x):
                        S.append(x)
                        Sset.add(x)
                        break
        return S

    def solve_instance(vals):
        all_set = set(vals)
        n = len(vals)
        best = []
        best_len = 0

        sorted_vals = sorted(vals)
        candidates = [
            [x for x in vals if x % 2],              # odd
            sorted_vals[n // 2:],                     # upper half
            [x for x in vals if x % 3 == 1],          # mod 3 == 1
            [x for x in vals if x % 3 == 2],          # mod 3 == 2
            [x for x in vals if x > 0],               # positive
            [x for x in vals if x < 0],               # negative
        ]

        for cand in candidates:
            valid = make_valid(cand, None)
            if len(valid) > best_len:
                best = valid
                best_len = len(valid)

        # Local search
        if time.time() < DEADLINE - 5:
            budget = min(DEADLINE - 5 - time.time(), 40)
            best = local_search(best, all_set, budget)

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        results.append(solve_instance(vals))

    return results
# EVOLVE-BLOCK-END
