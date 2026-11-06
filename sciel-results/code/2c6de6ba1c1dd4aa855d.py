"""Sum-free subset maximizer with local search and deadline guard."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin under 120s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if (a + b) in Sset:
                    return False
        return True

    def count_violations(S):
        Sset = set(S)
        cnt = 0
        for a in S:
            for b in S:
                if (a + b) in Sset:
                    cnt += 1
        return cnt

    def greedy_sum_free(vals):
        """Greedy: sort by |value| descending, add if no violation."""
        result = []
        rset = set()
        for x in sorted(vals, key=abs, reverse=True):
            # check if adding x creates a+b=c with a,b,c in result∪{x}
            ok = True
            # x = a+b: need a,b in rset with a+b==x
            for a in rset:
                if (x - a) in rset:
                    ok = False
                    break
            if ok:
                # a = x+b: need b in rset with x+b in rset
                for b in rset:
                    if (x + b) in rset:
                        ok = False
                        break
            if ok:
                # b = a+x: same as above symmetric
                # 2x = a: need a=2x in rset? no, 2*x = c means c=2x
                # if x+x in rset -> violation (a=b=x, c=2x)
                if (2 * x) in rset:
                    ok = False
            if ok:
                result.append(x)
                rset.add(x)
        return result

    def local_search(S, vals, max_iters=5000, time_limit=15.0):
        """Try to improve S by adding elements or swapping. Returns improved set."""
        Sset = set(S)
        remaining = [x for x in vals if x not in Sset]
        best = list(S)
        best_size = len(best)
        rng = random.Random(42)
        t_start = time.time()

        # Phase 1: try to add each remaining element
        for x in remaining:
            if time.time() > t_start + time_limit:
                break
            # check if adding x violates
            ok = True
            for a in Sset:
                if (x - a) in Sset:
                    ok = False
                    break
            if ok:
                for a in Sset:
                    if (x + a) in Sset:
                        ok = False
                        break
            if ok and (2 * x) in Sset:
                ok = False
            if ok:
                Sset.add(x)
                best = list(Sset)
                best_size = len(best)

        # Phase 2: swap - remove one, try to add two
        for _ in range(max_iters):
            if time.time() > t_start + time_limit:
                break
            if len(best) < 1:
                break
            # pick a random element to remove
            idx = rng.randrange(len(best))
            removed = best[idx]
            Sset.remove(removed)
            # try to add elements
            added = []
            for x in remaining:
                if x in Sset:
                    continue
                ok = True
                for a in Sset:
                    if (x - a) in Sset:
                        ok = False
                        break
                if ok:
                    for a in Sset:
                        if (x + a) in Sset:
                            ok = False
                            break
                if ok and (2 * x) in Sset:
                    ok = False
                if ok:
                    Sset.add(x)
                    added.append(x)
            if len(added) > 1:  # net gain
                best = list(Sset)
                best_size = len(best)
                remaining = [x for x in vals if x not in Sset]
            else:
                Sset.add(removed)
                if added:
                    Sset.discard(added[0])
                    remaining = [x for x in vals if x not in Sset]

        return best

    def solve_one(vals, time_budget):
        """Find large sum-free subset of vals within time_budget seconds."""
        n = len(vals)
        t_start = time.time()

        # Generate candidate solutions
        candidates = []

        # 1. Odd numbers
        odds = [x for x in vals if x % 2 != 0]
        candidates.append(odds)

        # 2. Even numbers (also sum-free? no: even+even=even)
        # Actually evens are NOT sum-free. Skip.

        # 3. Top half by value (positive large)
        sorted_vals = sorted(vals)
        half = n // 2
        top_half = sorted_vals[half:]
        # top_half might not be sum-free, check
        if is_sum_free(top_half):
            candidates.append(top_half)

        # 4. Bottom half
        bot_half = sorted_vals[:half]
        if is_sum_free(bot_half):
            candidates.append(bot_half)

        # 5. Greedy by absolute value
        greedy = greedy_sum_free(vals)
        candidates.append(greedy)

        # 6. Greedy by value (largest first)
        greedy2 = greedy_sum_free(sorted(vals, reverse=True))
        candidates.append(greedy2)

        # 7. All positive
        pos = [x for x in vals if x > 0]
        # positives not necessarily sum-free
        # try greedy on positives
        greedy_pos = greedy_sum_free(pos)
        candidates.append(greedy_pos)

        # 8. All negative
        neg = [x for x in vals if x < 0]
        greedy_neg = greedy_sum_free(neg)
        candidates.append(greedy_neg)

        # Find best candidate
        best = max(candidates, key=len)
        best_size = len(best)

        # Local search to improve
        remaining_time = time_budget - (time.time() - t_start)
        if remaining_time > 1.0:
            improved = local_search(best, vals, time_limit=min(remaining_time - 0.5, 20.0))
            if len(improved) > best_size:
                best = improved
                best_size = len(best)

        # Try local search on other good candidates
        remaining_time = time_budget - (time.time() - t_start)
        if remaining_time > 2.0:
            for cand in candidates:
                if len(cand) >= best_size - 2 and cand != best:
                    if time.time() - t_start > time_budget:
                        break
                    t_rem = time_budget - (time.time() - t_start)
                    if t_rem < 1.0:
                        break
                    improved = local_search(cand, vals, time_limit=min(t_rem - 0.5, 10.0))
                    if len(improved) > best_size:
                        best = improved
                        best_size = len(best)

        return best

    # Solve each instance
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]

    for size, seed in instances:
        vals = instance(size, seed)
        elapsed = time.time() - (deadline - 110.0)
        remaining = deadline - time.time()
        # Allocate time proportional to problem size, but at least some minimum
        if len(results) == 0:
            budget = min(remaining * 0.3, 30.0)
        elif len(results) == 1:
            budget = min(remaining * 0.35, 40.0)
        else:
            budget = min(remaining * 0.5, 50.0)
        budget = max(budget, 5.0)

        result = solve_one(vals, budget)
        results.append(result)

        if time.time() > deadline - 5.0:
            # Not enough time, use odd numbers for remaining
            for sz, sd in instances[len(results):]:
                v = instance(sz, sd)
                results.append([x for x in v if x % 2 != 0])
            break

    return results
# EVOLVE-BLOCK-END
