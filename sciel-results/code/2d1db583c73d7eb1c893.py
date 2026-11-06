"""Sum-free subset maximizer with fast conflict tracking and 2-swap local search."""
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

    def build_conflict_map(vals):
        """Precompute for each element which other elements conflict with it.
        conflict[x] = set of y such that x+y is in vals, or y-x is in vals, or x-y is in vals,
        or 2*x is in vals (for the a=b=x, c=2x case).
        Actually, we need: x conflicts with y if any of {x+y, x-y, y-x} is in vals, or 2x in vals and y==x.
        More precisely: x and y are incompatible if there exists z in vals such that:
          x+y=z, x+z=y, y+z=x, or (x==y and 2x in vals)
        For the local search, we track which elements in the current set conflict with a candidate.
        """
        vals_set = set(vals)
        # For each x, compute the set of y in vals such that x and y cannot both be in a sum-free set
        # x,y conflict if: x+y in vals_set, or x-y in vals_set, or y-x in vals_set
        # (these cover all three positions of a+b=c)
        # Note: x+y in vals means x+y=c, so {x,y,c} is a violation
        # x-y in vals means x=c+y, so {c,y,x} is a violation  
        # y-x in vals means y=c+x, so {c,x,y} is a violation
        # Also 2*x in vals means x+x=2x, so {x,x,2x} is a violation (a=b allowed)
        conflict = {}
        for x in vals:
            conf = set()
            for y in vals:
                if y == x:
                    if 2 * x in vals_set:
                        conf.add(y)
                else:
                    if (x + y) in vals_set:
                        conf.add(y)
                    elif (x - y) in vals_set:
                        conf.add(y)
                    elif (y - x) in vals_set:
                        conf.add(y)
            conflict[x] = conf
        return conflict, vals_set

    def greedy_sum_free(vals, order=None):
        """Greedy: add elements in given order if no conflict with already-added."""
        if order is None:
            order = sorted(vals, key=abs, reverse=True)
        result = []
        rset = set()
        for x in order:
            ok = True
            for a in rset:
                if (x - a) in rset or (x + a) in rset:
                    ok = False
                    break
            if ok and (2 * x) in rset:
                ok = False
            if ok:
                result.append(x)
                rset.add(x)
        return result

    def local_search(S, vals, conflict_map, vals_set, max_iters=10000, time_limit=15.0):
        """Improve S using add and 1-remove-2-add moves with fast conflict checking."""
        Sset = set(S)
        remaining = [x for x in vals if x not in Sset]
        best = list(S)
        best_size = len(best)
        rng = random.Random(42)
        t_start = time.time()

        # Phase 1: greedy add - repeatedly try to add any remaining element
        changed = True
        while changed:
            if time.time() > t_start + time_limit:
                break
            changed = False
            for x in remaining:
                if time.time() > t_start + time_limit:
                    break
                # Check if x conflicts with any element in Sset
                # x conflicts with a in Sset if: x+a in vals_set, x-a in vals_set, a-x in vals_set
                # or if 2x in vals_set (then x conflicts with itself, but x not in Sset yet)
                # Actually for adding x: we need no a,b in Sset∪{x} with a+b=c in Sset∪{x}
                # Since x is new, violations involving x are:
                #   x+a=c for a,c in Sset -> x=c-a, so check if x-a in Sset for any a in Sset
                #   a+x=c for a,c in Sset -> same as above
                #   a+b=x for a,b in Sset -> check if x-a in Sset for any a in Sset
                #   x+x=c for c in Sset -> check if 2x in Sset
                #   x+a=x -> a=0, but 0 not in vals
                ok = True
                for a in Sset:
                    if (x - a) in Sset or (x + a) in Sset:
                        ok = False
                        break
                if ok and (2 * x) in Sset:
                    ok = False
                if ok:
                    Sset.add(x)
                    remaining.remove(x)
                    best = list(Sset)
                    best_size = len(best)
                    changed = True

        # Phase 2: 1-remove-2-add swaps
        for _ in range(max_iters):
            if time.time() > t_start + time_limit:
                break
            if len(remaining) < 2:
                break
            # Pick a random element to remove
            idx = rng.randrange(len(best))
            removed = best[idx]
            Sset.remove(removed)
            
            # Try to add elements from remaining
            added = []
            for x in remaining:
                if x in Sset:
                    continue
                ok = True
                for a in Sset:
                    if (x - a) in Sset or (x + a) in Sset:
                        ok = False
                        break
                if ok and (2 * x) in Sset:
                    ok = False
                if ok:
                    Sset.add(x)
                    added.append(x)
                    if len(added) >= 2:
                        break  # We only need 2 to get net gain
            
            if len(added) >= 2:  # net gain of at least 1
                best = list(Sset)
                best_size = len(best)
                remaining = [x for x in vals if x not in Sset]
            elif len(added) == 1:
                # Net zero: swap removed with added[0]
                Sset.discard(added[0])
                Sset.add(removed)
                # Keep the swap if it might enable future improvements (heuristic: accept)
                remaining = [x for x in vals if x not in Sset]
            else:
                # Revert
                Sset.add(removed)

        return best

    def solve_one(vals, time_budget):
        """Find large sum-free subset of vals within time_budget seconds."""
        n = len(vals)
        t_start = time.time()

        # Precompute conflict map
        conflict_map, vals_set = build_conflict_map(vals)

        # Generate candidate solutions
        candidates = []

        # 1. Odd numbers
        odds = [x for x in vals if x % 2 != 0]
        candidates.append(odds)

        # 2. Top half by value
        sorted_vals = sorted(vals)
        half = n // 2
        top_half = sorted_vals[half:]
        if is_sum_free(top_half):
            candidates.append(top_half)

        # 3. Bottom half
        bot_half = sorted_vals[:half]
        if is_sum_free(bot_half):
            candidates.append(bot_half)

        # 4. Greedy by absolute value
        greedy = greedy_sum_free(vals)
        candidates.append(greedy)

        # 5. Greedy by value (largest first)
        greedy2 = greedy_sum_free(vals, order=sorted(vals, reverse=True))
        candidates.append(greedy2)

        # 6. Greedy by value (smallest first)
        greedy3 = greedy_sum_free(vals, order=sorted(vals))
        candidates.append(greedy3)

        # 7. All positive
        pos = [x for x in vals if x > 0]
        greedy_pos = greedy_sum_free(pos)
        candidates.append(greedy_pos)

        # 8. All negative
        neg = [x for x in vals if x < 0]
        greedy_neg = greedy_sum_free(neg)
        candidates.append(greedy_neg)

        # 9. Large positive (top 2/3)
        two_thirds = (2 * n) // 3
        top_two_thirds = sorted_vals[two_thirds:]
        if is_sum_free(top_two_thirds):
            candidates.append(top_two_thirds)

        # Find best candidate
        best = max(candidates, key=len)
        best_size = len(best)

        # Local search to improve
        remaining_time = time_budget - (time.time() - t_start)
        if remaining_time > 1.0:
            improved = local_search(best, vals, conflict_map, vals_set, 
                                     time_limit=min(remaining_time - 0.5, 25.0))
            if len(improved) > best_size:
                best = improved
                best_size = len(best)

        # Try local search on other good candidates
        remaining_time = time_budget - (time.time() - t_start)
        if remaining_time > 2.0:
            for cand in candidates:
                if len(cand) >= best_size - 3 and set(cand) != set(best):
                    if time.time() - t_start > time_budget:
                        break
                    t_rem = time_budget - (time.time() - t_start)
                    if t_rem < 1.5:
                        break
                    improved = local_search(cand, vals, conflict_map, vals_set,
                                            time_limit=min(t_rem - 0.5, 12.0))
                    if len(improved) > best_size:
                        best = improved
                        best_size = len(best)
                        remaining_time = time_budget - (time.time() - t_start)
                        if remaining_time < 2.0:
                            break

        return best

    # Solve each instance
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]

    for size, seed in instances:
        vals = instance(size, seed)
        remaining = deadline - time.time()
        # Allocate time proportional to problem size
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
