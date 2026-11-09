"""Stronger search for large sum-free subsets of three fixed sets."""
# EVOLVE-BLOCK-START
import time
import math
from random import Random

def construct():
    deadline = time.time() + 135.0  # safety margin under 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        s = set(S)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def sum_free_score(S):
        s = set(S)
        # count violations for diagnostics
        v = 0
        for a in s:
            for b in s:
                if a + b in s:
                    v += 1
        return v

    def greedy_ordering(S, order_key):
        """Greedy: add elements in order; keep if sum-free with current set."""
        chosen = []
        chosen_set = set()
        # Precompute for speed: for each candidate x, the set of y such that x+y in S or y+x in S or 2x in S
        # We'll just check incrementally.
        for x in sorted(S, key=order_key):
            ok = True
            for y in chosen_set:
                if x + y in chosen_set or x + y in set(S) and x + y == x:  # 2x check
                    # Actually need: for all a,b in chosen_set ∪ {x}, a+b != c in chosen_set ∪ {x}
                    # Simpler: check new violations introduced by adding x
                    pass
            # Do a proper check
            ok = True
            for y in chosen_set:
                if x + y in chosen_set:
                    ok = False
                    break
                if x + y == x:  # y=0, but 0 not in S
                    pass
            if ok and 2 * x in chosen_set:
                ok = False
            if ok:
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def greedy_with_check(S, order_key):
        """Greedy add in given order; verify sum-free after each add."""
        chosen = []
        chosen_set = set()
        S_set = set(S)
        for x in sorted(S, key=order_key):
            # Check if adding x creates a violation
            ok = True
            for y in chosen_set:
                s = x + y
                if s in chosen_set:
                    ok = False
                    break
            if ok and 2 * x in chosen_set:
                ok = False
            if ok:
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def local_search(S, initial, iterations=2000):
        """Local search: try swapping in/out to improve size while keeping sum-free."""
        chosen = list(initial)
        chosen_set = set(chosen)
        S_set = set(S)
        candidates = [x for x in S if x not in chosen_set]
        best_size = len(chosen)
        
        for _ in range(iterations):
            improved = False
            # Try adding a candidate
            for x in candidates:
                ok = True
                for y in chosen_set:
                    if x + y in chosen_set:
                        ok = False
                        break
                if ok and 2 * x in chosen_set:
                    ok = False
                if ok:
                    chosen.append(x)
                    chosen_set.add(x)
                    candidates.remove(x)
                    best_size = len(chosen)
                    improved = True
                    break
            if improved:
                continue
            # Try swapping: remove one, add one
            for i, a in enumerate(chosen):
                for x in candidates:
                    if x == a:
                        continue
                    # Remove a, add x
                    new_set = chosen_set - {a}
                    # Check x with new_set
                    ok = True
                    for y in new_set:
                        if x + y in new_set:
                            ok = False
                            break
                    if ok and 2 * x in new_set:
                        ok = False
                    if ok:
                        chosen[i] = x
                        chosen_set = new_set | {x}
                        candidates.remove(x)
                        candidates.append(a)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return chosen

    def solve_instance(S, time_budget):
        """Solve one instance: try multiple orderings, local search, return best."""
        t0 = time.time()
        n = len(S)
        S_set = set(S)
        
        # Base: odds
        odds = [x for x in S if x % 2 != 0]
        best = list(odds)
        best_set = set(best)
        
        # Try various orderings
        orderings = [
            lambda x: x,           # ascending
            lambda x: -x,          # descending
            lambda x: abs(x),      # by magnitude
            lambda x: -abs(x),     # by magnitude desc
            lambda x: x % 7,       # by mod 7
            lambda x: x % 5,       # by mod 5
            lambda x: (x * 7) % 13,  # pseudo-random
            lambda x: (x * 11) % 17,
            lambda x: math.sin(x),  # smooth
            lambda x: -math.sin(x),
        ]
        
        for order_key in orderings:
            if time.time() - t0 > time_budget:
                break
            cand = greedy_with_check(S, order_key)
            if len(cand) > len(best):
                best = cand
                best_set = set(cand)
        
        # Local search on best
        if time.time() - t0 < time_budget:
            improved = local_search(S, best, iterations=3000)
            if len(improved) > len(best):
                best = improved
                best_set = set(best)
        
        # Verify
        if not is_sum_free(best):
            # Fallback to odds
            best = odds
            best_set = set(best)
        
        return best

    # Solve each instance
    instances = [instance(60, 801), instance(120, 802), instance(240, 803)]
    results = []
    for i, S in enumerate(instances):
        remaining = deadline - time.time()
        budget = max(1.0, remaining * (1 - i * 0.1))
        res = solve_instance(S, budget)
        results.append(res)
        if time.time() > deadline - 5:
            # Not enough time, use odds for remaining
            for j in range(i + 1, 3):
                results.append([x for x in instances[j] if x % 2 != 0])
            break
    
    return results
# EVOLVE-BLOCK-END
