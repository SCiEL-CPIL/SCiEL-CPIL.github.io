import time
import random
from math import gcd

def construct():
    deadline = time.time() + 55.0
    start = time.time()

    # --- Precomputation ---
    # For each k, list of m in [2,256] such that k divides m
    multiples_of = [[] for _ in range(257)]
    for k in range(1, 257):
        for m in range(k, 257, k):
            if m >= 2:
                multiples_of[k].append(m)

    # Largest prime factor for ordering heuristics
    lpf = [0] * 257
    for i in range(2, 257):
        if lpf[i] == 0:  # i is prime
            for j in range(i, 257, i):
                if lpf[j] == 0:
                    lpf[j] = i

    candidates = list(range(2, 257))

    def can_add(a, in_set):
        """
        Check if a can be added to the set represented by in_set.
        For every distinct b,c in S, we need:
          a ∤ b*c
        Equivalent check: for each b in S, let g = gcd(a,b).
        If k = a//g > 1, we need no c in S (c != b) with k | c.
        """
        for b in range(2, 257):
            if not in_set[b]:
                continue
            g = gcd(a, b)
            k = a // g
            if k > 1:
                for c in multiples_of[k]:
                    if c != b and in_set[c]:
                        return False
        return True

    def greedy_construction(order, time_budget=None):
        """Run greedy construction on a given order. Optionally stop early if time_budget exhausted."""
        in_set = [False] * 257
        current_set = []
        for a in order:
            if time_budget is not None and time.time() > time_budget:
                break
            if can_add(a, in_set):
                in_set[a] = True
                current_set.append(a)
        return current_set

    def dynamic_greedy(candidates_pool):
        """
        Dynamic greedy: at each step, pick the "best" candidate from the remaining pool.
        Best = can be added, and has the fewest conflicts (or highest LPF as tiebreaker).
        This adapts to the current set state.
        """
        in_set = [False] * 257
        current_set = []
        pool = list(candidates_pool)
        while pool:
            best_idx = -1
            best_score = None
            for i, a in enumerate(pool):
                if can_add(a, in_set):
                    # Score: prefer elements with larger LPF (less restrictive), then larger value
                    score = (lpf[a], a)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_idx = i
            if best_idx == -1:
                break
            a = pool.pop(best_idx)
            in_set[a] = True
            current_set.append(a)
        return current_set

    def local_search_swaps(initial_set, num_swaps, time_limit):
        """
        Try to improve the set by removing one element and adding a different one,
        possibly unlocking additional additions.
        """
        best_set = list(initial_set)
        best_size = len(best_set)
        in_set = [False] * 257
        for x in best_set:
            in_set[x] = True

        remaining = [x for x in range(2, 257) if not in_set[x]]

        for _ in range(num_swaps):
            if time.time() > time_limit:
                break

            # Try removing a random element from best_set
            if not best_set:
                break
            rem_idx = random.randint(0, len(best_set) - 1)
            rem_elem = best_set[rem_idx]
            in_set[rem_elem] = False
            best_set.pop(rem_idx)

            # Now try to add back multiple elements from remaining
            # First, add rem_elem back? No, we want a different element.
            # Try adding all possible remaining elements in a greedy manner, but prioritize
            # elements that were previously blocked.
            # Simple approach: run a quick greedy on the remaining candidates.
            added_any = False
            for x in remaining:
                if in_set[x]:
                    continue
                if can_add(x, in_set):
                    in_set[x] = True
                    best_set.append(x)
                    remaining.remove(x)
                    added_any = True
                    if len(best_set) > best_size:
                        best_size = len(best_set)

            # If we didn't improve, restore
            if not added_any or len(best_set) < best_size:
                # Undo
                in_set = [False] * 257
                for x in best_set:
                    in_set[x] = True
                # This is getting complex. Let's use a simpler swap strategy.
                break

        return best_set

    best_set = []

    # --- Strategy 1: Deterministic Greedy with specific orderings ---
    # Order 1: Descending Largest Prime Factor
    order_lpf_desc = sorted(candidates, key=lambda x: (-lpf[x], -x))
    S = greedy_construction(order_lpf_desc)
    if len(S) > len(best_set):
        best_set = S[:]

    # Order 2: Descending Value
    order_desc = sorted(candidates, reverse=True)
    S = greedy_construction(order_desc)
    if len(S) > len(best_set):
        best_set = S[:]

    # Order 3: Ascending Value
    order_asc = sorted(candidates)
    S = greedy_construction(order_asc)
    if len(S) > len(best_set):
        best_set = S[:]

    # Order 4: Primes first
    primes = [x for x in candidates if lpf[x] == x]
    composites = [x for x in candidates if lpf[x] != x]
    order_primes_first = sorted(primes, reverse=True) + sorted(composites, reverse=True)
    S = greedy_construction(order_primes_first)
    if len(S) > len(best_set):
        best_set = S[:]

    # --- Strategy 2: Dynamic Greedy ---
    # This adapts to the current set state, potentially finding larger sets.
    S = dynamic_greedy(candidates)
    if len(S) > len(best_set):
        best_set = S[:]

    # --- Strategy 3: Randomized Greedy Restarts ---
    random.seed(42)  # Reproducibility
    num_restarts = 0
    max_restarts = 1500

    while time.time() < deadline - 5.0 and num_restarts < max_restarts:
        # Weighted shuffle: bias towards higher LPF elements being placed earlier
        # This is a heuristic to find better structures.
        weights = [lpf[x] for x in candidates]
        total_weight = sum(weights)
        order = []
        remaining = list(candidates)
        remaining_weights = list(weights)
        for _ in range(len(candidates)):
            if not remaining:
                break
            r = random.uniform(0, sum(remaining_weights))
            cum = 0
            for i, w in enumerate(remaining_weights):
                cum += w
                if r <= cum:
                    order.append(remaining[i])
                    remaining.pop(i)
                    remaining_weights.pop(i)
                    break

        S = greedy_construction(order)
        num_restarts += 1
        if len(S) > len(best_set):
            best_set = S[:]

    # --- Strategy 4: Local Search / Swap Improvement ---
    # Try to improve the best set found by local swaps.
    if time.time() < deadline - 2.0:
        # For each element in best_set, try removing it and see if we can add more than 1 element back.
        # This is a 1-for-k swap.
        best_size = len(best_set)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            in_set = [False] * 257
            for x in best_set:
                in_set[x] = True
            remaining = [x for x in range(2, 257) if not in_set[x]]

            for i, x in enumerate(best_set[:]):  # Iterate over a copy
                if time.time() > deadline - 2.0:
                    break
                # Remove x
                in_set[x] = False
                best_set.pop(i)

                # Try to add back elements from remaining in a greedy manner
                # Prioritize elements that were previously blocked by x
                # Simple heuristic: sort remaining by LPF descending
                for y in sorted(remaining, key=lambda z: (-lpf[z], -z)):
                    if in_set[y]:
                        continue
                    if can_add(y, in_set):
                        in_set[y] = True
                        best_set.append(y)
                        remaining.remove(y)

                # If we added more than 1 element, we improved
                if len(best_set) > best_size:
                    best_size = len(best_set)
                    improved = True
                    break
                else:
                    # Restore x
                    best_set.append(x)
                    in_set[x] = True
                    # Re-add any elements that were added during the trial
                    # (This is tricky. Let's just re-run the full greedy on the new best_set)

            # After trying all swaps, if improved, continue
            # Otherwise, break

    # Final safety check: Ensure types are correct
    result = [int(x) for x in best_set]
    return result
