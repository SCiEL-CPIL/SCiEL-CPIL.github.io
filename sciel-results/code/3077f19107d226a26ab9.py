"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 145

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if (a + b) in S_set:
                    return False
        return True

    def count_violations_fast(S_set):
        """Count violations efficiently."""
        count = 0
        S_list = list(S_set)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                if (a + b) in S_set:
                    count += 1
        return count

    def can_add_fast(S_set, x):
        """Check if x can be added without violation. O(|S|) check."""
        for a in S_set:
            if (a + x) in S_set:
                return False
            if (x - a) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True

    def greedy_from_candidates(candidates, V_set):
        """Greedily build sum-free set by trying candidates in order."""
        S_set = set()
        for x in candidates:
            if x not in S_set and can_add_fast(S_set, x):
                S_set.add(x)
        return S_set

    def local_search_1opt(S_set, V_set, max_iters=200, rng=None, deadline=None):
        """Local search: try adding elements, or removing+adding to improve."""
        if rng is None:
            rng = random.Random()
        best_size = len(S_set)
        no_improve = 0

        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            improved = False

            # Try to add elements
            candidates = [x for x in V_set if x not in S_set]
            rng.shuffle(candidates)
            for x in candidates[:50]:
                if can_add_fast(S_set, x):
                    S_set.add(x)
                    best_size = len(S_set)
                    improved = True
                    break

            # Try remove-and-add
            if not improved:
                S_list = list(S_set)
                rng.shuffle(S_list)
                for a in S_list[:20]:
                    S_set.discard(a)
                    added = False
                    candidates2 = [x for x in V_set if x not in S_set]
                    rng.shuffle(candidates2)
                    for x in candidates2[:60]:
                        if can_add_fast(S_set, x):
                            S_set.add(x)
                            added = True
                            if len(S_set) > best_size:
                                best_size = len(S_set)
                            improved = True
                            break
                    if not added:
                        S_set.add(a)

            if not improved:
                no_improve += 1
                if no_improve > 30:
                    break
            else:
                no_improve = 0

        return S_set

    def search_instance(elements, seed_offset=0, time_budget=30, deadline=None):
        """Search for a large sum-free subset of elements."""
        if deadline is None:
            deadline = start_time + DEADLINE
        rng = random.Random(seed_offset)
        V_set = set(elements)
        best = set()
        best_size = 0
        t_start = time.time()

        def try_candidate(cand):
            nonlocal best, best_size
            # Greedy repair: build sum-free set from candidate order
            S_set = greedy_from_candidates(cand, V_set)
            # Local search to improve
            S_set = local_search_1opt(S_set, V_set, max_iters=100, rng=rng, deadline=deadline)
            if len(S_set) > best_size:
                best_size = len(S_set)
                best = set(S_set)

        # Generate diverse structural candidates
        candidates = []
        elements_sorted = sorted(elements)
        n = len(elements_sorted)
        max_val = max(elements)
        min_val = min(elements)

        # 1. Odd elements (always sum-free)
        odd = [x for x in elements if x % 2 != 0]
        candidates.append(odd)

        # 2. Upper interval: (max/2, max] - sum-free
        upper_thresh = max_val / 2
        upper = [x for x in elements if x > upper_thresh]
        candidates.append(upper)

        # 3. Lower interval: [min, min/2) - sum-free (for negatives)
        if min_val < 0:
            lower_thresh = min_val / 2
            lower = [x for x in elements if x < lower_thresh]
            candidates.append(lower)

        # 4. Combination: odds in upper interval
        upper_odds = [x for x in elements if x > upper_thresh and x % 2 != 0]
        candidates.append(upper_odds)

        # 5. All positive
        pos = [x for x in elements if x > 0]
        candidates.append(pos)

        # 6. All negative
        neg = [x for x in elements if x < 0]
        candidates.append(neg)

        # 7. Top-k by value
        sorted_desc = sorted(elements, reverse=True)
        for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            k = int(n * frac)
            candidates.append(sorted_desc[:k])

        # 8. Top-k by magnitude
        sorted_mag = sorted(elements, key=lambda x: abs(x), reverse=True)
        for frac in [0.3, 0.4, 0.5, 0.6]:
            k = int(n * frac)
            candidates.append(sorted_mag[:k])

        # 9. Interval-based
        for lo_frac in [0.0, 0.1, 0.2, 0.3, 0.4]:
            for hi_frac in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if hi_frac <= lo_frac:
                    continue
                lo = int(n * lo_frac)
                hi = int(n * hi_frac)
                if hi > lo:
                    candidates.append(elements_sorted[lo:hi])

        # 10. Modular subsets
        for mod in [3, 4, 5, 6, 7, 8]:
            for rem in range(mod):
                subset = [x for x in elements if x % mod == rem]
                if len(subset) >= 3:
                    candidates.append(subset)

        # 11. High/low halves
        median_val = elements_sorted[n // 2]
        high_half = [x for x in elements if x > median_val]
        low_half = [x for x in elements if x < median_val]
        candidates.append(high_half)
        candidates.append(low_half)

        # 12. Random samples
        for _ in range(15):
            k = int(n * rng.uniform(0.3, 0.8))
            candidates.append(rng.sample(elements, k))

        # 13. Upper interval with different thresholds
        for frac in [0.3, 0.35, 0.4, 0.45, 0.5]:
            thresh = max_val * frac
            subset = [x for x in elements if x > thresh]
            candidates.append(subset)

        # 14. Lower interval with different thresholds (for negatives)
        if min_val < 0:
            for frac in [0.3, 0.35, 0.4, 0.45, 0.5]:
                thresh = min_val * frac
                subset = [x for x in elements if x < thresh]
                candidates.append(subset)

        # 15. Mixed: positive upper + negative lower
        pos_upper = [x for x in elements if x > 0 and x > max_val * 0.3]
        neg_lower = [x for x in elements if x < 0 and x < min_val * 0.3]
        if pos_upper or neg_lower:
            candidates.append(pos_upper + neg_lower)

        # 16. Odds in upper 60% by value
        sorted_by_val = sorted(elements, reverse=True)
        k60 = int(n * 0.6)
        upper60 = set(sorted_by_val[:k60])
        odds_upper60 = [x for x in elements if x in upper60 and x % 2 != 0]
        candidates.append(odds_upper60)

        # 17. Elements with |x| > threshold
        for thresh_frac in [0.3, 0.4, 0.5, 0.6]:
            thresh = 10000 * thresh_frac
            subset = [x for x in elements if abs(x) > thresh]
            candidates.append(subset)

        # 18. Positive elements above certain value
        for thresh in [1000, 2000, 3000, 4000, 5000]:
            subset = [x for x in elements if x > thresh]
            candidates.append(subset)

        # 19. Negative elements below certain value
        for thresh in [-1000, -2000, -3000, -4000, -5000]:
            subset = [x for x in elements if x < thresh]
            candidates.append(subset)

        # 20. Random with structural bias
        for _ in range(10):
            # Bias toward large magnitude
            weights = [abs(x) for x in elements]
            total_w = sum(weights)
            k = int(n * rng.uniform(0.4, 0.7))
            # Weighted sample
            sampled = []
            remaining = list(elements)
            rem_weights = list(weights)
            for _ in range(k):
                total = sum(rem_weights)
                if total == 0:
                    break
                r = rng.random() * total
                cum = 0
                for idx, w in enumerate(rem_weights):
                    cum += w
                    if cum >= r:
                        sampled.append(remaining[idx])
                        remaining.pop(idx)
                        rem_weights.pop(idx)
                        break
            candidates.append(sampled)

        # Process candidates with time awareness
        budget_check_interval = max(1, len(candidates) // 10)
        for i, cand in enumerate(candidates):
            if time.time() > deadline - 5:
                break
            if i > 0 and i % budget_check_interval == 0:
                if time.time() - t_start > time_budget * 0.6:
                    break
            try_candidate(cand)

        # Random restarts with remaining time
        restart_start = time.time()
        restart_count = 0
        while time.time() > 0 and time.time() - restart_start < time_budget * 0.35:
            if time.time() > deadline - 5:
                break
            # Mix of random and structural
            if rng.random() < 0.5:
                k = int(n * rng.uniform(0.3, 0.8))
                cand = rng.sample(elements, k)
            else:
                # Structural random
                if rng.random() < 0.5:
                    thresh = max_val * rng.uniform(0.2, 0.6)
                    cand = [x for x in elements if x > thresh]
                else:
                    thresh = min_val * rng.uniform(0.2, 0.6)
                    cand = [x for x in elements if x < thresh]
                    if not cand:
                        cand = rng.sample(elements, int(n * 0.5))
            try_candidate(cand)
            restart_count += 1
            if restart_count > 50:
                break

        return list(best)

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))

    # Allocate time budget: more time for larger instances
    # Total available: DEADLINE - 5 (safety) = 140s
    # Give: 12s, 28s, 60s = 100s, leaving buffer
    time_budgets = [12, 28, 60]

    results = []
    for i, elems in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - 5 - elapsed
        if remaining < 3:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        budget = min(time_budgets[i], max(3, remaining))
        result = search_instance(elems, seed_offset=801 + i, time_budget=budget, deadline=start_time + DEADLINE)
        results.append(result)

    # Final verification and fallback
    for i, (result, elems) in enumerate(zip(results, instances)):
        if not is_sum_free(result):
            # Fallback to odds
            results[i] = [x for x in elems if x % 2 != 0]

    return results
# EVOLVE-BLOCK-END
