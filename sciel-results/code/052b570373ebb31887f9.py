"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import Counter, defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def can_add(S_set, x):
        """Check if x can be added to sum-free set S without violation.
        Violation if: exists a in S such that a+x in S, or x-a in S, or 2x in S.
        """
        for a in S_set:
            if (a + x) in S_set:
                return False
            if (x - a) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True

    def is_sum_free(S):
        """Verify S is sum-free."""
        S_set = set(S)
        for a in S:
            for b in S:
                if (a + b) in S_set:
                    return False
        return True

    def violation_count(S_set):
        """Count total number of violating triples (a,b,c) with a+b=c in S."""
        count = 0
        S_list = list(S_set)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                c = a + b
                if c in S_set:
                    count += 1
        return count

    def element_violation_scores(S_set):
        """For each element, count how many violations it participates in.
        Returns dict: element -> count.
        """
        scores = defaultdict(int)
        S_list = list(S_set)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                c = a + b
                if c in S_set:
                    scores[a] += 1
                    scores[b] += 1
                    scores[c] += 1
        return dict(scores)

    def smooth_surrogate_score(S_set, V_set):
        """Smooth surrogate: for each element in V, compute a soft score based on
        how 'free' it is (fewer potential violations = higher score).
        Uses soft-max style: score = -log(1 + number_of_conflicts).
        """
        scores = {}
        S_list = list(S_set)
        for x in V_set:
            if x in S_set:
                continue
            # Count how many elements in S would conflict with x
            conflicts = 0
            for a in S_list:
                if (a + x) in S_set:
                    conflicts += 1
                if (x - a) in S_set:
                    conflicts += 1
            if (2 * x) in S_set:
                conflicts += 1
            # Soft score: negative log of (1 + conflicts)
            scores[x] = -math.log1p(conflicts)
        return scores

    def greedy_expand(S_list, V_set, order=None):
        """Add elements from V to S in the given order."""
        S_set = set(S_list)
        if order is None:
            order = sorted(V_set - S_set, key=lambda x: abs(x), reverse=True)
        for x in order:
            if x not in S_set and can_add(S_set, x):
                S_set.add(x)
        return list(S_set)

    def repair(S_list, V_set):
        """Remove elements to fix violations, then greedily expand."""
        S_set = set(S_list)

        # Phase 1: Remove elements causing most violations (weighted)
        changed = True
        while changed:
            changed = False
            scores = element_violation_scores(S_set)
            if scores:
                worst = max(scores, key=scores.get)
                S_set.discard(worst)
                changed = True

        # Phase 2: Greedy expand
        S_list = list(S_set)
        S_list = greedy_expand(S_list, V_set - S_set)
        return S_list

    def walksat_search(S_list, V_set, max_iters=1000, time_budget=30, seed=0):
        """WalkSAT-style local search for sum-free set.
        
        Each violating triple (a,b,c) with a+b=c is a clause that should be 'satisfied'
        (i.e., at least one of a,b,c is removed). We use clause-weighted random
        restarts to escape local optima.
        
        Objective: maximize |S| subject to no violations.
        Surrogate: minimize total violation weight, then maximize size.
        """
        rng = random.Random(seed)
        S_set = set(S_list)
        V_set = set(V_set)
        
        # Ensure start is valid
        if not is_sum_free(S_set):
            S_set = set(repair(list(S_set), V_set))
        
        best_size = len(S_set)
        best_set = set(S_set)
        t_start = time.time()
        no_improve = 0
        clause_weights = defaultdict(float)
        
        for it in range(max_iters):
            if time.time() - t_start > time_budget:
                break
            
            # Current violations
            S_list = list(S_set)
            n = len(S_list)
            violations = []
            for i in range(n):
                a = S_list[i]
                for j in range(i, n):
                    b = S_list[j]
                    c = a + b
                    if c in S_set:
                        violations.append((a, b, c))
            
            if not violations:
                # No violations: try to add more elements
                candidates = [x for x in V_set if x not in S_set]
                rng.shuffle(candidates)
                added = False
                for x in candidates[:50]:
                    if can_add(S_set, x):
                        S_set.add(x)
                        added = True
                        if len(S_set) > best_size:
                            best_size = len(S_set)
                            best_set = set(S_set)
                if added:
                    no_improve = 0
                    continue
                else:
                    # Local optimum: try a random restart with perturbation
                    no_improve += 1
                    if no_improve > 30:
                        # Perturb: remove 2-3 random elements, then re-expand
                        S_list = list(S_set)
                        rng.shuffle(S_list)
                        for _ in range(rng.randint(2, 3)):
                            if S_list:
                                S_set.discard(S_list.pop())
                        S_set = set(greedy_expand(list(S_set), V_set - S_set))
                        no_improve = 0
                    continue
            
            # Has violations: use WalkSAT-style move
            # Weight clauses by frequency
            for (a, b, c) in violations:
                clause_weights[(a, b, c)] += 1.0
            
            # Pick a violating triple with probability proportional to weight
            total_w = sum(clause_weights.values())
            r = rng.random() * total_w
            cumulative = 0
            chosen = violations[0]
            for (a, b, c) in violations:
                cumulative += clause_weights[(a, b, c)]
                if cumulative >= r:
                    chosen = (a, b, c)
                    break
            
            # Decide which element to remove from the chosen triple
            # Use element violation scores to guide the choice
            scores = element_violation_scores(S_set)
            a, b, c = chosen
            # Remove the element with the highest violation score
            candidates_remove = [a, b, c]
            candidates_remove.sort(key=lambda x: scores.get(x, 0), reverse=True)
            
            # With some probability, pick randomly to escape local optima
            if rng.random() < 0.3:
                to_remove = rng.choice(candidates_remove)
            else:
                to_remove = candidates_remove[0]
            
            S_set.discard(to_remove)
            
            # After removal, try to greedily add back some elements
            candidates = [x for x in V_set if x not in S_set]
            rng.shuffle(candidates)
            for x in candidates[:30]:
                if can_add(S_set, x):
                    S_set.add(x)
            
            if len(S_set) > best_size:
                best_size = len(S_set)
                best_set = set(S_set)
                no_improve = 0
            else:
                no_improve += 1
            
            # Decay clause weights to allow exploration
            for key in list(clause_weights.keys()):
                clause_weights[key] *= 0.95
                if clause_weights[key] < 0.01:
                    del clause_weights[key]
        
        # Final polish pass: high-precision local search
        S_set = set(best_set)
        S_set = polish_pass(S_set, V_set, time_budget=5, seed=seed + 1000)
        
        return list(S_set)

    def polish_pass(S_set, V_set, time_budget=5, seed=0):
        """High-precision polish pass: tighter tolerance, more iterations.
        Tries single-element swaps and double-element swaps with strict improvement.
        """
        rng = random.Random(seed)
        S_set = set(S_set)
        V_set = set(V_set)
        best_size = len(S_set)
        best_set = set(S_set)
        t_start = time.time()
        
        for it in range(200):
            if time.time() - t_start > time_budget:
                break
            
            improved = False
            
            # Try single-element removal + re-expansion
            S_list = list(S_set)
            rng.shuffle(S_list)
            for a in S_list[:10]:
                S_set.discard(a)
                temp = set(S_set)
                candidates = [x for x in V_set if x not in temp]
                rng.shuffle(candidates)
                for x in candidates[:50]:
                    if can_add(temp, x):
                        temp.add(x)
                if len(temp) > best_size:
                    S_set = temp
                    best_size = len(S_set)
                    best_set = set(S_set)
                    improved = True
                    break
                else:
                    S_set.add(a)  # restore
            
            if improved:
                continue
            
            # Try double-element swap
            S_list = list(S_set)
            if len(S_list) >= 2:
                rng.shuffle(S_list)
                for i in range(min(8, len(S_list))):
                    a = S_list[i]
                    for j in range(i + 1, min(i + 8, len(S_list))):
                        b = S_list[j]
                        S_set.discard(a)
                        S_set.discard(b)
                        temp = set(S_set)
                        candidates = [x for x in V_set if x not in temp]
                        rng.shuffle(candidates)
                        added = 0
                        for x in candidates[:60]:
                            if can_add(temp, x):
                                temp.add(x)
                                added += 1
                                if added >= 2:
                                    break
                        if len(temp) > best_size:
                            S_set = temp
                            best_size = len(S_set)
                            best_set = set(S_set)
                            improved = True
                            break
                        else:
                            S_set.add(a)
                            S_set.add(b)
                    if improved:
                        break
            
            if not improved:
                break
        
        return list(best_set)

    def local_search(S_list, V_set, max_iters=500, time_budget=30, seed=0):
        """Combined local search: WalkSAT + polish."""
        result = walksat_search(S_list, V_set, max_iters=max_iters, time_budget=time_budget, seed=seed)
        return result

    def search_instance(elements, seed_offset=0, time_budget=30):
        """Search for a large sum-free subset of elements."""
        rng = random.Random(seed_offset)
        best = []
        best_size = 0
        t_start = time.time()
        elements_set = set(elements)

        def try_candidate(cand, seed=0):
            nonlocal best, best_size
            if time.time() - t_start > time_budget - 2:
                return
            S = repair(cand, elements_set)
            remaining = max(0, time_budget - (time.time() - t_start) - 2)
            S = local_search(S, elements_set, max_iters=300, time_budget=remaining, seed=seed)
            if len(S) > best_size:
                best_size = len(S)
                best = list(S)

        # Generate diverse initial candidates
        candidates = []

        # 1. Odds (guaranteed sum-free)
        odd = [x for x in elements if x % 2 != 0]
        candidates.append(odd)

        # 2. All positive
        pos = [x for x in elements if x > 0]
        candidates.append(pos)

        # 3. All negative
        neg = [x for x in elements if x < 0]
        candidates.append(neg)

        # 4. Top-k by value (largest)
        sorted_desc = sorted(elements, reverse=True)
        for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            k = int(len(elements) * frac)
            candidates.append(sorted_desc[:k])

        # 5. Top-k by magnitude
        sorted_mag = sorted(elements, key=lambda x: abs(x), reverse=True)
        for frac in [0.3, 0.4, 0.5, 0.6]:
            k = int(len(elements) * frac)
            candidates.append(sorted_mag[:k])

        # 6. Interval-based: elements in [L, R]
        elements_sorted = sorted(elements)
        n = len(elements_sorted)
        for lo_frac in [0.0, 0.1, 0.2, 0.3]:
            for hi_frac in [0.5, 0.7, 0.9, 1.0]:
                lo = int(n * lo_frac)
                hi = int(n * hi_frac)
                if hi > lo and hi - lo >= 5:
                    candidates.append(elements_sorted[lo:hi])

        # 7. Random samples with varying sizes
        for _ in range(12):
            k = int(len(elements) * rng.uniform(0.2, 0.9))
            candidates.append(rng.sample(elements, k))

        # 8. Mixed: some high, some low
        for _ in range(4):
            k_high = int(len(elements) * rng.uniform(0.1, 0.3))
            k_low = int(len(elements) * rng.uniform(0.1, 0.3))
            high = sorted_desc[:k_high]
            low = sorted_desc[-k_low:]
            candidates.append(high + low)

        # Process each candidate
        for idx, cand in enumerate(candidates):
            if time.time() - t_start > time_budget - 3:
                break
            try_candidate(cand, seed=seed_offset + idx)

        # Additional random restarts if time permits
        restart_count = 0
        while time.time() - t_start < time_budget - 3:
            k = int(len(elements) * rng.uniform(0.2, 0.9))
            cand = rng.sample(elements, k)
            try_candidate(cand, seed=seed_offset + 100 + restart_count)
            restart_count += 1
            if restart_count > 60:
                break

        # Final sanity check on best
        if best and not is_sum_free(set(best)):
            best = repair(best, elements_set)
            best = polish_pass(best, elements_set, time_budget=3, seed=seed_offset + 999)
            
        return best

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))

    results = []
    # Allocate time budget: n=60 gets 25s, n=120 gets 40s, n=240 gets 55s (total 120s, leaving 25s margin)
    time_budgets = [25, 40, 55]
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        result = search_instance(elems, seed_offset=801 + i, time_budget=time_budgets[i])
        results.append(result)

    return results
# EVOLVE-BLOCK-END
