import time
from math import lcm
from fractions import Fraction
import bisect
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = [L // d for d in denominators]

    def check_time():
        return time.time() > deadline

    def validate(indices):
        """Check if indices are valid: nonempty, distinct, in range, sum < L"""
        if not indices:
            return False, 0
        if len(set(indices)) != len(indices):
            return False, 0
        if any(i < 0 or i >= n for i in indices):
            return False, 0
        s = sum(weights[i] for i in indices)
        if s >= L:
            return False, 0
        return True, s

    best_sum = -1
    best_indices = None

    def update_best(indices):
        nonlocal best_sum, best_indices
        valid, s = validate(indices)
        if valid and s > best_sum:
            best_sum = s
            best_indices = list(indices)
            return True
        return False

    # ============================================================
    # STRATEGY 1: Meet-in-the-middle (split into 2 halves)
    # ============================================================
    mid = n // 2  # 24
    half1 = list(range(mid))
    half2 = list(range(mid, n))

    w1 = [weights[i] for i in half1]
    w2 = [weights[i] for i in half2]

    def gen_subset_sums(weights_list, max_count=None):
        """Generate all subset sums < L with their masks"""
        sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            if check_time():
                break
            bit = 1 << i
            current_len = len(sums)
            for j in range(current_len):
                s, m = sums[j]
                ns = s + w
                if ns < L:
                    sums.append((ns, m | bit))
        return sums

    print("Strategy 1: Meet-in-the-middle...")
    sums1 = gen_subset_sums(w1)
    print(f"  Half 1: {len(sums1)} subsets")

    if not check_time():
        sums2 = gen_subset_sums(w2)
        print(f"  Half 2: {len(sums2)} subsets")

        if not check_time():
            # Sort sums2 by value for binary search
            sums2.sort(key=lambda x: x[0])
            sums2_vals = [s[0] for s in sums2]

            for s1, m1 in sums1:
                if check_time():
                    break
                target = L - s1 - 1  # We need s2 <= target for strict < L
                idx = bisect.bisect_right(sums2_vals, target)
                if idx > 0:
                    s2, m2 = sums2[idx - 1]
                    total = s1 + s2
                    if total > best_sum:
                        best_sum = total
                        # Convert masks to global indices
                        indices = []
                        for i in range(mid):
                            if m1 & (1 << i):
                                indices.append(i)
                        for i in range(mid, n):
                            if m2 & (1 << (i - mid)):
                                indices.append(i)
                        best_indices = indices

    # ============================================================
    # STRATEGY 2: Branch and Bound ILP solver
    # Maximize sum(w_i * x_i) subject to sum(w_i * x_i) <= L-1
    # ============================================================
    if not check_time():
        print("Strategy 2: Branch and Bound...")

        # Sort items by weight descending for better pruning
        order = sorted(range(n), key=lambda i: -weights[i])
        sorted_weights = [weights[i] for i in order]

        # Suffix sums for upper bound
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + sorted_weights[i]

        # Branch and bound
        def branch_and_bound():
            nonlocal best_sum, best_indices

            # State: (index, current_sum, chosen_mask)
            # Use iterative DFS to avoid recursion limits
            stack = [(0, 0, 0)]
            visited = set()

            while stack:
                if check_time():
                    break

                idx, cur_sum, mask = stack.pop()

                # Upper bound: current sum + all remaining weights
                ub = cur_sum + suffix[idx]
                if ub <= best_sum:
                    continue

                # If we've processed all items
                if idx == n:
                    if cur_sum > best_sum and cur_sum < L:
                        best_sum = cur_sum
                        indices = []
                        for i in range(n):
                            if mask & (1 << i):
                                indices.append(order[i])
                        best_indices = indices
                    continue

                # Option 1: Include item idx
                new_sum = cur_sum + sorted_weights[idx]
                if new_sum < L:
                    # Upper bound check
                    if new_sum + suffix[idx + 1] > best_sum:
                        new_mask = mask | (1 << idx)
                        state = (idx + 1, new_sum, new_mask)
                        if state not in visited:
                            visited.add(state)
                            stack.append(state)

                # Option 2: Exclude item idx
                # Upper bound: current sum + remaining weights
                if cur_sum + suffix[idx + 1] > best_sum:
                    state = (idx + 1, cur_sum, mask)
                    if state not in visited:
                        visited.add(state)
                        stack.append(state)

        branch_and_bound()

    # ============================================================
    # STRATEGY 3: Greedy + local search
    # ============================================================
    if not check_time():
        print("Strategy 3: Greedy + local search...")

        # Greedy: add smallest denominators that keep sum < L
        # Sort by weight descending (largest reciprocals first)
        order = sorted(range(n), key=lambda i: -weights[i])

        def greedy_add():
            """Greedy: add items in order if they fit"""
            s = 0
            chosen = []
            for i in order:
                if s + weights[i] < L:
                    s += weights[i]
                    chosen.append(i)
            return chosen, s

        def greedy_small():
            """Greedy: add items in order, prefer smaller weights (more items)"""
            order_small = sorted(range(n), key=lambda i: weights[i])
            s = 0
            chosen = []
            for i in order_small:
                if s + weights[i] < L:
                    s += weights[i]
                    chosen.append(i)
            return chosen, s

        # Try several greedy variants
        for _ in range(50):
            if check_time():
                break

            # Variant: random shuffle + greedy
            shuffled = order[:]
            random.shuffle(shuffled)
            s = 0
            chosen = []
            for i in shuffled:
                if s + weights[i] < L:
                    s += weights[i]
                    chosen.append(i)
            update_best(chosen)

        # Variant: reverse greedy (remove items to get close to L)
        for _ in range(50):
            if check_time():
                break
            shuffled = order[:]
            random.shuffle(shuffled)
            s = 0
            chosen = []
            for i in shuffled:
                if s + weights[i] < L:
                    s += weights[i]
                    chosen.append(i)
            update_best(chosen)

        # Local search: swap items to improve
        if best_indices is not None:
            for iteration in range(2000):
                if check_time():
                    break
                if not best_indices:
                    break

                chosen = set(best_indices)
                s = sum(weights[i] for i in chosen)

                # Try adding one item
                improved = False
                candidates_add = [i for i in range(n) if i not in chosen and s + weights[i] < L]
                if candidates_add:
                    # Pick the one that gets closest to L
                    best_add = max(candidates_add, key=lambda i: s + weights[i])
                    new_s = s + weights[best_add]
                    if new_s > best_sum:
                        chosen.add(best_add)
                        best_indices = list(chosen)
                        best_sum = new_s
                        improved = True

                if not improved:
                    # Try removing one item and adding a different one
                    candidates_remove = list(chosen)
                    random.shuffle(candidates_remove)
                    found = False
                    for rem in candidates_remove:
                        if found or check_time():
                            break
                        s_after = s - weights[rem]
                        candidates_add2 = [i for i in range(n)
                                           if i not in chosen and i != rem
                                           and s_after + weights[i] < L]
                        if candidates_add2:
                            best_add2 = max(candidates_add2, key=lambda i: s_after + weights[i])
                            new_s = s_after + weights[best_add2]
                            if new_s > best_sum:
                                chosen.discard(rem)
                                chosen.add(best_add2)
                                best_indices = list(chosen)
                                best_sum = new_s
                                found = True
                                break

                if not improved and not found:
                    # Try removing one item (if sum is already close)
                    if len(chosen) > 1:
                        for rem in list(chosen):
                            if check_time():
                                break
                            new_s = s - weights[rem]
                            if new_s > best_sum and new_s < L:
                                chosen.discard(rem)
                                best_indices = list(chosen)
                                best_sum = new_s
                                break

    # ============================================================
    # STRATEGY 4: Randomized subset sampling
    # ============================================================
    if not check_time():
        print("Strategy 4: Randomized sampling...")

        # Sample random subsets of various sizes
        for _ in range(50000):
            if check_time():
                break

            # Random size between 1 and n
            size = random.randint(1, n)
            subset = random.sample(range(n), size)
            s = sum(weights[i] for i in subset)
            if s < L and s > best_sum:
                best_sum = s
                best_indices = list(subset)

    # ============================================================
    # STRATEGY 5: Targeted search around known good solutions
    # ============================================================
    if not check_time() and best_indices is not None:
        print("Strategy 5: Targeted search...")

        for iteration in range(10000):
            if check_time():
                break

            # Start from current best, make small random changes
            chosen = set(best_indices)
            s = best_sum

            # Randomly remove 1-3 items and add 1-3 different items
            n_remove = random.randint(1, min(3, len(chosen)))
            n_add = random.randint(1, 3)

            to_remove = random.sample(list(chosen), n_remove)
            available = [i for i in range(n) if i not in chosen]
            if len(available) < n_add:
                continue
            to_add = random.sample(available, n_add)

            new_chosen = chosen.copy()
            for r in to_remove:
                new_chosen.discard(r)
            for a in to_add:
                new_chosen.add(a)

            new_s = sum(weights[i] for i in new_chosen)
            if new_s < L and new_s > best_sum:
                best_sum = new_s
                best_indices = list(new_chosen)

    # ============================================================
    # STRATEGY 6: Systematic single-item swaps from best
    # ============================================================
    if not check_time() and best_indices is not None:
        print("Strategy 6: Systematic swaps...")

        chosen = set(best_indices)
        s = best_sum

        # For each item in chosen, try replacing with every item not in chosen
        for rem in list(chosen):
            if check_time():
                break
            s_without = s - weights[rem]
            for add in range(n):
                if add in chosen:
                    continue
                new_s = s_without + weights[add]
                if new_s < L and new_s > best_sum:
                    best_sum = new_s
                    new_chosen = chosen.copy()
                    new_chosen.discard(rem)
                    new_chosen.add(add)
                    best_indices = list(new_chosen)
                    chosen = new_chosen
                    s = new_s

    # ============================================================
    # STRATEGY 7: Exhaustive small subsets
    # ============================================================
    if not check_time():
        print("Strategy 7: Exhaustive small subsets...")

        # Try all subsets of size 1-4
        for size in range(1, 5):
            if check_time():
                break
            for combo in itertools.combinations(range(n), size):
                s = sum(weights[i] for i in combo)
                if s < L and s > best_sum:
                    best_sum = s
                    best_indices = list(combo)

    # ============================================================
    # STRATEGY 8: Weighted random hill climbing
    # ============================================================
    if not check_time() and best_indices is not None:
        print("Strategy 8: Hill climbing...")

        for iteration in range(20000):
            if check_time():
                break

            chosen = set(best_indices)
            s = best_sum

            # Random move: add, remove, or swap
            move = random.choice(['add', 'remove', 'swap'])

            if move == 'add':
                candidates = [i for i in range(n) if i not in chosen and s + weights[i] < L]
                if candidates:
                    # Weighted by how close to L
                    max_w = max(s + weights[i] for i in candidates)
                    min_w = min(s + weights[i] for i in candidates)
                    # Bias toward larger sums
                    best_add = random.choice(candidates[:max(1, len(candidates)//3)])
                    new_s = s + weights[best_add]
                    if new_s > best_sum:
                        chosen.add(best_add)
                        best_sum = new_s
                        best_indices = list(chosen)

            elif move == 'remove' and len(chosen) > 1:
                rem = random.choice(list(chosen))
                new_s = s - weights[rem]
                if new_s > best_sum and new_s < L:
                    chosen.discard(rem)
                    best_sum = new_s
                    best_indices = list(chosen)

            elif move == 'swap' and len(chosen) > 0:
                rem = random.choice(list(chosen))
                available = [i for i in range(n) if i not in chosen]
                if available:
                    add = random.choice(available)
                    new_s = s - weights[rem] + weights[add]
                    if new_s < L and new_s > best_sum:
                        chosen.discard(rem)
                        chosen.add(add)
                        best_sum = new_s
                        best_indices = list(chosen)

    # ============================================================
    # STRATEGY 9: Final polish - try all single modifications
    # ============================================================
    if not check_time() and best_indices is not None:
        print("Strategy 9: Final polish...")

        for _ in range(5):
            if check_time():
                break
            changed = False
            chosen = set(best_indices)
            s = best_sum

            # Try adding any single item
            for i in range(n):
                if check_time():
                    break
                if i not in chosen and s + weights[i] < L and s + weights[i] > best_sum:
                    chosen.add(i)
                    best_sum = s + weights[i]
                    best_indices = list(chosen)
                    s = best_sum
                    changed = True

            # Try removing any single item
            for i in list(chosen):
                if check_time():
                    break
                if len(chosen) > 1:
                    new_s = s - weights[i]
                    if new_s > best_sum and new_s < L:
                        chosen.discard(i)
                        best_sum = new_s
                        best_indices = list(chosen)
                        s = new_s
                        changed = True

            if not changed:
                break

    # Final validation
    if best_indices is None or not validate(best_indices)[0]:
        # Fallback: find any valid subset
        # Try single items
        for i in range(n):
            if weights[i] < L:
                best_indices = [i]
                best_sum = weights[i]
                break
        if best_indices is None:
            best_indices = [0]

    print(f"Final best sum: {best_sum}/{L}, deficit: {L - best_sum}/{L}")
    print(f"Best indices: {best_indices}")
    print(f"Number of items: {len(best_indices)}")

    return best_indices
