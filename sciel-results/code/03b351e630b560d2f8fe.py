import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)

    weights = [L // d for d in denominators]

    def check_time():
        return time.time() > deadline

    def validate(indices):
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
    # STRATEGY 1: Branch and Bound DFS
    # ============================================================
    # Sort indices by weight descending (larger reciprocals first)
    # This helps find good solutions early for pruning
    order = sorted(range(n), key=lambda i: -weights[i])
    search_weights = [weights[i] for i in order]

    # Precompute suffix sums for upper bound pruning
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + search_weights[i]

    # Iterative DFS with explicit stack
    # Stack items: (idx, current_sum, chosen_mask)
    # Using mask to track which original indices are chosen
    stack = [(0, 0, 0)]

    while stack:
        if check_time():
            break

        idx, current_sum, mask = stack.pop()

        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                # Reconstruct indices from mask
                indices = []
                for i in range(n):
                    if mask & (1 << i):
                        indices.append(order[i])
                best_indices = indices
            continue

        # Pruning: upper bound check
        if current_sum + suffix_sums[idx] <= best_sum:
            continue

        # Option 1: Exclude current item
        stack.append((idx + 1, current_sum, mask))

        # Option 2: Include current item (if valid)
        new_sum = current_sum + search_weights[idx]
        if new_sum < L:
            stack.append((idx + 1, new_sum, mask | (1 << idx)))

    # ============================================================
    # STRATEGY 2: Local Search Polish
    # ============================================================
    if not check_time() and best_indices is not None:
        # Try to improve by swapping or adding/removing elements
        chosen_set = set(best_indices)
        improved = True
        while improved and not check_time():
            improved = False

            # Try adding one element
            for i in range(n):
                if i not in chosen_set:
                    new_indices = list(chosen_set) + [i]
                    s = sum(weights[j] for j in new_indices)
                    if s < L and s > best_sum:
                        best_sum = s
                        best_indices = new_indices
                        chosen_set = set(new_indices)
                        improved = True
                        break

            # Try swapping one element
            for i in list(chosen_set):
                for j in range(n):
                    if j not in chosen_set:
                        new_indices = [k for k in chosen_set if k != i] + [j]
                        s = sum(weights[k] for k in new_indices)
                        if s < L and s > best_sum:
                            best_sum = s
                            best_indices = new_indices
                            chosen_set = set(new_indices)
                            improved = True
                            break
                if improved:
                    break

            # Try removing one element (might allow adding a better one)
            for i in list(chosen_set):
                new_indices = [k for k in chosen_set if k != i]
                s = sum(weights[k] for k in new_indices)
                if s < L and s > best_sum:
                    best_sum = s
                    best_indices = new_indices
                    chosen_set = set(new_indices)
                    improved = True
                    break

    # ============================================================
    # STRATEGY 3: Targeted Greedy with Different Orderings
    # ============================================================
    if not check_time():
        # Try different greedy orderings
        orderings = [
            sorted(range(n), key=lambda i: -weights[i]),  # Descending weight
            sorted(range(n), key=lambda i: weights[i]),   # Ascending weight
            sorted(range(n), key=lambda i: -weights[i] / (i + 1)),  # Weight per position
        ]

        for ordering in orderings:
            if check_time():
                break
            s = 0
            chosen = []
            for i in ordering:
                if s + weights[i] < L:
                    s += weights[i]
                    chosen.append(i)
            if s > best_sum:
                best_sum = s
                best_indices = chosen

    # ============================================================
    # STRATEGY 4: Randomized Local Search
    # ============================================================
    if not check_time():
        import random
        random.seed(42)

        for _ in range(1000):
            if check_time():
                break

            # Random starting point
            s = 0
            chosen = []
            for i in range(n):
                if random.random() < 0.3:  # 30% chance to include
                    if s + weights[i] < L:
                        s += weights[i]
                        chosen.append(i)

            if s > best_sum:
                best_sum = s
                best_indices = list(chosen)

            # Local improvement
            chosen_set = set(best_indices)
            for _ in range(50):
                if check_time():
                    break
                improved = False

                # Try adding
                for i in range(n):
                    if i not in chosen_set:
                        new_indices = list(chosen_set) + [i]
                        s = sum(weights[j] for j in new_indices)
                        if s < L and s > best_sum:
                            best_sum = s
                            best_indices = new_indices
                            chosen_set = set(new_indices)
                            improved = True
                            break

                # Try swapping
                for i in list(chosen_set):
                    for j in range(n):
                        if j not in chosen_set:
                            new_indices = [k for k in chosen_set if k != i] + [j]
                            s = sum(weights[k] for k in new_indices)
                            if s < L and s > best_sum:
                                best_sum = s
                                best_indices = new_indices
                                chosen_set = set(new_indices)
                                improved = True
                                break
                    if improved:
                        break

                if not improved:
                    break

    # Final validation and return
    if best_indices is not None:
        valid, s = validate(best_indices)
        if valid:
            return best_indices

    # Fallback: return a valid subset
    # Greedy fallback
    s = 0
    chosen = []
    for i in range(n):
        if s + weights[i] < L:
            s += weights[i]
            chosen.append(i)
    return chosen
