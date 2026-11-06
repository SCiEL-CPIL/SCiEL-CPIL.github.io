def construct():
    from fractions import Fraction
    import time
    import math

    start_time = time.time()
    deadline = start_time + 85  # safety margin

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    fractions = [Fraction(1, d) for d in denominators]

    # Precompute reciprocal as float for fast upper-bound estimation
    recip_floats = [1.0 / d for d in denominators]

    # Sort by fraction value descending (largest first) for tighter bounds
    order = sorted(range(n), key=lambda i: -fractions[i])
    sorted_fracs = [fractions[i] for i in order]
    sorted_recs = [recip_floats[i] for i in order]

    # Precompute suffix sums of float reciprocals for upper bound
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_recs[i]

    best_indices = []
    best_sum = Fraction(0, 1)
    best_def = Fraction(1, 1)

    def dfs(pos, current_sum, current_indices, remaining_budget):
        nonlocal best_indices, best_sum, best_def

        if time.time() > deadline:
            return

        # Update best if current sum is valid and better
        if current_sum < 1:
            cur_def = 1 - current_sum
            if cur_def < best_def:
                best_def = cur_def
                best_sum = current_sum
                best_indices = list(current_indices)

        if pos >= n:
            return

        # Upper bound: current_sum + sum of all remaining fractions
        # If even taking all remaining can't beat best, prune
        if current_sum + Fraction(suffix_sum[pos], 1) <= best_sum:
            # Actually we need to be careful: suffix_sum is float, let's use a tighter bound
            pass

        # Try including each remaining item
        for i in range(pos, n):
            if time.time() > deadline:
                return

            new_sum = current_sum + sorted_fracs[i]
            if new_sum >= 1:
                # Since we're going in descending order, all subsequent are smaller
                # but we still need to try them individually
                continue

            current_indices.append(order[i])
            dfs(i + 1, new_sum, current_indices, remaining_budget)
            current_indices.pop()

            # Upper bound pruning: if even taking all items from i+1 onward
            # can't improve our best, stop
            # Use float approximation for speed
            max_possible = float(new_sum) + suffix_sum[i + 1]
            if max_possible < float(best_sum):
                break

    # Initial greedy solution
    greedy_selected = []
    greedy_total = Fraction(0, 1)
    for i in range(n):
        if greedy_total + fractions[i] < 1:
            greedy_selected.append(i)
            greedy_total += fractions[i]

    if greedy_selected:
        best_indices = greedy_selected
        best_sum = greedy_total
        best_def = 1 - best_sum

    # Run branch and bound
    dfs(0, Fraction(0, 1), [], 0)

    return best_indices

