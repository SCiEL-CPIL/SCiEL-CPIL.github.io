import time
import random

def construct():
    start_time = time.time()
    deadline = 75.0  # Safety margin before 90s kill

    MAX_N = 512
    TARGET_SIZE = 48
    MAX_SUM = TARGET_SIZE * MAX_N  # 24576

    def is_valid_addition(x, selected, dp, dp_excl, n):
        """
        Check if x can be added to selected without creating any mean conflicts.
        dp[k]: bitmask of sums achievable with exactly k elements from 'selected'
        dp_excl[a][k]: bitmask of sums achievable with exactly k elements from 'selected' excluding 'a'
        """
        # Check 1: Is x the mean of any subset of 'selected' of size k >= 2?
        for k in range(2, n + 1):
            target_sum = k * x
            if target_sum > MAX_SUM:
                break
            if (dp[k] >> target_sum) & 1:
                return False

        # Check 2: For each a in selected, is a the mean of a subset of (selected \ {a}) U {x} of size >= 2?
        # A subset of (selected \ {a}) U {x} of size k >= 2 that includes x and sums to k*a
        # is equivalent to a subset of (selected \ {a}) of size k-1 >= 1 summing to k*a - x.
        # Let j = k-1. Then j >= 1 and sum = (j+1)*a - x.
        for a in selected:
            for j in range(1, n):
                target_sum = (j + 1) * a - x
                if target_sum < 0 or target_sum > MAX_SUM:
                    continue
                if (dp_excl[a][j] >> target_sum) & 1:
                    return False

        return True

    def add_element(x, selected, dp, dp_excl, n):
        """
        Add x to selected and update dp and dp_excl.
        Returns updated (selected, dp, dp_excl, n).
        """
        selected.append(x)

        # Update dp
        new_dp = [0] * (n + 2)
        new_dp[0] = 1
        for k in range(1, n + 2):
            val = dp[k] if k <= n else 0
            if k - 1 >= 0 and k - 1 <= n:
                val |= (dp[k - 1] << x)
            new_dp[k] = val
        dp = new_dp

        # Update dp_excl for existing elements
        new_dp_excl = {}
        for a in selected[:-1]:  # All except x
            old_dp_w = dp_excl[a]
            new_list = [0] * (n + 2)
            new_list[0] = 1
            for k in range(1, n + 2):
                val = old_dp_w[k] if k <= n - 1 else 0
                if k - 1 >= 0 and k - 1 <= n - 1:
                    val |= (old_dp_w[k - 1] << x)
                new_list[k] = val
            new_dp_excl[a] = new_list

        # For new element x, dp_excl[x] is the state of dp BEFORE adding x
        new_dp_excl[x] = dp[:]
        dp_excl = new_dp_excl
        n += 1

        return selected, dp, dp_excl, n

    def compute_conflict_density(x, selected, dp, n):
        """
        Compute the number of distinct subset-sum targets k*x (for k >= 2)
        that are currently achievable by the selected set.
        Lower density means fewer conflicts.
        """
        density = 0
        for k in range(2, n + 1):
            target_sum = k * x
            if target_sum > MAX_SUM:
                break
            if (dp[k] >> target_sum) & 1:
                density += 1
        return density

    def greedy_with_density(order, selected, dp, dp_excl, n, deadline):
        """
        Greedily add elements from 'order', prioritizing those with lowest conflict density.
        """
        # Build a priority queue based on conflict density
        # Since order is dynamic, we re-evaluate density at each step
        remaining = list(order)

        while remaining and n < TARGET_SIZE:
            if time.time() - start_time > deadline:
                break

            # Find the element with minimum conflict density
            best_idx = -1
            best_density = float('inf')
            for i, x in enumerate(remaining):
                density = compute_conflict_density(x, selected, dp, n)
                if density < best_density:
                    best_density = density
                    best_idx = i

            if best_idx == -1:
                break

            x = remaining.pop(best_idx)

            # Check if x is valid
            if is_valid_addition(x, selected, dp, dp_excl, n):
                selected, dp, dp_excl, n = add_element(x, selected, dp, dp_excl, n)

        return selected, dp, dp_excl, n

    best_set = []
    best_len = 0

    # Strategy 1: Ascending order
    selected, dp, dp_excl, n = [], [1] + [0] * TARGET_SIZE, {}, 0
    dp[0] = 1
    res, _, _, _ = greedy_with_density(range(1, MAX_N + 1), selected, dp, dp_excl, n, deadline)
    if len(res) > best_len:
        best_set = res
        best_len = len(res)

    # Strategy 2: Descending order
    if time.time() - start_time < deadline - 5:
        selected, dp, dp_excl, n = [], [1] + [0] * TARGET_SIZE, {}, 0
        dp[0] = 1
        res, _, _, _ = greedy_with_density(range(MAX_N, 0, -1), selected, dp, dp_excl, n, deadline)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 3: Randomized restarts with density-based greedy
    attempts = 0
    while time.time() - start_time < deadline - 2:
        attempts += 1
        seed = attempts * 104729 + 7919
        random.seed(seed)

        candidates = list(range(1, MAX_N + 1))

        # Different ordering strategies
        if attempts % 4 == 0:
            random.shuffle(candidates)
        elif attempts % 4 == 1:
            # Sort by random hash
            candidates.sort(key=lambda x: hash((x, seed)))
        elif attempts % 4 == 2:
            # Sort by modulo small prime
            p = 7 + (attempts % 5)
            candidates.sort(key=lambda x: (x % p, x))
        else:
            # Sort by value
            candidates.sort()

        selected, dp, dp_excl, n = [], [1] + [0] * TARGET_SIZE, {}, 0
        dp[0] = 1
        res, _, _, _ = greedy_with_density(candidates, selected, dp, dp_excl, n, deadline)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= TARGET_SIZE:
                break

    return best_set
