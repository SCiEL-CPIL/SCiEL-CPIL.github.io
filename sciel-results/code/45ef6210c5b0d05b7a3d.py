import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    n = 5040
    target = 4979

    # Generate all divisors of 5040, sorted descending
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    div_set = set(divisors)

    # Precompute suffix sums: suffix[i] = sum of divisors[i:]
    suffix = [0] * (num_divs + 1)
    for i in range(num_divs - 1, -1, -1):
        suffix[i] = suffix[i + 1] + divisors[i]

    def feasible(k):
        """Check if there exist k distinct divisors summing to target."""
        if time.time() > deadline:
            return False
        if k < 1 or k > num_divs:
            return False
        # Max possible sum with k divisors
        if suffix[0] < target:
            return False
        # Min possible sum with k divisors (smallest k divisors)
        min_sum = suffix[num_divs - k] if num_divs >= k else 0
        if min_sum > target:
            return False
        # Max sum with k divisors (largest k divisors)
        max_sum = sum(divisors[:k])
        if max_sum < target:
            return False

        # DFS with pruning
        chosen = [0] * k
        path = []

        def dfs(idx, remaining_sum, slots_left):
            if time.time() > deadline:
                return False
            if slots_left == 0:
                return remaining_sum == 0
            if idx >= num_divs:
                return False
            if num_divs - idx < slots_left:
                return False

            # Max sum we can get with slots_left divisors from idx onwards
            # = sum of largest slots_left available from idx
            # Since divisors is sorted desc, the largest slots_left from idx
            # are divisors[idx:idx+slots_left]
            max_sum_here = suffix[idx] - suffix[idx + slots_left] if idx + slots_left <= num_divs else suffix[idx]
            if remaining_sum > max_sum_here:
                return False

            # Min sum: smallest slots_left from idx onwards
            # = divisors[num_divs - slots_left : num_divs] but must be >= idx
            # Actually, smallest slots_left from idx onwards are the last slots_left
            # that are >= idx
            if num_divs - idx >= slots_left:
                min_sum_here = suffix[num_divs - slots_left] if num_divs - slots_left >= idx else suffix[idx]
            else:
                min_sum_here = suffix[idx]
            if remaining_sum < min_sum_here:
                return False

            # Try including divisors[idx]
            if divisors[idx] <= remaining_sum:
                path.append(divisors[idx])
                if dfs(idx + 1, remaining_sum - divisors[idx], slots_left - 1):
                    return True
                path.pop()

            # Try excluding
            if num_divs - idx - 1 >= slots_left:
                if dfs(idx + 1, remaining_sum, slots_left):
                    return True

            return False

        if dfs(0, target, k):
            return True
        return False

    # Binary search for minimal k
    # Lower bound: at least ceil(target / max_divisor)
    min_k = max(1, (target + divisors[0] - 1) // divisors[0])
    # Upper bound: greedy gives us some solution
    # First get a greedy solution as upper bound
    def greedy_solve():
        rem = target
        chosen = []
        for d in divisors:
            if d <= rem:
                chosen.append(d)
                rem -= d
                if rem == 0:
                    break
        if rem == 0:
            return chosen
        return None

    greedy_sol = greedy_solve()
    if greedy_sol is None:
        # Should not happen for practical numbers, but fallback
        return [1] * target  # invalid but shouldn't happen

    max_k = len(greedy_sol)

    # Binary search for minimal feasible k
    lo = min_k
    hi = max_k
    best_k = max_k
    best_sol = greedy_sol

    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if feasible(mid):
            best_k = mid
            best_sol = greedy_sol  # placeholder, need to find actual solution
            hi = mid - 1
        else:
            lo = mid + 1

    # Now find the actual solution for best_k
    # Re-run DFS to extract the solution
    def find_solution(k):
        if time.time() > deadline:
            return None
        chosen = [0] * k
        path = []

        def dfs(idx, remaining_sum, slots_left):
            if time.time() > deadline:
                return False
            if slots_left == 0:
                return remaining_sum == 0
            if idx >= num_divs:
                return False
            if num_divs - idx < slots_left:
                return False

            max_sum_here = suffix[idx] - suffix[idx + slots_left] if idx + slots_left <= num_divs else suffix[idx]
            if remaining_sum > max_sum_here:
                return False

            if num_divs - idx >= slots_left:
                min_sum_here = suffix[num_divs - slots_left] if num_divs - slots_left >= idx else suffix[idx]
            else:
                min_sum_here = suffix[idx]
            if remaining_sum < min_sum_here:
                return False

            if divisors[idx] <= remaining_sum:
                path.append(divisors[idx])
                if dfs(idx + 1, remaining_sum - divisors[idx], slots_left - 1):
                    return True
                path.pop()

            if num_divs - idx - 1 >= slots_left:
                if dfs(idx + 1, remaining_sum, slots_left):
                    return True

            return False

        if dfs(0, target, k):
            return path[:]
        return None

    # Find solution for best_k
    sol = find_solution(best_k)
    if sol is not None and len(sol) == best_k and sum(sol) == target and len(set(sol)) == len(sol):
        return sol

    # Fallback: use greedy
    if greedy_sol and sum(greedy_sol) == target:
        return greedy_sol

    return [1] * target
