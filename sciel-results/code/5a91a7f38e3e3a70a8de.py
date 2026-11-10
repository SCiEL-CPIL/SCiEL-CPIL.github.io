"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import factorial, gcd
    from itertools import product
    from functools import lru_cache
    import sys
    sys.setrecursionlimit(10000)

    target = factorial(12)  # 479001600

    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]

    # Generate all integers > 12 that divide 12!
    # We can do this by generating all divisors of 12!
    # Number of divisors = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792
    # That's manageable.

    # Generate all divisors
    def gen_divisors(exp_bounds, idx=0, current=1):
        if idx == len(exp_bounds):
            return [current]
        result = []
        for e in range(exp_bounds[idx] + 1):
            result.extend(gen_divisors(exp_bounds, idx + 1, current * primes[idx] ** e))
        return result

    all_divisors = gen_divisors(target_exp)
    # Filter: > 12
    candidates = [d for d in all_divisors if d > 12]

    # Sort candidates in ascending order
    candidates.sort()

    # For each candidate, compute its exponent vector
    def exp_vector(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)

    cand_vecs = [exp_vector(c) for c in candidates]

    # We need to find a subset of distinct candidates whose exponent vectors sum to target_exp
    # and minimize the maximum candidate in the subset.
    #
    # Strategy: binary search on the maximum factor value M.
    # For a given M, check if there exists a subset of candidates (all <= M) with distinct values
    # whose exponent vectors sum to target_exp.
    #
    # Since the number of candidates is at most 791, and we need subset sum over 5D vectors,
    # we can use DP. The state is the exponent vector (tuple of 5 ints).
    # The number of possible states is (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 792.
    # For each state, we track whether it's reachable.
    #
    # But we also need distinct factors. Since each factor is used at most once, and we're doing
    # subset sum, we process each candidate once (0/1 knapsack style).
    #
    # DP: dp[state] = True if state is reachable using some subset of the first i candidates.
    # We iterate over candidates, and for each, update dp in reverse to avoid using the same candidate twice.

    # Precompute all possible states as a set for quick lookup
    # State is a tuple of 5 ints, each within bounds
    all_states = set()
    for e0 in range(11):
        for e1 in range(6):
            for e2 in range(3):
                for e3 in range(2):
                    for e4 in range(2):
                        all_states.add((e0, e1, e2, e3, e4))

    target_state = tuple(target_exp)

    def can_achieve(max_val):
        """Check if we can achieve target_state using distinct candidates all <= max_val."""
        # Get candidates <= max_val
        # Find the index up to which candidates are <= max_val
        # candidates is sorted, so use bisect
        import bisect
        idx = bisect.bisect_right(candidates, max_val)
        if idx == 0:
            return False

        # DP: set of reachable states
        reachable = {tuple([0] * 5)}

        for i in range(idx):
            cv = cand_vecs[i]
            # For each currently reachable state, try adding cv
            new_reachable = set()
            for state in reachable:
                new_state = tuple(state[j] + cv[j] for j in range(5))
                # Check if new_state is within bounds
                if all(new_state[j] <= target_exp[j] for j in range(5)):
                    new_reachable.add(new_state)
            reachable = reachable | new_reachable

            if target_state in reachable:
                return True

        return target_state in reachable

    # Binary search on max_val
    # Lower bound: at least 13 (smallest candidate > 12)
    # Upper bound: 12! itself
    lo = 13
    hi = target

    # But we can be smarter: the answer must be one of the candidate values
    # Let's just binary search over the sorted candidates
    # Actually, let's do a linear scan from smallest candidate upward and check feasibility
    # But that could be slow. Binary search is better.

    # Let's find the minimum max_val such that can_achieve(max_val) is True
    # We'll binary search over the candidate values

    # Get unique candidate values
    unique_vals = sorted(set(candidates))

    lo_idx = 0
    hi_idx = len(unique_vals) - 1
    best = None

    while lo_idx <= hi_idx:
        mid_idx = (lo_idx + hi_idx) // 2
        mid_val = unique_vals[mid_idx]
        if can_achieve(mid_val):
            best = mid_val
            hi_idx = mid_idx - 1
        else:
            lo_idx = mid_idx + 1

    if best is None:
        # Fallback: return [12!]
        return [target]

    # Now we need to actually find the subset that achieves target_state with max factor = best
    # We'll reconstruct by doing the DP again and tracking the path
    # For simplicity, we'll do a BFS/DFS on the state space with the constraint that all factors <= best

    # Let's redo the DP but track which candidates were used
    import bisect
    idx_limit = bisect.bisect_right(candidates, best)

    # dp[state] = list of candidates used (or None if not reachable)
    # To save memory, we'll store the last candidate added and the previous state
    from collections import defaultdict

    # We'll do forward DP: for each step i (candidate index), we maintain a dict
    # mapping state to (prev_state, candidate_index)
    # Actually, let's just do a simple reconstruction:
    # dp[i][state] = True if state is reachable using first i+1 candidates
    # parent[i][state] = (prev_state, candidate_i_used)

    # Since the number of states is small (792), and number of candidates is up to ~791,
    # this is feasible.

    # Let's use a different approach: since we know a solution exists, let's do a DFS
    # with pruning, trying to build the solution greedily.

    # Actually, let's just do the DP with parent tracking
    # dp[i] = dict mapping state to (prev_state, i) where i is the candidate index used to reach it
    # We process candidates in order

    # Initialize
    # dp[0] = {zero_state: None}
    # For i in range(idx_limit):
    #   dp[i+1] = dp[i]  # don't use candidate i
    #   for state in dp[i]:
    #     new_state = add cand_vecs[i] to state
    #     if valid and new_state not in dp[i+1]:
    #       dp[i+1][new_state] = (state, i)

    # But this is O(n * |states|) which is ~791 * 792 = ~626k, fine.

    zero_state = tuple([0] * 5)

    # dp[i] is a dict: state -> (prev_state, cand_idx) or None for zero_state at i=0
    # We'll store it as a list of dicts
    dp = [dict() for _ in range(idx_limit + 1)]
    dp[0][zero_state] = None

    for i in range(idx_limit):
        # Carry over
        for state, parent in dp[i].items():
            if state not in dp[i + 1]:
                dp[i + 1][state] = parent
        # Add candidate i
        cv = cand_vecs[i]
        for state in list(dp[i].keys()):
            new_state = tuple(state[j] + cv[j] for j in range(5))
            if all(new_state[j] <= target_exp[j] for j in range(5)):
                if new_state not in dp[i + 1]:
                    dp[i + 1][new_state] = (state, i)

    # Reconstruct
    if target_state not in dp[idx_limit]:
        # Should not happen
        return [target]

    # Trace back
    result_indices = []
    state = target_state
    i = idx_limit
    while i > 0:
        parent = dp[i][state]
        if parent is None:
            break
        prev_state, cand_idx = parent
        result_indices.append(cand_idx)
        state = prev_state
        i = cand_idx  # Wait, this is wrong. We need to go to i-1?

    # Actually, the DP as defined: dp[i+1] includes dp[i] states (not using cand i) and new states using cand i.
    # So when we trace back from dp[i+1][new_state] = (state, i), the previous state is in dp[i].
    # So we should set i = i (meaning we look at dp[i] next), not i = cand_idx.

    # Let me redo the reconstruction properly.
    result_indices = []
    state = target_state
    i = idx_limit
    while i > 0:
        parent = dp[i].get(state)
        if parent is None:
            # state was carried over from dp[i-1] without using candidate i-1
            i -= 1
            continue
        prev_state, cand_idx = parent
        # cand_idx should be i-1
        result_indices.append(cand_idx)
        state = prev_state
        i = i - 1  # Move to previous step

    result_indices.reverse()
    result = [candidates[i] for i in result_indices]

    # Verify
    prod = 1
    for r in result:
        prod *= r
    if prod != target:
        # Fallback
        return [target]

    # Ensure distinct
    if len(result) != len(set(result)):
        return [target]

    # Ensure all > 12
    if any(r <= 12 for r in result):
        return [target]

    return result

# EVOLVE-BLOCK-END
