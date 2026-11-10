import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 84  # 6s safety margin
    N = 512
    MAX_K = 48
    
    def compute_dp(nums):
        """
        For a list of nums, compute dp[k] where bit s is set iff there exists
        a subset of nums of size k with sum s.
        """
        n = len(nums)
        max_sum = sum(nums)
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in nums:
            for k in range(n, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << x)
        return dp

    def is_valid_set(S):
        """
        Verify that S is a valid set: no element is the mean of any subset
        of at least two OTHER elements.
        """
        n = len(S)
        if n <= 1:
            return True
        
        # For each element a in S, check if a is the mean of some subset of S\{a} of size k >= 2
        for i, a in enumerate(S):
            others = [S[j] for j in range(n) if j != i]
            m = len(others)
            if m < 2:
                continue
            dp = compute_dp(others)
            for k in range(2, m + 1):
                target_sum = k * a
                if target_sum > sum(others):
                    break
                if (dp[k] >> target_sum) & 1:
                    return False
        return True

    def is_valid_addition(S, x):
        """
        Check if x can be added to valid set S while maintaining validity.
        Two types of violations to check:
        1. x is the mean of a subset T of S with |T| >= 2: sum(T) = |T| * x
        2. Some a in S is the mean of a subset U of S ∪ {x} containing x:
           U = T ∪ {x} where T ⊆ S \ {a}, |U| >= 2
           sum(T) + x = (|T| + 1) * a
           sum(T) = (|T| + 1) * a - x
           where |T| >= 0 and |T| + 1 >= 2, so |T| >= 1
        """
        n = len(S)
        if n < 2:
            # Need to check: is x the mean of any subset of S of size >= 2?
            # If n < 2, no subset of size >= 2 exists, so only check type 2
            # Type 2: a in S, subset containing x. |U| >= 2 means T ⊆ S\{a} with |T| >= 1
            # But S has at most 1 element, so T must be {b} where b is the one element != a
            # sum({b}) = (1+1)*a - x = 2a - x
            # So b = 2a - x
            if n == 1:
                a = S[0]
                b = 2 * a - x
                if b == a:  # b would be in S\{a} = empty, so impossible
                    return True
                # Actually S\{a} is empty, so no T of size >= 1 exists
                return True
            return True
        
        # Type 1: x is mean of subset T of S, |T| >= 2
        dp_S = compute_dp(S)
        for k in range(2, n + 1):
            target_sum = k * x
            if target_sum > sum(S):
                break
            if (dp_S[k] >> target_sum) & 1:
                return False
        
        # Type 2: some a in S is mean of subset U = T ∪ {x} where T ⊆ S\{a}, |T| >= 1
        for i, a in enumerate(S):
            S_excl_a = [S[j] for j in range(n) if j != i]
            m = len(S_excl_a)
            if m < 1:
                continue
            dp_excl = compute_dp(S_excl_a)
            sum_excl = sum(S_excl_a)
            for j in range(1, m + 1):
                target_sum = (j + 1) * a - x
                if target_sum < 0:
                    continue
                if target_sum > sum_excl:
                    break
                if (dp_excl[j] >> target_sum) & 1:
                    return False
        return True

    # Greedy construction with different orderings
    def greedy_construct(order, S_start=None):
        if S_start is None:
            S = []
        else:
            S = S_start[:]
        for x in order:
            if time.time() > deadline - 2:
                break
            if x in S:
                continue
            if len(S) >= MAX_K:
                break
            if is_valid_addition(S, x):
                S.append(x)
        return S

    # Local search: try to swap elements to allow more additions
    def local_search(S, iterations=200):
        S = S[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n = len(S)
            if n >= MAX_K:
                break
            # Try removing one element and adding two
            for i in range(n):
                if time.time() > deadline - 3:
                    break
                S_minus = [S[j] for j in range(n) if j != i]
                # Try adding random candidates
                candidates = [x for x in range(1, N + 1) if x not in S_minus]
                random.shuffle(candidates)
                found = False
                for x in candidates[:50]:
                    if is_valid_addition(S_minus, x):
                        S_new = S_minus + [x]
                        if len(S_new) > n:
                            S = S_new
                            improved = True
                            found = True
                            break
                if found:
                    break
            if not improved:
                # Try single swap: remove one, add one different
                for i in range(n):
                    if time.time() > deadline - 3:
                        break
                    S_minus = [S[j] for j in range(n) if j != i]
                    candidates = [x for x in range(1, N + 1) if x not in S_minus]
                    random.shuffle(candidates)
                    for x in candidates[:30]:
                        if is_valid_addition(S_minus, x):
                            S = S_minus + [x]
                            improved = True
                            break
                    if improved:
                        break
        return S

    # Multi-start greedy with local search
    best_set = []
    best_size = 0

    # Strategy 1: Descending
    order_desc = list(range(N, 0, -1))
    S1 = greedy_construct(order_desc)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1[:]

    # Strategy 2: Ascending
    if time.time() < deadline - 10:
        order_asc = list(range(1, N + 1))
        S2 = greedy_construct(order_asc)
        if len(S2) > best_size:
            best_size = len(S2)
            best_set = S2[:]

    # Strategy 3: Randomized greedy (multiple restarts)
    for trial in range(50):
        if time.time() > deadline - 8:
            break
        order_rand = list(range(1, N + 1))
        random.shuffle(order_rand)
        S = greedy_construct(order_rand)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    # Local search on best
    if time.time() < deadline - 5 and len(best_set) < MAX_K:
        best_set = local_search(best_set)
        best_size = len(best_set)

    # Final attempt: try to fill remaining slots
    if len(best_set) < MAX_K and time.time() < deadline - 2:
        S = best_set[:]
        candidates = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 1:
                break
            if len(S) >= MAX_K:
                break
            if is_valid_addition(S, x):
                S.append(x)
        if len(S) > best_size:
            best_set = S
            best_size = best_size

    # Verify the final set
    if not is_valid_set(best_set):
        # Fallback: use powers of 2
        best_set = []
        val = 1
        while val <= N:
            best_set.append(val)
            val *= 2

    return best_set
