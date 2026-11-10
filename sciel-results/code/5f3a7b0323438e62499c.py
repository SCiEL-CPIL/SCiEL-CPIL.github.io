# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 82  # Leave 8s margin for safety
    MAX_N = 512
    TARGET = 48
    rng = np.random.default_rng(42)

    def compute_dp_bitsets(nums):
        """
        Compute dp[k] bitsets for subset sums of size k.
        Returns list of integers where bit s is 1 if some subset of size k sums to s.
        """
        n = len(nums)
        if n == 0:
            return [1]  # dp[0] = 1 (bit 0 set)
        
        # Use Python integers for arbitrary precision bitsets
        # dp[k] is an integer; bit s set means sum s achievable with k elements
        dp = [0] * (n + 1)
        dp[0] = 1  # empty subset sums to 0
        
        for x in nums:
            # Update in reverse to avoid using x multiple times
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        
        return dp

    def is_valid_set(A):
        """
        Check if set A (list of distinct ints) satisfies the constraint:
        No element is the exact arithmetic mean of any subset of at least 2 OTHER elements.
        """
        if len(A) < 3:
            return True
        
        n = len(A)
        A_set = set(A)
        total_sum = sum(A)
        
        # Precompute DP for the full set
        dp_full = compute_dp_bitsets(A)
        
        for idx, a in enumerate(A):
            # Check if a is the mean of some subset T of A\{a} with |T| >= 2
            # mean(T) = a => sum(T) = |T| * a
            # T is a subset of A excluding a
            # We need to check if there exists k >= 2 and a subset of size k from A\{a} summing to k*a
            
            # Build A without a
            # Instead of rebuilding DP each time, we can use the full DP and check
            # But we need to ensure the subset doesn't include a.
            # Since elements are distinct, if we find a subset of A of size k summing to k*a,
            # we need to verify it doesn't contain a.
            # Actually, if a is in the subset, then sum(T) = k*a and a in T.
            # Let T = T' union {a}, where T' subset of A\{a}, |T'| = k-1.
            # sum(T') + a = k*a => sum(T') = (k-1)*a.
            # So if there's a subset of A\{a} of size k-1 summing to (k-1)*a, that's a different condition.
            # The condition is: a is the mean of a subset of OTHER elements.
            # So the subset T must NOT contain a.
            
            # Efficient approach: For each a, compute DP of A\{a}? Too slow.
            # Alternative: Use the full DP. If there's a subset of size k summing to k*a in A,
            # it might include a or not.
            # If it includes a: then there's a subset of size k-1 from A\{a} summing to (k-1)*a.
            # If it doesn't include a: that's the violation we're looking for.
            
            # Let's just compute DP for A\{a} for each a. With n<=48, this is feasible if done smartly.
            # But 48 * DP computation might be slow.
            
            # Better: Precompute prefix and suffix DPs? Or just do it for each a.
            # Actually, let's think about the structure.
            # For a fixed a, we want to know if there exists k>=2 such that
            # there's a subset of A\{a} of size k summing to k*a.
            
            # Let's compute DP for A without a.
            A_excl_a = [x for i, x in enumerate(A) if i != idx]
            dp_excl = compute_dp_bitsets(A_excl_a)
            
            m = len(A_excl_a)
            for k in range(2, m + 1):
                target = k * a
                if target > sum(A_excl_a):
                    break
                if target < 0:
                    continue
                if (dp_excl[k] >> target) & 1:
                    return False
        
        return True

    def penalty(A):
        """
        Count the number of violated mean constraints.
        For each a in A, count the number of subsets T of A\{a} with |T|>=2
        such that sum(T) = |T|*a.
        """
        if len(A) < 3:
            return 0
        
        n = len(A)
        total_violations = 0
        
        for idx, a in enumerate(A):
            A_excl_a = [x for i, x in enumerate(A) if i != idx]
            dp_excl = compute_dp_bitsets(A_excl_a)
            
            m = len(A_excl_a)
            for k in range(2, m + 1):
                target = k * a
                if target > sum(A_excl_a):
                    break
                if target < 0:
                    continue
                if (dp_excl[k] >> target) & 1:
                    total_violations += 1
        
        return total_violations

    def sa_search(initial_set, time_limit):
        """
        Simulated Annealing to minimize penalty.
        """
        current = list(initial_set)
        current_pen = penalty(current)
        
        if current_pen == 0:
            return current
        
        best = list(current)
        best_pen = current_pen
        
        T = 10.0
        T_min = 0.1
        alpha = 0.995
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Move: swap one element with a random unused element
            if not current:
                break
            
            idx = random.randint(0, len(current) - 1)
            old_val = current[idx]
            
            # Pick a random candidate from [1, 512] not in current
            candidates = [x for x in range(1, 513) if x not in set(current)]
            if not candidates:
                break
            
            new_val = random.choice(candidates)
            
            # Try the swap
            current[idx] = new_val
            new_pen = penalty(current)
            
            delta = new_pen - current_pen
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_pen = new_pen
                if new_pen < best_pen:
                    best_pen = new_pen
                    best = list(current)
            else:
                current[idx] = old_val
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        return best

    # Initial set: use a greedy construction to get a reasonable starting point
    def greedy_initial():
        selected = []
        # Try to build a set greedily
        # Start with some spread-out numbers
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 30:
                break
            if len(selected) >= TARGET:
                break
            test_set = selected + [x]
            if is_valid_set(test_set):
                selected.append(x)
        
        return selected

    # Run multiple SA attempts with different initializations
    best_result = []
    best_size = 0
    
    # First, try to get a valid set via greedy
    initial = greedy_initial()
    if initial:
        best_result = initial
        best_size = len(initial)
    
    # If we already have a large valid set, we can try to improve
    if best_size >= 40:
        # Try to add more elements
        unused = [x for x in range(1, 513) if x not in set(best_result)]
        random.shuffle(unused)
        for x in unused:
            if time.time() > deadline - 10:
                break
            if len(best_result) >= TARGET:
                break
            test_set = best_result + [x]
            if is_valid_set(test_set):
                best_result.append(x)
        best_size = len(best_result)
        if best_size >= TARGET:
            return best_result
    
    # Run SA to improve
    if time.time() < deadline - 15:
        # Use the current best as initial, or generate a new random initial
        if best_result:
            sa_initial = best_result
        else:
            # Random initial set of size ~30
            sa_initial = random.sample(range(1, 513), min(30, 512))
        
        sa_time = min(20, max(5, deadline - 15 - time.time()))
        sa_result = sa_search(sa_initial, sa_time)
        
        if len(sa_result) > best_size or (len(sa_result) == best_size and penalty(sa_result) < penalty(best_result)):
            best_result = sa_result
            best_size = len(best_result)
        
        # Try to add more elements to the SA result
        if best_size < TARGET and time.time() < deadline - 5:
            unused = [x for x in range(1, 513) if x not in set(best_result)]
            random.shuffle(unused)
            for x in unused:
                if time.time() > deadline - 2:
                    break
                if len(best_result) >= TARGET:
                    break
                test_set = best_result + [x]
                if is_valid_set(test_set):
                    best_result.append(x)
            best_size = len(best_result)
    
    # Final check: if best_result is valid, return it
    if best_result and is_valid_set(best_result):
        return best_result
    
    # Fallback: return an empty list or a small valid set
    return best_result
# EVOLVE-BLOCK-END
