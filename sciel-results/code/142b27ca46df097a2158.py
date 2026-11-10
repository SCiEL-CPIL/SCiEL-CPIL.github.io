# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    N = 512
    MAX_K = 20  # Max subset size to consider. 20*512=10240, manageable bitset size.
    # Actually, if we have 48 elements, max subset size is 47.
    # But sums are bounded by 512*47 ~ 24000.
    # Bitsets of size 25000 are fine.
    
    # Precomputation:
    # For each candidate x in [1, 512], and each subset size k in [2, 47],
    # we want to know which subsets of OTHER elements could form a mean x.
    # This is hard to precompute for all subsets.
    
    # Alternative: Precompute "conflict" relationships.
    # A set S is valid if no x in S is the mean of any subset T of S\{x} with |T|>=2.
    # This is equivalent to: For all x in S, and all k>=2, sum(T) != k*x for any T subset S\{x}, |T|=k.
    
    # Let's use a different precomputation:
    # For a fixed set S, checking if x is valid requires checking if there is a subset of S\{x} of size k summing to k*x.
    # This is the same DP problem as before, but we can speed it up.
    
    # New Strategy: 
    # 1. Generate a random ordering of 1..512.
    # 2. Greedily add elements.
    # 3. Maintain the DP bitsets for the current set incrementally?
    #    The current code recomputes DP for A and A\{a} from scratch. This is O(N^2 * Sum).
    #    We can maintain DP for the current set S.
    #    When adding x:
    #    a) Check if x is mean of subset of S. (Uses DP[S])
    #    b) Check if any a in S is mean of subset of S U {x} containing x.
    #       This requires checking subsets of S\{a} U {x}.
    #       This is harder to update incrementally without storing DP for all S\{a}.
    
    # Let's try a different structural change:
    # Use a "Conflict Graph" for pairs (k=2) first, then extend?
    # No, the problem allows any k>=2.
    
    # Let's optimize the verification.
    # Instead of recomputing DP for A_excl_a, we can use the fact that:
    # sum(T) + x = (j+1)a => sum(T) = (j+1)a - x.
    # T is a subset of A_excl_a of size j.
    # We need DP for A_excl_a.
    
    # What if we precompute all "bad" configurations?
    # A configuration is (S, x) where x is mean of subset of S.
    # This is too large.
    
    # Let's stick to the DP approach but optimize the inner loops and reuse.
    # Actually, the biggest bottleneck is `compute_dp` being called O(N^2) times.
    # For N=48, that's ~2300 calls. Each call is O(N * Sum).
    # Sum ~ 48 * 250 = 12000.
    # 2300 * 48 * 12000 ~ 1.3e9 operations. Too slow in Python.
    
    # We MUST avoid recomputing DP from scratch.
    
    # Insight:
    # When we have a set S, we maintain DP_S.
    # To check if x can be added:
    # 1. Is x mean of subset of S? 
    #    Check DP_S[k] for bit k*x.
    # 2. Is any a in S mean of subset of S U {x} containing x?
    #    Subset T' = T U {x}, T subset S\{a}.
    #    Sum(T') = Sum(T) + x.
    #    Size(T') = |T| + 1.
    #    Mean = a => Sum(T) + x = (|T|+1)a => Sum(T) = (|T|+1)a - x.
    #    We need to check if there is a subset T of S\{a} of size j = |T| summing to (j+1)a - x.
    
    # We don't have DP for S\{a}.
    # However, we can compute DP for S\{a} from DP_S?
    # No, removing an element from DP is not straightforward (inverse of OR-shift).
    
    # Alternative:
    # Only check condition 2 for the NEW element x?
    # No, the condition is symmetric. If we add x, it might make an OLD element a invalid.
    # Example: S={1, 3}. 1 is not mean of {3} (size 1 not allowed). 3 is not mean of {1}.
    # Add 2. 
    # Check 2 for a=1: Is 1 mean of subset of {1,3,2} containing 2?
    # Subset {2, ?} ... 1 = (2+y)/2 => y=0 (not in set).
    # Subset {2, 3, ?} ...
    
    # Wait, if we maintain the set S such that it is VALID, then:
    # Adding x creates a new set S' = S U {x}.
    # We must ensure S' is VALID.
    # This requires checking:
    # 1. x is not mean of subset of S.
    # 2. For all a in S, a is not mean of subset of S' containing x.
    #    (a was not mean of subset of S not containing x, because S was valid).
    #    So we only need to check subsets that CONTAIN x.
    
    # So the check for a in S is:
    # Does there exist T subset S\{a} such that mean(T U {x}) = a?
    # This is exactly what the current code does.
    
    # Can we speed up the check for a?
    # We need DP of S\{a}.
    # If we precompute DP for all S\{a} every time we add an element?
    # Cost: N * (Cost to compute DP for N-1 elements).
    # Still expensive.
    
    # What if we limit the search to a specific structure?
    # E.g., powers of 2?
    # 1, 2, 4, 8, 16, 32, 64, 128, 256. (9 elements)
    # Can we add more?
    # If we pick numbers with large gaps, the chance of being a mean is low.
    
    # Let's try a "Sparse" greedy with a faster verification.
    # Approximate verification? No, must be exact.
    
    # Optimization:
    # Use Python integers as bitsets.
    # `dp[k]` is an int.
    # `(dp[k] >> target_sum) & 1` is fast.
    # The bottleneck is the `compute_dp` function itself.
    
    # Let's try to cache DP results?
    # The set S changes every time.
    
    # What if we use a different search:
    # Start with a random set of size 48.
    # Check validity.
    # If invalid, remove the element involved in the most conflicts.
    # This is a "Repair" strategy.
    
    # 1. Generate random set S of size 48 from [1, 512].
    # 2. Find all violations.
    #    A violation is (a, T) where a in S, T subset S\{a}, |T|>=2, mean(T)=a.
    # 3. If no violations, return S.
    # 4. If violations exist, pick an element a involved in a violation.
    #    Remove a from S.
    # 5. Repeat.
    
    # This guarantees a valid set.
    # The question is: how large can we make it?
    # If we start with 48, and remove few, we get a large set.
    # If we remove many, we get a small set.
    # We can restart with different random sets.
    
    # Finding violations:
    # For each a in S:
    #   For each k in [2, len(S)-1]:
    #     target = k * a
    #     Check if there is a subset of S\{a} of size k summing to target.
    #     This requires DP of S\{a}.
    #     We have to compute DP for S\{a} for each a.
    #     Cost: N * DP(N-1).
    #     Same cost as before!
    
    # BUT, in the repair strategy, we only do this ONCE per restart, not N times (once per addition).
    # Greedy: N additions * N checks * DP(N) = N^2 * DP(N).
    # Repair: 1 check * N * DP(N).
    # This is a factor of N improvement!
    
    # So:
    # 1. Randomly sample set S of size M (e.g., 48).
    # 2. Compute DP for S\{a} for all a in S. (N times)
    # 3. Check all violations.
    # 4. If valid, return S.
    # 5. If invalid, remove one element from a violating pair (a, T).
    #    Which one? 
    #    Removing a fixes all violations involving a as the mean.
    #    Removing an element from T fixes this specific violation, but might create others?
    #    Removing an element from S always preserves validity of the remaining set?
    #    YES. If S' is a subset of a valid set S, then S' is valid.
    #    Proof: If S' had a violation, S would have the same violation.
    #    So we can greedily remove elements to fix violations.
    
    # Algorithm:
    # 1. Start with S = random sample of 512, size 48.
    # 2. Loop:
    #    a. Find any violation in S.
    #    b. If none, return S.
    #    c. If found (a, T), remove a from S. (Or remove an element from T? Removing a is safer/cleaner?)
    #       Actually, removing a from S is always safe.
    #       Removing x from T (where x in S) is also safe.
    #       Which leads to larger set?
    #       Hard to say. Let's try removing a.
    # 3. If time runs out, return current S.
    
    # To find violations efficiently:
    # We need DP for S\{a} for all a.
    # We can compute DP for S once, but we need S\{a}.
    # Is there a way to get DP[S\{a}] from DP[S]?
    # No.
    
    # So we compute DP[S\{a}] for each a in S.
    # Cost: |S| * DP(|S|-1).
    # For |S|=48, this is 48 * DP(47).
    # DP(47) involves 47 iterations, each shifting bitsets of size ~24000.
    # 47 * 47 * 24000 ~ 53 million bit operations.
    # In Python, big integer shifts are fast (C implementation).
    # 53M operations might take ~0.1-0.5s?
    # Let's test this hypothesis.
    
    # If it's fast enough, we can do multiple restarts.
    
    def compute_dp(nums):
        n = len(nums)
        max_sum = sum(nums)
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in nums:
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        return dp

    def find_violation(S):
        """
        Returns (a, T_indices) if violation found, else None.
        S is a list of ints.
        """
        n = len(S)
        if n < 3:
            return None
        
        # Compute DP for S\{a} for each a?
        # Or just check one a at a time to save memory/time if we find a violation early?
        # Let's check one a at a time.
        
        # Optimization: 
        # If we just want to know IF there is a violation, we can stop early.
        # But to remove an element, we need to know WHICH element.
        
        # Let's iterate a in S.
        # For each a, compute DP of S\{a}.
        # Check if a is mean of any subset of S\{a}.
        
        # To speed up, we can precompute sum(S\{a})?
        # Sum(S\{a}) = sum(S) - a.
        
        # Let's try this.
        total_sum = sum(S)
        
        for a in S:
            S_excl_a = [x for x in S if x != a]
            if not S_excl_a:
                continue
            
            dp_excl = compute_dp(S_excl_a)
            m = len(S_excl_a)
            max_sum_excl = total_sum - a
            
            # Check if a is mean of subset T of S_excl_a with |T| >= 2
            # sum(T) = k * a
            for k in range(2, m + 1):
                target = k * a
                if target > max_sum_excl:
                    break
                if (dp_excl[k] >> target) & 1:
                    return a # Just return the mean a. We will remove a.
                    
        return None

    def repair_set(S):
        """
        Repairs S by removing elements until valid.
        Returns the valid subset.
        """
        S = S[:]
        while True:
            if time.time() > deadline:
                break
            v = find_violation(S)
            if v is None:
                return S
            # Remove v (the mean)
            S.remove(v)
            if len(S) == 0:
                return S

    best_set = []
    
    # Try multiple random restarts
    # Start with size 48. If it fails, the resulting set is smaller.
    # We want to maximize the size.
    
    for i in range(50): # Max 50 restarts
        if time.time() > deadline - 5:
            break
            
        # Random sample
        # To get better results, maybe sample from a "spread out" range?
        # Or just uniform.
        S_init = random.sample(range(1, 513), 48)
        
        S_valid = repair_set(S_init)
        
        if len(S_valid) > len(best_set):
            best_set = S_valid
            
        # If we found a set of size 48, we are done (max possible).
        if len(best_set) == 48:
            break
        
        # If we found a set of size 47, maybe try to add one more?
        # The repair process might have removed too many.
        # Let's try to "grow" the set.
        if len(best_set) >= 40 and time.time() < deadline - 10:
            # Try to add random elements to best_set
            current = best_set[:]
            unused = [x for x in range(1, 513) if x not in current]
            random.shuffle(unused)
            
            for x in unused:
                if time.time() > deadline - 2:
                    break
                # Check if x can be added to current
                # This requires the expensive check again.
                # Let's use the repair logic on current U {x}
                # If repair(current U {x}) has size len(current)+1, then x was added successfully?
                # No, repair might remove x or something else.
                # We want to keep current.
                
                # Let's just do the explicit check for x.
                # Is x mean of subset of current?
                # Is any a in current mean of subset of current U {x} containing x?
                
                # Reuse logic from before but optimized?
                # Let's just skip growth for now and rely on restarts.
                # The repair strategy from 48 should yield good results if 48 is feasible.
                # If 48 is not feasible, it will yield the max size.
                
                pass

    return best_set

# EVOLVE-BLOCK-END
