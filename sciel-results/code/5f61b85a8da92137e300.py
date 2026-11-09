import math
import time
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    max_val = P['maximum_window']
    max_size = P['maximum_size']
    
    # Target: 210 = 2 * 3 * 5 * 7. 
    # We need product(selected) * 210 to be a square.
    # This is equivalent to XOR of signatures(selected) == signature(210).
    
    # Step 1: Sieve to find smallest prime factor (SPF) for all numbers up to max_val
    # This allows O(log n) factorization for any number.
    limit = max_val
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_signature(x):
        """
        Returns a tuple of primes where x has an odd exponent.
        We represent the signature as a frozenset of primes.
        """
        sig = set()
        temp = x
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                sig.add(p)
        return frozenset(sig)
    
    # Precompute signatures for all numbers in (210, max_val]
    # We only care about numbers > 210
    sig_map = {}
    for x in range(211, max_val + 1):
        sig_map[x] = get_signature(x)
        
    target_sig = get_signature(210) # {2, 3, 5, 7}
    
    # Step 2: Binary search on the maximum value M
    # We want to find the smallest M such that there exists a subset S of {211, ..., M}
    # with len(S) <= 100, distinct elements, and XOR(S) == target_sig.
    
    # To check if a solution exists with max element <= M:
    # We can use a BFS/DFS on the state space of signatures, or Gaussian elimination.
    # Since the number of primes involved is small (primes <= 10210), the state space is large,
    # but we can use a "meet in the middle" or simple set expansion if the set size is small.
    
    # However, a simpler approach:
    # If we just want ANY subset, we can use linear algebra over GF(2).
    # But we have a constraint on the number of elements (<= 100) and distinctness.
    # Actually, the "distinctness" is handled by the fact that we are picking from a set of available numbers.
    
    # Let's refine the check:
    # For a given M, we have a pool of numbers P_M = {x in [211, M]}.
    # We want to know if there is a subset S of P_M such that XOR(S) = target_sig.
    # This is the "Subset XOR" problem.
    # We can solve this using a dictionary mapping signature -> list of numbers (or just existence).
    # But the number of possible signatures is huge (2^k where k is number of primes).
    
    # Alternative: Since we want to MINIMIZE max(S), and the numbers are small,
    # the optimal solution likely uses very few numbers.
    # Let's check for solutions of size 1, 2, 3, ... up to some limit K.
    # If K is small (e.g., < 10), we can do this efficiently.
    
    # Let's precompute for each signature, the smallest number having that signature.
    # sig_to_min_num = defaultdict(lambda: float('inf'))
    # for x in range(211, max_val + 1):
    #     sig = sig_map[x]
    #     if x < sig_to_min_num[sig]:
    #         sig_to_min_num[sig] = x
        
    # Wait, this only works for size 1.
    # For size 2: we need sig1 ^ sig2 = target.
    # For size 3: sig1 ^ sig2 ^ sig3 = target.
    
    # Let's use a BFS approach on the "signature graph" starting from the empty set (signature 0).
    # But we are constrained by the numbers we pick.
    
    # Better Strategy:
    # Iterate M from 211 upwards.
    # Maintain a set of achievable signatures using numbers in [211, M].
    # Let S_M be the set of signatures achievable by subsets of {211, ..., M}.
    # Initially S_210 = {frozenset()} (empty set, signature 0).
    # When we add number x with signature s_x:
    # S_new = S_old U {sig ^ s_x for sig in S_old}
    # We stop when target_sig is in S_new.
    # The size of S_new can grow exponentially, but the number of distinct signatures is limited
    # by the number of primes. Primes <= 10210: there are ~1250 primes.
    # So the state space is 2^1250, which is too big for a full BFS.
    
    # However, we only care about signatures that can be formed by SMALL subsets.
    # Let's limit the subset size.
    # If we restrict to subsets of size <= K, the number of states is much smaller.
    
    # Let's try K=4 or K=5.
    # If no solution found with K=5, we might need to go higher, but usually
    # the "parity completion" for small numbers is quick.
    
    # Actually, let's look at the structure.
    # 210 = 2*3*5*7.
    # If we pick x=210*k^2, that's size 1. Min is 840.
    # Can we do better?
    # Try to find 2 numbers a, b such that a*b = 210 * k^2.
    # a*b / 210 = k^2.
    # Let a = 210 * u, b = 210 * v? No, a,b just need to be > 210.
    
    # Let's implement the BFS with a limit on the number of states to avoid explosion.
    # We will track (signature, max_element_in_subset) ? No, we are iterating M.
    # For a fixed M, we just want to know if target_sig is reachable.
    
    # Let's restart the search logic:
    # We will iterate M from 211 to max_val.
    # For each M, we check if there is a subset of {211...M} with XOR = target.
    # We use a dictionary: sig -> min_max_element_to_achieve_sig? No.
    
    # Let's use a different representation:
    # For each signature, store the minimum possible MAX element of a subset that achieves it.
    # Let dp[sig] = min(max(S)) for all subsets S with XOR(S) = sig.
    # Initialize dp[0] = 0 (empty set, max element 0).
    # For x from 211 to max_val:
    #   s_x = sig_map[x]
    #   For each sig in current dp keys (snapshot):
    #       new_sig = sig ^ s_x
    #       new_max = max(dp[sig], x)
    #       if new_max < dp.get(new_sig, inf):
    #           dp[new_sig] = new_max
    #
    # This is a "Knapsack-like" DP but on the XOR group.
    # The number of distinct signatures might still be large, but let's see.
    # Primes involved: 2, 3, 5, 7, 11, 13, ...
    # If we only use numbers <= M, the primes involved are <= M.
    # The number of distinct signatures is at most 2^(number of primes <= M).
    # For M=100, primes ~ 25, so 2^25 states ~ 33 million. Too many.
    # But we only update states that are reachable.
    # Also, we can prune: if dp[sig] is already small, we don't need to update it with a larger max.
    
    # Let's try this DP. It is structurally different from the previous heuristic search.
    
    dp = {frozenset(): 0}  # sig -> min max element
    
    best_answer = None
    best_value = float('inf')
    
    # We need to reconstruct the solution, so we need to track parents.
    # parent[sig] = (prev_sig, x)
    parent = {}
    
    for x in range(211, max_val + 1):
        if time.time() > deadline - 5.0:
            break
            
        s_x = sig_map[x]
        
        # We iterate over a snapshot of current dp to avoid using x multiple times
        # (since elements must be distinct).
        current_states = list(dp.items())
        
        for sig, current_max in current_states:
            new_sig = sig ^ s_x
            new_max = max(current_max, x)
            
            if new_max < dp.get(new_sig, float('inf')):
                dp[new_sig] = new_max
                parent[new_sig] = (sig, x)
                
                # Check if we hit the target
                if new_sig == target_sig:
                    # We found a solution with max element x
                    # Reconstruct it
                    sol = []
                    cur_sig = new_sig
                    while cur_sig != frozenset():
                        prev_sig, last_x = parent[cur_sig]
                        sol.append(last_x)
                        cur_sig = prev_sig
                    sol.reverse()
                    
                    # Verify
                    if len(sol) <= max_size:
                        val = max(sol) - 210
                        if val < best_value:
                            best_value = val
                            best_answer = sol
                            # Since we are iterating x upwards, the first time we hit target_sig
                            # with a valid reconstruction, is it the minimal max?
                            # Yes, because x is the current max. Any solution found here has max=x.
                            # Any solution found in a previous iteration had max < x.
                            # So if we found one before, we would have returned it?
                            # No, we are looking for the GLOBAL minimum max.
                            # The DP updates dp[new_sig] only if new_max is smaller.
                            # So dp[target_sig] will hold the minimum max element for any subset.
                            # We can just check at the end?
                            # But we want to stop early if we find a very good one?
                            # Actually, since x increases, the first time we see target_sig in dp
                            # might not be the minimal max if a later x allows a "better" combination?
                            # No, "better" means smaller max.
                            # If we found target_sig with max=100, can we find it with max=90 later?
                            # No, because we process x in increasing order.
                            # When x=90, we processed all subsets with max<=90.
                            # If target_sig was achievable with max<=90, dp[target_sig] would have been set then.
                            # So the FIRST time dp[target_sig] is set (or updated to a smaller value, which is impossible as x grows)
                            # is the minimal max?
                            # Wait. dp[target_sig] is updated if new_max < dp[target_sig].
                            # Since x is increasing, new_max = max(prev_max, x) >= x.
                            # So new_max is non-decreasing with x for a fixed path?
                            # Not necessarily. A path using a large x might have a smaller max than a path using a slightly smaller x?
                            # No, max(S) is at least the largest element in S.
                            # If we add x, the new max is at least x.
                            # So if we process x=100, any new state has max >= 100.
                            # If dp[target_sig] was already 90, we won't update it to 100.
                            # So dp[target_sig] will hold the true minimum max element.
                            
                            # We can break early if we are confident?
                            # No, we must ensure no smaller max exists.
                            # But since we iterate x from 211 upwards, and dp[target_sig] is only updated if a SMALLER max is found,
                            # and smaller max implies smaller x,
                            # the first time we see target_sig in dp is NOT necessarily the minimal max?
                            # Example:
                            # x=10, sig=A. dp[A]=10.
                            # x=20, sig=B. A^B=Target. dp[Target]=20.
                            # x=15, sig=C. C^C=0? No.
                            # Is it possible that a solution with max=15 exists but was not found when x=15 was processed?
                            # Only if the subset required an element > 15. But max=15 means all elements <= 15.
                            # So if a subset has max 15, it was fully processed when x=15 was the loop variable.
                            # So yes, the first time dp[target_sig] is set is the minimal max?
                            # Let's verify.
                            # If a subset S has max M, it is considered when x=M is processed.
                            # At that point, all elements in S are <= M, so they have been added to the DP.
                            # The DP state for S's signature will be updated with max <= M.
                            # So if a solution with max M exists, dp[target_sig] <= M will be true after processing x=M.
                            # Since we iterate M upwards, the first M where dp[target_sig] becomes finite is the minimal M.
                            
                            # So we can break immediately?
                            # We need to make sure the reconstruction is valid (distinct, etc).
                            # The DP ensures distinctness because we iterate over a snapshot of states BEFORE adding x.
                            # So x is used at most once.
                            
                            # Let's do a final verification of the found solution.
                            if is_valid_answer(sol):
                                return [int(x) for x in sol]
                                
                            # If invalid (shouldn't happen), continue
                            
    # If loop finishes without breaking, return best found
    if best_answer:
        return [int(x) for x in best_answer]
        
    # Fallback: single element 840
    return [840]

def is_valid_answer(answer):
    """Check if answer is a valid list of distinct ints > 210, <= 10210, 
    with product * 210 being a perfect square."""
    if not isinstance(answer, (list, tuple)):
        return False
    if len(answer) == 0 or len(answer) > 100:
        return False
    if len(set(answer)) != len(answer):
        return False
    for x in answer:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= 210 or x > 10210:
            return False
    # Check product * 210 is perfect square
    prod = 210
    for x in answer:
        prod *= x
    sqrt_val = math.isqrt(prod)
    return sqrt_val * sqrt_val == prod
