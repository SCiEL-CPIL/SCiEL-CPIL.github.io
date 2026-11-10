# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    deadline = time.time() + 85  # 85s margin out of 90s
    n_max = 512
    target_size = 48
    
    # Precompute "bad" candidates for all pairs (a, b) in [1, 512]
    # If we have a set S containing a and b, then any x such that
    # x is the mean of {a, b} (i.e., 2x = a+b -> x = (a+b)/2) is bad.
    # More generally, if we add x, we must ensure x is not the mean of any subset of S.
    # The most restrictive constraints are often small subsets.
    # However, a more general constraint:
    # x is invalid if there exist a, b in S such that x is the mean of a subset containing a, b?
    # No, x is the mean of a subset T of S.
    # This implies sum(T) = |T| * x.
    # If |T|=2, x = (a+b)/2.
    # If |T|=3, x = (a+b+c)/3.
    
    # Strategy:
    # 1. Greedy construction with a "blacklist" of forbidden values.
    # 2. When we add a new element x, we update the blacklist for future elements.
    #    Actually, the blacklist depends on the current set S.
    #    If S is current, a new element y is valid iff y is not the mean of any subset of S.
    #    This is hard to check fast for large subsets.
    
    # Alternative Strategy: Randomized Greedy with Local Search
    # 1. Generate a random set S of size ~48.
    # 2. Check validity. If invalid, repair.
    # 3. Try to improve by swapping elements.
    
    # Let's stick to a robust Greedy + Repair.
    # Key Insight: The condition "x is mean of subset of S" is monotonic?
    # No.
    
    # Let's use the "bad pairs" heuristic for the most common violations (size 2 and 3).
    # Precompute for each pair (a, b) with a < b, the value m = (a+b)/2 if integer.
    # If a, b are in S, m cannot be added.
    # Also, if a, b, c are in S, (a+b+c)/3 cannot be added.
    
    # This suggests we should pick elements that are "far apart" or "independent".
    # Randomized greedy:
    # 1. Shuffle [1..512].
    # 2. Iterate, adding x if it doesn't create a mean violation with existing S.
    #    To check violation fast, we can maintain a set of "forbidden" values.
    #    Forbidden values are means of subsets of S.
    #    Updating forbidden set is expensive.
    
    # Better: Maintain S. To check if x is valid:
    #   For each k >= 2, check if k*x is a subset sum of S of size k.
    #   This is the original problem.
    
    # Let's try a different angle:
    # Use a set of "generators" that are unlikely to form means.
    # Powers of 2? 1, 2, 4, 8, ... 256. Size 9. Too small.
    # Primes? ~97 primes.
    # Let's try a greedy that prioritizes numbers with fewer "partners".
    
    # Actually, the most effective strategy found in similar problems is often:
    # 1. Greedy construction.
    # 2. If stuck, try to remove one element and replace it with a better one.
    
    # Let's implement a fast validity checker using bitsets for subset sums,
    # but optimized for the specific case of "is x a mean?".
    # We don't need all subset sums, we just need to know if k*x is achievable.
    
    def is_valid_candidate(x, S):
        # S is a list of ints
        n = len(S)
        if n < 2:
            return True
        
        # We need to check if there exists a subset T of S with size k >= 2
        # such that sum(T) = k * x.
        
        # DP for subset sums of S.
        # dp[j] is a bitset of sums achievable with j elements.
        # Max sum = 512 * 48 = 24576.
        # We can use Python integers as bitsets.
        
        # Optimization: Early exit if x is too large or small?
        # Min mean of 2 elements in S: min(S) + second_min(S) / 2.
        # Max mean: max(S) + second_max(S) / 2.
        # This doesn't help much for arbitrary k.
        
        # Let's just run the DP.
        # To speed it up, we can limit the range of sums we care about.
        # We only care about sums that are multiples of x? No, sum = k*x.
        
        # Standard DP:
        # dp[0] = 1 (bit 0)
        # for val in S:
        #   for j from current_max down to 1:
        #     dp[j] |= dp[j-1] << val
        
        # Then check if bit (k*x) is set in dp[k].
        
        # This is O(N^2) per check. N=48. 48^2 = 2304 ops.
        # 512 candidates * 2304 = ~1.2M ops per greedy pass.
        # This is fast enough for many restarts.
        
        dp = [0] * (n + 1)
        dp[0] = 1
        max_j = 0
        for val in S:
            # Update dp
            # We can limit j to max_j + 1
            for j in range(max_j + 1, 0, -1):
                dp[j] |= (dp[j-1] << val)
            max_j += 1
            
            # Early check? No, we need full DP for all k.
            # But we can check if x is already in S? No, distinct.
            
        # Check all k >= 2
        for k in range(2, n + 1):
            target_sum = k * x
            # Max possible sum with k elements is k * 512.
            # If target_sum > 512 * k: impossible? No, S elements are <= 512.
            # If target_sum > 512 * k: definitely not in S.
            # If target_sum < 1 * k: definitely not.
            if target_sum > 512 * k or target_sum < k:
                continue
            if (dp[k] >> target_sum) & 1:
                return False
                
        return True

    def get_best_set():
        best_set = []
        best_score = 0
        
        # Multiple random restarts
        # Use different orderings
        orders = [
            list(range(1, 513)),
            list(reversed(range(1, 513))),
            [x for x in range(1, 513) if x % 2 == 1],
            [x for x in range(1, 513) if x % 2 == 0],
        ]
        
        # Add some random shuffles
        for _ in range(10):
            rand_order = list(range(1, 513))
            random.shuffle(rand_order)
            orders.append(rand_order)
            
        for order in orders:
            if time.time() > deadline:
                break
                
            S = []
            for x in order:
                if len(S) >= target_size:
                    break
                if time.time() > deadline:
                    break
                if is_valid_candidate(x, S):
                    S.append(x)
            
            if len(S) > best_score:
                best_score = len(S)
                best_set = S[:]
                
            # If we reached target, we can try to improve or just return?
            # The task is to maximize cardinality. 48 is the max allowed in the prompt "1..maximum=48".
            # Wait, "At most 48 integers". So 48 is the cap.
            # If we reach 48, we should return it?
            # Yes, "Maximize cardinality. At most 48".
            # So 48 is the optimal score.
            if best_score >= 48:
                return best_set
                
        # If we didn't reach 48, try local search on the best set found
        if best_score < 48:
            # Try to swap elements to increase size or validity?
            # Since we only add valid elements, the set is always valid.
            # The issue is we might get stuck at a local maximum < 48.
            # Local search: Remove one element, try to add multiple new ones.
            # This is complex. Given the time, the greedy with restarts is usually sufficient
            # for these types of "mean avoiding" sets if the order is good.
            # Primes order is often good.
            
            # Let's try one more specific order: Primes
            primes = [x for x in range(1, 513) if all(x % i != 0 for i in range(2, int(x**0.5)+1))]
            S = []
            for x in primes:
                if len(S) >= target_size:
                    break
                if time.time() > deadline:
                    break
                if is_valid_candidate(x, S):
                    S.append(x)
            if len(S) > best_score:
                best_score = len(S)
                best_set = S[:]
                
            # Try powers of 2 mixed?
            # Just return the best found.
            
        return best_set

    return get_best_set()
# EVOLVE-BLOCK-END
