"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin for 60s limit
    
    # We will try multiple orderings to find a larger set
    # Orderings to try:
    # 1. Ascending
    # 2. Descending
    # 3. Primes
    # 4. Powers of 2
    # 5. Random shuffles
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    # Generate orderings
    all_nums = list(range(1, 513))
    orderings = []
    
    # 1. Ascending
    orderings.append(all_nums)
    
    # 2. Descending
    orderings.append(sorted(all_nums, reverse=True))
    
    # 3. Primes first
    primes = [x for x in all_nums if is_prime(x)]
    non_primes = [x for x in all_nums if not is_prime(x)]
    orderings.append(primes + non_primes)
    
    # 4. Powers of 2
    powers = [2**i for i in range(1, 10) if 2**i <= 512]
    others = [x for x in all_nums if x not in powers]
    orderings.append(powers + others)
    
    # 5. Random shuffles (limited number to save time)
    rng = random.Random(42)
    for _ in range(20):
        shuffled = all_nums.copy()
        rng.shuffle(shuffled)
        orderings.append(shuffled)

    best_set = []
    
    for ordering in orderings:
        if time.time() > deadline:
            break
            
        current_set = []
        # DP array size: max sum is 512 * 48 ~ 24576. 
        # We need dp[j] to be a bitset.
        # Max j is 48.
        
        for candidate in ordering:
            if time.time() > deadline:
                break
                
            # Try adding candidate
            new_set = current_set + [candidate]
            n = len(new_set)
            
            # Check validity of new_set
            # Build DP bitsets
            # dp[j] is an integer where bit s is 1 if sum s is achievable with j elements
            # Max sum needed: 48 * 512 = 24576. 
            # We can mask to keep integers small if needed, but Python big ints are fine.
            
            # Optimization: Early exit if sum of smallest k elements > k * max_val? 
            # Not necessary, bitset DP is fast.
            
            dp = [0] * (n + 1)
            dp[0] = 1
            
            # Sort new_set for potential early pruning? 
            # Bitset DP doesn't benefit from sorting for correctness, 
            # but might for cache? No, just iterate.
            
            for x in new_set:
                # Update dp in reverse to avoid using x multiple times in same step
                # We are building subsets of distinct elements.
                # Standard 0/1 knapsack style update.
                for j in range(n, 0, -1):
                    # Shift dp[j-1] by x
                    dp[j] |= (dp[j-1] << x)
                    
            # Check constraints
            valid = True
            # For each a in new_set, check if k*a is in dp[k] for any k >= 2
            # To speed up, we can precompute the bits to check?
            # Just iterate.
            
            for a in new_set:
                # k ranges from 2 to n
                # Target sum is k * a
                for k in range(2, n + 1):
                    target = k * a
                    # Check if bit 'target' is set in dp[k]
                    if (dp[k] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set = new_set
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
# EVOLVE-BLOCK-END
