import time
import random
import math
from collections import defaultdict

# Precompute Smallest Prime Factor (SPF) up to a reasonable limit
# Max sum of a_i is 20000. 
# If we use multipliers, k_i are smaller.
# Let's compute SPF up to 20000 to be safe for direct sums, 
# but we will work with k_i for efficiency.

MAX_VAL = 10000
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

def get_prime_factors(n):
    """Return set of prime factors of n using SPF."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = float('inf')
    best_set = None
    
    # Candidate multipliers
    # M=6 forces primes {2,3}
    # M=30 forces primes {2,3,5}
    # M=210 forces primes {2,3,5,7}
    # M=420 forces primes {2,3,5,7} (same set, larger step)
    # M=2 forces prime {2}
    
    multipliers = [6, 30, 210, 2, 10, 15]
    
    # Precompute factors for small numbers (k sums)
    # If M=210, max k = 10000/210 = 47. Max sum = 94.
    # If M=6, max k = 1666. Max sum = 3332.
    # We will compute factors on the fly or cache them.
    
    factor_cache = {}
    def get_factors_cached(n):
        if n in factor_cache:
            return factor_cache[n]
        f = get_prime_factors(n)
        factor_cache[n] = f
        return f

    def evaluate_and_search(M):
        """
        Local search on k values where a_i = M * k_i.
        """
        max_k = MAX_VAL // M
        if max_k < 12:
            return None, float('inf')
        
        # Initialize k values
        # Strategy: Pick small k values that are "smooth" or random
        current_k = [random.randint(1, max_k) for _ in range(12)]
        # Ensure distinctness
        attempts = 0
        while len(set(current_k)) < 12 and attempts < 100:
            dup_idx = random.randint(0, 11)
            current_k[dup_idx] = random.randint(1, max_k)
            attempts += 1
        if len(set(current_k)) < 12:
            return None, float('inf')
        
        # Initialize prime counts
        # We track the count of sums divisible by each prime.
        prime_counts = defaultdict(int)
        
        def update_counts(k_list, delta, i, new_k_val=None):
            """
            Update prime_counts for changes involving index i.
            If new_k_val is None, remove old k_list[i].
            If new_k_val is not None, add new sums with new_k_val.
            """
            pass # We'll do manual updates in the loop for speed
            
        # Initial calculation
        for i in range(12):
            for j in range(i+1, 12):
                s = current_k[i] + current_k[j]
                factors = get_factors_cached(s)
                for p in factors:
                    prime_counts[p] += 1
        
        current_score = len(prime_counts)
        best_local_score = current_score
        best_local_k = current_k[:]
        
        # Simulated Annealing / Hill Climbing
        temp = 10.0
        cooling = 0.99
        min_temp = 0.1
        
        no_improve = 0
        max_no_improve = 5000
        
        while no_improve < max_no_improve and temp > min_temp and time.time() < deadline:
            i = random.randint(0, 11)
            old_k = current_k[i]
            
            # Generate new k
            # Bias towards small numbers to keep sums small (fewer primes)
            if random.random() < 0.7:
                new_k = random.randint(1, min(50, max_k))
            else:
                new_k = random.randint(1, max_k)
                
            if new_k == old_k or new_k in current_k:
                no_improve += 1
                continue
            
            # Calculate delta in score
            # Sums involving i: (k_i, k_j) for all j != i
            # Remove old sums
            old_factors_counts = defaultdict(int)
            new_factors_counts = defaultdict(int)
            
            for j in range(12):
                if j == i: continue
                s_old = old_k + current_k[j]
                s_new = new_k + current_k[j]
                
                f_old = get_factors_cached(s_old)
                f_new = get_factors_cached(s_new)
                
                for p in f_old:
                    old_factors_counts[p] += 1
                for p in f_new:
                    new_factors_counts[p] += 1
            
            # Apply changes to prime_counts
            # We need to know which primes drop to 0 and which become > 0
            
            primes_to_check = set(old_factors_counts.keys()) | set(new_factors_counts.keys())
            
            delta_score = 0
            valid_move = True
            
            # Check if any prime would become negative (shouldn't happen if logic is right)
            # and count changes in the union size
            
            # For each prime p:
            # count_old = prime_counts[p]
            # count_new = count_old - old_factors_counts[p] + new_factors_counts[p]
            
            # If count_old > 0 and count_new == 0: score decreases by 1
            # If count_old == 0 and count_new > 0: score increases by 1
            
            # We can simulate this without modifying prime_counts yet
            
            for p in primes_to_check:
                c_old = prime_counts[p]
                c_new = c_old - old_factors_counts[p] + new_factors_counts[p]
                
                if c_old > 0 and c_new == 0:
                    delta_score -= 1
                elif c_old == 0 and c_new > 0:
                    delta_score += 1
                # if c_old > 0 and c_new > 0: no change
                # if c_old == 0 and c_new == 0: no change
                
                if c_new < 0:
                    valid_move = False
                    break
            
            if not valid_move:
                no_improve += 1
                continue
                
            new_score = current_score + delta_score
            
            # Acceptance criterion
            if new_score < current_score:
                # Accept
                for p in primes_to_check:
                    prime_counts[p] -= old_factors_counts[p]
                    prime_counts[p] += new_factors_counts[p]
                    if prime_counts[p] == 0:
                        del prime_counts[p]
                
                current_k[i] = new_k
                current_score = new_score
                no_improve = 0
                
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_k = current_k[:]
                    
            else:
                # Metropolis acceptance
                if random.random() < math.exp(-delta_score / temp):
                    for p in primes_to_check:
                        prime_counts[p] -= old_factors_counts[p]
                        prime_counts[p] += new_factors_counts[p]
                        if prime_counts[p] == 0:
                            del prime_counts[p]
                    
                    current_k[i] = new_k
                    current_score = new_score
                    no_improve += 1 # Count as no improvement for best tracking
                else:
                    no_improve += 1
            
            temp *= cooling
            
        return best_local_k, best_local_score

    # Run search for each multiplier
    for M in multipliers:
        if time.time() > deadline:
            break
            
        # Multiple restarts for each M
        restarts = 20 if M == 6 else 50 # More restarts for harder cases
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            # Add some structured initialization
            if r == 0:
                # Try small consecutive numbers
                k_init = list(range(1, 13))
                if max(k_init) <= MAX_VAL // M:
                    current_k = k_init[:]
                else:
                    current_k = [random.randint(1, MAX_VAL // M) for _ in range(12)]
            elif r == 1:
                # Try even numbers
                k_init = [2*i for i in range(1, 13)]
                if max(k_init) <= MAX_VAL // M:
                    current_k = k_init[:]
                else:
                    current_k = [random.randint(1, MAX_VAL // M) for _ in range(12)]
            else:
                current_k = None # Let evaluate_and_search handle it
            
            # We need to modify evaluate_and_search to accept initial k
            # Let's refactor slightly or just run it.
            # For now, just call it. It handles random init.
            # To use specific inits, we'd need to pass them.
            # Let's just run the standard search.
            
            k_res, score_res = evaluate_and_search(M)
            
            if k_res is not None:
                # Calculate actual score to be sure
                nums = [M * k for k in k_res]
                # Verify distinctness and bounds
                if len(set(nums)) == 12 and max(nums) <= MAX_VAL:
                    # Recalculate score from scratch to avoid bugs
                    mask = 0
                    for i in range(12):
                        for j in range(i+1, 12):
                            s = nums[i] + nums[j]
                            f = get_factors_cached(s)
                            for p in f:
                                # We just need count of distinct primes
                                pass 
                    # Simpler: just use the set union
                    all_primes = set()
                    for i in range(12):
                        for j in range(i+1, 12):
                            s = nums[i] + nums[j]
                            all_primes.update(get_factors_cached(s))
                    
                    actual_score = len(all_primes)
                    
                    if actual_score < best_score:
                        best_score = actual_score
                        best_set = nums[:]
                        
                        # If we hit 5 or 6, we can stop early
                        if best_score <= 6:
                            return best_set

    # Fallback: Return the best found or a default set
    if best_set is None:
        # Should not happen
        best_set = [6*i for i in range(1, 13)]
        
    return best_set

# EVOLVE-BLOCK-END
