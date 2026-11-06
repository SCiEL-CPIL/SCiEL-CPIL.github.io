import time
import random
import math

# Precompute Smallest Prime Factor (SPF) up to 20000
MAX_VAL = 10000
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:  # i is prime
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

# Precompute prime factors as bitmasks for all numbers up to MAX_SUM
# We'll use the first 64 primes as our "universe" for bitmask representation
primes_list = []
for i in range(2, MAX_SUM + 1):
    if spf[i] == i:
        primes_list.append(i)
        if len(primes_list) >= 64:
            break

prime_to_bit = {p: (1 << i) for i, p in enumerate(primes_list)}

# Precompute bitmask of prime factors for each number
pf_mask = [0] * (MAX_SUM + 1)
for i in range(2, MAX_SUM + 1):
    mask = 0
    n = i
    while n > 1:
        p = spf[n]
        if p in prime_to_bit:
            mask |= prime_to_bit[p]
        while n % p == 0:
            n //= p
    pf_mask[i] = mask

def count_bits(x):
    """Count set bits in an integer."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = None
    best_score = float('inf')
    
    def calc_score_mask(nums):
        """Calculate score using bitmask OR operation."""
        mask = 0
        n = len(nums)
        for i in range(n):
            xi = nums[i]
            for j in range(i+1, n):
                s = xi + nums[j]
                mask |= pf_mask[s]
        return count_bits(mask)

    # CSP / Clique Finding Strategy
    # We look for 12 integers such that all pairwise sums have prime factors in a small set S.
    # Since we want to minimize distinct primes, we start with small S.
    # If all numbers are multiples of 6, sums are divisible by 6 (primes 2, 3).
    # We check if we can restrict sums to only have primes 2, 3, and maybe 5.
    
    # Candidate pool: Multiples of 6 up to 10000.
    # To make the graph dense, we prefer numbers that are "smooth" relative to the target primes.
    # Let's generate candidates.
    candidates = []
    for k in range(1, MAX_VAL // 6 + 1):
        val = 6 * k
        candidates.append(val)
    
    # Sort candidates by "smoothness" or just keep a manageable subset?
    # 1666 candidates is small enough to build a graph or do backtracking.
    # However, checking all pairs is O(N^2) ~ 1.4M ops, which is fast.
    
    # Let's try to find a clique of size 12 in the graph where:
    # Edge (u, v) exists if prime_factors(u+v) is a subset of AllowedPrimes.
    
    # Try AllowedPrimes = {2, 3}
    # Then u+v must be of the form 2^a * 3^b.
    # Since u, v are multiples of 6, u+v = 6(k1+k2).
    # 6(k1+k2) = 2*3*(k1+k2).
    # So k1+k2 must be of the form 2^a * 3^b.
    
    # Let's define the compatibility based on the reduced sum.
    # If we pick u=6*k1, v=6*k2, sum = 6*(k1+k2).
    # Primes of sum = {2, 3} U primes(k1+k2).
    # We want primes(k1+k2) to be subset of ExtraPrimes.
    
    def find_clique_for_extra_primes(extra_primes_set):
        """
        Find 12 distinct k values (1..1666) such that for all pairs i<j,
        prime_factors(k_i + k_j) is a subset of extra_primes_set.
        Returns the list of k values or None.
        """
        # Precompute compatibility for k values
        # k ranges from 1 to 1666.
        # Sum k_i + k_j ranges from 1+2=3 to 1666+1665=3331.
        # We need to check if prime factors of sum are in {2,3} U extra_primes_set.
        
        allowed_primes = {2, 3} | extra_primes_set
        
        # Precompute a boolean array for valid sums
        # valid_sum[s] is True if all prime factors of s are in allowed_primes
        max_k_sum = 1666 + 1665
        valid_sum = [False] * (max_k_sum + 1)
        for s in range(2, max_k_sum + 1):
            # Check if s is smooth with respect to allowed_primes
            is_smooth = True
            temp_s = s
            for p in allowed_primes:
                while temp_s % p == 0:
                    temp_s //= p
            if temp_s != 1:
                is_smooth = False
            valid_sum[s] = is_smooth
        
        # Now we have a graph of k values (1..1666)
        # Node i is connected to Node j if valid_sum[i+j] is True.
        # We want a clique of size 12.
        
        # To speed up backtracking, we can order nodes by degree (number of valid neighbors).
        # Or just iterate.
        # Let's use a simple backtracking with pruning.
        
        # Optimization: If we restrict k to be multiples of some number, we might reduce search space?
        # No, let's just search the k space.
        
        # Degree calculation for ordering
        degrees = [0] * 1667
        # This O(N^2) is 1.4M checks, acceptable.
        for i in range(1, 1667):
            for j in range(i+1, 1667):
                if valid_sum[i+j]:
                    degrees[i] += 1
                    degrees[j] += 1
        
        # Sort k values by degree descending
        sorted_ks = sorted(range(1, 1667), key=lambda x: degrees[x], reverse=True)
        
        # Backtracking
        current_clique = []
        
        def backtrack(start_idx):
            if len(current_clique) == 12:
                return True
            
            # Pruning: if not enough nodes left
            if len(sorted_ks) - start_idx < 12 - len(current_clique):
                return False
                
            for idx in range(start_idx, len(sorted_ks)):
                k_val = sorted_ks[idx]
                
                # Check if k_val is compatible with all in current_clique
                compatible = True
                for c in current_clique:
                    if not valid_sum[c + k_val]:
                        compatible = False
                        break
                
                if compatible:
                    current_clique.append(k_val)
                    if backtrack(idx + 1):
                        return True
                    current_clique.pop()
                    
            return False
        
        # Start backtracking
        # We can also try to force small k values first as they are more likely to have small sums?
        # Actually, high degree means more connections, so sorting by degree is good.
        
        if backtrack(0):
            return current_clique[:]
        return None

    # Try Extra Primes = {} (Only 2, 3)
    # This requires k_i + k_j to be 2^a * 3^b.
    ks = find_clique_for_extra_primes(set())
    if ks:
        best_set = [6*k for k in ks]
        best_score = 2
        return best_set

    # Try Extra Primes = {5} (Primes 2, 3, 5)
    ks = find_clique_for_extra_primes({5})
    if ks:
        best_set = [6*k for k in ks]
        best_score = 3
        return best_set

    # Try Extra Primes = {5, 7} (Primes 2, 3, 5, 7)
    ks = find_clique_for_extra_primes({5, 7})
    if ks:
        best_set = [6*k for k in ks]
        best_score = 4
        return best_set

    # If CSP fails (timeout or no solution), fall back to random local search
    # (Re-implementing the original logic briefly or just returning a valid set)
    
    # Quick fallback: Random search
    def local_search_k(k_init, max_no_improve=200):
        current_k = k_init[:]
        current_score = calc_score_mask([6*k for k in current_k])
        best_k = current_k[:]
        best_sc = current_score
        no_improve = 0
        
        while no_improve < max_no_improve and time.time() < deadline:
            i = random.randint(0, 11)
            old_k = current_k[i]
            new_k = random.randint(1, 1666)
            
            if new_k == old_k or new_k in current_k:
                no_improve += 1
                continue
            
            current_k[i] = new_k
            nums = [6*k for k in current_k]
            new_score = calc_score_mask(nums)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if current_score < best_sc:
                    best_sc = current_score
                    best_k = current_k[:]
            else:
                current_k[i] = old_k
                no_improve += 1
        
        return best_k, best_sc

    # Generate a few random inits
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        k_init = random.sample(range(1, 1667), 12)
        bk, bsc = local_search_k(k_init)
        if bsc < best_score:
            best_score = bsc
            best_set = [6*k for k in bk]
            
    if best_set is None:
        # Last resort valid set
        best_set = [6*i for i in range(1, 13)]
        best_score = calc_score_mask(best_set)
        
    return best_set

# EVOLVE-BLOCK-END
