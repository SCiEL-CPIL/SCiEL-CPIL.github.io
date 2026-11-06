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
    
    # KEY INSIGHT: If all numbers are multiples of 6, then all sums are divisible by 6.
    # So prime factors always include 2 and 3. We need to minimize additional primes.
    # Let nums[i] = 6 * k[i], then sum = 6*(k[i]+k[j])
    # Prime factors of sum = {2, 3} ∪ prime_factors(k[i]+k[j])
    # So we need to find 12 distinct k[i] to minimize union of prime_factors(k[i]+k[j])
    
    # Strategy: Search over k values (smaller numbers), then multiply by 6
    # k[i] must be distinct positive integers, and 6*k[i] <= 10000, so k[i] <= 1666
    
    MAX_K = MAX_VAL // 6  # 1666
    
    # Precompute prime factor masks for sums of k values
    # k[i] + k[j] can be up to 2*MAX_K = 3332
    # We already have pf_mask for all numbers up to 20000, so we can use it
    
    def calc_score_k(k_list):
        """Calculate score for k values (before multiplying by 6).
        The actual score = 2 (for primes 2,3) + additional primes from k sums."""
        mask = 0
        n = len(k_list)
        for i in range(n):
            ki = k_list[i]
            for j in range(i+1, n):
                s = ki + k_list[j]
                mask |= pf_mask[s]
        # The mask includes primes 2 and 3 if they divide the sums, plus others
        # Since all sums are even (k_i + k_j), prime 2 is always included
        # We need to count all distinct primes in the mask
        return count_bits(mask)
    
    # Try structured initializations for k values
    # If k values are all even, then k_i + k_j is even, so 2 is in the sum's factors
    # If k values are all odd, then k_i + k_j is even, so 2 is in the sum's factors
    # Actually, k_i + k_j is always even if k_i and k_j have same parity, odd if different
    # To ensure 2 is always a factor, make all k_i same parity
    
    # Try: all k_i even (so k_i + k_j even, factor 2 guaranteed)
    # Try: all k_i odd (so k_i + k_j even, factor 2 guaranteed)
    
    # Let's try k_i = 2*i for i in 1..12 (all even)
    k_init1 = [2*i for i in range(1, 13)]
    # k_i = 2*i+1 for i in 0..11 (all odd)
    k_init2 = [2*i+1 for i in range(0, 12)]
    # k_i = i for i in 1..12 (mixed parity)
    k_init3 = list(range(1, 13))
    # k_i = 6*i for i in 1..12 (all multiples of 6, so sums divisible by 6)
    k_init4 = [6*i for i in range(1, 13)]
    # k_i = 3*i for i in 1..12 (all multiples of 3)
    k_init5 = [3*i for i in range(1, 13)]
    
    for k_init in [k_init1, k_init2, k_init3, k_init4, k_init5]:
        if all(0 < k <= MAX_K for k in k_init) and len(set(k_init)) == 12:
            sc = calc_score_k(k_init)
            # The actual nums are 6*k, and the score is the same as calc_score_mask
            # because pf_mask[6*s] includes factors of s plus possibly 2,3
            # But we want to count distinct primes in the actual sums
            nums = [6*k for k in k_init]
            actual_sc = calc_score_mask(nums)
            if actual_sc < best_score:
                best_score = actual_sc
                best_set = nums[:]
    
    # Local search on k values
    current_k = k_init1[:]  # Start with all even k values
    current_score = calc_score_mask([6*k for k in current_k])
    if current_score < best_score:
        best_score = current_score
        best_set = [6*k for k in current_k]
    
    no_improve = 0
    max_no_improve = 1000
    
    while no_improve < max_no_improve and time.time() < deadline:
        i = random.randint(0, 11)
        old_k = current_k[i]
        
        # Generate new k value with bias towards small, smooth numbers
        if random.random() < 0.5:
            # Small even numbers (to keep sums even)
            new_k = random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120])
            if random.random() < 0.3:
                # Scale up
                new_k *= random.randint(1, MAX_K // new_k)
        elif random.random() < 0.5:
            # Small odd numbers
            new_k = random.choice([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39])
            if random.random() < 0.3:
                new_k *= random.randint(1, MAX_K // new_k)
        else:
            new_k = random.randint(1, MAX_K)
        
        if new_k == old_k or new_k in current_k:
            no_improve += 1
            continue
        
        current_k[i] = new_k
        nums = [6*k for k in current_k]
        new_score = calc_score_mask(nums)
        
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_set = nums[:]
                if best_score <= 4:
                    break
        else:
            current_k[i] = old_k
            no_improve += 1
    
    # Try more restarts with different initial structures
    if best_score > 5:
        for r in range(30):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            strategy = random.randint(0, 3)
            if strategy == 0:
                # All even, small
                current_k = [2*random.randint(1, 100) for _ in range(12)]
            elif strategy == 1:
                # All odd, small
                current_k = [2*random.randint(1, 100)+1 for _ in range(12)]
            elif strategy == 2:
                # Multiples of 6
                current_k = [6*random.randint(1, 277) for _ in range(12)]
            else:
                # Random
                current_k = [random.randint(1, MAX_K) for _ in range(12)]
            
            # Ensure distinctness
            attempts = 0
            while len(set(current_k)) < 12 and attempts < 100:
                dup_idx = random.randint(0, 11)
                current_k[dup_idx] = random.randint(1, MAX_K)
                attempts += 1
            
            if len(set(current_k)) < 12:
                continue
            
            nums = [6*k for k in current_k]
            current_score = calc_score_mask(nums)
            
            if current_score < best_score:
                best_score = current_score
                best_set = nums[:]
                if best_score <= 4:
                    break
            
            no_improve = 0
            max_no_improve = 300
            
            while no_improve < max_no_improve and time.time() < deadline:
                i = random.randint(0, 11)
                old_k = current_k[i]
                
                if random.random() < 0.5:
                    new_k = random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120])
                    if random.random() < 0.3:
                        new_k *= random.randint(1, MAX_K // new_k)
                elif random.random() < 0.5:
                    new_k = random.choice([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39])
                    if random.random() < 0.3:
                        new_k *= random.randint(1, MAX_K // new_k)
                else:
                    new_k = random.randint(1, MAX_K)
                
                if new_k == old_k or new_k in current_k:
                    no_improve += 1
                    continue
                
                current_k[i] = new_k
                nums = [6*k for k in current_k]
                new_score = calc_score_mask(nums)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_set = nums[:]
                        if best_score <= 4:
                            break
                else:
                    current_k[i] = old_k
                    no_improve += 1
    
    return best_set

# EVOLVE-BLOCK-END
