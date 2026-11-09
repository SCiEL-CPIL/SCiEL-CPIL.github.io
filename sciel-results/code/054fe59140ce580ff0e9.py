"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 512
    candidates = list(range(N - 1, 0, -1))
    n = len(candidates)
    
    # Precompute prime factors for each candidate
    # prime_factors[i] is a set of prime factors of candidates[i]
    prime_factors = []
    for a in candidates:
        factors = set()
        temp = a
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                factors.add(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)
        prime_factors.append(factors)
        
    # Precompute bitmask for each candidate (index in prime list)
    # We only care about primes up to N
    primes = []
    is_prime = [True] * N
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, N):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N, i):
                is_prime[j] = False
                
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Bitmask for each candidate
    masks = [0] * n
    for i, a in enumerate(candidates):
        mask = 0
        for p in prime_factors[i]:
            if p in prime_index:
                mask |= (1 << prime_index[p])
        masks[i] = mask
        
    # Value of each candidate
    vals = [1.0 / (N - a) for a in candidates]
    
    # Initial solution: Greedy by value
    best_set = set()
    best_sum = 0.0
    used_mask = 0
    
    for i in range(n):
        if (masks[i] & used_mask) == 0:
            best_set.add(candidates[i])
            best_sum += vals[i]
            used_mask |= masks[i]
            
    # Local Search: Simulated Annealing on the "Foundation"
    # We treat the selection of small primes as the "foundation".
    # Actually, let's just do standard SA on the set itself, but with fast checks.
    # Since n=511, we can represent the set as a bitmask of size 511.
    
    current_set_mask = 0
    for a in best_set:
        idx = n - 1 - a  # candidates is reversed, so index of a is n-1-a
        current_set_mask |= (1 << idx)
        
    current_sum = best_sum
    
    def get_sum_from_mask(mask):
        s = 0.0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += vals[i]
            m >>= 1
            i += 1
        return s
        
    # Precompute valid neighbors? No, too many.
    # We will perform random moves: Add a random valid element, or remove a random element.
    
    T = 1.0
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline:
        if steps % 1000 == 0:
            T *= cooling
            
        # Random move
        # 1. Try to add a random element
        if random.random() < 0.5:
            # Pick a random index
            idx = random.randint(0, n - 1)
            if not (current_set_mask & (1 << idx)):
                # Check if coprime to all in set
                # This is O(n) in worst case, but average set size is small?
                # Actually, set size is around 100-200.
                # Let's check against the union of masks? No, we need pairwise.
                # We can maintain a global "forbidden mask" of primes used by the set.
                # Wait, if I use a global forbidden mask, I can check O(1).
                
                # Let's maintain global_forbidden_mask alongside current_set_mask
                # This requires re-initialization or careful tracking.
                # Let's restart the loop with a helper that maintains state properly.
                pass # Placeholder, logic below handles it
        
        # Better approach: Maintain global_forbidden_mask
        # Let's refactor to use a list for the set to allow O(1) add/remove checks?
        # No, O(n) gcd is fine if n is small. But bitmask is faster.
        
        # Let's stick to the bitmask of the SET, and compute the "prime usage mask" dynamically?
        # That's O(k) where k is set size.
        
        # Optimization: Precompute for each element i, the list of elements j that conflict with it.
        # conflict_mask[i] is a bitmask of all j such that gcd(candidates[i], candidates[j]) > 1.
        # Then adding i is valid if (current_set_mask & conflict_mask[i]) == 0.
        
    # Precompute conflict masks
    conflict_masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (masks[i] & masks[j]) != 0:
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i)
                
    # Re-initialize current state
    current_set_mask = 0
    for a in best_set:
        idx = n - 1 - a
        current_set_mask |= (1 << idx)
    current_sum = best_sum
    
    T = 0.01 # Start with small temperature since we have a good greedy start
    cooling = 0.999
    
    while time.time() < deadline:
        T *= cooling
        if T < 1e-9:
            T = 1e-9
            
        # 1. Try Add
        if random.random() < 0.6:
            idx = random.randint(0, n - 1)
            if not (current_set_mask & (1 << idx)):
                if (current_set_mask & conflict_masks[idx]) == 0:
                    new_mask = current_set_mask | (1 << idx)
                    new_sum = current_sum + vals[idx]
                    delta = new_sum - current_sum
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_set_mask = new_mask
                        current_sum = new_sum
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = set()
                            m = current_set_mask
                            i = 0
                            while m:
                                if m & 1:
                                    best_set.add(candidates[i])
                                m >>= 1
                                i += 1
        else:
            # 2. Try Remove
            # Pick a random element from the set
            # We can find a random set bit
            if current_set_mask != 0:
                # Get list of set bits? Or just pick a random index and check if set.
                # To be efficient, we can maintain a list of indices in the set.
                # But for now, random index check is O(1) probability of success proportional to density.
                # Density is ~200/511 ~ 40%. So expected 2.5 tries.
                while True:
                    idx = random.randint(0, n - 1)
                    if current_set_mask & (1 << idx):
                        break
                    
                new_mask = current_set_mask & ~(1 << idx)
                new_sum = current_sum - vals[idx]
                delta = new_sum - current_sum
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_set_mask = new_mask
                    current_sum = new_sum
                    
        # 3. Try Swap (Add one, Remove one) - might be complex, skip for now.
        
    return list(best_set)
# EVOLVE-BLOCK-END
