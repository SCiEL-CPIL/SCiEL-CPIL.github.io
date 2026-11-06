import numpy as np

def construct():
    # Define the range
    start = 10000
    end = 1000000  # n can be up to 1000000, so we need lpf for 1000001
    
    # We need lpf for numbers in [start, end+1]
    # Let's create an array for indices 0 to end+1 for easy indexing, 
    # or just offset the range. 
    # Using a full array of size end+2 is simplest and memory is ~8MB for int32, which is fine.
    
    max_val = end + 1
    # Initialize LPF array with 0. 0 indicates "not yet assigned" or "prime itself" later.
    lpf = np.zeros(max_val + 1, dtype=np.int64)
    
    # Generate primes up to sqrt(max_val) ~ 1000
    # Simple sieve for small primes
    limit = int(np.sqrt(max_val)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            # Mark multiples as not prime
            step = i
            start_multiple = i * i
            if start_multiple <= limit:
                is_prime[start_multiple:limit+1:step] = False

    # Fill LPF array
    # For each prime p, update lpf for all multiples of p in [start, max_val]
    for p in primes:
        # First multiple of p >= start
        first_multiple = ((start + p - 1) // p) * p
        for multiple in range(first_multiple, max_val + 1, p):
            lpf[multiple] = p
            
    # Handle primes and 1
    # If lpf[i] is 0, then i is either 1 or a prime.
    # In our range [10000, 1000001], 1 is not included.
    # So if lpf[i] == 0, i is prime, so lpf[i] = i.
    # We can do this efficiently:
    # Indices where lpf is 0
    zero_indices = np.where(lpf == 0)[0]
    # Filter to our range of interest [start, max_val]
    # Actually, we only care about [start, max_val]
    # But setting lpf[i]=i for all i where lpf[i]==0 is safe and fast enough if done carefully.
    # Let's just iterate over the zero indices within our range.
    # Since most numbers are composite, zero_indices will be small (primes).
    
    # More efficient: 
    # Create a boolean mask for the range of interest
    # But we need to update the array.
    # Let's just loop over primes found? No, the zeros are the primes > limit.
    # We can just do:
    # lpf[zero_indices] = zero_indices
    # But zero_indices includes 0, 1, and primes < start.
    # We only care about [start, max_val].
    
    # Let's refine:
    # The loop above set lpf for composites.
    # Primes > limit remain 0.
    # We need to set lpf[p] = p for all primes p in [start, max_val].
    
    # We can find them by checking where lpf is 0 in the range [start, max_val]
    range_mask = np.zeros(max_val + 1, dtype=bool)
    range_mask[start:max_val+1] = True
    prime_candidates = np.where((lpf == 0) & range_mask)[0]
    lpf[prime_candidates] = prime_candidates
    
    # Now compute score for n in [start, end]
    # score[n] = max(lpf[n], lpf[n+1])
    # n ranges from start to end
    # lpf indices needed: start to end+1 (which is max_val)
    
    lpf_n = lpf[start:end+1]
    lpf_n1 = lpf[start+1:end+2]
    
    scores = np.maximum(lpf_n, lpf_n1)
    
    # Find index of minimum score
    min_idx = np.argmin(scores)
    
    # The actual n is start + min_idx
    return start + min_idx

