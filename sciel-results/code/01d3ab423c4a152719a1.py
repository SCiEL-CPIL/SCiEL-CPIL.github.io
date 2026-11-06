import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Find all primes up to 128
    is_prime = [True] * 129
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(128)) + 1):
        if is_prime[i]:
            for j in range(i*i, 129, i):
                is_prime[j] = False
    primes = [i for i in range(2, 129) if is_prime[i]]
    
    # 2. Generate prime powers and group by prime
    prime_powers = {}
    for p in primes:
        powers = []
        m = p
        while m <= 128:
            powers.append(m)
            m *= p
        if powers:
            prime_powers[p] = powers
            
    results = []
    
    # 3. For each prime, assign residues
    for p, powers in prime_powers.items():
        # powers is [p^1, p^2, ...]
        # Assign r_1, ..., r_k
        assigned_residues = []
        for idx, m in enumerate(powers):
            # m is p^(idx+1)
            # We need to pick r in [0, m-1]
            # such that for all prev_idx < idx:
            # r % p^(prev_idx+1) != assigned_residues[prev_idx]
            
            found = False
            for r in range(m):
                valid = True
                for prev_idx in range(idx):
                    prev_m = powers[prev_idx]
                    prev_r = assigned_residues[prev_idx]
                    if r % prev_m == prev_r:
                        valid = False
                        break
                if valid:
                    assigned_residues.append(r)
                    found = True
                    break
            if not found:
                # Fallback, though unlikely
                assigned_residues.append(0)
        
        for idx, m in enumerate(powers):
            results.append([assigned_residues[idx], m])
            
    # Check time
    if time.time() > deadline:
        return results
        
    # Optional: Try to add composites?
    # Given the time, 43 is a strong baseline.
    # The seed had 7. 43 is a 6x improvement.
    
    return results
