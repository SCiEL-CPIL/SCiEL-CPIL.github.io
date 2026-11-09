"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n = len(primes)  # 168

    # Evaluate: find largest L such that every x in 1..L has >= 2 covering primes
    def evaluate(residues, max_check=100000):
        # For each x, count how many primes cover it
        # Use an array for counts up to some reasonable bound
        # We need to find the first x where count < 2
        # Efficient: for each prime p with residue r, it covers x = r + kp
        # We'll use a difference array or just check incrementally
        
        # Better approach: create an array of counts
        # But max_check could be large. Let's think about what L we're aiming for.
        # Start with a reasonable bound and expand if needed.
        
        # For efficiency, let's use the fact that we only need to find the first failure
        # Count coverage for x = 1, 2, 3, ... until we find one with < 2
        
        # Precompute: for each prime index i, residue r_i, prime p_i
        # x is covered by prime i if x % p_i == r_i
        
        # Use a list to track counts, but only up to a reasonable limit
        # Let's start with a limit and increase if all pass
        
        limit = 200  # start small
        while limit <= max_check:
            counts = [0] * (limit + 1)
            for i, p in enumerate(primes):
                r = residues[i]
                # x in [1, limit] where x ≡ r (mod p)
                # x = r + kp, need 1 <= x <= limit
                # k_min = ceil((1 - r) / p) if r < 1 else 0
                # Actually: x = r + kp >= 1 => k >= (1-r)/p
                # x = r + kp <= limit => k <= (limit - r)/p
                if r == 0:
                    # x = kp, k >= 1 (since x>=1)
                    k_min = 1
                else:
                    k_min = 0
                k_max = (limit - r) // p
                if k_min > k_max:
                    continue
                for k in range(k_min, k_max + 1):
                    x = r + k * p
                    if 1 <= x <= limit:
                        counts[x] += 1
            
            # Find first x in 1..limit with count < 2
            first_fail = limit + 1
            for x in range(1, limit + 1):
                if counts[x] < 2:
                    first_fail = x
                    break
            
            if first_fail <= limit:
                return first_fail - 1
            else:
                # All passed, try larger limit
                limit *= 2
        
        return max_check

    # Seed solution: assign each prime to cover ceil((i+1)/2)
    def seed_solution():
        return [((i + 1) // 2) % p for i, p in enumerate(primes)]

    # Random solution
    def random_solution():
        return [random.randint(0, p - 1) for p in primes]

    # Local search: hill climbing
    def local_search(residues, max_time=None):
        if max_time is None:
            max_time = deadline - 5.0
        
        current = list(residues)
        current_score = evaluate(current)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try all single-residue changes
            for i in range(n):
                p = primes[i]
                old_r = current[i]
                # Try a few random new residues
                for _ in range(5):
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        continue
                    current[i] = new_r
                    new_score = evaluate(current)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current[i] = old_r
                
                if time.time() >= max_time:
                    break
        
        return current, current_score

    # Main: multiple restarts with local search
    best_residues = seed_solution()
    best_score = evaluate(best_residues)
    
    random.seed(42)
    
    # First try the seed
    best_residues, best_score = local_search(best_residues, deadline - 10.0)
    
    # Multiple random restarts
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        cand = random_solution()
        cand, score = local_search(cand, deadline - 5.0)
        if score > best_score:
            best_score = score
            best_residues = cand
            if best_score >= 1000:
                break
    
    return best_residues
# EVOLVE-BLOCK-END
