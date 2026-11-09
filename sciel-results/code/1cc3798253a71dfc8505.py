import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Generate primes up to 1000
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
        
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Select a set of primes with sum of reciprocals <= 1
    # Strategy: Greedy smallest primes, excluding 2 to allow more primes
    # Let's try a few configurations
    configs = []
    
    # Config 1: Smallest primes starting from 3
    sum_rec = 0
    p_set = []
    for p in primes:
        if p == 2: continue
        if sum_rec + 1/p <= 1:
            p_set.append(p)
            sum_rec += 1/p
        else:
            break
    configs.append(p_set)
    
    # Config 2: Include 2, then smallest
    sum_rec = 0
    p_set = []
    for p in primes:
        if sum_rec + 1/p <= 1:
            p_set.append(p)
            sum_rec += 1/p
        else:
            break
    configs.append(p_set)
    
    # Config 3: Random subsets? Maybe later if time permits.
    
    best_L = 0
    best_residues = None
    best_primes = None
    
    for p_list in configs:
        if not p_list: continue
        
        # Initialize residues randomly
        residues = [random.randint(0, p-1) for p in p_list]
        
        # Local search
        current_L = 0
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        
        while time.time() < deadline:
            # Evaluate current
            # To speed up, we can cache or just re-eval
            # Evaluation is O(L * |P| / p_avg). L ~ 2000, |P| ~ 15. Very fast.
            
            def get_coverage(res, p_l, limit=5000):
                covered = [False] * (limit + 1)
                for r, p in zip(res, p_l):
                    for i in range(r, limit+1, p):
                        covered[i] = True
                # Find first uncovered
                for i in range(1, limit+1):
                    if not covered[i]:
                        return i - 1
                return limit

            current_L = get_coverage(residues, p_list, 5000)
            
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
                best_primes = p_list.copy()
                
            # If we covered 5000, try higher limit
            if current_L == 5000:
                # Increase limit for next evals
                pass # Simplified: just stick to 5000 for now, or dynamic
                
            # Pick a prime to change
            i = random.randint(0, len(p_list) - 1)
            old_r = residues[i]
            # Random new residue
            new_r = random.randint(0, p_list[i] - 1)
            residues[i] = new_r
            
            new_L = get_coverage(residues, p_list, 5000)
            
            if new_L > current_L:
                current_L = new_L
                T = max(T_min, T * cooling)
            else:
                # Accept with probability
                if random.random() < (new_L - current_L) / T: # Note: new_L - current_L is negative
                    current_L = new_L
                else:
                    residues[i] = old_r
                T = max(T_min, T * cooling)
                
            if T == T_min and new_L <= current_L:
                # Restart with new random residues if stuck?
                if random.random() < 0.01:
                    residues = [random.randint(0, p-1) for p in p_list]
                    T = 1.0
                    current_L = get_coverage(residues, p_list, 5000)

    # Final check with best found
    if best_primes:
        # Verify sum of reciprocals
        s = sum(Fraction(1, p) for p in best_primes)
        if s <= 1:
            return [[p, r] for p, r in zip(best_primes, best_residues)]
            
    # Fallback to seed-like solution
    primes_seed = [p for p in range(13, 1001) if is_prime(p)]
    return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
