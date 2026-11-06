import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Generate primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Target L to achieve
    TARGET_L = 1000
    MAX_L = 5000  # Search up to 5000 to see if we can exceed 1000
    
    def evaluate_residues(residues):
        """Evaluate how many integers 1..MAX_L are covered at least twice"""
        counts = [0] * (MAX_L + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            for x in range(r, MAX_L + 1, p):
                if x >= 1:
                    counts[x] += 1
        
        # Find the largest L such that all 1..L have count >= 2
        L = 0
        for x in range(1, MAX_L + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L
    
    # Greedy approach with local search
    # Start with random initialization
    residues = [random.randint(0, p - 1) for p in primes]
    best_score = evaluate_residues(residues)
    best_residues = residues.copy()
    
    # Local search: try to improve by changing one residue at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_primes):
            if time.time() > deadline:
                break
            p = primes[i]
            old_r = residues[i]
            best_local_r = old_r
            best_local_score = best_score
            
            # Try all possible residues for this prime
            for r in range(p):
                if r == old_r:
                    continue
                residues[i] = r
                score = evaluate_residues(residues)
                if score > best_local_score:
                    best_local_score = score
                    best_local_r = r
                residues[i] = old_r
            
            if best_local_r != old_r:
                residues[i] = best_local_r
                best_score = best_local_score
                best_residues = residues.copy()
                improved = True
    
    # If we haven't reached 1000, try a more aggressive search
    if best_score < 1000 and time.time() < deadline - 30:
        # Try to focus on extending the prefix
        for _ in range(5):  # 5 restarts
            if time.time() > deadline - 10:
                break
            
            # Start from best_residues and do targeted improvements
            residues = best_residues.copy()
            current_score = best_score
            
            # Find the first uncovered position
            counts = [0] * (MAX_L + 1)
            for i, p in enumerate(primes):
                r = residues[i]
                for x in range(r, MAX_L + 1, p):
                    if x >= 1:
                        counts[x] += 1
            
            # Find first x with count < 2
            first_bad = None
            for x in range(1, MAX_L + 1):
                if counts[x] < 2:
                    first_bad = x
                    break
            
            if first_bad is None:
                break
            
            # Try to fix first_bad by adjusting residues of primes that could cover it
            for i in range(n_primes):
                if time.time() > deadline - 10:
                    break
                p = primes[i]
                # Check if this prime can cover first_bad
                r_needed = first_bad % p
                if residues[i] == r_needed:
                    continue
                
                # Try changing this residue to cover first_bad
                old_r = residues[i]
                residues[i] = r_needed
                
                # Evaluate
                new_counts = [0] * (MAX_L + 1)
                for j, pj in enumerate(primes):
                    rj = residues[j]
                    for x in range(rj, MAX_L + 1, pj):
                        if x >= 1:
                            new_counts[x] += 1
                
                new_score = 0
                for x in range(1, MAX_L + 1):
                    if new_counts[x] >= 2:
                        new_score = x
                    else:
                        break
                
                if new_score > current_score:
                    current_score = new_score
                    best_score = max(best_score, new_score)
                    best_residues = residues.copy()
                else:
                    residues[i] = old_r
    
    return best_residues
