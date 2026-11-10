"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000
DEADLINE = 320  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    
    # 1. Precompute primes
    all_primes = _primes(N + 1)
    prime_set = set(all_primes)
    
    # 2. Initialize sparse set
    # Start with 1 and primes p > sqrt(N)
    sqrt_N = int(N ** 0.5) + 1
    points = {1}
    for p in all_primes:
        if p > sqrt_N:
            points.add(p)
    
    # Ensure we are within [0, N]
    points = {x for x in points if 0 <= x <= N}
    
    # 3. Initialize structures
    residue_counts = {}
    missing_counts = {}
    
    for p in all_primes:
        counts = [0] * p
        residue_counts[p] = counts
        missing_counts[p] = p  # initially all missing
        
    # Populate initial counts
    for x in points:
        for p in all_primes:
            r = x % p
            if residue_counts[p][r] == 0:
                missing_counts[p] -= 1
            residue_counts[p][r] += 1
            
    # 4. Repair Phase: Ensure admissibility
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > DEADLINE - 10:
            break
            
        n = len(points)
        # Check primes p <= n
        for p in all_primes:
            if p > n:
                break
            
            if missing_counts[p] == 0:
                # Need to remove elements to create a missing residue
                min_count = min(residue_counts[p])
                target_residues = [r for r, c in enumerate(residue_counts[p]) if c == min_count]
                target = target_residues[0]
                
                # Remove all points x such that x % p == target
                to_remove = [x for x in points if x % p == target]
                
                if not to_remove:
                    continue
                    
                for x in to_remove:
                    points.remove(x)
                    for pp in all_primes:
                        r = x % pp
                        residue_counts[pp][r] -= 1
                        if residue_counts[pp][r] == 0:
                            missing_counts[pp] += 1
                            
                changed = True
                break  # Restart check as points changed significantly
        
    # 5. Optimization Phase: Add back elements
    candidates = [x for x in range(N + 1) if x not in points]
    
    added_any = True
    while added_any:
        added_any = False
        if time.time() - start_time > DEADLINE - 10:
            break
        
        n = len(points)
        new_n = n + 1
        new_prime = None
        if new_n in prime_set:
            new_prime = new_n
            
        for x in candidates:
            if x in points:
                continue
                
            valid = True
            
            # Check 1: Existing primes p <= n
            for p in all_primes:
                if p > n:
                    break
                r = x % p
                if residue_counts[p][r] == 0:
                    if missing_counts[p] == 1:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Check 2: New prime bound
            if new_prime is not None:
                p = new_prime
                r_x = x % p
                if residue_counts[p][r_x] == 0:
                    final_missing = missing_counts[p] - 1
                else:
                    final_missing = missing_counts[p]
                
                if final_missing == 0:
                    valid = False
            
            if valid:
                points.add(x)
                for pp in all_primes:
                    r = x % pp
                    if residue_counts[pp][r] == 0:
                        missing_counts[pp] -= 1
                    residue_counts[pp][r] += 1
                    
                added_any = True
                break  # Restart to re-evaluate prime bound
        
    # 6. Targeted removal pass: For primes with exactly 1 missing residue,
    # remove the least-populated residue to create a second missing class.
    # This allows more additions in subsequent passes.
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > DEADLINE - 15:
            break
            
        n = len(points)
        for p in all_primes:
            if p > n:
                break
            
            if missing_counts[p] == 1:
                # Find the least populated residue
                min_count = min(residue_counts[p])
                target_residues = [r for r, c in enumerate(residue_counts[p]) if c == min_count]
                target = target_residues[0]
                
                to_remove = [x for x in points if x % p == target]
                if not to_remove:
                    continue
                    
                # Only remove if it helps (removes few elements)
                if len(to_remove) <= 3:
                    for x in to_remove:
                        points.remove(x)
                        for pp in all_primes:
                            r = x % pp
                            residue_counts[pp][r] -= 1
                            if residue_counts[pp][r] == 0:
                                missing_counts[pp] += 1
                    changed = True
                    break
    
    # 7. Second add-back pass
    candidates = [x for x in range(N + 1) if x not in points]
    
    added_any = True
    while added_any:
        added_any = False
        if time.time() - start_time > DEADLINE - 10:
            break
        
        n = len(points)
        new_n = n + 1
        new_prime = None
        if new_n in prime_set:
            new_prime = new_n
            
        for x in candidates:
            if x in points:
                continue
                
            valid = True
            
            for p in all_primes:
                if p > n:
                    break
                r = x % p
                if residue_counts[p][r] == 0:
                    if missing_counts[p] == 1:
                        valid = False
                        break
            
            if not valid:
                continue
                
            if new_prime is not None:
                p = new_prime
                r_x = x % p
                if residue_counts[p][r_x] == 0:
                    final_missing = missing_counts[p] - 1
                else:
                    final_missing = missing_counts[p]
                
                if final_missing == 0:
                    valid = False
            
            if valid:
                points.add(x)
                for pp in all_primes:
                    r = x % pp
                    if residue_counts[pp][r] == 0:
                        missing_counts[pp] -= 1
                    residue_counts[pp][r] += 1
                    
                added_any = True
                break
    
    return list(points)
# EVOLVE-BLOCK-END
