"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    N = 100000  # cap
    
    # Fast evaluation: given residues array, compute doubly-covered length
    def evaluate(residues):
        cover = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            # All x in [1, N] such that x % p == r
            # Start: smallest x >= 1 with x % p == r
            if r == 0:
                start = p  # x=p, x=2p, ... but x>=1, so if p>=1, start=p
                # Actually x % p == 0 means x = p, 2p, ...
                # But we need x >= 1, so start = p if p >= 1
                if start < 1:
                    start = 1
            else:
                start = r
                if start < 1:
                    start = 1
            # Generate all x = start, start+p, start+2p, ... <= N
            if start <= N:
                indices = np.arange(start, N + 1, p)
                cover[indices] += 1
        # Find largest L such that all cover[1..L] >= 2
        # Find first position where cover < 2
        if N < 1:
            return 0
        below_two = np.where(cover[1:N+1] < 2)[0]
        if len(below_two) == 0:
            return N
        else:
            return below_two[0]  # first index where cover < 2, so L = that index
    
    # Greedy initialization
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Try a few initial strategies and pick the best
    best_residues = None
    best_score = 0
    
    # Strategy 1: all zeros
    r1 = np.zeros(n_primes, dtype=np.int64)
    s1 = evaluate(r1)
    if s1 > best_score:
        best_score = s1
        best_residues = r1.copy()
    
    # Strategy 2: r_p = p//2
    r2 = np.array([p // 2 for p in primes], dtype=np.int64)
    s2 = evaluate(r2)
    if s2 > best_score:
        best_score = s2
        best_residues = r2.copy()
    
    # Strategy 3: r_p = 1
    r3 = np.ones(n_primes, dtype=np.int64)
    s3 = evaluate(r3)
    if s3 > best_score:
        best_score = s3
        best_residues = r3.copy()
    
    # Strategy 4: original seed
    r4 = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    s4 = evaluate(r4)
    if s4 > best_score:
        best_score = s4
        best_residues = r4.copy()
    
    residues = best_residues.copy()
    current_score = best_score
    
    # Local search with simulated annealing
    import random
    random.seed(42)
    np.random.seed(42)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    for step in range(200000):
        if time.time() > deadline:
            break
        
        # Pick a random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        
        # Try a random new residue
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        residues[i] = new_r
        new_score = evaluate(residues)
        
        if new_score >= current_score:
            current_score = new_score
            best_residues = residues.copy()
            best_score = max(best_score, current_score)
        else:
            # Simulated annealing: accept with probability exp((new-old)/T)
            delta = new_score - current_score
            if delta < 0 and random.random() < np.exp(delta / T):
                current_score = new_score
            else:
                residues[i] = old_r
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final hill-climbing polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_primes):
            if time.time() > deadline:
                break
            p = primes[i]
            old_r = residues[i]
            best_local_r = old_r
            best_local_s = current_score
            # Try a few random residues
            for _ in range(5):
                trial_r = random.randint(0, p - 1)
                if trial_r == old_r:
                    continue
                residues[i] = trial_r
                s = evaluate(residues)
                if s > best_local_s:
                    best_local_s = s
                    best_local_r = trial_r
            residues[i] = best_local_r
            if best_local_r != old_r:
                improved = True
                current_score = best_local_s
                best_residues = residues.copy()
                best_score = max(best_score, current_score)
    
    # Use the best found
    residues = best_residues
    return [int(r) for r in residues]
# EVOLVE-BLOCK-END
