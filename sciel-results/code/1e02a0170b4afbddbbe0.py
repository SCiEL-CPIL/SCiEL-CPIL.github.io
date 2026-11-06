"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)  # 168
    max_check = 100000

    # Build coverage array incrementally
    # coverage[x] = number of primes p such that x ≡ r_p (mod p)
    # We maintain this array and update it when a single residue changes.

    def build_coverage(residues):
        """Build full coverage array from scratch."""
        cov = np.zeros(max_check + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_check:
                cov[start::p] += 1
        return cov

    def get_L(cov):
        """Find largest L such that cov[1..L] >= 2."""
        # Find first index >= 1 where cov < 2
        mask = cov[1:] < 2
        if not mask.any():
            return max_check
        first_gap = int(np.argmax(mask)) + 1
        return first_gap - 1

    def update_coverage(cov, prime_idx, old_r, new_r):
        """Update coverage array when prime at prime_idx changes residue from old_r to new_r."""
        p = primes[prime_idx]
        # Remove old contribution
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start <= max_check:
            cov[old_start::p] -= 1
        # Add new contribution
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start <= max_check:
            cov[new_start::p] += 1

    # Initial assignment: structured heuristic
    # For integer x to be covered by prime p (p > x), we need r_p = x.
    # Assign residues to cover small integers as densely as possible.
    # Strategy: for each prime p, assign r_p to the smallest x >= 1 that currently has < 2 coverage
    # and for which p > x (so r_p = x works directly).
    
    def make_initial():
        residues = [0] * n_primes
        # Track coverage counts for small integers
        # For x < p, x is covered by p iff r_p == x
        # Process primes in decreasing order of p (larger primes can cover more small x values)
        # Actually, let's do a greedy: for each prime, pick residue to maximize min coverage
        
        # Simpler: assign r_p = (i+1) % p initially, then hill-climb
        for i, p in enumerate(primes):
            residues[i] = (i + 1) % p
        return residues

    residues = make_initial()
    cov = build_coverage(residues)
    best_L = get_L(cov)
    best_residues = list(residues)
    best_cov = cov.copy()

    random.seed(42)

    # Phase 1: Greedy local search with incremental updates
    # For each prime, try all residues 0..min(p-1, some_limit) and pick the best
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_primes):
            if time.time() > deadline:
                break
            p = primes[i]
            old_r = residues[i]
            best_local_r = old_r
            best_local_L = best_L
            
            # Try residues: 0 to min(p-1, 60) plus some random
            max_try = min(p, 60)
            for new_r in range(max_try):
                if new_r == old_r:
                    continue
                update_coverage(cov, i, old_r, new_r)
                L = get_L(cov)
                if L > best_local_L:
                    best_local_L = L
                    best_local_r = new_r
                update_coverage(cov, i, new_r, old_r)
            
            # Also try some random residues
            for _ in range(20):
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                update_coverage(cov, i, old_r, new_r)
                L = get_L(cov)
                if L > best_local_L:
                    best_local_L = L
                    best_local_r = new_r
                update_coverage(cov, i, new_r, old_r)
            
            if best_local_L > best_L:
                update_coverage(cov, i, old_r, best_local_r)
                residues[i] = best_local_r
                best_L = best_local_L
                best_residues = list(residues)
                best_cov = cov.copy()
                improved = True

    # Phase 2: Simulated annealing with incremental updates
    if time.time() < deadline:
        current_residues = list(best_residues)
        current_cov = best_cov.copy()
        current_L = best_L
        T = 10.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < deadline and T > T_min:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = current_residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            update_coverage(current_cov, i, old_r, new_r)
            new_L = get_L(current_cov)
            
            if new_L >= current_L:
                current_L = new_L
                current_residues[i] = new_r
                if new_L > best_L:
                    best_L = new_L
                    best_residues = list(current_residues)
                    best_cov = current_cov.copy()
            else:
                delta = new_L - current_L
                if random.random() < math.exp(delta / T):
                    current_L = new_L
                    current_residues[i] = new_r
                else:
                    update_coverage(current_cov, i, new_r, old_r)
            
            T *= cooling

    # Phase 3: Final greedy pass with wider search
    if time.time() < deadline:
        residues = list(best_residues)
        cov = best_cov.copy()
        best_L = get_L(cov)
        
        for i in range(n_primes):
            if time.time() > deadline:
                break
            p = primes[i]
            old_r = residues[i]
            best_local_r = old_r
            best_local_L = best_L
            
            # Try all residues for small primes, subset for large
            if p <= 50:
                candidates = list(range(p))
            else:
                candidates = list(range(min(100, p)))
                for _ in range(30):
                    candidates.append(random.randint(0, p - 1))
            
            for new_r in set(candidates):
                if new_r == old_r:
                    continue
                update_coverage(cov, i, old_r, new_r)
                L = get_L(cov)
                if L > best_local_L:
                    best_local_L = L
                    best_local_r = new_r
                update_coverage(cov, i, new_r, old_r)
            
            if best_local_L > best_L:
                update_coverage(cov, i, old_r, best_local_r)
                residues[i] = best_local_r
                best_L = best_local_L
                best_residues = list(residues)
                best_cov = cov.copy()

    return best_residues
# EVOLVE-BLOCK-END
