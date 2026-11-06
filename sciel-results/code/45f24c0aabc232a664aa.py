"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140
    
    # Generate primes up to 1000
    def primes_up_to(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, v in enumerate(sieve) if v]
    
    primes = primes_up_to(1000)
    n = len(primes)
    
    # Cap for coverage computation
    cap = 2000
    
    def compute_coverage(residues):
        coverage = [0] * (cap + 1)
        for r, p in zip(residues, primes):
            first = r if r > 0 else p
            x = first
            while x <= cap:
                coverage[x] += 1
                x += p
        return coverage
    
    def get_L(coverage):
        L = 0
        for i in range(1, cap + 1):
            if coverage[i] >= 2:
                L = i
            else:
                break
        return L
    
    def find_first_gap(coverage):
        for i in range(1, cap + 1):
            if coverage[i] < 2:
                return i
        return None
    
    # Initial assignment: use small values to get good coverage of small integers
    # Strategy: assign residues so that small primes cover small numbers densely
    residues = []
    for i, p in enumerate(primes):
        # For small primes, use residue that covers multiple small numbers
        # A good heuristic: r = i % p, but adjusted
        residues.append(i % p)
    
    coverage = compute_coverage(residues)
    best_L = get_L(coverage)
    best_residues = residues[:]
    
    # Hill climbing
    random.seed(42)
    step = 0
    while time.time() < deadline:
        step += 1
        gap = find_first_gap(coverage)
        if gap is None or gap > best_L:
            # No gap before best_L, try to extend
            # Find first position where coverage < 2 after best_L
            for i in range(best_L + 1, cap + 1):
                if coverage[i] < 2:
                    gap = i
                    break
            else:
                break
        
        if gap is None:
            break
        
        # Find primes that cover gap
        covering_indices = []
        for i, p in enumerate(primes):
            if (gap - residues[i]) % p == 0:
                covering_indices.append(i)
        
        # Try changing residues of primes that can cover the gap
        improved = False
        random.shuffle(covering_indices)
        
        for idx in covering_indices:
            if time.time() > deadline:
                return best_residues
            
            p = primes[idx]
            old_r = residues[idx]
            
            # Try new residue: we want residue r such that gap ≡ r (mod p)
            # That means r = gap % p
            new_r = gap % p
            
            if new_r == old_r:
                continue
            
            # Also try other residues that might help nearby
            # First, try the direct fix
            residues[idx] = new_r
            new_coverage = compute_coverage(residues)
            new_L = get_L(new_coverage)
            
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
                coverage = new_coverage
                improved = True
                break
            else:
                # Revert
                residues[idx] = old_r
        
        if not improved:
            # Try random perturbations
            for _ in range(20):
                if time.time() > deadline:
                    return best_residues
                
                idx = random.randint(0, n - 1)
                p = primes[idx]
                old_r = residues[idx]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
                
                residues[idx] = new_r
                new_coverage = compute_coverage(residues)
                new_L = get_L(new_coverage)
                
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues[:]
                    coverage = new_coverage
                    improved = True
                    break
                else:
                    residues[idx] = old_r
        
        if not improved and step > 50:
            # Stuck, try a bigger random restart
            for _ in range(10):
                if time.time() > deadline:
                    return best_residues
                
                # Perturb multiple residues
                for _ in range(5):
                    idx = random.randint(0, n - 1)
                    p = primes[idx]
                    residues[idx] = random.randint(0, p - 1)
                
                new_coverage = compute_coverage(residues)
                new_L = get_L(new_coverage)
                
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues[:]
                    coverage = new_coverage
                    break
    
    return best_residues
# EVOLVE-BLOCK-END
