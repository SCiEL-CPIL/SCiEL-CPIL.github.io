"""Rewritten program with local search + simulated annealing for double coverage."""
# EVOLVE-BLOCK-START
import time
import random

# Precompute primes up to 1000
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

PRIMES = _primes_upto(1000)
N_PRIMES = len(PRIMES)  # 168

# Precompute for each prime index, the residue class membership
# For prime p at index i, integer x is covered if x % p == r_i

def _initial_construct():
    """Greedy initial: assign residues to cover small integers twice."""
    residues = [0] * N_PRIMES
    # Start with the seed-like construction, then improve
    for i, p in enumerate(PRIMES):
        residues[i] = ((i + 2) // 2) % p
    return residues

def _compute_L(residues):
    """Compute the largest L such that every integer 1..L is covered at least twice."""
    # For each integer x, count how many primes p have x % p == residues[i]
    # We scan x = 1, 2, 3, ... until we find one with count < 2
    # Efficient: for each x, iterate over primes and check
    # But that's O(L * 168) which for L up to 1000 is 168000 ops - fine
    
    # More efficient: build a difference array approach
    # For each prime i with residue r, it covers numbers r, r+p, r+2p, ...
    # We need count[x] >= 2 for all x in 1..L
    
    # Use an array of counts
    # Max x we might check: let's cap at 100000 but usually L is small
    max_check = 2000  # we expect L to be in this range
    counts = [0] * (max_check + 1)
    
    for i, p in enumerate(PRIMES):
        r = residues[i]
        # Numbers covered: r, r+p, r+2p, ... but we want x >= 1
        # x = r + k*p for k >= 0, and x >= 1
        if r == 0:
            start = p  # smallest positive multiple
        else:
            start = r
        x = start
        while x <= max_check:
            counts[x] += 1
            x += p
    
    # Find largest L where all 1..L have count >= 2
    L = 0
    for x in range(1, max_check + 1):
        if counts[x] < 2:
            break
        L = x
    return L

def _compute_L_incremental(residues, changed_idx, old_r, new_r):
    """
    Recompute L after changing residue of prime at changed_idx from old_r to new_r.
    Returns new L.
    """
    p = PRIMES[changed_idx]
    
    # We need to update counts for the affected numbers
    # Old residue old_r covered: old_r, old_r+p, old_r+2p, ...
    # New residue new_r covers: new_r, new_r+p, new_r+2p, ...
    
    # This is complex to do incrementally without storing full counts.
    # For now, just recompute fully but with a tighter bound.
    # Actually, let's store counts globally and update them.
    return None  # Not used in current design; we'll recompute

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Initial construction
    best_residues = _initial_construct()
    best_L = _compute_L(best_residues)
    
    # Local search: try changing one residue at a time
    # For each prime, try a few random alternative residues
    
    rng = random.Random(42)
    
    # Precompute: for each prime index, which integers in 1..L_target are covered
    # We'll do a hill-climbing with occasional random restarts
    
    # Phase 1: Greedy improvement - for each prime, find the residue that maximizes L
    # This is expensive, so we'll do a limited version
    
    # Phase 2: Local search with random perturbations
    
    # Let's use a smarter approach: 
    # For a given target L, check if it's achievable, then binary search or increment
    
    # Actually, let's just do local search on the full problem.
    # Key insight: changing residue of prime i affects which numbers it covers.
    # The score L is determined by the first x where count[x] < 2.
    
    # Efficient local search:
    # 1. Compute current counts for x = 1..max_x
    # 2. Find current L
    # 3. For each prime i, try changing residue to a value that helps cover the "bottleneck"
    
    max_x = 500  # We'll work in this range; extend if needed
    
    def compute_counts(residues, max_x):
        counts = [0] * (max_x + 1)
        for i, p in enumerate(PRIMES):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            x = start
            while x <= max_x:
                counts[x] += 1
                x += p
        return counts
    
    def find_L(counts, max_x):
        L = 0
        for x in range(1, max_x + 1):
            if counts[x] < 2:
                break
            L = x
        return L
    
    counts = compute_counts(best_residues, max_x)
    current_L = find_L(counts, max_x)
    
    # Local search: iterate over primes, try to improve
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N_PRIMES):
            if time.time() > deadline:
                break
            p = PRIMES[i]
            old_r = best_residues[i]
            
            # Try a few candidate residues
            candidates = []
            # Try residue that covers the current bottleneck
            if current_L < max_x:
                bottleneck = current_L + 1
                # We want residue r such that bottleneck % p == r
                r_needed = bottleneck % p
                if r_needed != old_r:
                    candidates.append(r_needed)
            
            # Also try a few random residues
            for _ in range(3):
                candidates.append(rng.randint(0, p - 1))
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                if new_r >= p:
                    continue
                
                # Update counts: remove old coverage, add new coverage
                # Old: old_r, old_r+p, ...
                if old_r == 0:
                    old_start = p
                else:
                    old_start = old_r
                x = old_start
                while x <= max_x:
                    counts[x] -= 1
                    x += p
                
                # New: new_r, new_r+p, ...
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r
                x = new_start
                while x <= max_x:
                    counts[x] += 1
                    x += p
                
                new_L = find_L(counts, max_x)
                
                if new_L > current_L:
                    current_L = new_L
                    best_residues[i] = new_r
                    improved = True
                    # Extend max_x if we're hitting the boundary
                    if current_L == max_x:
                        # Extend the range
                        old_max = max_x
                        max_x = min(max_x * 2, 5000)
                        # Extend counts
                        old_counts = counts
                        counts = [0] * (max_x + 1)
                        for x in range(1, old_max + 1):
                            counts[x] = old_counts[x]
                        # Recompute the full counts to be safe
                        counts = compute_counts(best_residues, max_x)
                        current_L = find_L(counts, max_x)
                    break  # Move to next prime
                else:
                    # Revert
                    if old_r == 0:
                        old_start = p
                    else:
                        old_start = old_r
                    x = old_start
                    while x <= max_x:
                        counts[x] += 1
                        x += p
                    
                    if new_r == 0:
                        new_start = p
                    else:
                        new_start = new_r
                    x = new_start
                    while x <= max_x:
                        counts[x] -= 1
                        x += p
    
    # Phase 2: Random restarts / simulated annealing
    if time.time() < deadline:
        # Try several random perturbations from the best solution
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Start from best, perturb a few random primes
            residues = best_residues[:]
            n_perturb = rng.randint(3, 10)
            for _ in range(n_perturb):
                idx = rng.randint(0, N_PRIMES - 1)
                residues[idx] = rng.randint(0, PRIMES[idx] - 1)
            
            # Quick local improvement
            counts = compute_counts(residues, max_x)
            L = find_L(counts, max_x)
            
            # Greedy pass
            for i in range(N_PRIMES):
                if time.time() > deadline:
                    break
                p = PRIMES[i]
                old_r = residues[i]
                if L < max_x:
                    bottleneck = L + 1
                    r_needed = bottleneck % p
                    if r_needed != old_r:
                        # Try it
                        if old_r == 0:
                            old_start = p
                        else:
                            old_start = old_r
                        x = old_start
                        while x <= max_x:
                            counts[x] -= 1
                            x += p
                        
                        if r_needed == 0:
                            new_start = p
                        else:
                            new_start = r_needed
                        x = new_start
                        while x <= max_x:
                            counts[x] += 1
                            x += p
                        
                        new_L = find_L(counts, max_x)
                        if new_L > L:
                            L = new_L
                            residues[i] = r_needed
                            if L == max_x:
                                old_max = max_x
                                max_x = min(max_x * 2, 5000)
                                counts = compute_counts(residues, max_x)
                                L = find_L(counts, max_x)
                        else:
                            # Revert
                            if old_r == 0:
                                old_start = p
                            else:
                                old_start = old_r
                            x = old_start
                            while x <= max_x:
                                counts[x] += 1
                                x += p
                            
                            if r_needed == 0:
                                new_start = p
                            else:
                                new_start = r_needed
                            x = new_start
                            while x <= max_x:
                                counts[x] -= 1
                                x += p
                
                if L == max_x and max_x < 5000:
                    # Extend
                    old_max = max_x
                    max_x = min(max_x * 2, 5000)
                    counts = compute_counts(residues, max_x)
                    L = find_L(counts, max_x)
                
                if L >= 1000:
                    break
            
            if L > current_L:
                current_L = L
                best_residues = residues[:]
    
    # Final verification with a larger range
    final_L = _compute_L(best_residues)
    
    return best_residues
# EVOLVE-BLOCK-END
