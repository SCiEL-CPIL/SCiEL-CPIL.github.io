import time
import random
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_primes(limit):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

def _compute_coverage(primes, residues, max_check=100000):
    """
    Computes the largest L such that all integers 1..L are covered.
    """
    if not primes:
        return 0
    
    covered = np.zeros(max_check + 1, dtype=bool)
    
    for p, r in zip(primes, residues):
        if p > max_check:
            if 1 <= r <= max_check:
                covered[r] = True
        else:
            start = r
            if start == 0:
                start = p
            if start > max_check:
                if r == 0 and p <= max_check:
                    start = p
                else:
                    continue
            
            indices = np.arange(start, max_check + 1, p)
            if len(indices) > 0:
                covered[indices] = True
        
    uncovered = np.where(~covered[1:])[0]
    if len(uncovered) == 0:
        return max_check
    else:
        return uncovered[0] + 1

def _greedy_residue_assignment(primes, max_check=100000):
    """
    Greedy Set Cover approach for residue assignment.
    We want to cover 1..L. We start with no coverage.
    For each prime, we pick the residue that maximizes new coverage of the "frontier".
    However, simple sequential greedy might be weak. 
    Instead, we use an iterative improvement:
    1. Initialize residues randomly or greedily.
    2. Find first uncovered number U.
    3. Find all primes p that can cover U (i.e., U % p = r).
    4. For each such prime, check if changing its residue to U%p improves total coverage 
       (specifically, if it covers U and doesn't uncover too many other numbers).
    5. Accept the best change.
    """
    n = len(primes)
    if n == 0:
        return []
    
    # Initialize with a simple heuristic: residue 0 for all? Or random?
    # Residue 0 covers multiples. 
    # Let's start with random to break symmetry, then optimize.
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Precompute coverage state
    # We will maintain a count of how many times each number is covered?
    # No, we just need to know if it's covered.
    # For speed, we will compute full coverage after each "batch" of changes or use incremental updates.
    # Given 168 primes and 10^5 range, full recomputation is ~168 * (10^5/p) ops.
    # Sum(1/p) ~ 1. So total ops ~ 10^5. 
    # 10^5 ops in Python/Numpy is very fast (< 1ms).
    # So we can afford to recompute coverage frequently.
    
    current_coverage = _compute_coverage(primes, residues, max_check)
    
    # Local Search: Hill Climbing
    # While we can improve:
    # 1. Find first uncovered number U.
    # 2. Try to cover U by changing a prime's residue.
    
    improved = True
    while improved:
        improved = False
        # Get current coverage to find first hole
        # Actually, we should maintain the 'covered' array to avoid recomputing from scratch every time
        # But recomputing is cheap enough for 10^5.
        
        # Find first uncovered
        # We can reuse the logic from _compute_coverage but return the array too?
        # Let's just recompute for simplicity in the loop, but optimized.
        
        # Optimization: Keep the covered array from the last computation?
        # Let's just call _compute_coverage.
        
        U = current_coverage
        if U > max_check:
            break
            
        # U is the first uncovered number.
        # We want to cover U.
        # Candidates: Primes p such that we can set r = U % p.
        # Note: If U % p == 0, r=0.
        
        best_gain = 0
        best_prime_idx = -1
        best_new_residue = -1
        
        # We only need to check primes p <= U? 
        # If p > U, U % p = U. So r=U. 
        # Changing a large prime's residue to U covers U and potentially other numbers > U.
        # But small primes are more powerful.
        
        # To speed up, we can limit candidates to primes p <= U * 2 or something?
        # Actually, any prime can cover U.
        
        # Let's evaluate the "cost" of changing each prime's residue to cover U.
        # Cost = (Numbers newly covered) - (Numbers newly uncovered)
        # Since we are covering U, newly covered >= 1.
        
        # We can approximate by just trying a few random candidates or all of them.
        # With 168 primes, checking all is fine.
        
        for i, p in enumerate(primes):
            new_r = U % p
            if new_r == residues[i]:
                continue # Already covers U? No, if it covered U, U would be covered.
                         # So this implies current residue does NOT cover U.
                         
            # Simulate change
            old_r = residues[i]
            residues[i] = new_r
            
            new_cov = _compute_coverage(primes, residues, max_check)
            
            gain = new_cov - current_coverage
            
            # Revert for next iteration's comparison (or keep if best)
            residues[i] = old_r
            
            if gain > best_gain:
                best_gain = gain
                best_prime_idx = i
                best_new_residue = new_r
        
        if best_gain > 0:
            # Apply the best change
            residues[best_prime_idx] = best_new_residue
            current_coverage += best_gain
            improved = True
        else:
            # No single change improves coverage.
            # We might be stuck in a local optimum where we need to change 2 residues simultaneously.
            # Or we've reached the max possible with this prime set.
            break
            
    return residues

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    primes_all = _get_primes(1000)
    
    # The seed uses primes >= 13.
    # Sum(1/p for p in 13..1000) is approx 0.92 < 1.
    # This is the known optimal set of primes for the Erdos-Ruzsa covering at x=1000.
    # Using smaller primes (2, 3, 5, 7, 11) increases the sum.
    # Sum(1/2 + 1/3 + 1/5 + 1/7 + 1/11) = 0.5 + 0.33 + 0.2 + 0.14 + 0.09 = 1.26 > 1.
    # So we MUST exclude some small primes.
    # The standard result uses primes >= 13.
    
    current_primes = [p for p in primes_all if p >= 13]
    
    # Verify budget
    budget = sum(Fraction(1, p) for p in current_primes)
    if budget > 1:
        # Fallback: remove largest primes? No, remove smallest?
        # If we remove small primes, budget drops fast.
        # But we want to keep the set that allows covering 999.
        # The set 13..1000 is valid.
        pass
        
    # Strategy: 
    # 1. Greedy/Local Search for residues on this fixed set.
    # 2. If we get stuck, try swapping a prime (e.g., swap a large prime for a smaller one if budget allows) 
    #    to see if it helps cover the "hard" numbers.
    
    # Phase 1: Greedy Local Search
    residues = _greedy_residue_assignment(current_primes, 100000)
    
    current_coverage = _compute_coverage(current_primes, residues, 100000)
    best_primes = list(current_primes)
    best_residues = list(residues)
    best_coverage = current_coverage
    
    # Phase 2: Randomized Restart / Hill Climbing with Prime Swaps
    # If we didn't reach 999, we might need a different set of primes.
    # But 13..1000 is very likely the optimal set for x=1000.
    # Let's try to improve the residue assignment further with a more aggressive local search.
    
    # We will run a simple simulated annealing but focused on "covering the next hole".
    # This is effectively what the greedy did, but SA allows accepting worse moves to escape local minima.
    
    T = 10.0
    cooling = 0.995
    steps = 0
    max_steps = 20000
    
    # Precompute 1/p for budget checks if we swap primes
    # But we are keeping the prime set fixed for now to ensure validity and speed.
    # If we swap primes, we must check budget.
    
    # Let's allow prime swaps in the later stages if we are stuck.
    # Available primes: 2, 3, 5, 7, 11 (small, expensive) and primes > 1000 (not allowed).
    # So we can only swap primes from the set {2,3,5,7,11} INTO the set, and remove some from 13..1000.
    # Or swap within 13..1000.
    
    # Since 13..1000 is the "standard" set, let's stick to it and just optimize residues harder.
    
    # Aggressive Local Search:
    # Find first uncovered U.
    # Try changing residue of ANY prime to cover U.
    # If no single change works, try changing 2 primes? (Too slow).
    # Instead, use random perturbations + local repair.
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # 1. Find first uncovered
        U = current_coverage
        if U > 100000:
            break
            
        # 2. Randomly perturb a few residues to escape local minima
        # Or, just try to cover U again but with a random selection of candidate primes?
        
        # Let's try a "Greedy with Randomization"
        # Pick a random prime that CAN cover U (all primes can).
        # Actually, let's pick a random prime and change its residue to cover U.
        # This is a random walk in the space of "assignments that cover U".
        
        idx = random.randint(0, len(current_primes) - 1)
        p = current_primes[idx]
        new_r = U % p
        
        if new_r == residues[idx]:
            continue
            
        old_r = residues[idx]
        residues[idx] = new_r
        
        new_cov = _compute_coverage(current_primes, residues, 100000)
        
        # Accept if better, or with probability exp((delta)/T) if worse
        delta = new_cov - current_coverage
        if delta > 0:
            current_coverage = new_cov
            if current_coverage > best_coverage:
                best_coverage = current_coverage
                best_primes = list(current_primes)
                best_residues = list(residues)
                T *= cooling # Cool down on success
        else:
            # Accept worse move to escape
            prob = np.exp(delta / T)
            if random.random() < prob:
                current_coverage = new_cov
                T *= cooling
            else:
                # Revert
                residues[idx] = old_r
                T *= cooling
                
        # Safety check
        if time.time() > deadline - 10.0:
            break
            
    # Final Verification
    best_classes = [[p, r] for p, r in zip(best_primes, best_residues)]
    
    # Check budget one last time
    budget = sum(Fraction(1, p) for p in best_primes)
    if budget > 1:
        # Fallback to seed if somehow budget exceeded (should not happen)
        return [[p, (i + 1) % p] for i, p in enumerate(current_primes)]
        
    return best_classes

# EVOLVE-BLOCK-END
