"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate all primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    primes = _sieve(1000)
    max_check = 100000

    def compute_L(covered):
        """Compute the largest L such that 1..L are all covered."""
        sub = covered[1:]
        false_indices = np.where(~sub)[0]
        if len(false_indices) == 0:
            return max_check
        return false_indices[0]

    def apply_residue(covered, p, r):
        """Apply a residue class to the covered array (in-place)."""
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p
        covered[start::p] = True

    def remove_residue(covered, p, r):
        """Remove a residue class from the covered array (in-place)."""
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p
        covered[start::p] = False

    def greedy_residues_numpy(prime_list, max_check=100000):
        """Assign residues greedily using numpy for speed."""
        if not prime_list:
            return []

        covered = np.zeros(max_check + 1, dtype=bool)
        classes = []

        for p in prime_list:
            if time.time() > deadline:
                break

            # Find the first uncovered number
            sub = covered[1:]
            false_indices = np.where(~sub)[0]
            if len(false_indices) == 0:
                # All covered up to max_check, just pick any residue
                classes.append([p, 0])
                continue
            
            first_uncovered = int(false_indices[0])
            r_target = first_uncovered % p
            best_r = r_target
            best_L = first_uncovered - 1

            # For small primes, try all residues; for large, try target + neighbors
            if p <= 50:
                residues_to_try = list(range(p))
            else:
                residues_to_try = [r_target]
                for d in range(-10, 11):
                    r_alt = (r_target + d) % p
                    if r_alt != r_target:
                        residues_to_try.append(r_alt)
                # Add some random samples for diversity
                for _ in range(min(15, p // 10)):
                    rand_r = random.randint(0, p - 1)
                    if rand_r not in residues_to_try:
                        residues_to_try.append(rand_r)

            for r in residues_to_try:
                if r == 0:
                    start = p
                else:
                    start = r
                if start < 1:
                    start += p
                mask = np.zeros(max_check + 1, dtype=bool)
                mask[start::p] = True
                new_covered = covered | mask
                new_L = compute_L(new_covered)
                if new_L > best_L:
                    best_L = new_L
                    best_r = r

            classes.append([p, int(best_r)])

            # Update covered array
            r = best_r
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            covered[start::p] = True

        return classes

    def optimize_residues(classes, max_check=100000, time_limit=None):
        """
        Perform Cyclic Coordinate Descent to optimize residues.
        Repeatedly tries all residues for each prime to maximize L.
        """
        if not classes:
            return classes, 0
        
        if time_limit is None:
            time_limit = deadline - 5
            
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            apply_residue(covered, p, r)
        
        current_L = compute_L(covered)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            for idx, (p, r) in enumerate(classes):
                if time.time() > time_limit:
                    return classes, current_L
                
                best_r = r
                best_L = current_L
                
                # Try all residues for this prime
                for new_r in range(p):
                    if new_r == r:
                        continue
                    
                    # Remove current residue
                    remove_residue(covered, p, r)
                    # Add new residue
                    apply_residue(covered, p, new_r)
                    
                    new_L = compute_L(covered)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = new_r
                        # Keep the new residue applied for now to check next ones
                        # But we need to revert if this isn't the best at the end of the loop?
                        # Actually, Cyclic Coordinate Descent usually keeps the best change immediately.
                        # Let's keep it.
                        r = new_r 
                        classes[idx][1] = new_r
                        improved = True
                    else:
                        # Revert to the best found so far (which might be original r or a previous new_r)
                        # Wait, the logic above updates r and classes immediately if better.
                        # If not better, we need to revert to the state BEFORE this trial.
                        # But the state BEFORE this trial was the one with 'r' (which might have been updated in a previous iteration of this inner loop).
                        # Let's re-structure:
                        # Save state, try, if better update, else revert.
                        
                        # Correction: The code above is slightly flawed in revert logic if we update r inside.
                        # Let's do it cleanly:
                        pass

                # The inner loop above was messy. Let's rewrite the inner logic clearly.
                # We will restart the optimization function with a cleaner inner loop implementation below.
                # For now, break to avoid using the buggy version.
                # Actually, I will just implement the clean version in the final code.
                
        return classes, current_L

    # Refined Optimize Function
    def optimize_residues_v2(classes, max_check=100000, time_limit=None):
        if not classes:
            return classes, 0
        
        if time_limit is None:
            time_limit = deadline - 5
            
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            apply_residue(covered, p, r)
        
        current_L = compute_L(covered)
        
        # Cyclic Coordinate Descent with optimized inner loop
        while time.time() < time_limit:
            any_improvement = False
            for idx in range(len(classes)):
                if time.time() > time_limit:
                    break
                
                p, r = classes[idx]
                
                # Quick check: if this prime doesn't affect the gap, skip detailed search
                # First, compute what L would be without this prime
                remove_residue(covered, p, r)
                L_without = compute_L(covered)
                
                # If removing this prime doesn't change L, try to find a better residue
                # that maintains or improves coverage
                best_r = r
                best_L = current_L
                
                # For small primes, try all residues; for large, sample around current
                if p <= 30:
                    candidates = list(range(p))
                else:
                    # Sample residues: current, and neighbors, plus some random
                    candidates = [r]
                    for d in range(-10, 11):
                        alt = (r + d) % p
                        if alt != r:
                            candidates.append(alt)
                    # Add some random samples
                    for _ in range(min(20, p // 5)):
                        rand_r = random.randint(0, p - 1)
                        if rand_r not in candidates:
                            candidates.append(rand_r)
                
                for new_r in candidates:
                    if new_r == r:
                        continue
                    
                    apply_residue(covered, p, new_r)
                    new_L = compute_L(covered)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = new_r
                    else:
                        # Revert
                        remove_residue(covered, p, new_r)
                
                # Apply the best residue found
                if best_r != r:
                    # If we didn't keep the best applied, apply it now
                    if best_r not in [c for c in candidates if c != r]:
                        apply_residue(covered, p, best_r)
                    elif best_r != r:
                        # Already applied or need to reapply
                        remove_residue(covered, p, r)
                        apply_residue(covered, p, best_r)
                    
                    current_L = best_L
                    classes[idx][1] = best_r
                    any_improvement = True
                else:
                    # No improvement, ensure original is applied
                    if not covered[1:p+1].any() or True:
                        # Check if r is still applied
                        test_start = r if r != 0 else p
                        if test_start >= 1 and test_start <= max_check:
                            if not covered[test_start]:
                                apply_residue(covered, p, r)
            
            if not any_improvement:
                break
                
        return classes, current_L

    best_classes = []
    best_L = 0

    def evaluate_strategy(selected_primes, optimize_time=10):
        nonlocal best_classes, best_L
        if time.time() > deadline - 5:
            return
        if not selected_primes:
            return
        
        # 1. Greedy Init
        classes = greedy_residues_numpy(selected_primes, max_check)
        
        # 2. Optimize
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        opt_time = min(remaining - 5, optimize_time)
        if opt_time < 1:
            return
            
        classes, L = optimize_residues_v2(classes, max_check, time_limit=time.time() + opt_time)
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]

    # Generate Candidate Strategies
    
    # Strategy 1: Smallest primes (Dense)
    primes_asc = sorted(primes)
    budget = Fraction(1)
    selected = []
    for p in primes_asc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    evaluate_strategy(selected, optimize_time=25)

    # Strategy 2: Largest primes (Many)
    primes_desc = sorted(primes, reverse=True)
    budget = Fraction(1)
    selected = []
    for p in primes_desc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    evaluate_strategy(selected, optimize_time=25)

    # Strategy 2b: Medium primes (balanced)
    if time.time() < deadline - 40:
        primes_mid = sorted(primes, key=lambda x: abs(x - 100))  # Around 100
        budget = Fraction(1)
        selected = []
        for p in primes_mid:
            cost = Fraction(1, p)
            if budget >= cost:
                budget -= cost
                selected.append(p)
            else:
                break
        evaluate_strategy(selected, optimize_time=20)

    # Strategy 3: Mixed (Small core, large fill)
    if time.time() < deadline - 30:
        for num_small in [2, 3, 4, 5]:
            if time.time() > deadline - 30:
                break
            small_primes = primes_asc[:num_small]
            budget = Fraction(1)
            for p in small_primes:
                budget -= Fraction(1, p)
            if budget < 0:
                continue

            selected = list(small_primes)
            for p in primes_desc:
                if p in small_primes:
                    continue
                cost = Fraction(1, p)
                if budget >= cost:
                    budget -= cost
                    selected.append(p)
            
            evaluate_strategy(selected, optimize_time=15)

    # Strategy 4: Randomized
    if time.time() < deadline - 25:
        random.seed(42)
        for restart in range(8):
            if time.time() > deadline - 25:
                break
            
            random_primes = primes.copy()
            random.shuffle(random_primes)
            budget = Fraction(1)
            selected = []
            for p in random_primes:
                cost = Fraction(1, p)
                if budget >= cost:
                    budget -= cost
                    selected.append(p)
                else:
                    continue
            
            if selected:
                evaluate_strategy(selected, optimize_time=12)

    # Strategy 5: Hybrid with weighted selection
    if time.time() < deadline - 30:
        # Weight by inverse of prime (cheaper primes get higher weight)
        weights = [1.0 / p for p in primes]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        
        random.seed(123)
        for restart in range(6):
            if time.time() > deadline - 30:
                break
            
            selected = []
            budget = Fraction(1)
            used = set()
            
            # Select primes with probability proportional to weight
            for p in primes:
                if p in used:
                    continue
                # Probability of selecting this prime
                if random.random() < 0.3:  # 30% chance to consider each prime
                    cost = Fraction(1, p)
                    if budget >= cost:
                        budget -= cost
                        selected.append(p)
                        used.add(p)
            
            if selected:
                evaluate_strategy(selected, optimize_time=15)

    return best_classes if best_classes else [[2, 1]]

# EVOLVE-BLOCK-END
