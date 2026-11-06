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
        
    # Find first uncovered
    # covered[0] is unused, we care about 1..max_check
    uncovered = np.where(~covered[1:])[0]
    if len(uncovered) == 0:
        return max_check
    else:
        return uncovered[0] + 1

def _get_uncovered_mask(primes, residues, max_check=100000):
    """
    Returns a boolean mask of uncovered numbers in 1..max_check.
    """
    covered = np.zeros(max_check + 1, dtype=bool)
    for p, r in zip(primes, residues):
        if p > max_check:
            if 1 <= r <= max_check:
                covered[r] = True
        else:
            start = r
            if start == 0:
                start = p
            if start <= max_check:
                indices = np.arange(start, max_check + 1, p)
                if len(indices) > 0:
                    covered[indices] = True
    return ~covered[1:]

def _greedy_repair(primes, residues, max_check=100000, budget_limit=Fraction(1, 1)):
    """
    Attempts to improve coverage by greedily adjusting residues or swapping primes.
    This is the 'error_repair' mechanism.
    It iterates: find first uncovered, try to cover it with existing primes (change residue) 
    or add a new prime if budget allows.
    """
    current_primes = list(primes)
    current_residues = list(residues)
    
    # Calculate current budget
    current_budget = sum(Fraction(1, p) for p in current_primes)
    
    # Available primes not in current set
    all_primes = _get_primes(1000)
    available_primes = [p for p in all_primes if p not in current_primes]
    # Sort available by value (smaller primes are generally more valuable for density)
    available_primes.sort()
    
    improved = True
    while improved and time.time() < _deadline - 2:
        improved = False
        coverage = _compute_coverage(current_primes, current_residues, max_check)
        
        if coverage >= max_check:
            break
            
        # Find first uncovered number
        unc_mask = _get_uncovered_mask(current_primes, current_residues, max_check)
        first_unc_idx = np.argmax(unc_mask)
        first_unc = first_unc_idx + 1
        
        # Strategy A: Try to cover first_unc by changing residue of an existing prime
        # Pick a few random existing primes to try
        candidates_idx = random.sample(range(len(current_primes)), min(5, len(current_primes)))
        best_gain = 0
        best_move = None
        
        for idx in candidates_idx:
            p = current_primes[idx]
            # Residue needed to cover first_unc
            new_r = first_unc % p
            
            # Check if this residue is already used? No, residues can repeat for different primes.
            # But does this change improve coverage?
            # We can't easily know without computing. 
            # Heuristic: If new_r is different from current_r, it might help.
            if new_r == current_residues[idx]:
                continue
                
            # Tentative change
            old_r = current_residues[idx]
            current_residues[idx] = new_r
            
            new_cov = _compute_coverage(current_primes, current_residues, max_check)
            
            if new_cov > coverage + best_gain:
                best_gain = new_cov - coverage
                best_move = ('change_residue', idx, new_r)
            else:
                current_residues[idx] = old_r # Revert if not best yet
        
        if best_move:
            current_residues[best_move[1]] = best_move[2]
            improved = True
            continue
            
        # Strategy B: If no existing prime can easily cover it, try adding a new prime
        # Only if we have budget
        if current_budget < budget_limit:
            # Try a few smallest available primes
            for p in available_primes[:5]:
                cost = Fraction(1, p)
                if current_budget + cost > budget_limit:
                    break
                
                new_r = first_unc % p
                test_primes = current_primes + [p]
                test_residues = current_residues + [new_r]
                
                new_cov = _compute_coverage(test_primes, test_residues, max_check)
                
                if new_cov > coverage:
                    current_primes = test_primes
                    current_residues = test_residues
                    current_budget += cost
                    available_primes.remove(p)
                    improved = True
                    break
                    
        if not improved:
            # Stuck. Try a random kick to escape? 
            # For now, greedy repair stops. The outer loop will do the kick.
            break
            
    return current_primes, current_residues

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 110.0  # 10s margin for cleanup/return
    
    random.seed(42)
    
    primes_all = _get_primes(1000)
    
    # Initial configuration: Small primes with random residues
    # Sum 1/p for p <= 100 is ~0.6. We can afford more.
    initial_primes = [p for p in primes_all if p <= 100]
    initial_residues = [random.randint(0, p - 1) for p in initial_primes]
    
    # Initial repair to get a good starting point
    current_primes, current_residues = _greedy_repair(initial_primes, initial_residues)
    current_coverage = _compute_coverage(current_primes, current_residues)
    
    best_primes = list(current_primes)
    best_residues = list(current_residues)
    best_coverage = current_coverage
    
    # Iterated Local Search Loop
    # 1. Kick: Perturb current solution
    # 2. Repair: Greedy improvement
    # 3. Accept if better
    
    kick_size = 3 # Number of random residue changes
    
    while time.time() < _deadline - 5:
        # 1. Kick
        perturbed_primes = list(current_primes)
        perturbed_residues = list(current_residues)
        
        # Randomly change kick_size residues
        for _ in range(kick_size):
            idx = random.randint(0, len(perturbed_primes) - 1)
            p = perturbed_primes[idx]
            perturbed_residues[idx] = random.randint(0, p - 1)
            
        # Occasionally kick the prime set itself? 
        # E.g., swap a large prime for a smaller unused one if budget allows
        if random.random() < 0.1:
            current_budget = sum(Fraction(1, p) for p in perturbed_primes)
            if current_budget < Fraction(1, 1):
                # Find a prime to remove (largest p, smallest cost)
                # And a prime to add (smallest available p)
                if perturbed_primes:
                    # Remove largest prime
                    max_p_idx = max(range(len(perturbed_primes)), key=lambda i: perturbed_primes[i])
                    removed_p = perturbed_primes.pop(max_p_idx)
                    perturbed_residues.pop(max_p_idx)
                    
                    # Add smallest available prime
                    used = set(perturbed_primes)
                    available = [p for p in primes_all if p not in used]
                    if available:
                        add_p = available[0]
                        cost = Fraction(1, add_p)
                        new_budget = sum(Fraction(1, p) for p in perturbed_primes) + cost
                        if new_budget <= Fraction(1, 1):
                            perturbed_primes.append(add_p)
                            perturbed_residues.append(random.randint(0, add_p - 1))
        
        # 2. Repair
        repaired_primes, repaired_residues = _greedy_repair(perturbed_primes, perturbed_residues)
        repaired_coverage = _compute_coverage(repaired_primes, repaired_residues)
        
        # 3. Accept
        if repaired_coverage > current_coverage:
            current_primes = repaired_primes
            current_residues = repaired_residues
            current_coverage = repaired_coverage
            
            if current_coverage > best_coverage:
                best_coverage = current_coverage
                best_primes = list(current_primes)
                best_residues = list(current_residues)
                
            # If we hit the cap, we can stop or continue trying to optimize budget? 
            # Task says maximize L. If L=100000, we are done.
            if best_coverage >= 100000:
                break
        else:
            # Reject, keep current as base for next kick
            pass
            
    # Final verification
    best_primes = [c[0] for c in best_primes]
    best_residues = [c[1] for c in best_residues]
    
    # Verify budget
    budget = sum(Fraction(1, p) for p in best_primes)
    if budget > 1:
        # Fallback: Remove primes until budget <= 1
        # Sort by p descending to remove least impactful (largest p) first? 
        # Actually removing small p hurts coverage more. 
        # But we must be valid. 
        # Greedy removal: remove p that causes least drop in coverage.
        # For simplicity in fallback, just remove largest primes.
        sorted_indices = sorted(range(len(best_primes)), key=lambda i: best_primes[i], reverse=True)
        for i in sorted_indices:
            if budget <= 1:
                break
            budget -= Fraction(1, best_primes[i])
            best_primes.pop(i)
            best_residues.pop(i)
            
    return [[p, r] for p, r in zip(best_primes, best_residues)]

# EVOLVE-BLOCK-END
