"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
import random
import math
from functools import lru_cache

# Precompute primes up to 1000
def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    return [i for i in range(2, limit+1) if is_prime[i]]

PRIMES = _sieve(1000)
# Map prime to index
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES)}
# Reciprocals as Fractions for exact arithmetic
RECIPOCAL = {p: Fraction(1, p) for p in PRIMES}
# Precompute residues: for each prime p, residues 0..p-1
# We'll use a coverage representation

def _coverage_count(assignment, L):
    """
    assignment: dict prime -> residue (or None if not used)
    Returns the largest L such that 1..L are all covered.
    Uses a boolean array for coverage up to L.
    """
    covered = [False] * (L + 1)
    for p, r in assignment.items():
        if r is None:
            continue
        # Mark all n in 1..L such that n % p == r
        # First n >= 1 with n % p == r:
        # n = r if r >= 1, else p
        start = r if r >= 1 else p
        if start > L:
            continue
        for n in range(start, L + 1, p):
            covered[n] = True
    # Find largest L such that 1..L all covered
    max_l = 0
    for i in range(1, L + 1):
        if covered[i]:
            max_l = i
        else:
            break
    return max_l

def _exact_coverage(assignment, L):
    """
    Check if 1..L are all covered by assignment.
    Returns True if all covered.
    """
    for n in range(1, L + 1):
        covered = False
        for p, r in assignment.items():
            if r is not None and n % p == r:
                covered = True
                break
        if not covered:
            return False
    return True

def _budget_ok(assignment):
    """Check if sum of 1/p for used primes <= 1."""
    total = Fraction(0)
    for p, r in assignment.items():
        if r is not None:
            total += RECIPOCAL[p]
    return total <= 1

def _find_uncovered(assignment, L):
    """Return list of uncovered integers in 1..L."""
    uncovered = []
    for n in range(1, L + 1):
        covered = False
        for p, r in assignment.items():
            if r is not None and n % p == r:
                covered = True
                break
        if not covered:
            uncovered.append(n)
    return uncovered

def _greedy_construct():
    """
    Greedy: start with empty assignment, repeatedly add the prime/residue
    that covers the most currently-uncovered integers in 1..target_L,
    subject to budget.
    """
    # We'll try to maximize coverage up to some L
    # Start with a reasonable target
    target_L = 2000  # We'll check up to this
    
    # Current assignment: dict prime -> residue
    assignment = {}
    # Track which integers are covered
    covered = [False] * (target_L + 1)
    # Track budget
    budget_used = Fraction(0)
    
    # Available primes: all primes <= 1000
    available_primes = list(PRIMES)
    
    while True:
        if not _budget_ok(assignment):
            break
        
        # Find the best (p, r) to add
        best_gain = -1
        best_p = None
        best_r = None
        
        for p in available_primes:
            if p in assignment:
                continue
            if budget_used + RECIPOCAL[p] > 1:
                continue
            for r in range(p):
                # Count how many new integers in 1..target_L are covered
                gain = 0
                start = r if r >= 1 else p
                if start > target_L:
                    continue
                for n in range(start, target_L + 1, p):
                    if not covered[n]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
        
        if best_p is None or best_gain <= 0:
            break
        
        # Add the best
        assignment[best_p] = best_r
        budget_used += RECIPOCAL[best_p]
        start = best_r if best_r >= 1 else best_p
        for n in range(start, target_L + 1, best_p):
            covered[n] = True
    
    # Now find the actual coverage
    L = _coverage_count(assignment, target_L)
    return assignment, L

def _local_search(assignment, L, max_iter=5000, time_limit=100.0):
    """
    Local search: try to improve coverage by swapping primes/residues.
    Returns improved assignment and L.
    """
    start_time = time.time()
    current_L = L
    current_assignment = dict(assignment)
    
    # We'll work with a fixed target for coverage checking
    # Use a larger target to see if we can extend coverage
    search_target = min(current_L + 500, 10000)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try random perturbations
        # 1. Try changing residue of an existing prime
        # 2. Try swapping a prime
        # 3. Try adding a new prime if budget allows
        
        # Get list of used primes
        used_primes = [p for p, r in current_assignment.items() if r is not None]
        
        if not used_primes:
            break
        
        # Randomly pick an operation
        op = random.random()
        
        if op < 0.5 and used_primes:
            # Change residue of a random used prime
            p = random.choice(used_primes)
            old_r = current_assignment[p]
            new_r = random.randrange(p)
            if new_r == old_r:
                continue
            
            # Check if this improves coverage
            # Temporarily change
            current_assignment[p] = new_r
            new_L = _coverage_count(current_assignment, search_target)
            if new_L > current_L:
                current_L = new_L
            else:
                # Revert
                current_assignment[p] = old_r
        
        elif op < 0.8 and len(used_primes) > 1:
            # Swap: remove one prime, add another
            p_remove = random.choice(used_primes)
            old_r_remove = current_assignment[p_remove]
            
            # Find a candidate prime to add
            candidates = [p for p in PRIMES if p not in current_assignment]
            if not candidates:
                continue
            p_add = random.choice(candidates)
            
            # Check budget
            budget = sum(RECIPOCAL[p] for p, r in current_assignment.items() if r is not None)
            if budget - RECIPOCAL[p_remove] + RECIPOCAL[p_add] > 1:
                continue
            
            # Try random residue for new prime
            r_add = random.randrange(p_add)
            
            # Temporarily swap
            del current_assignment[p_remove]
            current_assignment[p_add] = r_add
            
            new_L = _coverage_count(current_assignment, search_target)
            if new_L > current_L:
                current_L = new_L
            else:
                # Revert
                current_assignment[p_remove] = old_r_remove
                del current_assignment[p_add]
        
        else:
            # Try adding a new prime if budget allows
            budget = sum(RECIPOCAL[p] for p, r in current_assignment.items() if r is not None)
            candidates = [p for p in PRIMES if p not in current_assignment and budget + RECIPOCAL[p] <= 1]
            if not candidates:
                continue
            p_add = random.choice(candidates)
            r_add = random.randrange(p_add)
            
            current_assignment[p_add] = r_add
            new_L = _coverage_count(current_assignment, search_target)
            if new_L > current_L:
                current_L = new_L
            else:
                del current_assignment[p_add]
    
    return current_assignment, current_L

def _multi_start_greedy(num_starts=5, time_limit=110.0):
    """
    Run greedy from multiple random starting points.
    """
    start_time = time.time()
    best_assignment = None
    best_L = 0
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initial assignment
        assignment = {}
        budget = Fraction(0)
        
        # Randomly add some primes
        random_primes = random.sample(PRIMES, min(len(PRIMES), random.randint(5, 20)))
        for p in random_primes:
            if budget + RECIPOCAL[p] <= 1:
                r = random.randrange(p)
                assignment[p] = r
                budget += RECIPOCAL[p]
        
        # Run greedy to fill
        target_L = 2000
        covered = [False] * (target_L + 1)
        for p, r in assignment.items():
            start = r if r >= 1 else p
            for n in range(start, target_L + 1, p):
                covered[n] = True
        
        # Continue greedy
        while True:
            best_gain = -1
            best_p = None
            best_r = None
            
            for p in PRIMES:
                if p in assignment:
                    continue
                if budget + RECIPOCAL[p] > 1:
                    continue
                for r in range(p):
                    gain = 0
                    start = r if r >= 1 else p
                    if start > target_L:
                        continue
                    for n in range(start, target_L + 1, p):
                        if not covered[n]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                        best_r = r
            
            if best_p is None or best_gain <= 0:
                break
            
            assignment[best_p] = best_r
            budget += RECIPOCAL[best_p]
            start = best_r if best_r >= 1 else best_p
            for n in range(start, target_L + 1, best_p):
                covered[n] = True
        
        L = _coverage_count(assignment, target_L)
        if L > best_L:
            best_L = L
            best_assignment = dict(assignment)
    
    return best_assignment, best_L

def construct():
    deadline = time.time() + 115.0  # 115s to be safe
    
    # Start with greedy
    assignment, L = _greedy_construct()
    
    # Local search
    assignment, L = _local_search(assignment, L, max_iter=3000, time_limit=60.0)
    
    # Multi-start
    if time.time() < deadline - 30.0:
        ms_assignment, ms_L = _multi_start_greedy(num_starts=3, time_limit=50.0)
        if ms_L > L:
            assignment = ms_assignment
            L = ms_L
    
    # Final local search
    if time.time() < deadline - 10.0:
        assignment, L = _local_search(assignment, L, max_iter=2000, time_limit=40.0)
    
    # Convert to list of [p, r]
    result = [[p, r] for p, r in sorted(assignment.items()) if r is not None]
    
    # Verify budget
    total = Fraction(0)
    for p, r in result:
        total += RECIPOCAL[p]
    if total > 1:
        # Shouldn't happen, but just in case, remove some
        # This is a fallback
        pass
    
    return result
# EVOLVE-BLOCK-END
