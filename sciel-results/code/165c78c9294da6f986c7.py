import numpy as np
import time
import random
from collections import defaultdict

N = 5000
RANGE = N + 1

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _is_admissible(points, primes):
    """Check if a set of points is admissible."""
    n = len(points)
    if n == 0:
        return True
    point_set = set(points)
    for p in primes:
        if p > n:
            break
        # Check if all residue classes mod p are covered
        residues = set(x % p for x in point_set)
        if len(residues) == p:
            return False
    return True

def _greedy_construct(seed, deadline):
    """
    Construct an admissible tuple using greedy residue removal.
    Returns the list of points.
    """
    random.seed(seed)
    active = np.ones(RANGE, dtype=bool)
    x_indices = np.arange(RANGE)
    current_size = RANGE
    
    for p in PRIMES:
        if time.time() > deadline:
            break
        if p > current_size:
            break
        
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        residues = active_indices % p
        if len(residues) == 0:
            break
        
        counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(counts)
        
        if min_count == 0:
            continue
        
        candidate_classes = np.where(counts == min_count)[0]
        
        r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
        
        mask = (x_indices % p == r) & active
        current_size -= np.sum(mask)
        active = active & ~mask
        
        if current_size == 0:
            break
    
    return np.where(active)[0].tolist()

def _sa_polish(points, deadline):
    """
    Simulated Annealing polish to maximize size while maintaining admissibility.
    Uses a penalty method to allow temporary violations if it helps escape local optima,
    but tracks and returns the best VALID state found.
    """
    if not points:
        return points
    
    # Initialize state
    active = np.zeros(RANGE, dtype=bool)
    for p in points:
        active[p] = True
    
    current_size = len(points)
    best_size = current_size
    best_active = active.copy()
    
    # Penalty weight
    # Energy = -size + penalty * violations
    # We want to minimize Energy.
    # Violations: number of primes p <= size where all residues are covered.
    
    # To speed up violation checking, we can just check the primes up to current_size.
    # But in SA, size changes.
    
    # Helper to check violations and return count
    def get_violations(act):
        sz = np.sum(act)
        if sz == 0:
            return 0
        # Get active indices
        # For speed, we can maintain a list, but numpy sum is fast.
        # Let's just iterate primes.
        viol = 0
        # Only primes <= sz matter
        # We can precompute primes up to RANGE
        for p in PRIMES:
            if p > sz:
                break
            # Check if all residues mod p are present in act
            # act is a bool array of size RANGE
            # We need to check if for each r in 0..p-1, there is an x in act such that x%p == r
            # This is equivalent to: 
            # mask_p = np.zeros(RANGE, dtype=bool)
            # for r in range(p):
            #   mask_p[r::p] = True
            # if np.all(act & mask_p) == False: # Wait, we want to know if ALL residues are covered.
            #   # If any residue is NOT covered, it's not a violation.
            #   pass
            # else:
            #   viol += 1
            
            # Optimization:
            # Check if any residue is missing.
            # missing = False
            # for r in range(p):
            #   if not np.any(act[r::p]):
            #     missing = True
            #     break
            # if not missing:
            #   viol += 1
            
            # Faster numpy way:
            # reshape act to (RANGE//p, p) ? No, RANGE is not divisible by p.
            # act_flat = act.flatten()
            # We can just do:
            # covered = np.zeros(p, dtype=bool)
            # idx = np.where(act)[0]
            # covered[idx % p] = True
            # if np.all(covered):
            #   viol += 1
            
            # This allocates arrays. 
            # Given RANGE=5000, this is small.
            
            idx = np.where(act)[0]
            if len(idx) == 0:
                break
            covered = np.zeros(p, dtype=bool)
            covered[idx % p] = True
            if np.all(covered):
                viol += 1
        return viol

    # SA Parameters
    T = 10.0
    cooling = 0.999
    # Number of steps depends on time
    steps = 0
    max_steps = 50000 # Approx
    
    # Precompute active indices for speed?
    # We will update active array.
    
    while time.time() < deadline - 2 and steps < max_steps:
        steps += 1
        
        # Generate move
        # 1. Add random inactive
        # 2. Remove random active
        # 3. Swap
        
        move_type = random.random()
        
        old_size = current_size
        old_viol = get_violations(active)
        old_energy = -old_size + 5.0 * old_viol # Penalty weight 5.0
        
        if move_type < 0.5:
            # Try Add
            inactive_indices = np.where(~active)[0]
            if len(inactive_indices) == 0:
                continue
            x = inactive_indices[random.randint(0, len(inactive_indices) - 1)]
            active[x] = True
            new_size = old_size + 1
            new_viol = get_violations(active)
            new_energy = -new_size + 5.0 * new_viol
        elif move_type < 0.7:
            # Try Remove
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                continue
            x = active_indices[random.randint(0, len(active_indices) - 1)]
            active[x] = False
            new_size = old_size - 1
            new_viol = get_violations(active)
            new_energy = -new_size + 5.0 * new_viol
        else:
            # Try Swap
            active_indices = np.where(active)[0]
            inactive_indices = np.where(~active)[0]
            if len(active_indices) == 0 or len(inactive_indices) == 0:
                continue
            x = active_indices[random.randint(0, len(active_indices) - 1)]
            y = inactive_indices[random.randint(0, len(inactive_indices) - 1)]
            active[x] = False
            active[y] = True
            new_size = old_size
            new_viol = get_violations(active)
            new_energy = -new_size + 5.0 * new_viol
        
        # Acceptance
        delta_e = new_energy - old_energy
        if delta_e < 0:
            accept = True
        else:
            accept = random.random() < np.exp(-delta_e / T)
        
        if accept:
            current_size = new_size
            # Update best if valid
            if new_viol == 0 and new_size > best_size:
                best_size = new_size
                best_active = active.copy()
        else:
            # Revert
            if move_type < 0.5:
                active[x] = False
            elif move_type < 0.7:
                active[x] = True
            else:
                active[x] = True
                active[y] = False
        
        T *= cooling
        if T < 0.01:
            break
            
    # Return best valid state
    return np.where(best_active)[0].tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_size = 0
    best_points = []
    
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        seed = attempts
        points = _greedy_construct(seed, deadline)
        
        # Polish the best candidate so far? Or just this one?
        # Polishing takes time. 
        # Let's polish if we have time.
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            
        # If we found a good one, try to polish it
        # Limit polishing to avoid timeout
        polish_deadline = time.time() + min(10, deadline - time.time() - 5)
        if polish_deadline > time.time() + 1:
            polished = _sa_polish(points, polish_deadline)
            if len(polished) > best_size:
                best_size = len(polished)
                best_points = polished
        
        # Also try polishing the global best occasionally?
        # For now, just polish the current greedy result.
        
    return best_points
