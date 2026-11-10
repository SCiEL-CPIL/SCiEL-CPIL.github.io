"""Rewritten program for the narrow admissible tuple task."""
import time
import random
from math import isqrt

N = 5000
DEADLINE = 300.0  # seconds, safety margin from 330s kill


def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


# Precompute primes up to N+1 (max possible tuple size)
_PRIMES_MAX = _primes(N + 1)
# Primes we will explicitly control (miss a residue for)
_CTRL_PRIMES = [2, 3, 5, 7, 11, 13]


def _is_admissible(points, primes_needed):
    """Check if points form an admissible tuple for the given primes."""
    n = len(points)
    if n == 0:
        return True
    for p in primes_needed:
        if p > n:
            break
        residues = set()
        for x in points:
            residues.add(x % p)
        if len(residues) == p:
            return False
    return True


def _construct_with_control(control_primes, missed_residues, deadline):
    """
    Construct a set S of points in [0, N] such that for each p in control_primes,
    x % p != missed_residues[p] for all x in S.
    Then repair S to be admissible for all other primes.
    """
    # 1. Generate candidate set C
    # C = { x in [0, N] | for all p in control_primes, x % p != missed_residues[p] }
    candidates = []
    for x in range(N + 1):
        valid = True
        for p, r in zip(control_primes, missed_residues):
            if x % p == r:
                valid = False
                break
        if valid:
            candidates.append(x)
    
    if not candidates:
        return set()

    # Start with all candidates
    S = set(candidates)
    
    # 2. Repair S for primes not in control_primes
    # We need to check primes p <= |S|.
    # We only need to check primes p > max(control_primes) because we already ensured
    # that for p in control_primes, a residue is missed.
    # However, if we remove elements, we might accidentally fill a class for a control prime?
    # No, we started with a set where a specific class is EMPTY. Removing elements keeps it empty.
    # So we only need to repair for primes p > max(control_primes).
    
    max_ctrl = max(control_primes)
    primes_to_check = [p for p in _PRIMES_MAX if p > max_ctrl]
    
    # Greedy repair
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(S)
        if n == 0:
            break
        
        # Find first violating prime
        for p in primes_to_check:
            if p > n:
                break
            
            # Count residues
            counts = [0] * p
            for x in S:
                counts[x % p] += 1
            
            # Check if full
            if min(counts) > 0:
                # Find class with min count
                min_count = min(counts)
                # Find all classes with min count
                min_classes = [i for i, c in enumerate(counts) if c == min_count]
                # Pick one randomly
                target_class = random.choice(min_classes)
                
                # Find an element in this class to remove
                # We can pick randomly from S
                # Optimization: pick the one that contributes least to other constraints?
                # For now, random is fine.
                to_remove = None
                for x in S:
                    if x % p == target_class:
                        to_remove = x
                        break
                if to_remove is not None:
                    S.remove(to_remove)
                    changed = True
                    break
                    
    return S


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin
    
    best_set = set()
    best_size = 0
    
    # Try different assignments of missed residues for control primes
    # Number of combinations: 2 * 3 * 5 * 7 * 11 * 13 = 28380
    # This is too many to try all if each takes time.
    # But the construction is fast (O(N * len(ctrl_primes))).
    # The repair is the bottleneck.
    
    # Let's try a subset of assignments or random ones.
    # Or iterate through all if fast enough.
    # Let's estimate time.
    # N=5000. Loop 5000 * 6 = 30000 ops. ~0.01s.
    # Repair: depends on size. Size ~ 5000 * (1/2)*(2/3)*(4/5)*(6/7)*(10/11)*(12/13)
    # Product approx 0.16. Size ~ 800.
    # Primes to check: up to 800. ~130 primes.
    # Check cost: 800 * 130 = 100k ops.
    # Repair loop might run a few times. Say 10 times. 1M ops. ~0.05s.
    # Total per assignment ~0.06s.
    # 28380 * 0.06 = 1700s. Too slow.
    
    # We need to sample assignments or optimize repair.
    # Let's sample random assignments.
    
    num_trials = 0
    while time.time() < deadline - 10.0:
        # Randomly assign missed residues
        missed_residues = []
        for p in _CTRL_PRIMES:
            missed_residues.append(random.randint(0, p - 1))
            
        S = _construct_with_control(_CTRL_PRIMES, missed_residues, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        num_trials += 1
        
    # One final check with a specific "good" assignment?
    # e.g., always miss 0 for all primes?
    # S = { x | x not divisible by 2,3,5,7,11,13 }
    # This is a known good structure.
    if time.time() < deadline - 5.0:
        missed_residues = [0] * len(_CTRL_PRIMES)
        S = _construct_with_control(_CTRL_PRIMES, missed_residues, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return list(best_set)


# To ensure the function is defined and callable
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
