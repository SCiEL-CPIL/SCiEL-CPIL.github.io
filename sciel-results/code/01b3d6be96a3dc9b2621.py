"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes_upto(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _check_admissible(points, all_primes):
    """Check if points is admissible for all primes p <= len(points)."""
    m = len(points)
    for p in all_primes:
        if p > m:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


def _get_conflicting_primes(points, all_primes):
    """Return list of primes p <= len(points) that are violated."""
    m = len(points)
    conflicts = []
    for p in all_primes:
        if p > m:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                conflicts.append(p)
    return conflicts


def _randomized_local_search(points, all_primes, deadline):
    """
    Stochastic local search:
    1. Repair violations by removing elements.
    2. Try to add elements.
    3. If stuck, apply a random kick (remove random element) to escape local optima.
    """
    points = list(points)
    points_set = set(points)
    
    # Precompute primes relevant to max possible size
    max_relevant_primes = [p for p in all_primes if p <= N]
    
    def get_violations(p_list):
        m = len(p_list)
        viol = []
        for p in max_relevant_primes:
            if p > m:
                break
            res = set()
            for v in p_list:
                res.add(v % p)
                if len(res) == p:
                    viol.append(p)
                    break
        return viol

    current_violations = get_violations(points)
    
    # Phase 1: Repair
    while current_violations and time.time() < deadline:
        p = current_violations[0]
        # Remove an element from the smallest residue class? No, just pick a random one from the violating set
        # Actually, to be fast, pick the element that appears in the most violating classes?
        # Simple: pick a random element and remove it.
        # Better: pick an element that is "responsible" for the conflict.
        # Let's just remove a random element from points.
        if not points:
            break
        idx = random.randint(0, len(points) - 1)
        removed = points.pop(idx)
        points_set.discard(removed)
        current_violations = get_violations(points)
        
    # Phase 2: Improve
    improved = True
    last_kick_time = time.time()
    
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        
        # Check if we are stuck (no improvements in last 10 seconds)
        if time.time() - last_kick_time > 10:
            # Kick: Remove a random element to change the landscape
            if points:
                idx = random.randint(0, len(points) - 1)
                removed = points.pop(idx)
                points_set.discard(removed)
            last_kick_time = time.time()
            improved = True # Force another loop iteration
            
        # Try to add elements
        # Sample candidates
        candidates = list(range(N + 1))
        random.shuffle(candidates)
        
        # Limit candidates to keep it fast
        num_candidates = 100
        for c in candidates[:num_candidates]:
            if c in points_set:
                continue
            if time.time() >= deadline:
                break
            
            # Check if adding c is admissible
            # We need to check primes p <= len(points) + 1
            new_m = m + 1
            admissible = True
            for p in all_primes:
                if p > new_m:
                    break
                # Check if c fills the last empty class for p
                # We need to know the current residue counts for p
                # This is expensive to compute from scratch every time.
                # Optimization: Maintain residue counts?
                # For simplicity in this rewrite, we do a quick check.
                # If the set of residues of points mod p is size p-1, and c % p is the missing one, it fails.
                # Otherwise it's safe.
                
                # Let's compute residues for p
                res = set()
                for v in points:
                    res.add(v % p)
                    if len(res) == p:
                        # This shouldn't happen if we are admissible, but just in case
                        admissible = False
                        break
                if not admissible:
                    break
                
                if len(res) == p - 1:
                    # Check if c fills the gap
                    missing = [r for r in range(p) if r not in res]
                    if c % p in missing:
                        admissible = False
                        break
            
            if admissible:
                points.append(c)
                points_set.add(c)
                m += 1
                improved = True
                last_kick_time = time.time()
                break # Restart candidate loop with new state
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 290
    all_primes = _primes_upto(N + 1)
    
    best_points = []
    best_size = 0
    
    # Run multiple restarts
    # Each restart: random init + local search
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random Initialization
        # Start with a random subset of size ~ N/4 to N/2
        init_size = random.randint(N // 3, N // 2)
        points = random.sample(range(N + 1), init_size)
        
        # Run local search
        current_points = _randomized_local_search(points, all_primes, deadline)
        
        if len(current_points) > best_size:
            best_size = len(current_points)
            best_points = current_points
            
    # Final verification
    if best_points and not _check_admissible(best_points, all_primes):
        # Emergency repair
        while not _check_admissible(best_points, all_primes) and best_points:
            # Remove random element
            idx = random.randint(0, len(best_points) - 1)
            best_points.pop(idx)
            
    return best_points
# EVOLVE-BLOCK-END
