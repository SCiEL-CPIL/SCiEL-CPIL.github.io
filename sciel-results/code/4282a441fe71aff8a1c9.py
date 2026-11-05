"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

N = 5000
_DEADLINE = 300.0  # seconds wall-clock limit

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = False
    return [int(p) for p in np.where(sieve)[0]]


def _is_admissible(points):
    """Check if points form an admissible tuple."""
    n = len(points)
    if n == 0:
        return True
    primes = _primes(n)
    for p in primes:
        residues = set()
        for v in points:
            residues.add(v % p)
        if len(residues) == p:
            return False
    return True


def _count_residues(points, p):
    """Count how many points fall in each residue class mod p using NumPy."""
    arr = np.array(points, dtype=np.int32)
    counts = np.bincount(arr % p, minlength=p)
    return counts


def _greedy_remove(points, primes_list, deadline, rng=None):
    """Greedy removal with randomized tie-breaking to escape local optima.
    
    For each violating prime, choose the residue class with the fewest points,
    with small random perturbation to explore different trade-offs.
    """
    start_time = time.time()
    points_list = list(points)
    
    # Precompute primes up to N+1
    all_primes = [p for p in primes_list if p <= N + 1]
    
    changed = True
    while changed:
        if time.time() - start_time > deadline:
            break
        changed = False
        n = len(points_list)
        if n == 0:
            break
        
        # Find all violating primes and their residue counts
        violating = []
        for p in all_primes:
            if p > n:
                break
            counts = _count_residues(points_list, p)
            if np.all(counts > 0):
                violating.append((p, counts))
        
        if not violating:
            break
        
        # For each violating prime, compute a score for each residue class
        # Score = points_in_class + small random noise
        best_p = None
        best_r = None
        best_score = float('inf')
        
        for p, counts in violating:
            for r in range(p):
                if counts[r] > 0:
                    score = float(counts[r])
                    if rng is not None:
                        score += rng.random() * 0.05
                    if score < best_score:
                        best_score = score
                        best_p = p
                        best_r = r
        
        if best_p is None:
            break
        
        # Remove all points in residue class best_r mod best_p
        p = best_p
        r = best_r
        points_list = [v for v in points_list if v % p != r]
        changed = True
    
    return points_list


def _deterministic_greedy(points, primes_list, deadline):
    """Original deterministic greedy: remove least-populated residue class."""
    start_time = time.time()
    points_list = list(points)
    all_primes = [p for p in primes_list if p <= N + 1]
    
    changed = True
    while changed:
        if time.time() - start_time > deadline:
            break
        changed = False
        n = len(points_list)
        if n == 0:
            break
        
        for p in all_primes:
            if p > n:
                break
            counts = _count_residues(points_list, p)
            if np.all(counts > 0):
                min_idx = int(np.argmin(counts))
                points_list = [v for v in points_list if v % p != min_idx]
                changed = True
                break
    
    return points_list


def construct():
    start_time = time.time()
    deadline = _DEADLINE - 30  # safety margin
    
    best_result = None
    best_size = -1
    
    primes_list = _primes(N + 1)
    
    # First, run deterministic greedy as baseline
    det_result = _deterministic_greedy(list(range(N + 1)), primes_list, deadline - (time.time() - start_time))
    if _is_admissible(det_result):
        best_size = len(det_result)
        best_result = det_result
    
    # Multiple randomized restarts to escape local optima
    remaining_time = deadline - (time.time() - start_time)
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() - start_time > deadline:
            break
        
        rng = random.Random(restart * 12345 + 678)
        points = list(range(N + 1))
        
        result = _greedy_remove(points, primes_list, deadline - (time.time() - start_time), rng)
        
        if _is_admissible(result):
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
        else:
            # Fallback: try deterministic on the partial result
            fixed = _deterministic_greedy(result, primes_list, deadline - (time.time() - start_time))
            if _is_admissible(fixed):
                if len(fixed) > best_size:
                    best_size = len(fixed)
                    best_result = fixed
    
    if best_result is None:
        return []
    
    return best_result
# EVOLVE-BLOCK-END
