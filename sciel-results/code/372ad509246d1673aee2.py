"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time
import random

_DEADLINE = 320.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple: for every prime p in primes_to_check,
    some residue class mod p is missed."""
    point_set = set(points)
    for p in primes_to_check:
        seen = set()
        for v in point_set:
            seen.add(v % p)
        if len(seen) == p:
            return False
    return True


def _get_covered_primes(points, primes_to_check):
    """Return list of primes p where all p residue classes are covered."""
    point_set = set(points)
    covered = []
    for p in primes_to_check:
        seen = set()
        for v in point_set:
            seen.add(v % p)
        if len(seen) == p:
            covered.append(p)
    return covered


def _greedy_removal(points, primes_to_check, deadline):
    """Greedy removal: repeatedly find a covered prime, remove least-populated class."""
    points = list(points)
    point_set = set(points)
    
    while time.time() < deadline:
        # Find all covered primes
        covered = []
        for p in primes_to_check:
            seen = set()
            for v in point_set:
                seen.add(v % p)
            if len(seen) == p:
                covered.append(p)
        
        if not covered:
            break
        
        # Pick the smallest covered prime (most constrained)
        p = min(covered)
        
        # Count elements in each residue class mod p
        class_counts = {}
        for v in point_set:
            r = v % p
            class_counts[r] = class_counts.get(r, 0) + 1
        
        # Remove the least populated class
        min_class = min(class_counts, key=lambda r: class_counts[r])
        to_remove = [v for v in point_set if v % p == min_class]
        
        for v in to_remove:
            point_set.discard(v)
        
        points = list(point_set)
    
    return points


def _add_back_search(points, primes_to_check, deadline):
    """Try to add back removed elements to increase tuple size while maintaining admissibility."""
    points = list(points)
    point_set = set(points)
    all_candidates = set(range(N + 1)) - point_set
    
    # Precompute primes we need to check
    max_possible = len(points) + len(all_candidates)
    check_primes = [p for p in primes_to_check if p <= max_possible]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle candidates for diversity
        candidates = list(all_candidates)
        random.shuffle(candidates)
        
        for v in candidates:
            if time.time() >= deadline:
                break
            # Try adding v
            new_set = point_set | {v}
            # Check admissibility for all relevant primes
            admissible = True
            for p in check_primes:
                seen = set()
                for x in new_set:
                    seen.add(x % p)
                if len(seen) == p:
                    admissible = False
                    break
            
            if admissible:
                point_set.add(v)
                all_candidates.discard(v)
                improved = True
                # Update check_primes if needed
                new_max = len(point_set) + len(all_candidates)
                check_primes = [p for p in primes_to_check if p <= new_max]
    
    return list(point_set)


def _initial_dense_set():
    """Create a dense initial set: 1 plus all primes > sqrt(N)."""
    primes = _primes(N)
    sqrt_n = int(N ** 0.5)
    return [1] + [p for p in primes if p > sqrt_n]


def _initial_random_set(fraction=0.5, seed=None):
    """Create a random initial set of given fraction of [0, N]."""
    if seed is not None:
        random.seed(seed)
    return random.sample(range(N + 1), int(fraction * (N + 1)))


def _full_pipeline(points, deadline):
    """Run greedy removal then add-back search."""
    primes_to_check = _primes(N + 1)
    
    # Phase 1: Greedy removal to make admissible
    points = _greedy_removal(points, primes_to_check, deadline)
    
    # Phase 2: Add-back search to maximize size
    points = _add_back_search(points, primes_to_check, deadline)
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Get primes up to N+1 for checking
    all_primes = _primes(N + 1)
    
    best = []
    
    # Strategy 1: Dense initial set (1 + primes > sqrt(N))
    if time.time() < deadline - 5:
        init1 = _initial_dense_set()
        result1 = _full_pipeline(init1, deadline)
        if len(result1) > len(best):
            best = result1
    
    # Strategy 2: Multiple random restarts with different densities
    attempts = 0
    while time.time() < deadline - 10 and attempts < 20:
        # Vary the fraction for diversity
        fraction = 0.3 + 0.4 * (attempts % 4) / 4.0  # 0.3, 0.4, 0.5, 0.6
        seed = attempts * 7919 + 12345
        init = _initial_random_set(fraction=fraction, seed=seed)
        result = _full_pipeline(init, deadline)
        if len(result) > len(best):
            best = result
        attempts += 1
    
    # Final: if we have time, do one more add-back on the best
    if time.time() < deadline - 5 and best:
        best = _add_back_search(best, all_primes, deadline)
    
    return best
# EVOLVE-BLOCK-END
