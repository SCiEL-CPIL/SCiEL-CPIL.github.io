"""Combined program: vectorized greedy set-reduction with randomized tie-breaking and local improvement."""
import time
import numpy as np

N = 5000
DEADLINE = 300.0  # seconds, safety margin below 330s kill


def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _greedy_repair(points, all_primes, start_time, deadline, rng):
    """Greedily remove least-populated residue classes to achieve admissibility.
    Uses randomized tie-breaking among equally-minimal residue classes."""
    changed = True
    while changed:
        if time.time() - start_time > deadline - 10:
            break
        changed = False
        current_len = len(points)
        
        for p in all_primes:
            if p > current_len:
                break
            
            residues = points % p
            class_counts = np.bincount(residues, minlength=p)
            
            if np.all(class_counts > 0):
                # Find the minimum count
                min_count = np.min(class_counts)
                # Find all classes with the minimum count
                min_classes = np.where(class_counts == min_count)[0]
                # Randomly pick one of the tied minimum classes
                if len(min_classes) > 1:
                    min_class = min_classes[rng.integers(len(min_classes))]
                else:
                    min_class = min_classes[0]
                
                # Remove all points with that residue
                points = points[points % p != min_class]
                changed = True
                break
    
    # Final polish pass: re-check all primes up to len(points) one more time
    current_len = len(points)
    for p in all_primes:
        if p > current_len:
            break
        residues = points % p
        class_counts = np.bincount(residues, minlength=p)
        if np.all(class_counts > 0):
            min_count = np.min(class_counts)
            min_classes = np.where(class_counts == min_count)[0]
            if len(min_classes) > 1:
                min_class = min_classes[rng.integers(len(min_classes))]
            else:
                min_class = min_classes[0]
            points = points[points % p != min_class]
            current_len = len(points)
    
    return points


def _local_improve(points, all_primes, start_time, deadline, rng):
    """Try to add back removed points to increase set size while maintaining admissibility."""
    points = np.sort(points)
    n = len(points)
    
    # Build a boolean mask of which points are present
    present = np.zeros(N + 1, dtype=bool)
    present[points] = True
    
    # Candidates to try adding: points not in current set
    candidates = np.where(~present)[0]
    
    # Shuffle candidates for diversity
    rng.shuffle(candidates)
    
    improved = True
    while improved:
        if time.time() - start_time > deadline - 15:
            break
        improved = False
        current_len = len(points)
        
        for cand in candidates:
            if time.time() - start_time > deadline - 15:
                break
            
            # Try adding this candidate
            test_points = np.append(points, cand)
            test_len = len(test_points)
            
            # Check admissibility
            admissible = True
            for p in all_primes:
                if p > test_len:
                    break
                residues = test_points % p
                class_counts = np.bincount(residues, minlength=p)
                if np.all(class_counts > 0):
                    admissible = False
                    break
            
            if admissible:
                points = test_points
                present[cand] = True
                candidates = np.where(~present)[0]
                rng.shuffle(candidates)
                improved = True
                break
    
    return points


def construct():
    start_time = time.time()
    
    # Precompute primes up to N+1 (max possible len is N+1)
    all_primes = _primes(N + 1)
    
    best_points = None
    best_len = -1
    
    # Try multiple random restarts with different seeds for tie-breaking
    rng = np.random.default_rng(seed=42)
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Start with all integers in [0, N]
        points = np.arange(N + 1, dtype=np.int32)
        
        # Run greedy repair with randomized tie-breaking
        points = _greedy_repair(points, all_primes, start_time, DEADLINE, rng)
        
        # Try local improvement
        points = _local_improve(points, all_primes, start_time, DEADLINE, rng)
        
        current_len = len(points)
        if current_len > best_len:
            best_len = current_len
            best_points = points.copy()
        
        # If we have significant time left, continue; otherwise stop
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 60:
            break
    
    # Use the best result found
    result = best_points if best_points is not None else np.arange(N + 1, dtype=np.int32)
    
    # Convert to list of distinct integers
    result = np.sort(result).tolist()
    
    # Verify admissibility as a safety check
    n_final = len(result)
    for p in all_primes:
        if p > n_final:
            break
        residues = set(v % p for v in result)
        if len(residues) == p:
            # Should not happen, but if it does, do one more removal
            counts = [0] * p
            for v in result:
                counts[v % p] += 1
            min_c = min(counts)
            min_r = counts.index(min_c)
            result = [v for v in result if v % p != min_r]
            n_final = len(result)
    
    return result
