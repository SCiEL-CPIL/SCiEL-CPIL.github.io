"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import random

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

_ALL_PRIMES = _primes(N)

def construct():
    start_time = time.time()
    
    def repair(points):
        points = list(points)
        changed = True
        while changed:
            if time.time() - start_time > 250:
                break
            changed = False
            current_len = len(points)
            for p in _ALL_PRIMES:
                if p > current_len:
                    break
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                if all(c > 0 for c in counts):
                    # Find least populated class
                    min_size = min(counts)
                    # Remove all values in the class(es) with min size
                    target_residues = [r for r in range(p) if counts[r] == min_size]
                    # To be deterministic, pick the smallest residue
                    target_res = min(target_residues)
                    points = [v for v in points if v % p != target_res]
                    changed = True
                    break
        return points

    def add_points(points):
        points = list(points)
        point_set = set(points)
        
        # Precompute M for all primes up to N to be safe, or dynamically.
        # Dynamic is better for memory if N is large, but N=5000 is small.
        # Let's compute M for all primes p <= N initially.
        M = {}
        for p in _ALL_PRIMES:
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            M[p] = set(r for r in range(p) if counts[r] == 0)
            
        candidates = [i for i in range(N + 1) if i not in point_set]
        # Shuffle candidates to avoid bias? Or just iterate.
        # Iterating in random order might help find larger sets.
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > 280:
                break
            
            new_len = len(points) + 1
            safe = True
            for p in _ALL_PRIMES:
                if p > new_len:
                    break
                if M[p] == {x % p}:
                    safe = False
                    break
            
            if safe:
                points.append(x)
                point_set.add(x)
                # Update M
                for p in _ALL_PRIMES:
                    if p > new_len:
                        break
                    r = x % p
                    if r in M[p]:
                        M[p].remove(r)
        
        return points

    best_points = []
    
    # Start with the full set [0, 5000] and repair it
    # This strategy was found to yield much larger tuples than starting from small seeds
    full_set = list(range(N + 1))
    repaired_full = repair(full_set)
    added_full = add_points(repaired_full)
    
    if len(added_full) > len(best_points):
        best_points = added_full
    
    # Try a few random initializations as well to see if they improve upon the full-set repair
    for attempt in range(3):
        if time.time() - start_time > 250:
            break
        
        # Random initialization
        size = 2500 + attempt * 200
        initial_points = random.sample(range(N + 1), min(size, N + 1))
        
        repaired = repair(initial_points)
        added = add_points(repaired)
        
        if len(added) > len(best_points):
            best_points = added
        
    return best_points
# EVOLVE-BLOCK-END
