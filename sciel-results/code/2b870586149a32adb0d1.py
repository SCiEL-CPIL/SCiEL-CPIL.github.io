"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = False
    return np.where(sieve)[0].tolist()


def construct():
    start_time = time.time()
    deadline = 290.0  # safety margin before 300s kill
    
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    
    # Start with all integers in [0, N]
    mask = np.ones(N + 1, dtype=bool)
    all_points = np.arange(N + 1)
    
    # Repair: iteratively remove least-populated residue class for any prime with full coverage
    changed = True
    while changed:
        if time.time() - start_time > deadline:
            break
        changed = False
        current_len = int(np.sum(mask))
        for p in primes:
            if p > current_len:
                break
            if time.time() - start_time > deadline:
                break
            residues = all_points[mask] % p
            counts = np.bincount(residues, minlength=p)
            if np.all(counts > 0):
                min_res = int(np.argmin(counts))
                mask[(all_points % p) == min_res] = False
                changed = True
                break
    
    # Polish: try to add back removed elements if admissibility is preserved
    if time.time() - start_time < deadline:
        removed = all_points[~mask]
        # Shuffle removed for variety
        rng = np.random.default_rng(42)
        rng.shuffle(removed)
        for v in removed:
            if time.time() - start_time > deadline:
                break
            # Try adding v
            test_mask = mask.copy()
            test_mask[v] = True
            test_len = int(np.sum(test_mask))
            ok = True
            for p in primes:
                if p > test_len:
                    break
                residues = all_points[test_mask] % p
                counts = np.bincount(residues, minlength=p)
                if np.all(counts > 0):
                    ok = False
                    break
            if ok:
                mask = test_mask
    
    return all_points[mask].tolist()
# EVOLVE-BLOCK-END
