import math
import time
import random
import heapq
from collections import defaultdict

MAX_MOD = 128
DEADLINE = 60.0
MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _solve_greedy(moduli, t_limit, seed_val):
    """
    Greedy with backtracking limited to depth 1 (swap with last).
    Returns list of (r, m) or None.
    """
    random.seed(seed_val)
    n = len(moduli)
    if n == 0:
        return []
    
    # Order moduli to help backtracking.
    # Heuristic: Moduli with many small factors (high gcd with others) are more constrained.
    # Or simply sort by value?
    # Let's try sorting by decreasing "constraint density" - i.e., number of other moduli it shares factors with.
    # But for speed, let's just use the input order or a random shuffle.
    # The caller shuffles.
    
    residues = [None] * n
    assigned_mods = [False] * n
    
    def backtrack(idx, depth):
        if time.time() > t_limit:
            return False
        if idx == n:
            return True
        
        m = moduli[idx]
        # Try residues in random order
        residues_to_try = list(range(m))
        random.shuffle(residues_to_try)
        
        for r in residues_to_try:
            valid = True
            for j in range(idx):
                if not _disjoint(r, m, residues[j], moduli[j]):
                    valid = False
                    break
            if valid:
                residues[idx] = r
                if backtrack(idx + 1, depth + 1):
                    return True
                residues[idx] = None
                # If we are at a shallow depth, we could try swapping with a previous assignment?
                # That's complex. Let's stick to simple backtracking but with a limit on total nodes or time.
                # Since we have a time limit, we just return False to trigger restarts or deeper search if allowed.
                # But deep backtracking is exponential.
                # Let's add a node count limit or just rely on time.
                # To make it robust, if we fail at a certain depth, we might want to restart.
                # But the caller handles restarts by calling _solve_greedy again with a different seed.
                return False
        
        return False

    if backtrack(0, 0):
        return [(residues[i], moduli[i]) for i in range(n)]
    return None

def _solve_maxflow_subset(moduli, t_limit):
    """
    Uses a Max-Flow like heuristic based on prime factors to select a subset of moduli
    that are likely solvable, then solves that subset.
    """
    # This is a placeholder for a more structural approach.
    # For now, we rely on the greedy solver with multiple restarts and different orderings.
    pass

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Pre-compute a heuristic for moduli "difficulty".
    # Moduli that are powers of 2 are very constrained?
    # Moduli that are primes are less constrained?
    
    # Strategy:
    # 1. Try to solve the full set with a fast greedy/backtracking hybrid.
    # 2. If that fails or times out, try subsets.
    # 3. Use multiple random seeds and orderings.
    
    best_pairs = []
    best_len = 0
    
    # Heuristic ordering: Sort by number of prime factors?
    # Let's try a few fixed orderings first.
    
    # Ordering 1: Ascending
    orderings = [
        sorted(all_mods),
        sorted(all_mods, reverse=True),
        sorted(all_mods, key=lambda x: len([f for f in range(2, int(x**0.5)+1) if x%f==0])),
        all_mods[:] # Randomized by shuffle later
    ]
    
    # We will run the solver with different seeds and orderings.
    # To maximize speed, we use a very fast check.
    
    # Optimization: Pre-compute gcds for all pairs?
    # n=127, n^2/2 ~ 8000. Very fast.
    gcd_table = {}
    for i, m1 in enumerate(all_mods):
        for j, m2 in enumerate(all_mods):
            if i != j:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def fast_disjoint(r1, m1, r2, m2):
        g = gcd_table[(m1, m2)]
        return (r2 - r1) % g != 0
    
    def _solve_fast(moduli, t_limit, seed_val):
        random.seed(seed_val)
        n = len(moduli)
        if n == 0:
            return []
        
        residues = [None] * n
        
        def backtrack(idx):
            if time.time() > t_limit:
                return False
            if idx == n:
                return True
            
            m = moduli[idx]
            residues_to_try = list(range(m))
            random.shuffle(residues_to_try)
            
            for r in residues_to_try:
                valid = True
                for j in range(idx):
                    if not fast_disjoint(r, m, residues[j], moduli[j]):
                        valid = False
                        break
                if valid:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = None
                    # If we fail, we continue trying other residues for this modulus.
                    # This is standard backtracking.
                    # To avoid exponential blowup, we rely on the time limit.
                    # However, if we get stuck deep, we waste time.
                    # A simple heuristic: if we fail at idx, and idx > 10, we might want to restart.
                    # But let's just let the time limit handle it.
            return False
        
        if backtrack(0):
            return [(residues[i], moduli[i]) for i in range(n)]
        return None

    # Try different orderings and seeds
    # We have ~55 seconds.
    # Each attempt should be capped at a small time, e.g., 0.5s, to allow many restarts.
    # Or, do a few long attempts.
    
    # Let's do 50 attempts of 0.5s each.
    attempts = 0
    max_attempts = 100
    time_per_attempt = 0.5
    
    for seed in range(1000):
        if time.time() > deadline - 1.0 or attempts >= max_attempts:
            break
        
        # Pick an ordering
        # Alternate between sorted and shuffled
        if attempts % 2 == 0:
            mods = sorted(all_mods)
        else:
            mods = all_mods[:]
            random.seed(seed)
            random.shuffle(mods)
        
        # Also try reversing
        if attempts % 4 == 1:
            mods = mods[::-1]
            
        pairs = _solve_fast(mods, t0 + time_per_attempt, seed)
        attempts += 1
        
        if pairs is not None:
            if len(pairs) > best_len:
                best_pairs = pairs
                best_len = len(pairs)
                # If we found a perfect solution, we can stop
                if best_len == len(all_mods):
                    break
        else:
            # If it failed, maybe try a smaller subset?
            # For now, just continue.
            pass
    
    # If we didn't find a solution for the full set, try to greedily build a large one.
    if best_len < len(all_mods):
        # Greedy construction
        # Sort by "constraint" - e.g., modulus with most divisors first?
        # Or just random.
        for seed in range(100):
            if time.time() > deadline - 0.5:
                break
            random.seed(seed)
            mods = all_mods[:]
            random.shuffle(mods)
            
            pairs = []
            for m in mods:
                if time.time() > deadline - 0.2:
                    break
                found = False
                r_candidates = list(range(m))
                random.shuffle(r_candidates)
                for r in r_candidates:
                    ok = True
                    for (r2, m2) in pairs:
                        if not fast_disjoint(r, m, r2, m2):
                            ok = False
                            break
                    if ok:
                        pairs.append((r, m))
                        found = True
                        break
                if not found:
                    continue
            
            if len(pairs) > best_len:
                best_pairs = pairs
                best_len = len(pairs)
                if best_len == len(all_mods):
                    break
    
    if not best_pairs:
        # Fallback
        best_pairs = [(0, 2)]
        
    return [[r, m] for r, m in best_pairs]
