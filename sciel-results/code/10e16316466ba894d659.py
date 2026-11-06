"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _fast_is_sidon(lst):
    """Fast Sidon check for sorted list (all unordered pair sums unique including diagonals)."""
    n = len(lst)
    sums = set()
    for i in range(n):
        a = lst[i]
        for j in range(i, n):
            s = a + lst[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient):
    """Compute max Sidon subset size for sorted 12-element list."""
    n = len(ambient)
    # We know a 2-element set is always Sidon.
    # Check from largest possible k down to 3. If no 3-element Sidon found, return 2.
    # For 12 elements, checking k=6..3 is the main cost.
    for k in range(min(n, 6), 2, -1):
        for combo in itertools.combinations(ambient, k):
            if _fast_is_sidon(list(combo)):
                return k
    # Check for k=3
    for combo in itertools.combinations(ambient, 3):
        if _fast_is_sidon(list(combo)):
            return 3
    return 2

def _construct_candidate():
    """
    Generates a candidate with a small max Sidon subset size.
    Dense sets (small range) tend to have smaller max Sidon subsets due to sum collisions.
    """
    deadline = time.time() + 50  # Safety margin for 60s limit
    
    # Start with known good dense structures
    candidates = []
    
    # Arithmetic Progressions in small ranges
    for start in range(0, 5):
        for step in range(1, 5):
            cand = [start + i * step for i in range(12)]
            if max(cand) <= 128:
                candidates.append(cand)
                
    # Small random dense sets
    rng = random.Random(42)
    for _ in range(20):
        # Sample from a small range to force collisions
        max_val = rng.randint(15, 40)
        while True:
            cand = sorted(rng.sample(range(max_val + 1), 12))
            if len(set(cand)) == 12:
                candidates.append(cand)
                break

    best_cand = None
    best_k = float('inf')
    
    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        k = _max_sidon_size(cand)
        if k < best_k:
            best_k = k
            best_cand = cand
            if best_k <= 3: # 3 is very good, 4 is acceptable, 5 is worse
                break

    # If we didn't find a very good one, try a quick random search in dense spaces
    if best_k > 4:
        for _ in range(100):
            if time.time() > deadline:
                break
            max_val = rng.randint(15, 35)
            while True:
                cand = sorted(rng.sample(range(max_val + 1), 12))
                if len(set(cand)) == 12:
                    break
            
            # Quick check: does it have a 5-Sidon?
            # If not, max is <= 4.
            has_5 = False
            for combo in itertools.combinations(cand, 5):
                if _fast_is_sidon(list(combo)):
                    has_5 = True
                    break
            
            if not has_5:
                k = _max_sidon_size(cand)
                if k < best_k:
                    best_k = k
                    best_cand = cand
                    if best_k <= 3:
                        break

    if best_cand is None:
        # Fallback: simple AP
        return list(range(12))
        
    return best_cand

def construct():
    return _construct_candidate()
# EVOLVE-BLOCK-END
