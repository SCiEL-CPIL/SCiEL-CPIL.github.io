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

def _local_search(ambient, rng, deadline, current_best_k):
    """Local search: swap one element at a time to minimize max Sidon size."""
    best = list(ambient)
    best_k = current_best_k
    no_improve = 0
    max_iters = 150
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if best_k <= 3:
            break
        
        idx = rng.randint(0, 11)
        new_val = rng.randint(0, 128)
        if new_val in best:
            no_improve += 1
            continue
        
        trial = best[:]
        trial[idx] = new_val
        trial.sort()
        
        k = _max_sidon_size(trial)
        if k < best_k:
            best = trial
            best_k = k
            no_improve = 0
        else:
            no_improve += 1
            # Occasionally accept equal to escape local minima
            if k == best_k and rng.random() < 0.05:
                best = trial
                no_improve = 0
        
        if no_improve >= 60:
            break
    
    return best, best_k

def _construct_candidate():
    """
    Generates a candidate with a small max Sidon subset size.
    Combines dense structured candidates with local search refinement.
    """
    deadline = time.time() + 80  # Safety margin for 90s limit (evaluator kills at 90s)
    
    best_cand = None
    best_k = float('inf')
    
    # Phase 1: Test structured dense candidates
    candidates = []
    
    # Arithmetic Progressions in small ranges
    for start in range(0, 5):
        for step in range(1, 6):
            cand = [start + i * step for i in range(12)]
            if max(cand) <= 128:
                candidates.append(cand)
                
    # Small random dense sets
    rng = random.Random(42)
    for _ in range(30):
        # Sample from a small range to force collisions
        max_val = rng.randint(15, 45)
        while True:
            cand = sorted(rng.sample(range(max_val + 1), 12))
            if len(set(cand)) == 12:
                candidates.append(cand)
                break

    # Additional structured candidates
    candidates.append(list(range(12)))
    candidates.append(list(range(0, 24, 2)))
    candidates.append(list(range(0, 36, 3)))
    candidates.append([0,1,2,3,4,5,6,7,8,9,10,12])
    candidates.append([0,1,2,3,4,5,6,7,8,9,11,13])
    candidates.append([0,1,2,3,4,5,6,7,8,10,12,14])
    candidates.append([0,1,2,4,5,7,8,10,11,13,14,16])
    
    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        k = _max_sidon_size(cand)
        if k < best_k:
            best_k = k
            best_cand = cand
            if best_k <= 3:
                break

    # Phase 2: Local search refinement on best candidate
    if best_cand is not None and best_k > 3:
        improved, ik = _local_search(best_cand, rng, deadline - 5, best_k)
        if ik < best_k:
            best_cand = improved
            best_k = ik

    # Phase 3: If still not great, try more random starts with local search
    if best_k > 4:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            if best_k <= 3:
                break
            
            max_val = rng.randint(15, 40)
            while True:
                cand = sorted(rng.sample(range(max_val + 1), 12))
                if len(set(cand)) == 12:
                    break
            
            k = _max_sidon_size(cand)
            if k < best_k:
                best_k = k
                best_cand = cand
                if best_k <= 3:
                    break
            
            # Local search from this candidate
            if time.time() < deadline - 5:
                improved, ik = _local_search(cand, rng, deadline - 5, best_k)
                if ik < best_k:
                    best_cand = improved
                    best_k = ik
                    if best_k <= 3:
                        break

    if best_cand is None:
        # Fallback: simple AP
        return list(range(12))
        
    return best_cand

def construct():
    return _construct_candidate()
# EVOLVE-BLOCK-END
