"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _is_sidon_with_diagonals_fast(subset):
    """Check if subset has all unordered pair sums unique including diagonals (i.e. i<=j).
    Uses a set for O(n^2) check."""
    n = len(subset)
    seen = set()
    for i in range(n):
        ai = subset[i]
        for j in range(i, n):
            v = ai + subset[j]
            if v in seen:
                return False
            seen.add(v)
    return True

def _max_sidon_size_ambient(ambient, timeout=30.0):
    """Find maximum k such that some k-subset of ambient is Sidon (with diagonals).
    ambient has 12 elements, so k is in [1..12].
    Uses early termination and timeout guard."""
    n = len(ambient)
    start_time = time.time()
    # Try k from n down to 1; return first k with any k-subset being Sidon
    for k in range(n, 0, -1):
        for combo in itertools.combinations(ambient, k):
            if _is_sidon_with_diagonals_fast(list(combo)):
                return k
            if time.time() - start_time > timeout:
                # If we've spent too long, return k (conservative: assume there might be one)
                return k
    return 0

def _random_ambient(seed=None):
    """Generate a random ambient set of 12 distinct integers in [0,128]."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    return sorted(rng.sample(range(129), 12))

def _local_search(initial, deadline):
    """Local search: try to improve by swapping elements to reduce max Sidon size."""
    current = list(initial)
    current_k = _max_sidon_size_ambient(current, timeout=15.0)
    best = current
    best_k = current_k
    
    while time.time() < deadline:
        improved = False
        # Try single swaps
        for i in range(12):
            for j in range(i+1, 12):
                # Try swapping current[i] with a random value
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    new_val = random.randint(0, 128)
                    if new_val in current:
                        continue
                    candidate = list(current)
                    candidate[i] = new_val
                    if len(set(candidate)) != 12:
                        continue
                    candidate.sort()
                    k = _max_sidon_size_ambient(candidate, timeout=10.0)
                    if k < best_k:
                        best_k = k
                        best = candidate
                        improved = True
                        if best_k <= 3:
                            return best, best_k
                    elif k == best_k:
                        # Accept equal with some probability for diversity
                        if random.random() < 0.1:
                            current = candidate
                            improved = True
        # Try single replacements
        for i in range(12):
            for _ in range(3):
                if time.time() > deadline:
                    break
                new_val = random.randint(0, 128)
                if new_val in current:
                    continue
                candidate = list(current)
                candidate[i] = new_val
                if len(set(candidate)) != 12:
                    continue
                candidate.sort()
                k = _max_sidon_size_ambient(candidate, timeout=10.0)
                if k < best_k:
                    best_k = k
                    best = candidate
                    improved = True
                    if best_k <= 3:
                        return best, best_k
                elif k == best_k:
                    if random.random() < 0.1:
                        current = candidate
                        improved = True
        if not improved:
            break
    
    return best, best_k

def _select_best_ambient():
    deadline = time.time() + 80
    best = None
    best_k = None
    
    # Start with some structured candidates
    cands = [
        list(range(12)),
        list(range(0, 24, 2)),
        list(range(0, 36, 3)),
        list(range(0, 48, 4)),
        list(range(0, 60, 5)),
        list(range(0, 72, 6)),
        [0,1,2,3,4,5,6,7,8,9,10,12],
        [0,1,2,3,4,5,6,7,8,9,11,13],
    ]
    
    for cand in cands:
        if len(cand) != 12 or len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        k = _max_sidon_size_ambient(cand, timeout=10.0)
        if best_k is None or k < best_k:
            best = cand
            best_k = k
            if best_k <= 3:
                break
        if time.time() > deadline:
            break
    
    if best is None:
        best = list(range(12))
        best_k = _max_sidon_size_ambient(best, timeout=10.0)
    
    # Local search from best
    if time.time() < deadline - 10:
        best, best_k = _local_search(best, deadline)
    
    # Try some random starts if time permits
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            rand_cand = _random_ambient()
            k = _max_sidon_size_ambient(rand_cand, timeout=10.0)
            if k < best_k:
                best_k = k
                best = rand_cand
                if best_k <= 3:
                    break
            # Local search from random start
            if time.time() < deadline - 10:
                local_best, local_k = _local_search(rand_cand, deadline - 5)
                if local_k < best_k:
                    best_k = local_k
                    best = local_best
                    if best_k <= 3:
                        break
    
    return best

def construct():
    # Return the best ambient set found
    return _select_best_ambient()
# EVOLVE-BLOCK-END
