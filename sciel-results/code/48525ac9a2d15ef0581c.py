"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _score(A):
    """Return max(|A+A|, |A*A|) for sorted list A of length 20."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    return max(len(ss), len(ps))

def _incremental_swap(A, i, old_val, new_val):
    """Compute new score after replacing A[i] with new_val, given current sumset/prodset info.
    Returns (new_score, new_ss, new_ps) or just new_score for efficiency."""
    n = len(A)
    # Remove contributions of old_val at position i
    # Add contributions of new_val at position i
    # This is complex; let's just recompute for simplicity but with early exit
    
    A[i] = new_val
    ss = set()
    ps = set()
    for k in range(n):
        ak = A[k]
        for l in range(k, n):
            ss.add(ak + A[l])
            ps.add(ak * A[l])
    A[i] = old_val
    return max(len(ss), len(ps))

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 20
    maximum = 10000
    
    # Start with a geometric progression base
    # 1, 2, 4, 8, ..., 8192 (14 elements)
    base = [2**i for i in range(14)]
    
    # Need 6 more. Try various additions.
    # Strategy: use smooth numbers (products of 2,3,5)
    smooth = set()
    for a in range(1, maximum + 1):
        x = a
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.add(a)
    
    smooth_list = sorted(smooth)
    # Filter to reasonable range
    smooth_list = [x for x in smooth_list if x <= maximum]
    
    # Initial: try a few candidate sets
    candidates = []
    
    # Candidate 1: geometric + small odd numbers
    for extra in [[3,5,7,9,11,13], [3,5,7,9,11,15], [1,3,5,7,9,11], [3,5,7,15,21,25]]:
        A = sorted(set(base + extra))
        if len(A) == n and all(x <= maximum for x in A):
            candidates.append(A)
    
    # Candidate 2: Pure smooth number set
    # Take first 20 smooth numbers
    if len(smooth_list) >= n:
        candidates.append(sorted(smooth_list[:n]))
    
    # Candidate 3: Mix
    mix_base = [2**i for i in range(10)]  # 1 to 512
    mix_extra = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    A3 = sorted(set(mix_base + mix_extra))
    if len(A3) >= n:
        A3 = A3[:n]
    if len(A3) == n and all(x <= maximum for x in A3):
        candidates.append(A3)
    
    # Evaluate candidates
    best_A = None
    best_score = float('inf')
    for A in candidates:
        if _valid(A, n, maximum):
            sc = _score(A)
            if sc < best_score:
                best_score = sc
                best_A = A
    
    if best_A is None:
        best_A = list(range(1, n+1))
        best_score = _score(best_A)
    
    # Local search
    rng = random.Random(12345)
    A = list(best_A)
    
    # Precompute smooth list for candidates (limit size for speed)
    # Use a focused candidate pool
    candidate_pool = set()
    # Include all current elements
    for x in A:
        candidate_pool.add(x)
    # Add smooth numbers in reasonable range
    for x in smooth_list:
        if x <= maximum:
            candidate_pool.add(x)
    # Add some random numbers
    for _ in range(500):
        candidate_pool.add(rng.randint(1, maximum))
    candidate_pool = sorted(candidate_pool)
    
    # Limit pool size for speed
    if len(candidate_pool) > 2000:
        # Keep a subset
        candidate_pool = candidate_pool[:1000] + candidate_pool[-1000:]
        candidate_pool = sorted(set(candidate_pool))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            old_val = A[i]
            # Try replacing A[i] with values from candidate_pool
            # Sample a subset for speed
            pool_subset = candidate_pool
            for new_val in pool_subset:
                if new_val == old_val:
                    continue
                if new_val in A:
                    continue
                A[i] = new_val
                sc = _score(A)
                if sc < best_score:
                    best_score = sc
                    improved = True
                    break
                A[i] = old_val
            if improved:
                break
        # If no improvement, try random restarts of local search
        if not improved:
            # Try a few random perturbations
            for _ in range(50):
                i = rng.randint(0, n-1)
                old_val = A[i]
                new_val = rng.choice(candidate_pool)
                if new_val == old_val or new_val in A:
                    continue
                A[i] = new_val
                sc = _score(A)
                if sc < best_score:
                    best_score = sc
                    improved = True
                    break
                A[i] = old_val
            if not improved:
                break
    
    # Final validation
    if not _valid(A, n, maximum):
        A = list(range(1, n+1))
    
    return A

def _valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 or x > maximum for x in A):
        return False
    return True
# EVOLVE-BLOCK-END
