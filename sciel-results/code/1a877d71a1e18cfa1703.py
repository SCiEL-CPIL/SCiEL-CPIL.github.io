"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _sumset_products(A):
    """Compute |A+A| and |A*A| for a set A (list of ints)."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def _objective(A):
    s, p = _sumset_products(A)
    return max(s, p)

def _valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x <= 0 or x > 10000 for x in A):
        return False
    return True

def _geometric_progression(start, ratio, n=20):
    """Generate a geometric-like progression with integer values."""
    A = []
    val = start
    for _ in range(n):
        if val > 10000:
            break
        A.append(val)
        val = int(val * ratio + 0.5)
        if val <= A[-1]:
            val = A[-1] + 1
    while len(A) < 20:
        A.append(A[-1] + 1)
    return sorted(A[:20])

def _local_search(A, max_steps=5000, deadline=None):
    """Local search: try swapping elements to reduce objective."""
    A = list(A)
    best_obj = _objective(A)
    best_A = list(A)
    rng = random.Random(42)
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        i = rng.randrange(20)
        j = rng.randrange(20)
        if i == j:
            continue
        A[i], A[j] = A[j], A[i]
        if not _valid(A):
            A[i], A[j] = A[j], A[i]
            continue
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A)
        else:
            A[i], A[j] = A[j], A[i]
    
    return best_A, best_obj

def _hill_climb(A, max_steps=3000, deadline=None):
    """Hill climbing: try replacing individual elements."""
    A = list(A)
    best_obj = _objective(A)
    best_A = list(A)
    rng = random.Random(123)
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        i = rng.randrange(20)
        old = A[i]
        new_val = rng.randint(1, 10000)
        if new_val in A and new_val != old:
            continue
        A[i] = new_val
        if not _valid(A):
            A[i] = old
            continue
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A)
        else:
            A[i] = old
    
    return best_A, best_obj

def _simulated_annealing(A, max_steps=20000, deadline=None):
    """Simulated annealing."""
    A = list(A)
    best_obj = _objective(A)
    best_A = list(A)
    cur_obj = best_obj
    rng = random.Random(7)
    
    T = 10.0
    cooling = 0.999
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        i = rng.randrange(20)
        old = A[i]
        new_val = rng.randint(1, 10000)
        if new_val in A and new_val != old:
            continue
        A[i] = new_val
        if not _valid(A):
            A[i] = old
            continue
        obj = _objective(A)
        delta = obj - cur_obj
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
            cur_obj = obj
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)
        else:
            A[i] = old
        T *= cooling
    
    return best_A, best_obj

def _generate_candidates(deadline):
    """Generate diverse initial candidates."""
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, 21)))
    
    # 2. Geometric progressions with various ratios
    for start in [1, 2, 3]:
        for ratio in [1.1, 1.2, 1.3, 1.5, 2.0]:
            gp = _geometric_progression(start, ratio)
            if _valid(gp):
                candidates.append(gp)
    
    # 3. Powers of small bases
    for base in [2, 3]:
        A = [base ** i for i in range(20)]
        if max(A) <= 10000 and len(set(A)) == 20:
            candidates.append(A)
    
    # 4. Random geometric-like sequences
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        ratio = rng.uniform(1.1, 3.0)
        start = rng.randint(1, 5)
        A = []
        for i in range(20):
            val = int(start * (ratio ** i))
            if val > 10000:
                break
            if val not in A and val > 0:
                A.append(val)
        if len(A) == 20 and _valid(A):
            candidates.append(A)
    
    # 5. Random sets with small numbers
    for _ in range(100):
        if time.time() > deadline:
            break
        A = sorted(rng.sample(range(1, 200), 20))
        if _valid(A):
            candidates.append(A)
    
    # 6. Mixed: some small, some spread out
    for seed_val in [1, 2, 5]:
        rng2 = random.Random(seed_val)
        A = sorted(rng2.sample(range(1, 10001), 20))
        candidates.append(A)
    
    # 7. Logarithmic spacing
    for base in [1.5, 2.0, 2.5]:
        A = [max(1, int(base ** i)) for i in range(20)]
        # Ensure distinct
        for i in range(1, len(A)):
            if A[i] <= A[i-1]:
                A[i] = A[i-1] + 1
        if _valid(A):
            candidates.append(A)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Generate diverse initial candidates
    candidates = _generate_candidates(deadline)
    
    # Evaluate initial candidates
    best_A = None
    best_obj = float('inf')
    for cand in candidates:
        if not _valid(cand):
            continue
        obj = _objective(cand)
        if obj < best_obj:
            best_obj = obj
            best_A = list(cand)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    # Local search from best
    if best_A:
        best_A, best_obj = _local_search(best_A, max_steps=3000, deadline=deadline)
    
    # Try hill climbing from several starting points
    for cand in candidates[:8]:
        if time.time() > deadline:
            break
        if not _valid(cand):
            continue
        A, obj = _hill_climb(cand, max_steps=1500, deadline=deadline)
        if obj < best_obj:
            best_obj = obj
            best_A = A
    
    # Simulated annealing from best
    if best_A and time.time() < deadline:
        A, obj = _simulated_annealing(best_A, max_steps=15000, deadline=deadline)
        if obj < best_obj:
            best_obj = obj
            best_A = A
    
    # Final local polish
    if best_A and time.time() < deadline:
        best_A, best_obj = _local_search(best_A, max_steps=2000, deadline=deadline)
    
    # One more hill climb round
    if best_A and time.time() < deadline:
        A, obj = _hill_climb(best_A, max_steps=1000, deadline=deadline)
        if obj < best_obj:
            best_obj = obj
            best_A = A
    
    return best_A

# EVOLVE-BLOCK-END
