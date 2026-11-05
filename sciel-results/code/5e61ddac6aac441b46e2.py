import time
import random
from itertools import combinations

def _set_sizes(A):
    """Compute |A+A| and |A*A| with equal summands/factors allowed."""
    A = list(A)
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def _objective(A):
    s, p = _set_sizes(A)
    return max(s, p)

def _validate(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _grid_set(base1, base2, i_max, j_max):
    """Generate {base1^i * base2^j : i in 0..i_max, j in 0..j_max}"""
    A = []
    for i in range(i_max + 1):
        for j in range(j_max + 1):
            v = (base1 ** i) * (base2 ** j)
            if v <= 10000:
                A.append(v)
    return A

def _local_search(A, max_iters=5000, deadline=None):
    """Local search: replace one element to improve objective."""
    A = list(A)
    best_obj = _objective(A)
    best_set = list(A)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try replacing each element
        for idx in range(20):
            old_val = A[idx]
            # Try some candidate replacements
            candidates = set()
            # Random candidates
            for _ in range(5):
                candidates.add(random.randint(1, 10000))
            # Neighbors of current value
            for delta in [-1, -2, -3, 1, 2, 3, -5, 5, -10, 10, -20, 20, -50, 50]:
                nv = old_val + delta
                if 1 <= nv <= 10000:
                    candidates.add(nv)
            # Multiplicative neighbors
            for mult in [2, 3, 5]:
                if old_val * mult <= 10000:
                    candidates.add(old_val * mult)
                if old_val % mult == 0:
                    candidates.add(old_val // mult)
            
            for cand in candidates:
                if cand in A:
                    continue
                A[idx] = cand
                obj = _objective(A)
                if obj < best_obj:
                    best_obj = obj
                    best_set = list(A)
                    improved = True
                A[idx] = old_val
            
            if improved:
                break
        
        if not improved:
            # Try random restart of one element
            idx = random.randint(0, 19)
            old_val = A[idx]
            A[idx] = random.randint(1, 10000)
            if A[idx] in A:
                A[idx] = old_val
    
    return best_set, best_obj

def construct():
    start = time.time()
    deadline = start + 50.0  # Safety margin under 60s
    
    random.seed(42)
    
    # Generate several initial candidates
    candidates = []
    
    # 4x5 grid with 2,3
    A1 = _grid_set(2, 3, 3, 4)
    if len(A1) == 20 and _validate(A1):
        candidates.append(A1)
    
    # 5x4 grid with 2,3
    A2 = _grid_set(2, 3, 4, 3)
    if len(A2) == 20 and _validate(A2):
        candidates.append(A2)
    
    # 4x5 grid with 2,5
    A3 = _grid_set(2, 5, 3, 4)
    if len(A3) == 20 and _validate(A3):
        candidates.append(A3)
    
    # 5x4 grid with 2,5
    A4 = _grid_set(2, 5, 4, 3)
    if len(A4) == 20 and _validate(A4):
        candidates.append(A4)
    
    # 4x5 grid with 3,5
    A5 = _grid_set(3, 5, 3, 4)
    if len(A5) == 20 and _validate(A5):
        candidates.append(A5)
    
    # 2x10 grid with 2,3
    A6 = _grid_set(2, 3, 1, 9)
    if len(A6) == 20 and _validate(A6):
        candidates.append(A6)
    
    # 10x2 grid with 2,3
    A7 = _grid_set(2, 3, 9, 1)
    if len(A7) == 20 and _validate(A7):
        candidates.append(A7)
    
    # 4x5 with 2,7
    A8 = _grid_set(2, 7, 3, 4)
    if len(A8) == 20 and _validate(A8):
        candidates.append(A8)
    
    # 5x4 with 2,7
    A9 = _grid_set(2, 7, 4, 3)
    if len(A9) == 20 and _validate(A9):
        candidates.append(A9)
    
    # Arithmetic progression
    A10 = list(range(1, 21))
    candidates.append(A10)
    
    # Random sets
    for _ in range(5):
        A = random.sample(range(1, 10001), 20)
        candidates.append(A)
    
    # Evaluate all candidates
    best_set = None
    best_obj = float('inf')
    
    for cand in candidates:
        if not _validate(cand):
            continue
        obj = _objective(cand)
        if obj < best_obj:
            best_obj = obj
            best_set = list(cand)
    
    # Local search on best candidate
    if best_set is not None:
        best_set, best_obj = _local_search(best_set, max_iters=3000, deadline=deadline)
    
    # Try local search on other good candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if not _validate(cand):
            continue
        obj = _objective(cand)
        if obj <= best_obj + 20:  # Only if close to best
            s, p = _set_sizes(cand)
            if obj < best_obj:
                best_set, best_obj = _local_search(cand, max_iters=2000, deadline=deadline)
            else:
                # Try to improve
                new_set, new_obj = _local_search(cand, max_iters=1000, deadline=deadline)
                if new_obj < best_obj:
                    best_set = new_set
                    best_obj = new_obj
    
    # Final random restarts if time permits
    while time.time() < deadline - 5.0:
        A = random.sample(range(1, 10001), 20)
        obj = _objective(A)
        if obj < best_obj:
            best_set, best_obj = _local_search(A, max_iters=2000, deadline=deadline)
        else:
            # Quick local search
            new_set, new_obj = _local_search(A, max_iters=500, deadline=deadline)
            if new_obj < best_obj:
                best_set = new_set
                best_obj = new_obj
    
    return best_set
