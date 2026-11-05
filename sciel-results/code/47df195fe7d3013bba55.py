# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations_with_replacement

def _set_sizes(A):
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

def _geometric_init(base, n=20, max_val=10000):
    """Generate a geometric-like sequence to keep product set small."""
    A = []
    for i in range(n):
        val = int(base * (1.3 ** i))
        if val > max_val:
            val = max_val
        if val not in A:
            A.append(val)
    # If we don't have enough, fill in gaps
    while len(A) < n:
        candidate = random.randint(1, max_val)
        if candidate not in A:
            A.append(candidate)
    A.sort()
    return A

def _random_init(n=20, max_val=10000):
    """Random initialization with some structure."""
    # Mix of geometric and random
    A = set()
    # Add a geometric progression
    base = random.randint(1, 10)
    ratio = random.choice([1.2, 1.3, 1.4, 1.5, 2.0])
    for i in range(n):
        val = int(base * (ratio ** i))
        if 1 <= val <= max_val:
            A.add(val)
    # Fill remaining with random
    while len(A) < n:
        candidate = random.randint(1, max_val)
        A.add(candidate)
    return sorted(list(A))

def _local_search(A, max_iter=5000, deadline=None):
    """Local search with simulated annealing."""
    A = sorted(A[:])
    current_obj = _objective(A)
    best_A = A[:]
    best_obj = current_obj
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random move: pick one element and change it
        idx = random.randint(0, len(A) - 1)
        old_val = A[idx]
        new_val = random.randint(1, 10000)
        
        if new_val == old_val or new_val in A:
            continue
        
        A[idx] = new_val
        A.sort()
        
        new_obj = _objective(A)
        
        if new_obj < current_obj:
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = A[:]
        else:
            # Accept with probability based on temperature
            delta = new_obj - current_obj
            if delta > 0:
                prob = 1.0 / (1.0 + delta) * T
                if random.random() > prob:
                    A[idx] = old_val
                    A.sort()
            else:
                # Shouldn't happen but handle it
                pass
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_A, best_obj

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    best_A = None
    best_obj = float('inf')
    
    # Try multiple initializations
    initializations = [
        _geometric_init(1),
        _geometric_init(2),
        _geometric_init(3),
        _random_init(),
        _random_init(),
        list(range(1, 21)),  # Baseline
    ]
    
    for init_A in initializations:
        if time.time() > deadline:
            break
        
        A, obj = _local_search(init_A, max_iter=3000, deadline=deadline)
        if obj < best_obj:
            best_obj = obj
            best_A = A
    
    # If we still have time, do more rounds
    while time.time() < deadline:
        init_A = _random_init()
        A, obj = _local_search(init_A, max_iter=2000, deadline=deadline)
        if obj < best_obj:
            best_obj = obj
            best_A = A
    
    if best_A is None:
        return list(range(1, 21))
    
    return best_A
# EVOLVE-BLOCK-END
