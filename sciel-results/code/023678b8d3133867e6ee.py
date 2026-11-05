"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

def _set_sizes(A):
    """Compute |A+A| and |A*A| for set A (with equal elements allowed)."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def _objective(A):
    """Return max(|A+A|, |A*A|)."""
    s, p = _set_sizes(A)
    return max(s, p)

# EVOLVE-BLOCK-START
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin under 60s
    
    random.seed(42)
    
    best_set = list(range(1, 21))
    best_obj = _objective(best_set)
    
    def try_candidate(A):
        """Validate and evaluate candidate, update best if improved."""
        nonlocal best_set, best_obj
        if len(A) != 20:
            return
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = list(A)
            
    # Search in a small range to find good "shapes"
    # Since objective is invariant under scaling (for integer scaling),
    # searching in [1, 100] is sufficient to find the optimal structure.
    SEARCH_MAX = 100
    
    # --- Initial Candidates in Small Range ---
    # 1. Consecutive
    try_candidate(list(range(1, 21)))
    
    # 2. Arithmetic Progressions in small range
    for start in range(1, 10):
        for step in range(1, 10):
            cand = [start + i * step for i in range(20)]
            if max(cand) <= SEARCH_MAX:
                try_candidate(sorted(cand))
                
    # 3. Geometric Progressions in small range
    for base in [2, 3]:
        cand = []
        val = 1
        while len(cand) < 20 and val <= SEARCH_MAX:
            cand.append(val)
            val *= base
        if len(cand) == 20:
            try_candidate(sorted(cand))
            
    # 4. Random sets in small range
    for _ in range(100):
        cand = sorted(random.sample(range(1, SEARCH_MAX + 1), 20))
        try_candidate(cand)
        
    # --- Simulated Annealing in Small Range ---
    current_set = best_set[:]
    current_obj = best_obj
    T = 10.0  # Lower temperature since space is smaller
    cooling = 0.99
    
    while time.time() < deadline:
        if T < 0.1:
            T = 10.0
            current_set = best_set[:]
            current_obj = best_obj
            
        new_set = current_set[:]
        move_type = random.random()
        
        if move_type < 0.4:
            # Nudge one element by small amount
            i = random.randint(0, 19)
            delta = random.choice([-2, -1, 1, 2, 3, -3])
            new_val = current_set[i] + delta
            if 1 <= new_val <= SEARCH_MAX and new_val not in current_set:
                new_set[i] = new_val
            else:
                continue
        elif move_type < 0.7:
            # Replace one element with random small value
            i = random.randint(0, 19)
            new_val = random.randint(1, SEARCH_MAX)
            if new_val not in current_set:
                new_set[i] = new_val
            else:
                continue
        elif move_type < 0.9:
            # Swap two elements
            i, j = random.sample(range(20), 2)
            new_i = random.randint(1, SEARCH_MAX)
            new_j = random.randint(1, SEARCH_MAX)
            if new_i != new_j and new_i not in current_set and new_j not in current_set:
                new_set[i] = new_i
                new_set[j] = new_j
            else:
                continue
        else:
            # Replace 2 elements
            i, j = random.sample(range(20), 2)
            new_i = random.randint(1, SEARCH_MAX)
            new_j = random.randint(1, SEARCH_MAX)
            if new_i != new_j and new_i not in current_set and new_j not in current_set:
                new_set[i] = new_i
                new_set[j] = new_j
            else:
                continue
                
        new_set.sort()
        
        if len(new_set) != 20 or len(set(new_set)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in new_set):
            continue
            
        new_obj = _objective(new_set)
        
        if new_obj < current_obj or random.random() < math.exp(-(new_obj - current_obj) / T):
            current_set = new_set
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_set = new_set[:]
                
        T *= cooling
        
    return best_set
# EVOLVE-BLOCK-END
