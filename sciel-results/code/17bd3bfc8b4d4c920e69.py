"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

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
    
    # --- Diverse structured initializations ---
    # 1. Small consecutive
    try_candidate(list(range(1, 21)))
    
    # 2. Arithmetic progressions
    for start in [1, 2, 3]:
        for step in [1, 2, 3, 4, 5]:
            cand = [start + i * step for i in range(20)]
            if all(1 <= x <= 10000 for x in cand) and len(set(cand)) == 20:
                try_candidate(sorted(cand))
    
    # 3. Sets with 1 and small numbers (1 makes products overlap)
    try_candidate(sorted([1] + random.sample(range(2, 100), 19)))
    try_candidate(sorted([1] + list(range(2, 21))))
    
    # 4. Multiples of small numbers
    for m in [2, 3, 4, 5]:
        cand = [m * i for i in range(1, 21)]
        if all(x <= 10000 for x in cand) and len(set(cand)) == 20:
            try_candidate(sorted(cand))
    
    # 5. Mixed: 1, small primes, and a few larger
    try_candidate(sorted([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 2, 4, 6, 8, 9, 10, 12, 14, 15, 16]))
    
    # 6. Random small-value sets
    for _ in range(40):
        cand = sorted(random.sample(range(1, 80), 20))
        try_candidate(cand)
    
    # 7. Random with 1 included
    for _ in range(20):
        rest = random.sample(range(2, 100), 19)
        cand = sorted([1] + rest)
        try_candidate(cand)
    
    # 8. Near-consecutive perturbations
    for _ in range(30):
        cand = list(range(1, 21))
        for _ in range(random.randint(1, 6)):
            idx = random.randint(0, 19)
            cand[idx] += random.randint(-4, 4)
        if len(set(cand)) == 20 and all(1 <= x <= 10000 for x in cand):
            try_candidate(sorted(cand))
    
    # --- Simulated Annealing refinement ---
    current_set = best_set[:]
    current_obj = best_obj
    T = 30.0  # initial temperature
    cooling = 0.998
    min_T = 0.5
    
    while time.time() < deadline:
        if T < min_T:
            # Restart from best
            current_set = best_set[:]
            current_obj = best_obj
            T = 30.0
        
        # Generate a neighbor
        new_set = current_set[:]
        move_type = random.random()
        
        if move_type < 0.55:
            # Perturb one element
            i = random.randint(0, 19)
            orig = current_set[i]
            if random.random() < 0.6:
                # Local perturbation
                scale = max(3, int(0.15 * orig))
                delta = random.randint(-scale, scale)
            else:
                # Medium jump
                scale = max(10, int(0.5 * orig))
                delta = random.randint(-scale, scale)
            
            new_val = orig + delta
            if new_val < 1 or new_val > 10000 or new_val in current_set:
                continue
            new_set[i] = new_val
            new_set.sort()
        
        elif move_type < 0.75:
            # Swap two elements
            i, j = random.sample(range(20), 2)
            orig_i, orig_j = current_set[i], current_set[j]
            
            if random.random() < 0.5:
                # Local swaps
                si = random.randint(-max(3, int(0.1*orig_i)), max(3, int(0.1*orig_i)))
                sj = random.randint(-max(3, int(0.1*orig_j)), max(3, int(0.1*orig_j)))
                new_i, new_j = orig_i + si, orig_j + sj
            else:
                # Random swaps
                new_i = random.randint(1, 10000)
                new_j = random.randint(1, 10000)
            
            if (1 <= new_i <= 10000 and 1 <= new_j <= 10000 and 
                new_i != new_j and new_i not in current_set and new_j not in current_set):
                new_set[i] = new_i
                new_set[j] = new_j
                new_set.sort()
            else:
                continue
        
        elif move_type < 0.9:
            # Replace 2-4 elements
            k = random.randint(2, 4)
            indices = random.sample(range(20), k)
            existing = set(current_set)
            new_vals = set()
            for _ in range(k * 3):
                v = random.randint(1, 10000)
                if v not in existing and v not in new_vals:
                    new_vals.add(v)
                    if len(new_vals) == k:
                        break
            if len(new_vals) != k:
                continue
            for idx, val in zip(indices, list(new_vals)):
                new_set[idx] = val
            new_set.sort()
        
        else:
            # Large random restart-like move: replace half the set
            k = 10
            indices = random.sample(range(20), k)
            existing = set(current_set)
            new_vals = set()
            for _ in range(k * 5):
                v = random.randint(1, 10000)
                if v not in existing and v not in new_vals:
                    new_vals.add(v)
                    if len(new_vals) == k:
                        break
            if len(new_vals) != k:
                continue
            for idx, val in zip(indices, list(new_vals)):
                new_set[idx] = val
            new_set.sort()
        
        # Evaluate
        if len(set(new_set)) != 20:
            continue
        new_obj = _objective(new_set)
        
        # Acceptance criterion
        if new_obj <= current_obj:
            current_set = new_set
            current_obj = new_obj
            try_candidate(new_set)
        else:
            # Probabilistic acceptance of worse moves
            delta_obj = new_obj - current_obj
            if delta_obj < 100:  # don't accept catastrophically bad moves
                accept_prob = math.exp(-delta_obj / max(T, 0.01))
                if random.random() < accept_prob:
                    current_set = new_set
                    current_obj = new_obj
        
        T *= cooling
    
    return best_set
# EVOLVE-BLOCK-END
