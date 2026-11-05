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
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s
    
    random.seed(42)
    
    best_set = None
    best_obj = float('inf')
    
    def try_candidate(A):
        """Validate and evaluate candidate, update best if improved."""
        nonlocal best_set, best_obj
        if len(A) != 20:
            return False
        if len(set(A)) != 20:
            return False
        if any(x < 1 or x > 10000 for x in A):
            return False
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = list(A)
            return True
        return False
    
    # Generate diverse initial candidates
    # 1. Small consecutive
    try_candidate(list(range(1, 21)))
    
    # 2. Even numbers
    try_candidate(list(range(2, 41, 2)))
    
    # 3. Multiples of 3
    try_candidate(list(range(3, 61, 3)))
    
    # 4. Numbers with many small factors: 2^a * 3^b combinations
    candidates_23 = []
    for a in range(0, 6):
        for b in range(0, 6):
            val = (2**a) * (3**b)
            if 1 <= val <= 10000 and val not in candidates_23:
                candidates_23.append(val)
    candidates_23.sort()
    if len(candidates_23) >= 20:
        try_candidate(sorted(candidates_23[:20]))
    
    # 5. Powers of 2
    geo2 = [2**i for i in range(20)]
    if all(x <= 10000 for x in geo2):
        try_candidate(sorted(geo2))
    
    # 6. Small random sets
    for _ in range(50):
        cand = sorted(random.sample(range(1, 50), 20))
        try_candidate(cand)
    
    # 7. Random sets from highly composite regions
    for _ in range(50):
        pool = list(range(1, 100))
        cand = sorted(random.sample(pool, 20))
        try_candidate(cand)
    
    # 8. Structured: mix of small and medium
    for _ in range(30):
        small = random.sample(range(1, 30), 10)
        med = random.sample(range(30, 200), 10)
        cand = sorted(small + med)
        if len(set(cand)) == 20:
            try_candidate(cand)
    
    # 9. Geometric with ratio close to 1 (small integers in arithmetic progression with gaps)
    for gap in [1, 2, 3, 4, 5]:
        cand = list(range(1, 1 + 20*gap, gap))
        if len(cand) == 20 and all(x <= 10000 for x in cand):
            try_candidate(cand)
    
    # 10. Set based on divisors of highly composite numbers
    for hcn in [2520, 5040, 7560, 10080]:
        divs = [d for d in range(1, 10001) if hcn % d == 0]
        divs.sort()
        if len(divs) >= 20:
            try_candidate(divs[:20])
            # Also try last 20
            try_candidate(divs[-20:])
            # Random 20 from divisors
            for _ in range(5):
                if len(divs) >= 20:
                    try_candidate(sorted(random.sample(divs, 20)))
    
    # 11. Products of small primes
    primes = [2, 3, 5, 7]
    prod_candidates = set()
    for a in range(0, 5):
        for b in range(0, 4):
            for c in range(0, 3):
                for d in range(0, 2):
                    val = (2**a) * (3**b) * (5**c) * (7**d)
                    if 1 <= val <= 10000:
                        prod_candidates.add(val)
    prod_list = sorted(prod_candidates)
    if len(prod_list) >= 20:
        try_candidate(prod_list[:20])
        for _ in range(10):
            try_candidate(sorted(random.sample(prod_list, 20)))
    
    # Now do aggressive local search / simulated annealing
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = _objective(best_set)
    
    current_set = best_set[:]
    current_obj = best_obj
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    iteration = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        iteration += 1
        
        # Choose perturbation type
        r = random.random()
        
        if r < 0.5:
            # Replace one element
            i = random.randint(0, 19)
            # Try values near current or random
            if random.random() < 0.5:
                # Local perturbation
                delta = random.randint(-max(5, current_set[i]//4), max(5, current_set[i]//4))
                new_val = current_set[i] + delta
            else:
                # Random value
                new_val = random.randint(1, 10000)
            
            if new_val < 1 or new_val > 10000 or new_val in current_set:
                continue
            
            new_set = current_set[:]
            new_set[i] = new_val
            new_set.sort()
            
        elif r < 0.75:
            # Replace two elements
            i, j = random.sample(range(20), 2)
            new_i = random.randint(1, 10000)
            new_j = random.randint(1, 10000)
            
            if new_i == new_j or new_i in current_set or new_j in current_set:
                continue
            
            new_set = current_set[:]
            new_set[i] = new_i
            new_set[j] = new_j
            new_set.sort()
            
        elif r < 0.9:
            # Replace three elements
            indices = random.sample(range(20), 3)
            new_vals = random.sample(range(1, 10000), 3)
            
            if any(v in current_set for v in new_vals):
                continue
            
            new_set = current_set[:]
            for idx, val in zip(indices, new_vals):
                new_set[idx] = val
            new_set.sort()
            
        else:
            # Swap two elements with nearby values
            i, j = random.sample(range(20), 2)
            delta_i = random.randint(-max(5, current_set[i]//3), max(5, current_set[i]//3))
            delta_j = random.randint(-max(5, current_set[j]//3), max(5, current_set[j]//3))
            new_i = current_set[i] + delta_i
            new_j = current_set[j] + delta_j
            
            if new_i < 1 or new_i > 10000 or new_j < 1 or new_j > 10000:
                continue
            if new_i in current_set or new_j in current_set or new_i == new_j:
                continue
            
            new_set = current_set[:]
            new_set[i] = new_i
            new_set[j] = new_j
            new_set.sort()
        
        # Evaluate
        new_obj = _objective(new_set)
        delta_obj = new_obj - current_obj
        
        # Acceptance criterion
        if delta_obj <= 0 or random.random() < pow(2.718281828, -delta_obj / T):
            current_set = new_set
            current_obj = new_obj
            no_improve = 0
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_set = current_set[:]
                # Cool down more aggressively on improvement
                T *= 0.95
        else:
            no_improve += 1
        
        # Cooling
        T = max(T_min, T * cooling)
        
        # Restart if stuck
        if no_improve > max_no_improve:
            no_improve = 0
            T = 10.0
            # Jump to a new random neighborhood
            current_set = best_set[:]
            current_obj = best_obj
            for _ in range(random.randint(3, 8)):
                i = random.randint(0, 19)
                new_val = random.randint(1, 10000)
                if new_val not in current_set:
                    current_set[i] = new_val
            current_set.sort()
            current_obj = _objective(current_set)
    
    return best_set
# EVOLVE-BLOCK-END
