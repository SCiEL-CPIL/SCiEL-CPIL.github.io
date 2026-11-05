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

def _fast_set_sizes(A):
    """Compute |A+A| and |A*A| using numpy for speed."""
    n = len(A)
    # Use list of tuples for sums and products
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin under 60s
    
    random.seed(42)
    
    def validate_and_obj(A):
        if len(A) != 20:
            return None
        if len(set(A)) != 20:
            return None
        if any(x < 1 or x > 10000 for x in A):
            return None
        s, p = _fast_set_sizes(A)
        return max(s, p)
    
    # Generate structured initial candidates
    candidates = []
    
    # 1. Arithmetic progressions with different starts and steps
    for start in [1, 2, 3, 5, 10]:
        for step in [1, 2, 3, 4, 5, 7, 10, 13, 17, 20]:
            cand = [start + i * step for i in range(20)]
            if all(1 <= x <= 10000 for x in cand):
                candidates.append(sorted(cand))
    
    # 2. Near-arithmetic: AP with small perturbations
    base_ap = list(range(1, 21))
    for _ in range(50):
        cand = base_ap[:]
        for _ in range(random.randint(1, 5)):
            idx = random.randint(0, 19)
            cand[idx] += random.randint(-3, 3)
        if len(set(cand)) == 20 and all(1 <= x <= 10000 for x in cand):
            candidates.append(sorted(cand))
    
    # 3. Mixed: small AP + some geometric elements
    for small_n in [10, 12, 15]:
        ap_part = list(range(1, small_n + 1))
        geo_start = small_n + 1
        geo_part = []
        val = geo_start
        while len(geo_part) < 20 - small_n and val <= 10000:
            geo_part.append(val)
            val *= 2
        if len(geo_part) == 20 - small_n:
            cand = sorted(ap_part + geo_part)
            if len(set(cand)) == 20:
                candidates.append(cand)
    
    # 4. Random sets in various ranges
    for rng_max in [50, 100, 200, 500, 1000]:
        for _ in range(20):
            cand = sorted(random.sample(range(1, rng_max + 1), 20))
            candidates.append(cand)
    
    # 5. Powers of 2, 3, 4 (truncated to 20 elements)
    for base in [2, 3, 4]:
        cand = []
        val = 1
        while len(cand) < 20 and val <= 10000:
            cand.append(val)
            val *= base
        if len(cand) >= 20:
            candidates.append(sorted(cand[:20]))
    
    # Evaluate all candidates
    best_set = None
    best_obj = float('inf')
    
    for cand in candidates:
        obj = validate_and_obj(cand)
        if obj is not None and obj < best_obj:
            best_obj = obj
            best_set = cand[:]
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = _objective(best_set)
    
    # Simulated annealing
    current_set = best_set[:]
    current_obj = best_obj
    
    T = 100.0
    T_min = 0.1
    cooling = 0.995
    steps_per_temp = 50
    
    while time.time() < deadline and T > T_min:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Generate neighbor: perturb 1-3 elements
            n_perturb = random.randint(1, 3)
            new_set = current_set[:]
            perturbed_indices = random.sample(range(20), n_perturb)
            
            valid = True
            for idx in perturbed_indices:
                # Try different perturbation strategies
                strategy = random.random()
                if strategy < 0.3:
                    # Small local change
                    delta = random.randint(-10, 10)
                    if delta == 0:
                        continue
                    new_val = new_set[idx] + delta
                elif strategy < 0.5:
                    # Medium change
                    new_val = new_set[idx] + random.randint(-50, 50)
                elif strategy < 0.7:
                    # Replace with random value in similar range
                    lo = max(1, new_set[idx] // 2)
                    hi = min(10000, new_set[idx] * 2)
                    new_val = random.randint(lo, hi)
                else:
                    # Random from full range
                    new_val = random.randint(1, 10000)
                
                if new_val < 1 or new_val > 10000:
                    valid = False
                    break
                if new_val in new_set and new_set.index(new_val) != idx:
                    valid = False
                    break
                new_set[idx] = new_val
            
            if not valid:
                continue
            
            new_set.sort()
            new_obj = validate_and_obj(new_set)
            if new_obj is None:
                continue
            
            # Accept if better, or with probability exp(-delta/T) if worse
            delta = new_obj - current_obj
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_set = new_set
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_set = current_set[:]
        
        T *= cooling
    
    return best_set
# EVOLVE-BLOCK-END
