import time
import random
import math

def _objective(A):
    """
    Compute max(|A+A|, |A*A|) using set operations.
    """
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _generate_5_smooth(limit):
    """Generate all 5-smooth numbers up to limit."""
    smooths = set([1])
    powers2 = [1]
    powers3 = [1]
    powers5 = [1]
    
    # Generate powers
    while powers2[-1] * 2 <= limit:
        powers2.append(powers2[-1] * 2)
    while powers3[-1] * 3 <= limit:
        powers3.append(powers3[-1] * 3)
    while powers5[-1] * 5 <= limit:
        powers5.append(powers5[-1] * 5)
        
    for p2 in powers2:
        for p3 in powers3:
            val = p2 * p3
            if val > limit:
                break
            for p5 in powers5:
                prod = val * p5
                if prod > limit:
                    break
                smooths.add(prod)
                
    return sorted(smooths)

def _objective_fast(A):
    """
    Faster objective: compute sumset and product set sizes.
    Uses the fact that we only need the cardinalities, not the sets themselves
    for comparison purposes during search.
    """
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _local_search(A, smooths, deadline):
    """
    Local search restricting moves to 5-smooth numbers.
    Supports both single-element replacement and swap moves.
    """
    current = list(A)
    current_cost = _objective_fast(current)
    best = list(current)
    best_cost = current_cost
    
    # Pre-filter smooths to those <= 10000
    valid_smooths = [x for x in smooths if x <= 10000]
    valid_smooth_set = set(valid_smooths)
    
    improved = True
    max_passes = 50
    passes = 0
    while improved and passes < max_passes and time.time() < deadline - 3:
        improved = False
        passes += 1
        indices = list(range(20))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 3:
                break
            old = current[idx]
            
            # Try replacing with other 5-smooth numbers (single-element move)
            # Sample a subset for speed
            candidates = random.sample(valid_smooths, min(50, len(valid_smooths)))
            found_improvement = False
            for new_val in candidates:
                if new_val == old or new_val in current:
                    continue
                current[idx] = new_val
                new_cost = _objective_fast(current)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = list(current)
                    found_improvement = True
                    break
                else:
                    current[idx] = old
            if found_improvement:
                continue
            
            # Try swap moves: swap current[idx] with another element
            for idx2 in range(idx + 1, 20):
                if time.time() > deadline - 3:
                    break
                old2 = current[idx2]
                # Try swapping old and old2 with two new smooth values
                # This is expensive, so limit attempts
                if random.random() < 0.3:  # Only try swap moves 30% of the time
                    c1 = random.choice(valid_smooths)
                    c2 = random.choice(valid_smooths)
                    if c1 in current or c2 in current or c1 == c2 or c1 == old or c1 == old2 or c2 == old or c2 == old2:
                        continue
                    current[idx] = c1
                    current[idx2] = c2
                    new_cost = _objective_fast(current)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best = list(current)
                    else:
                        current[idx] = old
                        current[idx2] = old2
    
    return best, best_cost

def _generate_structured_candidates():
    """
    Generate diverse structured candidates:
    - Windows of 5-smooth numbers
    - Arithmetic progressions
    - Geometric progressions (approximate with integers)
    - Mixed structures
    """
    candidates = []
    
    # 5-smooth windows
    smooths = _generate_5_smooth(10000)
    for i in range(len(smooths) - 20 + 1):
        window = smooths[i:i+20]
        if _is_valid(window):
            candidates.append(list(window))
    
    # Arithmetic progressions
    for start in [1, 2, 3, 5, 7, 10, 15, 20]:
        for step in [1, 2, 3, 4, 5, 7, 8, 10, 12, 15, 20]:
            ap = [start + i * step for i in range(20)]
            if all(1 <= x <= 10000 for x in ap) and len(set(ap)) == 20:
                candidates.append(ap)
    
    # Geometric-like progressions (use 5-smooth numbers in geometric pattern)
    for base in [2, 3, 4, 5]:
        geo = []
        val = 1
        while len(geo) < 20 and val <= 10000:
            if val not in geo:
                geo.append(val)
            val = val * base
        if len(geo) >= 20:
            candidates.append(sorted(geo[:20]))
    
    # Mixed: combine small arithmetic progression with larger geometric
    for small_n in [5, 10, 15]:
        for start in [1, 2, 3]:
            for step in [1, 2, 3]:
                ap_part = [start + i * step for i in range(small_n)]
                remaining = 20 - small_n
                geo_part = []
                val = max(ap_part) + 1
                while len(geo_part) < remaining and val <= 10000:
                    geo_part.append(val)
                    val = val * 2
                if len(geo_part) >= remaining:
                    candidate = sorted(ap_part + geo_part[:remaining])
                    if _is_valid(candidate):
                        candidates.append(candidate)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    random.seed(42)
    
    smooths = _generate_5_smooth(10000)
    
    best_set = None
    best_cost = float('inf')
    
    # Generate structured candidates
    candidates = _generate_structured_candidates()
    
    # Evaluate all structured candidates
    for cand in candidates:
        if not _is_valid(cand):
            continue
        cost = _objective_fast(cand)
        if cost < best_cost:
            best_cost = cost
            best_set = list(cand)
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_cost = _objective_fast(best_set)
        
    # Local Search Phase with multiple restarts
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        if r == 0:
            start_A = list(best_set)
        elif r == 1:
            # Start from a different structured candidate
            if len(candidates) > 1:
                start_A = list(candidates[min(r, len(candidates)-1)])
            else:
                start_A = list(best_set)
        else:
            # Perturb best set by swapping a few elements with other smooths
            start_A = list(best_set)
            num_swaps = random.randint(1, 6)
            for _ in range(num_swaps):
                idx = random.randrange(20)
                new_val = random.choice(smooths)
                if new_val not in start_A:
                    start_A[idx] = new_val
            start_A = sorted(start_A)
            if not _is_valid(start_A):
                continue
                
        result, cost = _local_search(start_A, smooths, deadline)
        if cost < best_cost:
            best_cost = cost
            best_set = result
    
    return best_set
