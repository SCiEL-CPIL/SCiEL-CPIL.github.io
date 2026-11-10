import time
import math
import random

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def calc_sumprod_sizes(A):
    """Return (len(A+A), len(A*A)) for set A."""
    if not A:
        return (0, 0)
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return (len(sumset), len(prodset))

def soft_max(x, y, temperature=1.0):
    """Smooth approximation of max(x, y) using log-sum-exp."""
    if x == y:
        return x
    m = max(x, y)
    return m + temperature * math.log(math.exp((x - m) / temperature) + math.exp((y - m) / temperature))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Strategy: Start with geometric progression base, then perturb multiplicatively
    # A = {b^e_i * k_i} where b is base, e_i are exponents, k_i are small coprime multipliers
    
    # Generate candidate pool
    candidates = set()
    
    # Small integers
    for i in range(1, 201):
        candidates.add(i)
        
    # Powers of 2
    for a in range(0, 14):
        candidates.add(2**a)
        
    # 3-smooth
    for a in range(0, 14):
        for b in range(0, 14):
            val = (2**a) * (3**b)
            if val <= 10000:
                candidates.add(val)
                
    # 5-smooth
    for a in range(0, 12):
        for b in range(0, 12):
            for c in range(0, 12):
                val = (2**a) * (3**b) * (5**c)
                if val <= 10000:
                    candidates.add(val)
                    
    # 7-smooth
    for a in range(0, 10):
        for b in range(0, 10):
            for c in range(0, 10):
                for d in range(0, 10):
                    val = (2**a) * (3**b) * (5**c) * (7**d)
                    if val <= 10000:
                        candidates.add(val)
                        
    # Add some randoms
    for _ in range(300):
        candidates.add(random.randint(1, 10000))
        
    candidates = sorted(list(candidates))
    
    # Phase 1: Try multiple geometric progression starts
    best_A = None
    best_score = float('inf')
    
    # Try different bases and exponent patterns
    bases_to_try = [2, 3, 4, 5]
    exponent_patterns = [
        list(range(0, 20)),  # 0,1,2,...,19
        list(range(1, 21)),  # 1,2,3,...,20
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ]
    
    for base in bases_to_try:
        for exps in exponent_patterns:
            if time.time() > deadline - 30:
                break
                
            # Generate geometric progression
            A_init = []
            for e in exps:
                val = base ** e
                if val <= 10000 and val > 0:
                    A_init.append(val)
            
            # Ensure 20 distinct values
            if len(set(A_init)) < 20:
                continue
                
            A_init = sorted(list(set(A_init)))
            if len(A_init) > 20:
                A_init = A_init[:20]
            elif len(A_init) < 20:
                # Fill with small integers
                for i in range(1, 100):
                    if i not in A_init:
                        A_init.append(i)
                    if len(A_init) == 20:
                        break
                if len(A_init) < 20:
                    continue
                A_init = sorted(A_init[:20])
            
            score = calc_score(A_init)
            if score < best_score:
                best_score = score
                best_A = A_init[:]
        
        if time.time() > deadline - 30:
            break
    
    # If no good geometric progression found, use arithmetic progression as fallback
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = calc_score(best_A)
    
    A = best_A[:]
    current_score = best_score
    
    # Phase 2: Multiplicative perturbation
    # Try A' = {a_i * k_i} where k_i are small integers
    # This exploits the multiplicative structure
    
    # Generate small multipliers
    multipliers = list(range(1, 11))
    
    # Try perturbing each element with different multipliers
    improved = True
    max_perturb_passes = 5
    perturb_pass = 0
    
    while improved and time.time() < deadline - 20 and perturb_pass < max_perturb_passes:
        improved = False
        perturb_pass += 1
        
        for i in range(len(A)):
            if time.time() > deadline - 20:
                break
                
            original = A[i]
            best_mult = 1
            best_mult_score = current_score
            
            for mult in multipliers:
                if mult == 1:
                    continue
                    
                new_val = original * mult
                if new_val > 10000 or new_val in A:
                    continue
                
                # Try this perturbation
                A_new = A[:]
                A_new[i] = new_val
                A_new = sorted(A_new)
                
                s = calc_score(A_new)
                if s < best_mult_score:
                    best_mult_score = s
                    best_mult = mult
            
            if best_mult != 1:
                A[i] = original * best_mult
                A = sorted(A)
                current_score = best_mult_score
                improved = True
        
        # Also try division by small factors
        for i in range(len(A)):
            if time.time() > deadline - 20:
                break
                
            original = A[i]
            best_div = 1
            best_div_score = current_score
            
            for div in range(2, 6):
                if original % div != 0:
                    continue
                    
                new_val = original // div
                if new_val < 1 or new_val in A:
                    continue
                
                A_new = A[:]
                A_new[i] = new_val
                A_new = sorted(A_new)
                
                s = calc_score(A_new)
                if s < best_div_score:
                    best_div_score = s
                    best_div = div
            
            if best_div != 1:
                A[i] = original // best_div
                A = sorted(A)
                current_score = best_div_score
                improved = True
    
    # Phase 3: Local search with smooth surrogate
    # Use log-sum-exp to balance sumset and product-set sizes more smoothly
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_swap = None
        best_swap_score = current_score
        
        # Use a subset of candidates for speed
        # Focus on smooth numbers and values near existing elements
        local_candidates = set()
        for x in A:
            # Add neighbors
            for delta in range(-5, 6):
                if 1 <= x + delta <= 10000:
                    local_candidates.add(x + delta)
            # Add multiples and factors
            for m in range(2, 4):
                if x * m <= 10000:
                    local_candidates.add(x * m)
                if x % m == 0:
                    local_candidates.add(x // m)
            # Add smooth numbers near x
            for smooth in [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 25, 30, 32, 36, 40, 45, 48, 50, 60, 64, 72, 75, 80, 90, 96, 100]:
                if 1 <= smooth <= 10000:
                    local_candidates.add(smooth)
        
        local_candidates = sorted([c for c in local_candidates if c not in A])
        
        for i, x in enumerate(A):
            if time.time() > deadline - 10:
                break
                
            for y in local_candidates:
                if y == x or y in A:
                    continue
                
                A_new = A[:i] + A[i+1:] + [y]
                A_new = sorted(A_new)
                
                # Use smooth surrogate for evaluation
                sum_size, prod_size = calc_sumprod_sizes(A_new)
                # Temperature annealing: start with high temp, decrease
                temp = 1.0
                smooth_score = soft_max(sum_size, prod_size, temp)
                
                # Also compute exact score for comparison
                exact_score = max(sum_size, prod_size)
                
                # Prefer exact improvement, but use smooth score for tie-breaking
                if exact_score < best_swap_score or (exact_score == best_swap_score and smooth_score < best_swap_score):
                    best_swap_score = exact_score
                    best_swap = (i, x, y)
                    improved = True
        
        if improved:
            i, x, y = best_swap
            A[i] = y
            A = sorted(A)
            current_score = best_swap_score
        else:
            break
    
    # Phase 4: Final exact local search
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_swap = None
            best_swap_score = current_score
            
            # Targeted candidates
            targeted = set()
            for x in A:
                for delta in range(-10, 11):
                    if 1 <= x + delta <= 10000:
                        targeted.add(x + delta)
                if x > 1:
                    targeted.add(x * 2)
                    if x % 2 == 0:
                        targeted.add(x // 2)
                    if x % 3 == 0:
                        targeted.add(x // 3)
                    if x * 3 <= 10000:
                        targeted.add(x * 3)
            
            targeted = sorted([t for t in targeted if t not in A])
            
            for i, x in enumerate(A):
                for y in targeted:
                    if y == x or y in A:
                        continue
                    
                    A_new = A[:i] + A[i+1:] + [y]
                    A_new = sorted(A_new)
                    s = calc_score(A_new)
                    
                    if s < best_swap_score:
                        best_swap_score = s
                        best_swap = (i, x, y)
                        improved = True
            
            if improved:
                i, x, y = best_swap
                A[i] = y
                A = sorted(A)
                current_score = best_swap_score
            else:
                break

    # Ensure valid output
    best_set = sorted(list(set(A)))
    if len(best_set) < 20:
        for i in range(1, 200):
            if i not in best_set:
                best_set.append(i)
            if len(best_set) == 20:
                break
        best_set = best_set[:20]
    elif len(best_set) > 20:
        best_set = best_set[:20]
        
    return best_set

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = calc_score(res)
    print(f"Time: {end-start:.2f}s")
    print(f"Score: {score}")
    print(f"Set: {res}")
