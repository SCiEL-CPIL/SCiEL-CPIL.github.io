"""Rewritten program for integer sum-product extremal sets."""
import time
import random
import math

def _calc_score(A):
    """Compute max(|A+A|, |A*A|) for a set of distinct positive integers."""
    A = list(A)
    sumset = set()
    prodset = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def _generate_smooth_candidates(max_val, factors, count=20):
    """Generate 'count' smallest smooth numbers with given prime factors <= max_val."""
    vals = set()
    # Simple recursive/iterative generation of smooth numbers
    # For small factor sets, this is fast
    def gen_smooth(max_val, factors, current=1, idx=0):
        if current > max_val:
            return
        vals.add(current)
        if idx < len(factors):
            f = factors[idx]
            # Multiply by f repeatedly
            v = current
            while v * f <= max_val:
                v *= f
                gen_smooth(max_val, factors, v, idx)
        if idx + 1 < len(factors):
            gen_smooth(max_val, factors, current, idx + 1)
            
    # This simple recursion might be slow or miss combinations if not careful.
    # Better: generate all 2^a * 3^b * 5^c ... <= 10000
    # Let's stick to the explicit loops from the previous version which worked.
    pass

def _generate_2smooth_candidates():
    """Generate sets of 20 distinct 2-smooth numbers (of form 2^a * 3^b) <= 10000."""
    vals = set()
    a = 0
    while 2**a <= 10000:
        b = 0
        v = 2**a
        while v <= 10000:
            vals.add(v)
            b += 1
            v = 2**a * 3**b
        a += 1
    vals = sorted(vals)
    return vals

def _generate_3smooth_candidates():
    """Generate sets of 20 distinct 3-smooth numbers (of form 2^a * 3^b * 5^c) <= 10000."""
    vals = set()
    a = 0
    while 2**a <= 10000:
        b = 0
        v2 = 2**a
        while v2 <= 10000:
            c = 0
            v = v2
            while v <= 10000:
                vals.add(v)
                c += 1
                v = v2 * 5**c
            b += 1
            v2 = 2**a * 3**b
        a += 1
    vals = sorted(vals)
    return vals

def _simulated_annealing(A, max_iters=2000, start_temp=100.0, end_temp=1.0, timeout=5.0):
    """Simulated Annealing local search."""
    A = sorted(A)
    current_score = _calc_score(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Cooling schedule
        T = start_temp * (end_temp / start_temp) ** (i / max_iters)
        
        # Pick a random index
        idx = random.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Generate a new value
        # Bias towards nearby values for local refinement, occasionally jump
        if random.random() < 0.8:
            # Local move
            delta = random.randint(-50, 50)
            new_val = old_val + delta
        else:
            # Global jump
            new_val = random.randint(1, 10000)
            
        if new_val < 1 or new_val > 10000:
            continue
        if new_val in A and new_val != old_val:
            continue
            
        A[idx] = new_val
        new_score = _calc_score(A)
        
        delta_e = new_score - current_score
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = list(A)
        else:
            A[idx] = old_val
            
    return sorted(best_A), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    candidates = []
    
    # 1. Consecutive integers
    candidates.append(list(range(1, 21)))
    
    # 2. Powers of 2
    candidates.append([2**i for i in range(20)])
    
    # 3. Powers of 3
    candidates.append([3**i for i in range(20)])
    
    # 4. Small 2-smooth numbers
    smooth2 = _generate_2smooth_candidates()
    if len(smooth2) >= 20:
        candidates.append(smooth2[:20])
        # Try different subsets of smooth numbers
        # e.g., first 10 small, last 10 large
        candidates.append(smooth2[:10] + smooth2[-10:])
        
    # 5. Small 3-smooth numbers
    smooth3 = _generate_3smooth_candidates()
    if len(smooth3) >= 20:
        candidates.append(smooth3[:20])
        candidates.append(smooth3[:10] + smooth3[-10:])
        
    # 6. Geometric with ratio 1.5 (integers)
    geo = [1]
    for i in range(19):
        geo.append(int(round(geo[-1] * 1.5)))
    # Ensure distinct
    geo = sorted(set(geo))
    if len(geo) >= 20:
        candidates.append(geo[:20])
        
    # 7. Arithmetic Progressions
    # Try different starts and steps
    for start in [1, 2, 3, 5, 10]:
        for step in [2, 3, 5, 10]:
            ap = [start + i * step for i in range(20)]
            if max(ap) <= 10000:
                candidates.append(ap)
                
    # 8. Mix: Small integers + Powers
    mix1 = list(range(1, 11)) + [2**i for i in range(3, 13)]
    mix1 = sorted(set(x for x in mix1 if x <= 10000))
    if len(mix1) >= 20:
        candidates.append(mix1[:20])
        
    # 9. Random sets
    random.seed(42)
    for _ in range(10):
        rand_set = sorted(random.sample(range(1, 10001), 20))
        candidates.append(rand_set)
        
    best_A = None
    best_score = float('inf')
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        cand = sorted(set(x for x in cand if 1 <= x <= 10000))
        if len(cand) < 20:
            continue
        score = _calc_score(cand)
        if score < best_score:
            best_score = score
            best_A = list(cand)
            
    # Refine the best candidate with Simulated Annealing
    if best_A is not None:
        if time.time() < deadline - 5.0:
            # Run SA for a while
            time_budget = deadline - time.time() - 2.0
            best_A, best_score = _simulated_annealing(best_A, max_iters=5000, timeout=time_budget)
            
    # Try a few more random restarts if time permits
    if best_A is not None and time.time() < deadline - 10.0:
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            # Generate a random set biased towards small numbers (smooth-like)
            # or just random
            A_rand = sorted(random.sample(range(1, 10001), 20))
            score_rand = _calc_score(A_rand)
            if score_rand < best_score:
                best_score = score_rand
                best_A = list(A_rand)
                # Local search
                if time.time() < deadline - 5.0:
                    best_A, best_score = _simulated_annealing(best_A, max_iters=2000, timeout=deadline - time.time() - 3.0)
            
    if best_A is None:
        # Fallback
        best_A = list(range(1, 21))
        
    return best_A

# EVOLVE-BLOCK-END
