import time
import random
import math

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    # n=20, so 400 pairs
    sumset = set()
    prodset = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def eval_poly(coeffs, t_values):
    """Evaluate polynomial p(t) = sum(c_k * t^k) at each t in t_values."""
    result = []
    for t in t_values:
        val = 0
        for k, c in enumerate(coeffs):
            val += c * (t ** k)
        result.append(val)
    return result

def poly_search(time_budget, rng):
    """
    Search over polynomial coefficients to minimize max(|A+A|, |A*A|).
    A = {p(0), p(1), ..., p(19)} where p is a low-degree polynomial.
    """
    t_values = list(range(20))
    best_score = float('inf')
    best_A = None
    start = time.time()
    
    # Try degrees 1 through 4
    for degree in range(1, 5):
        # Time allocation per degree
        deg_budget = time_budget / 4.0
        deg_deadline = start + deg_budget
        
        # Initial coefficient guess: small positive integers
        # p(t) = c0 + c1*t + c2*t^2 + ...
        # Start with simple AP-like: p(t) = a + b*t
        base_coeffs = [1, 1] + [0] * (degree - 1)
        
        # Local search on coefficients
        current_coeffs = list(base_coeffs)
        # Ensure p(t) > 0 for all t in 0..19
        # p(t) must be positive and <= 10000 for all t
        
        def valid_coeffs(coeffs):
            vals = eval_poly(coeffs, t_values)
            if any(v < 1 for v in vals):
                return False
            if any(v > 10000 for v in vals):
                return False
            if len(set(vals)) != 20:
                return False
            return True
        
        # Hill climbing on coefficients
        current_coeffs = list(base_coeffs)
        if not valid_coeffs(current_coeffs):
            # Find a valid starting point
            found = False
            for _ in range(1000):
                trial = [rng.randint(1, 5) if i == 0 else rng.randint(-3, 3) for i in range(degree)]
                if valid_coeffs(trial):
                    current_coeffs = trial
                    found = True
                    break
            if not found:
                continue
        
        current_A = eval_poly(current_coeffs, t_values)
        current_score = calc_score(current_A)
        best_score = current_score
        best_A = list(current_A)
        
        improved = True
        while improved and (time.time() - start) < deg_deadline:
            improved = False
            # Try perturbing each coefficient
            for k in range(degree):
                if (time.time() - start) > deg_deadline:
                    break
                
                # Try small integer perturbations
                for delta in [-2, -1, 1, 2]:
                    if (time.time() - start) > deg_deadline:
                        break
                    
                    new_coeffs = list(current_coeffs)
                    new_coeffs[k] += delta
                    
                    if not valid_coeffs(new_coeffs):
                        continue
                    
                    new_A = eval_poly(new_coeffs, t_values)
                    new_score = calc_score(new_A)
                    
                    if new_score < current_score:
                        current_score = new_score
                        current_coeffs = new_coeffs
                        current_A = new_A
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_A = list(new_A)
        
        # Also try random restarts within this degree
        remaining = deg_deadline - time.time()
        if remaining > 1.0:
            for _ in range(10):
                if (time.time() - start) > deg_deadline:
                    break
                
                # Random coefficients
                trial = [rng.randint(1, 10)] + [rng.randint(-5, 5) for _ in range(degree - 1)]
                if not valid_coeffs(trial):
                    continue
                
                trial_A = eval_poly(trial, t_values)
                trial_score = calc_score(trial_A)
                
                if trial_score < best_score:
                    best_score = trial_score
                    best_A = list(trial_A)
                
                # Quick local search from this point
                inner_deadline = min(deg_deadline, time.time() + 0.5)
                cur_c = list(trial)
                cur_s = trial_score
                imp = True
                while imp and (time.time() - start) < inner_deadline:
                    imp = False
                    for k in range(degree):
                        if (time.time() - start) > inner_deadline:
                            break
                        for delta in [-1, 1]:
                            nc = list(cur_c)
                            nc[k] += delta
                            if not valid_coeffs(nc):
                                continue
                            na = eval_poly(nc, t_values)
                            ns = calc_score(na)
                            if ns < cur_s:
                                cur_s = ns
                                cur_c = nc
                                imp = True
                                if ns < best_score:
                                    best_score = ns
                                    best_A = list(na)
    
    return best_A, best_score

def construct():
    start_time = time.time()
    # Safety margin: 85 seconds
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    best_A = None
    best_score = float('inf')
    
    # Phase 1: Polynomial-based search (primary strategy)
    poly_time = min(40.0, deadline - time.time())
    if poly_time > 5.0:
        poly_A, poly_score = poly_search(poly_time, rng)
        if poly_A is not None and poly_score < best_score:
            best_score = poly_score
            best_A = poly_A
    
    # Phase 2: Local search refinement on best found
    remaining = deadline - time.time()
    if remaining > 5.0 and best_A is not None:
        # Refine using local search with a candidate pool
        # Generate candidate pool
        candidates = set()
        candidates.update(range(1, 101))
        # Smooth numbers
        for a in range(15):
            p2 = 2 ** a
            if p2 > 10000:
                break
            for b in range(15):
                p23 = p2 * (3 ** b)
                if p23 > 10000:
                    break
                for c in range(15):
                    val = p23 * (5 ** c)
                    if val > 10000:
                        break
                    candidates.add(val)
        candidate_pool = sorted(candidates)
        
        # Local search on best_A
        A = list(best_A)
        A.sort()
        cur_score = best_score
        best_local = list(A)
        best_local_score = cur_score
        local_deadline = time.time() + remaining * 0.8
        
        improved = True
        while improved and (time.time() < local_deadline):
            improved = False
            A_set = set(A)
            for i in range(len(A)):
                if time.time() > local_deadline:
                    break
                old_val = A[i]
                for new_val in candidate_pool:
                    if time.time() > local_deadline:
                        break
                    if new_val in A_set or new_val == old_val:
                        continue
                    new_A = A.copy()
                    new_A[i] = new_val
                    score = calc_score(new_A)
                    if score < cur_score:
                        cur_score = score
                        A = new_A
                        A_set = set(A)
                        improved = True
                        if score < best_local_score:
                            best_local_score = score
                            best_local = list(A)
        
        if best_local_score < best_score:
            best_score = best_local_score
            best_A = best_local
    
    # Fallback
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = calc_score(best_A)
    
    return best_A

# Quick self-test logic (not executed in construct, just for safety)
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Result: {sorted(res)}")
    print(f"Score: {calc_score(res)}")
    print(f"Time: {end - start:.2f}s")
