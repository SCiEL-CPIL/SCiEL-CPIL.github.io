import time
import random
import numpy as np
from collections import Counter

def _square_coeffs(c):
    """Return coefficient list of c*c (convolution), trimmed to remove trailing zeros."""
    c_arr = np.array(c, dtype=np.int64)
    result = np.convolve(c_arr, c_arr)
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def _nnz(c):
    return sum(1 for x in c if x != 0)

def _is_valid(poly):
    if len(poly) > 41:
        return False
    if len(poly) == 0 or poly[-1] == 0:
        return False
    if sum(1 for x in poly if x != 0) != 12:
        return False
    if any(abs(x) > 64 for x in poly):
        return False
    return True

def _evaluate_poly(poly):
    if not _is_valid(poly):
        return float('inf')
    sq = _square_coeffs(poly)
    return _nnz(sq)

def _build_poly(exps, coeffs):
    """Build polynomial from exponents and coefficients."""
    if not exps:
        return [0]
    poly = [0] * (max(exps) + 1)
    for e, c in zip(exps, coeffs):
        if 0 <= e < len(poly):
            poly[e] = c
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly

def _evaluate_exps_coeffs(exps, coeffs):
    """Evaluate a candidate given by exponents and coefficients."""
    poly = _build_poly(exps, coeffs)
    if len(poly) > 41:
        return float('inf')
    if not _is_valid(poly):
        return float('inf')
    return _evaluate_poly(poly)

def _local_search_v2(exps, coeffs, max_steps=3000, time_limit=3.0):
    """
    Improved local search with simulated annealing-like acceptance.
    """
    start = time.time()
    current_exps = exps[:]
    current_coeffs = coeffs[:]
    
    # Sort exponents for consistency
    order = np.argsort(current_exps)
    current_exps = [current_exps[i] for i in order]
    current_coeffs = [current_coeffs[i] for i in order]
    
    poly = _build_poly(current_exps, current_coeffs)
    if len(poly) > 41:
        return current_exps, current_coeffs, float('inf')
    
    current_score = _evaluate_poly(poly)
    best_score = current_score
    best_exps = current_exps[:]
    best_coeffs = current_coeffs[:]
    
    temperature = 2.0
    cooling = 0.995
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        
        # Choose a random move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Change a coefficient
            i = random.randint(0, 11)
            old_c = current_coeffs[i]
            # Try small changes
            deltas = [-2, -1, 1, 2]
            random.shuffle(deltas)
            moved = False
            for d in deltas:
                new_c = old_c + d
                if new_c == 0 or abs(new_c) > 64:
                    continue
                new_coeffs = current_coeffs[:]
                new_coeffs[i] = new_c
                p = _build_poly(current_exps, new_coeffs)
                if len(p) > 41:
                    continue
                new_score = _evaluate_poly(p)
                
                # Accept if better, or with probability exp(-delta/T)
                if new_score < current_score:
                    current_score = new_score
                    current_coeffs = new_coeffs
                    poly = p
                    if new_score < best_score:
                        best_score = new_score
                        best_exps = current_exps[:]
                        best_coeffs = current_coeffs[:]
                    moved = True
                    break
                else:
                    delta = new_score - current_score
                    if random.random() < np.exp(-delta / temperature):
                        current_score = new_score
                        current_coeffs = new_coeffs
                        poly = p
                        moved = True
                        break
            if moved:
                temperature *= cooling
                continue
                
        elif move_type < 0.7:
            # Change an exponent
            i = random.randint(0, 11)
            old_e = current_exps[i]
            # Try nearby exponents
            candidates = []
            for d in range(-3, 4):
                if d == 0:
                    continue
                new_e = old_e + d
                if 0 <= new_e <= 40 and new_e not in current_exps:
                    candidates.append(new_e)
            random.shuffle(candidates)
            moved = False
            for new_e in candidates:
                new_exps = current_exps[:]
                new_exps[i] = new_e
                # Re-sort
                order = np.argsort(new_exps)
                new_exps = [new_exps[j] for j in order]
                new_coeffs = [current_coeffs[j] for j in order]
                
                p = _build_poly(new_exps, new_coeffs)
                if len(p) > 41:
                    continue
                new_score = _evaluate_poly(p)
                
                if new_score < current_score:
                    current_score = new_score
                    current_exps = new_exps
                    current_coeffs = new_coeffs
                    poly = p
                    if new_score < best_score:
                        best_score = new_score
                        best_exps = current_exps[:]
                        best_coeffs = current_coeffs[:]
                    moved = True
                    break
                else:
                    delta = new_score - current_score
                    if random.random() < np.exp(-delta / temperature):
                        current_score = new_score
                        current_exps = new_exps
                        current_coeffs = new_coeffs
                        poly = p
                        moved = True
                        break
            if moved:
                temperature *= cooling
                continue
                
        elif move_type < 0.85:
            # Swap two exponents
            i, j = random.sample(range(12), 2)
            if current_exps[i] != current_exps[j]:
                new_exps = current_exps[:]
                new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
                p = _build_poly(new_exps, current_coeffs)
                if len(p) <= 41:
                    new_score = _evaluate_poly(p)
                    if new_score < current_score:
                        current_score = new_score
                        current_exps = new_exps
                        poly = p
                        if new_score < best_score:
                            best_score = new_score
                            best_exps = current_exps[:]
                            best_coeffs = current_coeffs[:]
                        temperature *= cooling
                        continue
                    else:
                        delta = new_score - current_score
                        if random.random() < np.exp(-delta / temperature):
                            current_score = new_score
                            current_exps = new_exps
                            poly = p
                            temperature *= cooling
                            continue
        
        else:
            # Change multiple coefficients at once
            indices = random.sample(range(12), 3)
            new_coeffs = current_coeffs[:]
            for i in indices:
                new_coeffs[i] = random.choice([-2, -1, 1, 2])
            p = _build_poly(current_exps, new_coeffs)
            if len(p) <= 41:
                new_score = _evaluate_poly(p)
                if new_score < current_score:
                    current_score = new_score
                    current_coeffs = new_coeffs
                    poly = p
                    if new_score < best_score:
                        best_score = new_score
                        best_exps = current_exps[:]
                        best_coeffs = current_coeffs[:]
                    temperature *= cooling
                    continue
        
        temperature *= cooling
    
    return best_exps, best_coeffs, best_score

def _structured_search(deadline):
    """
    Search for structured solutions:
    1. Palindromic polynomials
    2. Polynomials with symmetric exponent sets
    3. Products of simpler sparse polynomials
    """
    best_score = float('inf')
    best_exps = None
    best_coeffs = None
    
    # Strategy 1: Palindromic coefficients with symmetric exponents
    # P(x) = sum c_i x^{e_i} where e_i are symmetric around some center
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        
        # Choose a center and symmetric exponents
        center = random.randint(5, 20)
        offsets = random.sample(range(1, 15), 6)
        offsets.sort()
        exps = []
        for off in offsets:
            e1 = center - off
            e2 = center + off
            if 0 <= e1 <= 40 and 0 <= e2 <= 40:
                exps.append(e1)
                exps.append(e2)
        
        if len(exps) != 12:
            continue
        
        exps.sort()
        
        # Palindromic coefficients: c_i = c_{11-i} or c_i = -c_{11-i}
        base_coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(6)]
        coeffs = []
        for i in range(6):
            sign = random.choice([1, -1])
            coeffs.append(base_coeffs[i])
            coeffs.append(sign * base_coeffs[i])
        
        score = _evaluate_exps_coeffs(exps, coeffs)
        if score < best_score:
            best_score = score
            best_exps = exps[:]
            best_coeffs = coeffs[:]
        
        # Local search from this point
        e, c, s = _local_search_v2(exps, coeffs, max_steps=1000, time_limit=1.0)
        if s < best_score:
            best_score = s
            best_exps = e
            best_coeffs = c
    
    # Strategy 2: Exponents in arithmetic progression
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        
        start = random.randint(0, 20)
        step = random.randint(1, 5)
        exps = [start + i * step for i in range(12)]
        
        if max(exps) > 40:
            continue
        
        coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        
        score = _evaluate_exps_coeffs(exps, coeffs)
        if score < best_score:
            best_score = score
            best_exps = exps[:]
            best_coeffs = coeffs[:]
        
        e, c, s = _local_search_v2(exps, coeffs, max_steps=1000, time_limit=1.0)
        if s < best_score:
            best_score = s
            best_exps = e
            best_coeffs = c
    
    return best_exps, best_coeffs, best_score

def _random_search(deadline):
    """
    Pure random search with local optimization.
    """
    best_score = float('inf')
    best_exps = None
    best_coeffs = None
    
    attempts = 0
    while time.time() < deadline - 10 and attempts < 3000:
        attempts += 1
        
        # Random exponents
        exps = random.sample(range(0, 41), 12)
        exps.sort()
        
        # Small coefficients
        coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        
        score = _evaluate_exps_coeffs(exps, coeffs)
        if score < best_score:
            best_score = score
            best_exps = exps[:]
            best_coeffs = coeffs[:]
        
        # Local search
        e, c, s = _local_search_v2(exps, coeffs, max_steps=1500, time_limit=1.5)
        if s < best_score:
            best_score = s
            best_exps = e
            best_coeffs = c
    
    return best_exps, best_coeffs, best_score

def _refine_solution(exps, coeffs, time_limit=10.0):
    """
    Intensive refinement of a good solution.
    """
    start = time.time()
    current_exps = exps[:]
    current_coeffs = coeffs[:]
    
    # Sort
    order = np.argsort(current_exps)
    current_exps = [current_exps[i] for i in order]
    current_coeffs = [current_coeffs[i] for i in order]
    
    poly = _build_poly(current_exps, current_coeffs)
    current_score = _evaluate_poly(poly)
    best_score = current_score
    best_exps = current_exps[:]
    best_coeffs = current_coeffs[:]
    
    # Iterated local search with restarts
    for restart in range(5):
        if time.time() - start > time_limit:
            break
        
        # Run local search
        e, c, s = _local_search_v2(current_exps, current_coeffs, 
                                   max_steps=5000, time_limit=time_limit/5)
        if s < best_score:
            best_score = s
            best_exps = e
            best_coeffs = c
            current_exps = e
            current_coeffs = c
        
        # Perturb for next restart
        if restart < 4:
            # Change 1-2 exponents
            for _ in range(random.randint(1, 2)):
                i = random.randint(0, 11)
                old_e = current_exps[i]
                for d in [-2, -1, 1, 2]:
                    new_e = old_e + d
                    if 0 <= new_e <= 40 and new_e not in current_exps:
                        current_exps[i] = new_e
                        break
            
            # Change 1-2 coefficients
            for _ in range(random.randint(1, 2)):
                i = random.randint(0, 11)
                current_coeffs[i] = random.choice([-3, -2, -1, 1, 2, 3])
            
            # Re-sort
            order = np.argsort(current_exps)
            current_exps = [current_exps[i] for i in order]
            current_coeffs = [current_coeffs[i] for i in order]
    
    return best_exps, best_coeffs, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = float('inf')
    best_exps = None
    best_coeffs = None
    
    # Phase 1: Structured search
    if time.time() < deadline - 20:
        e, c, s = _structured_search(deadline - 20)
        if e is not None and s < best_score:
            best_score = s
            best_exps = e
            best_coeffs = c
    
    # Phase 2: Random search
    if time.time() < deadline - 10:
        e, c, s = _random_search(deadline - 10)
        if e is not None and s < best_score:
            best_score = s
            best_exps = e
            best_coeffs = c
    
    # Phase 3: Refine best solution
    if best_exps is not None and time.time() < deadline - 2:
        e, c, s = _refine_solution(best_exps, best_coeffs, time_limit=10.0)
        if s < best_score:
            best_score = s
            best_exps = e
            best_coeffs = c
    
    # Construct final polynomial
    if best_exps is None or best_coeffs is None:
        # Fallback: simple 12-term polynomial
        return [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    poly = _build_poly(best_exps, best_coeffs)
    
    if len(poly) > 41:
        poly = poly[:41]
    
    if not _is_valid(poly):
        # Fallback
        return [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    return poly

# EVOLVE-BLOCK-END
