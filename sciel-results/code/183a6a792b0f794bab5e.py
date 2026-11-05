import time
import random
import copy

def _square_nonzero_count(coeffs):
    """Count nonzero terms in the exact square of a polynomial given as a dict {exp: coeff}."""
    exps = list(coeffs.keys())
    square = {}
    for i in range(len(exps)):
        for j in range(i, len(exps)):
            deg = exps[i] + exps[j]
            if i == j:
                square[deg] = square.get(deg, 0) + coeffs[exps[i]] * coeffs[exps[j]]
            else:
                square[deg] = square.get(deg, 0) + 2 * coeffs[exps[i]] * coeffs[exps[j]]
    return sum(1 for v in square.values() if v != 0)

def _square_coeffs(coeffs_dict):
    """Return the square as a dict {degree: coefficient}."""
    exps = sorted(coeffs_dict.keys())
    square = {}
    for i in range(len(exps)):
        for j in range(i, len(exps)):
            deg = exps[i] + exps[j]
            if i == j:
                square[deg] = square.get(deg, 0) + coeffs_dict[exps[i]] * coeffs_dict[exps[j]]
            else:
                square[deg] = square.get(deg, 0) + 2 * coeffs_dict[exps[i]] * coeffs_dict[exps[j]]
    return square

def _to_list(coeffs_dict, max_deg=40):
    """Convert dict to coefficient list in ascending degree."""
    lst = [0] * (max_deg + 1)
    for exp, coeff in coeffs_dict.items():
        if 0 <= exp <= max_deg:
            lst[exp] = coeff
    # Trim trailing zeros but keep at least 1
    while len(lst) > 1 and lst[-1] == 0:
        lst.pop()
    return lst

def _validate(lst):
    """Check if the coefficient list is valid."""
    if len(lst) > 41:
        return False
    if len(lst) == 0 or lst[-1] == 0:
        return False
    if any(abs(c) > 64 for c in lst):
        return False
    if sum(1 for c in lst if c != 0) != 12:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    best_score = float('inf')
    best_result = None
    
    # Strategy: Use arithmetic progression exponents, optimize coefficients
    # Try multiple base structures
    
    random.seed(42)
    
    # Initial candidates: various AP-based structures
    base_exponents_list = [
        list(range(12)),          # 0..11
        list(range(0, 24, 2)),    # 0,2,4,...,22
        list(range(0, 36, 3)),    # 0,3,6,...,33
        list(range(1, 25, 2)),    # 1,3,5,...,23
        list(range(0, 40, 4)),    # 0,4,8,...,44 -> too big, skip
    ]
    
    # Filter valid base exponents (max degree <= 40)
    valid_base_exps = []
    for ex in base_exponents_list:
        if max(ex) <= 40 and len(ex) == 12:
            valid_base_exps.append(ex)
    
    # Also try some structured patterns
    # Symmetric: exponents around a center
    valid_base_exps.append([0,1,2,3,4,5,6,7,8,9,10,11])
    valid_base_exps.append([0,2,4,6,8,10,12,14,16,18,20,22])
    valid_base_exps.append([0,1,3,4,6,7,9,10,12,13,15,16])  # clustered
    
    for base_exps in valid_base_exps:
        if time.time() - start_time > deadline:
            break
        
        # Initialize with random coefficients
        coeffs_dict = {e: random.randint(1, 10) for e in base_exps}
        
        # Local search / simulated annealing
        current_score = _square_nonzero_count(coeffs_dict)
        
        # Try to improve
        for iteration in range(200000):
            if time.time() - start_time > deadline:
                break
            
            # Pick a random exponent to modify
            idx = random.randint(0, 11)
            exp = base_exps[idx]
            old_val = coeffs_dict[exp]
            
            # Propose new value
            if random.random() < 0.3:
                # Small perturbation
                new_val = old_val + random.randint(-3, 3)
            else:
                # Random new value
                new_val = random.randint(-64, 64)
            
            if new_val == 0 or abs(new_val) > 64:
                continue
            
            coeffs_dict[exp] = new_val
            new_score = _square_nonzero_count(coeffs_dict)
            
            if new_score < current_score:
                current_score = new_score
            else:
                # Revert
                coeffs_dict[exp] = old_val
            
            if current_score < best_score:
                best_score = current_score
                # Build the result list
                lst = [0] * 41
                for e, c in coeffs_dict.items():
                    if 0 <= e <= 40:
                        lst[e] = c
                # Trim
                while len(lst) > 1 and lst[-1] == 0:
                    lst.pop()
                if _validate(lst):
                    best_result = lst[:]
        
        # After local search, try more aggressive moves
        if time.time() - start_time > deadline:
            break
        
        # Try coefficient sign flips and small adjustments
        for _ in range(50000):
            if time.time() - start_time > deadline:
                break
            idx = random.randint(0, 11)
            exp = base_exps[idx]
            old_val = coeffs_dict[exp]
            
            # Try negating
            new_val = -old_val
            if new_val == 0 or abs(new_val) > 64:
                continue
            coeffs_dict[exp] = new_val
            new_score = _square_nonzero_count(coeffs_dict)
            if new_score < current_score:
                current_score = new_score
            else:
                coeffs_dict[exp] = old_val
            
            if current_score < best_score:
                best_score = current_score
                lst = [0] * 41
                for e, c in coeffs_dict.items():
                    if 0 <= e <= 40:
                        lst[e] = c
                while len(lst) > 1 and lst[-1] == 0:
                    lst.pop()
                if _validate(lst):
                    best_result = lst[:]
    
    # Fallback: return the seed answer if nothing better found
    if best_result is None:
        return [1] * 12
    
    return best_result
