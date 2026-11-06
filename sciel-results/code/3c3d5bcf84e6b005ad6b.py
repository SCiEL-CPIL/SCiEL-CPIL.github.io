import time
import random

def _square(coeffs):
    """Compute exact integer square of polynomial given as coefficient list (ascending degree)."""
    n = len(coeffs)
    # Only iterate over nonzero terms
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    if not nz:
        return [0]
    max_deg = 2 * (len(coeffs) - 1)
    result = [0] * (max_deg + 1)
    for i, ci in nz:
        for j, cj in nz:
            result[i + j] += ci * cj
    # Trim trailing zeros but keep at least length 1
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result

def _nonzero_count(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _is_valid(coeffs):
    """Check all constraints."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    nz = _nonzero_count(coeffs)
    if nz != 12:
        return False
    return True

def _score(coeffs):
    """Lower is better: number of nonzero terms in the square."""
    sq = _square(coeffs)
    return _nonzero_count(sq)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best_poly = None
    best_score = float('inf')
    
    def update_best(poly):
        nonlocal best_poly, best_score
        if _is_valid(poly):
            s = _score(poly)
            if s < best_score:
                best_score = s
                best_poly = poly[:]
    
    random.seed(42)
    
    # Try structured candidates first
    # Family: alternating signs with various patterns
    for base in range(0, 30):
        for d in [1, 2, 3, 4, 5, 6]:
            for sign_pattern in [[1]*12, [1,-1]*6, [1,1,-1,-1]*3, [1,-1,1,-1]*6, [1,-1,-1,1]*3]:
                poly = [0] * 41
                for i in range(12):
                    deg = base + i * d
                    if deg < 41:
                        poly[deg] = sign_pattern[i]
                if poly[-1] != 0:
                    update_best(poly)
    
    # Family: symmetric with specific coefficients
    for base in range(0, 25):
        for d in [1, 2, 3]:
            for coeff_pattern in [[1]*12, [1,-1]*6, [1,2,-1,-2]*3, [1,-2,2,-1]*3]:
                poly = [0] * 41
                for i in range(12):
                    deg = base + i * d
                    if deg < 41:
                        poly[deg] = coeff_pattern[i]
                if poly[-1] != 0:
                    update_best(poly)
    
    # Random search with local improvement
    def random_poly():
        # Choose 12 distinct degrees from 0..40
        degrees = random.sample(range(41), 12)
        degrees.sort()
        poly = [0] * 41
        for d in degrees:
            poly[d] = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6])
        # Ensure last coefficient nonzero
        if poly[40] == 0:
            poly[degrees[-1]] = random.choice([-1, 1, 2, 3])
        # Trim
        last_nz = max(i for i in range(40, -1, -1) if poly[i] != 0)
        return poly[:last_nz+1]
    
    def local_improve(poly, max_iters=50):
        """Try flipping signs, changing coefficients, moving terms."""
        poly = poly[:]
        best_local = _score(poly)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            nz_indices = [i for i, c in enumerate(poly) if c != 0]
            
            # 1. Flip sign
            for idx in nz_indices:
                old = poly[idx]
                poly[idx] = -old
                s = _score(poly)
                if s < best_local:
                    best_local = s
                    improved = True
                else:
                    poly[idx] = old

            # 2. Change coefficient
            for idx in nz_indices:
                old = poly[idx]
                for new_val in [-64, -16, -8, -4, -3, -2, -1, 1, 2, 3, 4, 8, 16, 64]:
                    if new_val == old:
                        continue
                    poly[idx] = new_val
                    s = _score(poly)
                    if s < best_local:
                        best_local = s
                        improved = True
                    else:
                        poly[idx] = old

            # 3. Move term to nearby degree
            for idx in nz_indices:
                old_val = poly[idx]
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_deg = idx + delta
                    if new_deg < 0 or new_deg >= 41:
                        continue
                    if new_deg == idx:
                        continue
                    
                    # Check if target is occupied in current poly
                    if new_deg < len(poly) and poly[new_deg] != 0:
                        continue
                    
                    # Create new poly
                    new_poly = poly[:]
                    # Extend if necessary
                    while len(new_poly) <= new_deg:
                        new_poly.append(0)
                    
                    new_poly[idx] = 0
                    new_poly[new_deg] = old_val
                    
                    # Trim trailing zeros
                    while len(new_poly) > 1 and new_poly[-1] == 0:
                        new_poly.pop()
                    
                    if not _is_valid(new_poly):
                        continue
                        
                    s = _score(new_poly)
                    if s < best_local:
                        best_local = s
                        poly = new_poly
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return poly, best_local
    
    # Random search loop
    while time.time() < deadline:
        poly = random_poly()
        if not _is_valid(poly):
            continue
        s = _score(poly)
        if s < best_score:
            best_score = s
            best_poly = poly[:]
        
        # Local improvement for promising candidates
        if s <= best_score + 3:
            improved_poly, improved_score = local_improve(poly)
            if _is_valid(improved_poly) and improved_score < best_score:
                best_score = improved_score
                best_poly = improved_poly[:]
        
        if time.time() > deadline:
            break
    
    # If we found something good, return it
    if best_poly is not None:
        return best_poly
    
    # Fallback: return a valid 12-term polynomial
    fallback = [0] * 41
    for i in range(12):
        fallback[i] = 1
    return fallback
