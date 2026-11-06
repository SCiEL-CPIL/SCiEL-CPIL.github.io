import time
import random

def _square(coeffs):
    n = len(coeffs)
    result = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(i, n):
            cj = coeffs[j]
            if cj == 0:
                continue
            result[i + j] += ci * cj
            if i != j:
                result[i + j] += ci * cj  # wait no, that's wrong
    # Let me redo this properly
    result = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            cj = coeffs[j]
            if cj == 0:
                continue
            result[i + j] += ci * cj
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result

def _nonzero_count(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _is_valid(coeffs):
    if len(coeffs) > 41 or len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    if _nonzero_count(coeffs) != 12:
        return False
    return True

def _score(coeffs):
    return _nonzero_count(_square(coeffs))

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best_poly = None
    best_score = float('inf')
    
    def try_poly(poly):
        nonlocal best_poly, best_score
        if not _is_valid(poly):
            return
        s = _score(poly)
        if s < best_score:
            best_score = s
            best_poly = poly[:]
    
    random.seed(12345)
    
    # Try structured candidates
    # 1. All ones at consecutive degrees
    for start in range(0, 30):
        poly = [0] * 41
        for i in range(12):
            if start + i < 41:
                poly[start + i] = 1
        last_nz = max(i for i in range(40, -1, -1) if poly[i] != 0)
        try_poly(poly[:last_nz+1])
    
    # 2. Alternating signs
    for start in range(0, 30):
        for d in [1, 2, 3]:
            poly = [0] * 41
            for i in range(12):
                deg = start + i * d
                if deg < 41:
                    poly[deg] = 1 if i % 2 == 0 else -1
            last_nz = max(i for i in range(40, -1, -1) if poly[i] != 0)
            try_poly(poly[:last_nz+1])
    
    # 3. Geometric with different ratios
    for start in range(0, 20):
        for d in range(1, 4):
            poly = [0] * 41
            for i in range(12):
                deg = start + i * d
                if deg < 41:
                    poly[deg] = 1
            last_nz = max(i for i in range(40, -1, -1) if poly[i] != 0)
            try_poly(poly[:last_nz+1])
    
    # 4. Symmetric: terms at positions that are symmetric
    for center in range(10, 30):
        for d in [1, 2, 3]:
            poly = [0] * 41
            # 6 terms on each side
            for i in range(6):
                deg1 = center - i * d
                deg2 = center + (i + 1) * d
                if 0 <= deg1 < 41:
                    poly[deg1] = 1
                if 0 <= deg2 < 41:
                    poly[deg2] = 1
            last_nz = max(i for i in range(40, -1, -1) if poly[i] != 0)
            if last_nz >= 0:
                try_poly(poly[:last_nz+1])
    
    # Random search
    def make_random_poly():
        degrees = sorted(random.sample(range(41), 12))
        poly = [0] * 41
        for d in degrees:
            poly[d] = random.choice([-1, 1, -2, 2, -3, 3])
        last_nz = max(i for i in range(40, -1, -1) if poly[i] != 0)
        return poly[:last_nz+1]
    
    def improve(poly, max_steps=50):
        poly = poly[:]
        cur_score = _score(poly)
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            nz = [i for i, c in enumerate(poly) if c != 0]
            
            # Flip sign
            for idx in nz:
                old = poly[idx]
                poly[idx] = -old
                s = _score(poly)
                if s < cur_score:
                    cur_score = s
                    improved = True
                else:
                    poly[idx] = old
            
            # Change magnitude
            for idx in nz:
                old = poly[idx]
                for nv in [-64, -16, -8, -4, -3, -2, -1, 1, 2, 3, 4, 8, 16, 64]:
                    if nv == old or nv == 0:
                        continue
                    poly[idx] = nv
                    s = _score(poly)
                    if s < cur_score:
                        cur_score = s
                        improved = True
                    else:
                        poly[idx] = old
            
            if not improved:
                break
        return poly, cur_score
    
    n_iter = 0
    while time.time() < deadline:
        n_iter += 1
        poly = make_random_poly()
        if not _is_valid(poly):
            continue
        s = _score(poly)
        if s < best_score:
            best_score = s
            best_poly = poly[:]
        
        if s <= best_score + 3:
            improved_poly, imp_score = improve(poly)
            if _is_valid(improved_poly) and imp_score < best_score:
                best_score = imp_score
                best_poly = improved_poly[:]
        
        if time.time() > deadline:
            break
    
    if best_poly is not None:
        return best_poly
    
    # Fallback
    fb = [0] * 41
    for i in range(12):
        fb[i] = 1
    return fb
