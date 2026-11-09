"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START
import numpy as np

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
    """Check if polynomial satisfies all constraints."""
    if len(poly) > 41:
        return False
    if len(poly) == 0 or poly[-1] == 0:
        return False
    if sum(1 for x in poly if x != 0) != 12:
        return False
    if any(abs(x) > 64 for x in poly):
        return False
    return True

def _local_search(poly, max_steps=500, time_limit=5.0):
    """Hill-climb: try flipping signs or changing individual coefficients."""
    start = time.time()
    current = poly[:]
    current_sq_nnz = _nnz(_square_coeffs(current))
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        improved = False
        # Try changing one coefficient at a time
        for i in range(len(current)):
            old_val = current[i]
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                if new_val == 0 or abs(new_val) > 64:
                    continue
                candidate = current[:]
                candidate[i] = new_val
                if not _is_valid(candidate):
                    continue
                sq = _square_coeffs(candidate)
                nnz_sq = _nnz(sq)
                if nnz_sq < current_sq_nnz:
                    current = candidate
                    current_sq_nnz = nnz_sq
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try sign flips
            improved = False
            for i in range(len(current)):
                if current[i] != 0:
                    candidate = current[:]
                    candidate[i] = -candidate[i]
                    if not _is_valid(candidate):
                        continue
                    sq = _square_coeffs(candidate)
                    nnz_sq = _nnz(sq)
                    if nnz_sq < current_sq_nnz:
                        current = candidate
                        current_sq_nnz = nnz_sq
                        improved = True
                        break
            if not improved:
                break
    
    return current, current_sq_nnz

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s
    
    best_sq_nnz = float('inf')
    best_poly = [1]*12
    random.seed(42)
    
    # Phase 1: Evaluate structured candidates
    candidates = []
    
    # All ones
    candidates.append([1]*12)
    
    # Alternating signs palindromic
    candidates.append([1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1])
    
    # Various palindromic sign patterns
    for signs in [(1,1,1,1,1,1), (1,1,1,-1,-1,-1), (1,-1,1,-1,1,-1), 
                  (1,1,-1,-1,1,1), (1,-1,-1,-1,-1,1), (-1,1,1,1,1,-1)]:
        pal = list(signs) + [signs[4], signs[3], signs[2], signs[1], signs[0]]
        if all(x != 0 for x in pal) and len(pal) == 12:
            candidates.append(pal)
    
    # Try polynomials that are products of sparse factors
    # e.g., (x^a + x^b + x^c) * (x^d + x^e + x^f + x^g + x^h + x^i)
    # This gives up to 18 terms, but we need exactly 12
    
    # Try binomial-like structures
    # P(x) = x^k * (1 + x)^m has m+1 terms. For 12 terms, m=11.
    # (1+x)^11 = sum C(11,i) x^i, all coeffs nonzero, 12 terms.
    # Square: (1+x)^22, 23 terms. Not great.
    
    # Better: try P(x) = (x^a - x^b + x^c) * (something with 4 terms)
    # or use the fact that (x^a + x^b)^2 = x^{2a} + 2x^{a+b} + x^{2b} (3 terms)
    
    # Key: if P = A + B where A*B has few terms, P^2 = A^2 + 2AB + B^2
    # If A^2, AB, B^2 all have few terms with cancellation, P^2 is sparse.
    
    # Try P(x) = (x^6 + x^3 + 1) * (x^5 + x^4 + x + 1)
    # First factor: 3 terms, second: 4 terms, product: up to 12 terms
    # Let's compute this
    f1 = [0]*7
    f1[0] = 1; f1[3] = 1; f1[6] = 1
    f2 = [0]*6
    f2[0] = 1; f2[1] = 1; f2[3] = 1; f2[5] = 1
    prod = np.convolve(f1, f2)
    prod_list = prod.tolist()
    while len(prod_list) > 1 and prod_list[-1] == 0:
        prod_list.pop()
    if _is_valid(prod_list):
        candidates.append(prod_list)
    
    # More product candidates
    for a in range(2, 8):
        for b in range(0, a):
            f1 = [0]*(a+1)
            f1[0] = 1; f1[a] = 1
            if b > 0:
                f1[b] = 1
            # f1 has 2 or 3 terms
            n_terms_f1 = sum(1 for x in f1 if x != 0)
            if n_terms_f1 < 2:
                continue
            # f2 should have 12//n_terms_f1 terms approximately
            target_f2_terms = 12 // n_terms_f1
            if target_f2_terms < 2:
                continue
            # Generate random f2 with target_f2_terms nonzero terms
            for _ in range(10):
                f2_len = random.randint(target_f2_terms, 10)
                f2 = [0]*f2_len
                positions = random.sample(range(f2_len), min(target_f2_terms, f2_len))
                for p in positions:
                    f2[p] = random.choice([-1, 1])
                prod = np.convolve(f1, f2)
                prod_list = prod.tolist()
                while len(prod_list) > 1 and prod_list[-1] == 0:
                    prod_list.pop()
                if _is_valid(prod_list):
                    candidates.append(prod_list)
    
    # Evaluate all candidates
    for c in candidates:
        if time.time() > deadline:
            break
        if not _is_valid(c):
            continue
        sq = _square_coeffs(c)
        nnz_sq = _nnz(sq)
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = c[:]
    
    # Phase 2: Random search with local optimization
    # Try different structures: palindromic, anti-palindromic, random
    
    non_zero_vals = list(range(-8, 9))
    non_zero_vals.remove(0)
    
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if attempts % 5000 == 0 and time.time() > deadline:
            break
        
        # Choose structure randomly
        r = random.random()
        
        if r < 0.4:
            # Palindromic degree 11
            free = [random.choice(non_zero_vals) for _ in range(6)]
            poly = free + [free[5], free[4], free[3], free[2], free[1], free[0]]
        elif r < 0.6:
            # Anti-palindromic: c[i] = -c[11-i] for i<5, c[5] and c[6] free
            # This gives c[0..4] = -c[11..7], c[5], c[6] free
            # 6 free params: c[0..4] and c[5], c[6] determined
            # Actually: c[0],c[1],c[2],c[3],c[4] free, c[11]=-c[0],c[10]=-c[1],...,c[7]=-c[4]
            # c[5] and c[6] are the middle, free
            # Total: 7 free params, but we need exactly 12 nonzero
            free = [random.choice(non_zero_vals) for _ in range(7)]
            poly = free[:5] + [free[5], free[6]] + [-free[4], -free[3], -free[2], -free[1], -free[0]]
        elif r < 0.8:
            # Random 12-term polynomial of random degree
            degree = random.randint(11, 20)
            poly = [0]*(degree+1)
            positions = random.sample(range(degree+1), 12)
            for p in positions:
                poly[p] = random.choice(non_zero_vals)
            # Ensure last coeff nonzero
            if poly[-1] == 0:
                poly[-1] = random.choice(non_zero_vals)
        else:
            # Product of two sparse polynomials
            n1 = random.choice([2, 3, 4])
            n2 = 12 // n1
            if n2 < 2:
                n2 = 2
            f1_len = random.randint(n1, 8)
            f1 = [0]*f1_len
            pos1 = random.sample(range(f1_len), min(n1, f1_len))
            for p in pos1:
                f1[p] = random.choice([-1, 1])
            f2_len = random.randint(n2, 8)
            f2 = [0]*f2_len
            pos2 = random.sample(range(f2_len), min(n2, f2_len))
            for p in pos2:
                f2[p] = random.choice([-1, 1])
            prod = np.convolve(f1, f2)
            poly = prod.tolist()
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
        
        if not _is_valid(poly):
            continue
        
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
            
            # Do local search from this candidate
            if time.time() < deadline - 10:
                improved_poly, improved_nnz = _local_search(poly, max_steps=200, time_limit=3.0)
                if improved_nnz < best_sq_nnz:
                    best_sq_nnz = improved_nnz
                    best_poly = improved_poly[:]
    
    # Phase 3: More intensive local search on best found
    if time.time() < deadline - 5:
        improved_poly, improved_nnz = _local_search(best_poly, max_steps=1000, time_limit=15.0)
        if improved_nnz < best_sq_nnz:
            best_sq_nnz = improved_nnz
            best_poly = improved_poly[:]
    
    # Ensure output is valid
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    if len(best_poly) == 0 or best_poly[-1] == 0:
        best_poly = [1]*12
    
    if sum(1 for x in best_poly if x != 0) != 12:
        best_poly = [1]*12
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    return best_poly
# EVOLVE-BLOCK-END
