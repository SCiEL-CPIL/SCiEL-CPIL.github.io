import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

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

def _local_search(poly, max_steps=2000, time_limit=10.0):
    """Hill-climb with single and pairwise moves."""
    start = time.time()
    current = poly[:]
    current_sq_nnz = _nnz(_square_coeffs(current))
    n = len(current)
    
    # Precompute indices for speed
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        improved = False
        
        # 1. Try changing one coefficient
        for i in range(n):
            old_val = current[i]
            # Try small deltas
            for delta in [-1, 1, -2, 2, -3, 3]:
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
            
            # Sign flip
            if old_val != 0:
                candidate = current[:]
                candidate[i] = -old_val
                if _is_valid(candidate):
                    sq = _square_coeffs(candidate)
                    nnz_sq = _nnz(sq)
                    if nnz_sq < current_sq_nnz:
                        current = candidate
                        current_sq_nnz = nnz_sq
                        improved = True
                        break

        if not improved:
            # 2. Try changing two coefficients (pairwise)
            # Sample a few pairs to keep it fast
            if random.random() < 0.5:
                i, j = random.sample(range(n), 2)
                old_i, old_j = current[i], current[j]
                for delta_i in [-1, 1, -2, 2]:
                    for delta_j in [-1, 1, -2, 2]:
                        new_i = old_i + delta_i
                        new_j = old_j + delta_j
                        if new_i == 0 or new_j == 0 or abs(new_i) > 64 or abs(new_j) > 64:
                            continue
                        candidate = current[:]
                        candidate[i] = new_i
                        candidate[j] = new_j
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
             # Stagnation: break
             break
    
    return current, current_sq_nnz

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_sq_nnz = float('inf')
    best_poly = [1]*12
    random.seed(42)
    
    # Phase 1: Exhaustive search on structured candidates (Palindromic/Anti-palindromic)
    # Degree 11 Palindromic: 6 free coeffs
    # Degree 12 Anti-palindromic: 6 free coeffs (middle is 0)
    
    search_values = [-3, -2, -1, 1, 2, 3]
    
    # 1. Palindromic degree 11
    # c = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
    for coeffs in itertools.product(search_values, repeat=6):
        if time.time() > deadline - 10: break
        a0, a1, a2, a3, a4, a5 = coeffs
        poly = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
        if not _is_valid(poly):
            continue
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
    
    # 2. Anti-palindromic degree 12 (13 coeffs, middle 0)
    # c = [a0, a1, a2, a3, a4, a5, 0, -a5, -a4, -a3, -a2, -a1, -a0]
    # This has 12 nonzero terms if all ai != 0.
    for coeffs in itertools.product(search_values, repeat=6):
        if time.time() > deadline - 10: break
        a0, a1, a2, a3, a4, a5 = coeffs
        poly = [a0, a1, a2, a3, a4, a5, 0, -a5, -a4, -a3, -a2, -a1, -a0]
        if not _is_valid(poly):
            continue
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
            
    # 3. Slightly larger coefficients for Palindromic degree 11 (if time permits)
    # Only if best is still > 16
    if best_sq_nnz > 16 and time.time() < deadline - 20:
        search_values_large = [-4, -3, -2, -1, 1, 2, 3, 4]
        # Subsample to avoid explosion
        for coeffs in itertools.product(search_values_large, repeat=6):
            if time.time() > deadline - 20: break
            a0, a1, a2, a3, a4, a5 = coeffs
            poly = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
            if not _is_valid(poly):
                continue
            sq = _square_coeffs(poly)
            nnz_sq = _nnz(sq)
            if nnz_sq < best_sq_nnz:
                best_sq_nnz = nnz_sq
                best_poly = poly[:]

    # Phase 2: Random search with local optimization
    non_zero_vals = list(range(-10, 11))
    non_zero_vals.remove(0)
    
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        if attempts % 1000 == 0 and time.time() > deadline - 5:
            break
        
        r = random.random()
        
        if r < 0.4:
            # Palindromic degree 11
            free = [random.choice(non_zero_vals) for _ in range(6)]
            poly = free + [free[5], free[4], free[3], free[2], free[1], free[0]]
        elif r < 0.6:
            # Anti-palindromic degree 12
            free = [random.choice(non_zero_vals) for _ in range(6)]
            poly = free + [0] + [-free[5], -free[4], -free[3], -free[2], -free[1], -free[0]]
        elif r < 0.8:
            # Random 12-term polynomial
            degree = random.randint(11, 25)
            poly = [0]*(degree+1)
            positions = random.sample(range(degree+1), 12)
            for p in positions:
                poly[p] = random.choice(non_zero_vals)
            if poly[-1] == 0:
                poly[-1] = random.choice(non_zero_vals)
        else:
            # Product of sparse factors
            n1 = random.choice([2, 3, 4])
            n2 = 12 // n1
            if n2 < 2: n2 = 2
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
            
            # Local search
            if time.time() < deadline - 10:
                improved_poly, improved_nnz = _local_search(poly, max_steps=500, time_limit=5.0)
                if improved_nnz < best_sq_nnz:
                    best_sq_nnz = improved_nnz
                    best_poly = improved_poly[:]
    
    # Phase 3: Final polish on best found
    if time.time() < deadline - 2:
        improved_poly, improved_nnz = _local_search(best_poly, max_steps=1000, time_limit=5.0)
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
