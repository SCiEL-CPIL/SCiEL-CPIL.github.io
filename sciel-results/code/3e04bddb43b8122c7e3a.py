import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def is_valid_coefficient(x):
        """Check if x is an integer or half-integer"""
        if x is None:
            return False
        val = 2 * x
        nearest = round(val)
        return abs(val - nearest) < 1e-9
    
    def to_half_integer(x):
        """Convert to integer or half-integer representation"""
        val = 2 * x
        nearest = round(val)
        if abs(val - nearest) < 1e-9:
            if nearest % 2 == 0:
                return nearest // 2
            else:
                return nearest / 2.0
        return x
    
    def try_find_decomposition(R, max_iterations=3000):
        """Try to find a decomposition with exactly R terms"""
        if time.time() - start_time > time_limit:
            return None
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                return None
            
            # Random initialization with sparsity
            density = 0.25
            U = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(R)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(R)] for _ in range(mp)]
            
            # Build linear system for W
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(R):
                            coeff = U[i * m + j][c] * V[j * p + k][c]
                            w_idx = (k * n + i) * R + c
                            A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            W_sol = W_flat.reshape((p * n, R))
            
            # Check if W_sol entries are close to half-integers
            all_valid = True
            W_clean = [[0] * R for _ in range(p * n)]
            for row_idx in range(p * n):
                for col_idx in range(R):
                    val = W_sol[row_idx, col_idx]
                    if not is_valid_coefficient(val):
                        all_valid = False
                        break
                    W_clean[row_idx][col_idx] = to_half_integer(val)
                if not all_valid:
                    break
            
            if not all_valid:
                continue
            
            # Verify the decomposition
            if verify_decomposition(U, V, W_clean, R):
                return {"U": U, "V": V, "W": W_clean}
        
        return None
    
    # Search ranks from low to high
    # Known lower bound for 3x4x5 is around 25
    ranks_to_try = list(range(25, 35)) + list(range(35, 45, 2)) + list(range(45, 61, 3))
    ranks_to_try = sorted(set(ranks_to_try))
    
    for R in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
        
        result = try_find_decomposition(R)
        
        if result is not None:
            return result
    
    # Return schoolbook if no better found
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
