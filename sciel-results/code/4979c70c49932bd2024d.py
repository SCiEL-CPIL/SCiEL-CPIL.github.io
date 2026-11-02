import numpy as np
import random
import time
from itertools import product

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
    time_limit = 115  # seconds
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def is_half_integer(matrix, tol=1e-9):
        for row in matrix:
            for val in row:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > tol:
                    return False, None
        result = []
        for row in matrix:
            res_row = []
            for val in row:
                nearest = round(val * 2) / 2
                if nearest == int(nearest):
                    res_row.append(int(nearest))
                else:
                    res_row.append(nearest)
            result.append(res_row)
        return True, result

    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # ILP-inspired approach:
    # We search over sparse U and V with entries in {-1, 0, 1}
    # For each candidate (U, V), we solve the linear system for W
    # and check if W has half-integer entries.
    #
    # Key insight: the tensor T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # We fix U and V, then W is determined by:
    #   For each (i,k): sum_j sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 for all j? 
    #   Actually: T_{i,j,k} = sum_c U_{i,m+j,c} * V_{j,p+k,c} * W_{k,n+i,c}
    #   So for fixed (i,k), we have m equations (one per j) in R*1 unknowns? No.
    #   W has shape (p*n, R). For fixed (i,k), the relevant W entries are W[k*n+i, :] of length R.
    #   The equations are: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #   This is m equations in R unknowns (the R values of W[k*n+i, :]).
    #   If m <= R, we can solve (or least-squares).
    #
    # Better: treat all W entries as unknowns. Total unknowns: p*n*R.
    # Total equations: n*m*p = 60.
    # Build A and b, solve via lstsq, check if solution is half-integer.

    def try_candidate(U, V, R):
        """Given U (nm x R) and V (mp x R), solve for W and check validity."""
        # Build linear system: A * W_flat = b
        # W_flat is flattened W of shape (p*n, R) in row-major: index = (k*n+i)*R + c
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
            return None
        
        W_sol = W_flat.reshape((p * n, R))
        
        # Check if W_sol is close to half-integers
        ok, W_nice = is_half_integer(W_sol)
        if not ok:
            return None
        
        # Verify with rounded values
        if verify_decomposition(U, V, W_nice, R):
            return W_nice
        return None

    # Search strategy:
    # Try ranks from a lower bound upward.
    # For each rank, generate random sparse U and V and test.
    
    # Lower bound estimate: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But actually the tensor rank is at least max(n*m, m*p, p*n) / something...
    # For 3x4x5, known results suggest rank around 35-45.
    # Let's try from 25 to 55.
    
    ranks_to_try = list(range(25, 56))
    
    # Use a smarter randomization: bias towards sparse matrices
    def random_sparse_matrix(rows, cols, density=0.3, seed=None):
        """Generate a random sparse matrix with entries in {-1, 0, 1}."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        mat = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if rng.random() < density:
                    mat[r][c] = rng.choice([-1, 1])
        return mat

    # Also try structured patterns: some rows/columns with specific patterns
    def generate_structured_U(R, variant):
        """Generate structured U matrices."""
        U = [[0] * R for _ in range(nm)]
        if variant == 0:
            # Random sparse
            return random_sparse_matrix(nm, R, density=0.4)
        elif variant == 1:
            # Each row has exactly 2 non-zeros
            for i in range(nm):
                cols = random.sample(range(R), min(2, R))
                for c in cols:
                    U[i][c] = random.choice([-1, 1])
            return U
        elif variant == 2:
            # Block structure
            block_size = max(1, R // (nm // 2 + 1))
            for i in range(nm):
                start = (i * block_size) % R
                for offset in range(min(block_size, R - start)):
                    U[i][start + offset] = random.choice([-1, 1])
            return U
        else:
            return random_sparse_matrix(nm, R, density=0.35)

    def generate_structured_V(R, variant):
        """Generate structured V matrices."""
        V = [[0] * R for _ in range(mp)]
        if variant == 0:
            return random_sparse_matrix(mp, R, density=0.4)
        elif variant == 1:
            for j in range(mp):
                cols = random.sample(range(R), min(2, R))
                for c in cols:
                    V[j][c] = random.choice([-1, 1])
            return V
        elif variant == 2:
            block_size = max(1, R // (mp // 2 + 1))
            for j in range(mp):
                start = (j * block_size) % R
                for offset in range(min(block_size, R - start)):
                    V[j][start + offset] = random.choice([-1, 1])
            return V
        else:
            return random_sparse_matrix(mp, R, density=0.35)

    # Main search loop
    found = False
    for R in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
        
        if R >= best_rank:
            continue
        
        # Number of attempts per rank, adjusted by time remaining
        time_remaining = time_limit - (time.time() - start_time)
        attempts = min(500, max(50, int(time_remaining * 10)))
        
        for attempt in range(attempts):
            if time.time() - start_time > time_limit:
                break
            
            variant = attempt % 4
            
            U_cand = generate_structured_U(R, variant)
            V_cand = generate_structured_V(R, variant)
            
            W_cand = try_candidate(U_cand, V_cand, R)
            if W_cand is not None:
                best_rank = R
                best_U = U_cand
                best_V = V_cand
                best_W = W_cand
                found = True
                break  # Found valid decomposition at this rank, try lower rank
        
        if found:
            # We found a valid decomposition at rank R.
            # Since we're iterating from low to high, this is the minimal found so far.
            # But we should continue to try lower ranks? No, we're going from low to high.
            # Actually, we want the MINIMAL rank, so we should try from low to high
            # and return the first valid one.
            return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
