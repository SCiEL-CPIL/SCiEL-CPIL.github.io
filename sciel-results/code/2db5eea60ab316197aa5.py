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
    
    def is_half_integer(matrix, tol=1e-9):
        """Check if all elements are close to k/2. Return rounded list-of-lists or None."""
        rounded = np.round(matrix * 2) / 2
        if not np.all(np.abs(matrix - rounded) < tol):
            return None
        res = []
        for row in rounded:
            res_row = []
            for val in row:
                if val == int(val):
                    res_row.append(int(val))
                else:
                    res_row.append(float(val))
            res.append(res_row)
        return res

    def verify_decomposition_fast(U_np, V_np, W_np):
        """Vectorized verification: check sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k."""
        # U_np: (nm, R), V_np: (mp, R), W_np: (pn, R)
        # For each (i,j,k): dot(U[i*m+j], V[j*p+k] * W[k*n+i])
        # Build all products efficiently
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U3 = U_np.reshape(n, m, -1)
        V3 = V_np.reshape(m, p, -1)
        W3 = W_np.reshape(p, n, -1)
        
        # For each (i,j,k): sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # Use einsum: result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # W3[k,i,c] needs to be indexed as [k, i, c]
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(np.abs(result - 1) < 1e-9)

    start_time = time.time()
    time_limit = 140  # 10s safety margin from 150s kill
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try ranks in a strategic order: start with mid-range (easier to find), then go lower
    # Known lower bounds suggest rank >= ~30 for 3x4x5
    # Try: 45, 40, 35, 50, 42, 38, 33, 48, 44, 39, 36, 32, 55, 52, 47, 43, 41, 37, 34, 31
    ranks_to_try = [45, 40, 35, 50, 42, 38, 33, 48, 44, 39, 36, 32, 55, 52, 47, 43, 41, 37, 34, 31]
    
    # Different sampling strategies for diversity
    def sample_UV(R, strategy_idx):
        """Sample U and V matrices using different strategies."""
        if strategy_idx % 5 == 0:
            # Sparse: 60% zeros, 20% each ±1
            U_rand = np.random.choice([-1, 0, 1], size=(nm, R), p=[0.2, 0.6, 0.2])
            V_rand = np.random.choice([-1, 0, 1], size=(mp, R), p=[0.2, 0.6, 0.2])
        elif strategy_idx % 5 == 1:
            # Less sparse: 40% zeros
            U_rand = np.random.choice([-1, 0, 1], size=(nm, R), p=[0.3, 0.4, 0.3])
            V_rand = np.random.choice([-1, 0, 1], size=(mp, R), p=[0.3, 0.4, 0.3])
        elif strategy_idx % 5 == 2:
            # Dense with some structure: rows have limited non-zeros
            U_rand = np.zeros((nm, R))
            V_rand = np.zeros((mp, R))
            for i in range(nm):
                nz = np.random.randint(2, max(3, R // 2) + 1)
                idx = np.random.choice(R, nz, replace=False)
                U_rand[i, idx] = np.random.choice([-1, 1], nz)
            for i in range(mp):
                nz = np.random.randint(2, max(3, R // 2) + 1)
                idx = np.random.choice(R, nz, replace=False)
                V_rand[i, idx] = np.random.choice([-1, 1], nz)
        elif strategy_idx % 5 == 3:
            # Binary: 0 or 1 only
            U_rand = np.random.choice([0, 1], size=(nm, R), p=[0.5, 0.5])
            V_rand = np.random.choice([0, 1], size=(mp, R), p=[0.5, 0.5])
        else:
            # Mixed: some rows sparse, some dense
            U_rand = np.zeros((nm, R))
            V_rand = np.zeros((mp, R))
            for i in range(nm):
                if random.random() < 0.5:
                    nz = np.random.randint(1, 3)
                    idx = np.random.choice(R, nz, replace=False)
                    U_rand[i, idx] = np.random.choice([-1, 1], nz)
                else:
                    nz = np.random.randint(R // 2, R)
                    idx = np.random.choice(R, nz, replace=False)
                    U_rand[i, idx] = np.random.choice([-1, 0, 1], nz, p=[0.4, 0.2, 0.4])
            for i in range(mp):
                if random.random() < 0.5:
                    nz = np.random.randint(1, 3)
                    idx = np.random.choice(R, nz, replace=False)
                    V_rand[i, idx] = np.random.choice([-1, 1], nz)
                else:
                    nz = np.random.randint(R // 2, R)
                    idx = np.random.choice(R, nz, replace=False)
                    V_rand[i, idx] = np.random.choice([-1, 0, 1], nz, p=[0.4, 0.2, 0.4])
        return U_rand, V_rand

    for R in ranks_to_try:
        if R >= best_rank:
            continue
        
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive restarts: more for higher ranks (easier), fewer for lower
        base_restarts = 300
        # Higher ranks get more restarts since they're more likely to succeed
        restarts = int(base_restarts * (1.0 + (60 - R) / 30.0))
        restarts = max(50, min(restarts, 800))
        
        found = False
        for r_idx in range(restarts):
            if time.time() - start_time > time_limit:
                break
            
            U_rand, V_rand = sample_UV(R, r_idx)
            
            # Build the linear system for W
            # A is (n*m*p, R*n*p), b is ones
            # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        coeffs = U_rand[i * m + j] * V_rand[j * p + k]
                        start_col = (k * n + i) * R
                        end_col = start_col + R
                        A[eq_idx, start_col:end_col] = coeffs
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            W_sol = W_flat.reshape((p * n, R))
            
            # Check for half-integers
            W_nice = is_half_integer(W_sol)
            if W_nice is not None:
                # Verify with vectorized check
                U_np = np.array(U_rand)
                V_np = np.array(V_rand)
                W_np = np.array(W_nice)
                if verify_decomposition_fast(U_np, V_np, W_np):
                    best_rank = R
                    best_U = U_rand.tolist()
                    best_V = V_rand.tolist()
                    best_W = W_nice
                    found = True
                    break
        
        if found:
            # Found a valid decomposition at rank R; continue to try lower ranks
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
