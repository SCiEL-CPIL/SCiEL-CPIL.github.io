import time
import itertools
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = 2, 3, 6
    
    # Trivial rank 36 decomposition
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    # Try to find rank-32 decomposition using rank-16 for 2x3x3
    import numpy as np
    
    found_233 = False
    U233 = None
    V233 = None
    W233 = None
    
    # Strategy: try various random U, V with entries in {-1, 0, 1}
    # Then solve linear system for W and check if solution is in half-integers
    random.seed(42)
    np.random.seed(42)
    
    max_attempts = 500
    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break
        
        # Generate random U and V
        U_try = np.array([[random.choice([-1, 0, 1]) for _ in range(16)] for _ in range(6)])
        V_try = np.array([[random.choice([-1, 0, 1]) for _ in range(16)] for _ in range(9)])
        
        # Build linear system: A_mat @ W_flat = b_vec
        num_vars = 6 * 16  # W has 6 rows, 16 cols
        num_eqs = 18
        
        A_mat = np.zeros((num_eqs, num_vars))
        b_vec = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    for l in range(16):
                        var_idx = (k * 2 + i) * 16 + l
                        A_mat[eq_idx, var_idx] = U_try[i*3+j, l] * V_try[j*3+k, l]
                    eq_idx += 1
        
        # Solve using least squares
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        # Check exactness
        W_check = A_mat @ W_flat
        if not np.allclose(W_check, b_vec, atol=1e-10):
            continue
        
        # Try rounding to half-integers
        W_rounded = np.round(W_flat * 2) / 2
        W_check2 = A_mat @ W_rounded
        if np.allclose(W_check2, b_vec, atol=1e-10):
            found_233 = True
            U233 = U_try.tolist()
            V233 = V_try.tolist()
            W233 = W_rounded.tolist()
            break
        
        # Also try rounding to integers
        W_int = np.round(W_flat)
        W_check3 = A_mat @ W_int
        if np.allclose(W_check3, b_vec, atol=1e-10):
            found_233 = True
            U233 = U_try.tolist()
            V233 = V_try.tolist()
            W233 = W_int.tolist()
            break
    
    if found_233:
        # Construct rank-32 decomposition for 2x3x6
        U32 = [[0] * 32 for _ in range(6)]
        V32 = [[0] * 32 for _ in range(18)]
        W32 = [[0] * 32 for _ in range(12)]
        
        # Block 1: l in 0..15, corresponds to k in {0,1,2}
        for l in range(16):
            for i in range(2):
                for j in range(3):
                    U32[i*3+j][l] = U233[i*3+j][l]
                for k in range(3):
                    V32[j*3+k][l] = V233[j*3+k][l]
            for k in range(3):
                for i in range(2):
                    W32[k*2+i][l] = W233[k*2+i][l]
        
        # Block 2: l in 16..31, corresponds to k in {3,4,5}
        for l in range(16):
            l2 = l + 16
            for i in range(2):
                for j in range(3):
                    U32[i*3+j][l2] = U233[i*3+j][l]
                for k in range(3):
                    V32[j*3+(k+3)][l2] = V233[j*3+k][l]
            for k in range(3):
                for i in range(2):
                    W32[(k+3)*2+i][l2] = W233[k*2+i][l]
        
        return {"U": U32, "V": V32, "W": W32}
    else:
        return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
# EVOLVE-BLOCK-END
