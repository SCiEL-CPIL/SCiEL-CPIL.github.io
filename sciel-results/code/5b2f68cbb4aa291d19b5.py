import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 24
    # 3 blocks, each 2x3x2 multiplication
    # Each 2x3x2 is decomposed into 2x2x2 (rank 7) + 2x1x2 (rank 1) = rank 8
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 Strassen terms for 2x2x2
    # u, v, w vectors for the 7 terms
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global mapping for block l:
    # i -> i
    # j -> j
    # k -> 2l + k
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u is (i, j), v is (j, k), w is (i, k)
        # We store as dict or list of (idx, val)
        
        # M1
        ([(0,0, 1), (1,1, 1)], [(0,0, 1), (1,1, 1)], [(0,0, 1), (0,1, 0), (1,0, 0), (1,1, 0)]), # Placeholder, will expand
    ]
    
    # It's easier to just fill the matrices directly for each block
    
    for l in range(3):
        base_r = l * 8
        
        # --- Strassen Part (j=0,1) ---
        # r = base_r + 0 to 6
        
        # M1: (A00+A11)(B00+B11)
        r = base_r + 0
        U[0*m + 0][r] += 1
        U[1*m + 1][r] += 1
        V[0*p + (2*l + 0)][r] += 1
        V[1*p + (2*l + 1)][r] += 1
        # C contributions:
        # C00 += 1, C11 += 1
        W[(2*l + 0)*n + 0][r] += 1
        W[(2*l + 1)*n + 1][r] += 1
        
        # M2: (A10+A11)B00
        r = base_r + 1
        U[1*m + 0][r] += 1
        U[1*m + 1][r] += 1
        V[0*p + (2*l + 0)][r] += 1
        # C10 += 1
        W[(2*l + 0)*n + 1][r] += 1
        
        # M3: A00(B01-B11)
        r = base_r + 2
        U[0*m + 0][r] += 1
        V[0*p + (2*l + 1)][r] += 1
        V[1*p + (2*l + 1)][r] += -1
        # C01 += 1
        W[(2*l + 1)*n + 0][r] += 1
        
        # M4: A11(B10-B00)
        r = base_r + 3
        U[1*m + 1][r] += 1
        V[1*p + (2*l + 0)][r] += 1
        V[0*p + (2*l + 0)][r] += -1
        # C00 += 1, C10 += 1
        W[(2*l + 0)*n + 0][r] += 1
        W[(2*l + 0)*n + 1][r] += 1
        
        # M5: (A00+A01)B11
        r = base_r + 4
        U[0*m + 0][r] += 1
        U[0*m + 1][r] += 1
        V[1*p + (2*l + 1)][r] += 1
        # C00 -= 1, C01 += 1
        W[(2*l + 0)*n + 0][r] += -1
        W[(2*l + 1)*n + 0][r] += 1
        
        # M6: (A10-A00)(B00+B01)
        r = base_r + 5
        U[1*m + 0][r] += 1
        U[0*m + 0][r] += -1
        V[0*p + (2*l + 0)][r] += 1
        V[0*p + (2*l + 1)][r] += 1
        # C11 += 1
        W[(2*l + 1)*n + 1][r] += 1
        
        # M7: (A01-A11)(B10+B11)
        r = base_r + 6
        U[0*m + 1][r] += 1
        U[1*m + 1][r] += -1
        V[1*p + (2*l + 0)][r] += 1
        V[1*p + (2*l + 1)][r] += 1
        # C00 += 1, C11 += 1
        W[(2*l + 0)*n + 0][r] += 1
        W[(2*l + 1)*n + 1][r] += 1
        
        # --- Extra Part (j=2) ---
        # r = base_r + 7
        # C_ik = A_i2 B_2k
        r = base_r + 7
        # A_i2
        U[0*m + 2][r] += 1
        U[1*m + 2][r] += 1
        # B_2k
        V[2*p + (2*l + 0)][r] += 1
        V[2*p + (2*l + 1)][r] += 1
        # C_ik
        W[(2*l + 0)*n + 0][r] += 1
        W[(2*l + 1)*n + 0][r] += 1
        W[(2*l + 0)*n + 1][r] += 1
        W[(2*l + 1)*n + 1][r] += 1
        
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W, num_tests=10):
    for _ in range(num_tests):
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_direct = A @ B
        
        U_arr = np.array(U).reshape(n, m, -1)
        V_arr = np.array(V).reshape(m, p, -1)
        W_arr = np.array(W).reshape(p, n, -1)
        
        LA = np.einsum('ij,ijr->ir', A, U_arr)
        LB = np.einsum('jk,jkr->kr', B, V_arr)
        
        C_decomp = np.einsum('kir,ir,kr->ik', W_arr, LA, LB)
        
        if not np.allclose(C_direct, C_decomp, atol=1e-9):
            return False
    return True
# EVOLVE-BLOCK-END
