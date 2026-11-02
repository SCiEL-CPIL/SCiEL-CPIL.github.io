import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-59 exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses Strassen's algorithm for the top-left 2x2 sub-block and schoolbook for the rest.
    """
    n, m, p = DIMENSIONS
    rank = 59
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx is (i, j) in A
        # v_idx is (j, k) in B
        # w_idx is (i, k) in C (stored as k*n + i)
        pass # Placeholder

    # We will build the list of terms
    terms = []
    
    # 1. Strassen terms for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen M1: (A00+A11)(B00+B11)
    # U: A00, A11. V: B00, B11. W: C00, C01, C10, C11?
    # Wait, the contribution of M1 to C is:
    # C00 += M1
    # C01 += M1
    # C10 += M1
    # C11 += M1
    # Is this correct?
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # C00 needs A00B00 + A11B11? No, C00 = A00B00 + A01B10 + A10B01 + A11B11.
    # Strassen formulas:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So W vectors for Strassen terms are not just 1s. They are coefficients.
    
    # Let's define the 7 Strassen terms (U, V, W)
    # U is in R^(2x2) flattened. V in R^(2x2). W in R^(2x2).
    # We map these to the global indices.
    
    # Term 1: M1
    # U1: A00=1, A11=1. Others 0.
    # V1: B00=1, B11=1. Others 0.
    # W1: C00=1, C11=1, C01=-1, C10=-1. (From C00=M1..., C11=M1..., C01=-M1? No, C01=M3+M5. C10=M2+M4. C00=M1-M4+M5-M7. C11=M1-M2+M3+M6.)
    # Wait, I need the exact W coefficients for each M_r in the sum for each C_ij.
    
    # Let's just use the known Strassen decomposition.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each M_r, we have a W vector.
    # W1 (for M1): C00=1, C11=1, C01=0, C10=0.
    # W2 (for M2): C10=1, C11=-1, C00=0, C01=0.
    # W3 (for M3): C01=1, C11=1, C00=0, C10=0.
    # W4 (for M4): C00=-1, C10=1, C01=0, C11=0.
    # W5 (for M5): C00=1, C01=1, C10=0, C11=0.
    # W6 (for M6): C11=1, C00=0, C01=0, C10=0.
    # W7 (for M7): C00=-1, C01=0, C10=0, C11=0.
    
    # Now define U and V for each M_r.
    # M1: U=(1,0,0,1), V=(1,0,0,1)
    # M2: U=(0,0,1,1), V=(1,0,0,0)
    # M3: U=(1,0,0,0), V=(0,1,-1,0)
    # M4: U=(0,0,0,1), V=(0,1,-1,0) -- Wait, B01-B00 is (0,1) - (1,0) in flattened?
    # Flattened B: B00, B01, B10, B11.
    # B01 is index 1. B00 is index 0.
    # So V4 = ( -1, 1, 0, 0 )? No, M4 = A11(B01-B00).
    # V4: B01=1, B00=-1. So ( -1, 1, 0, 0 ).
    # M5: U=(1,1,0,0), V=(0,0,0,1)
    # M6: U=( -1, 0, 1, 0 ), V=(1,1,0,0)
    # M7: U=(0,1,-1,0), V=(0,1,1,0) -- Wait, B10+B11. B10 is index 2, B11 is index 3.
    # V7: (0,0,1,1).
    
    # Let's add these 7 terms to the global arrays.
    # Map local (i,j) to global (i, j) for U.
    # Map local (j,k) to global (j, k) for V.
    # Map local (i,k) to global (k*n + i) for W.
    
    col = 0
    for r in range(7):
        # U_r
        if r == 0: u_vec = [1,0,0,1]
        elif r == 1: u_vec = [0,0,1,1]
        elif r == 2: u_vec = [1,0,0,0]
        elif r == 3: u_vec = [0,0,0,1]
        elif r == 4: u_vec = [1,1,0,0]
        elif r == 5: u_vec = [-1,0,1,0]
        else: u_vec = [0,1,-1,0]
        
        for li in range(2):
            for lj in range(2):
                if u_vec[li*2+lj] != 0:
                    U[li*m + lj, col] = u_vec[li*2+lj]
                    
        # V_r
        if r == 0: v_vec = [1,0,0,1]
        elif r == 1: v_vec = [1,0,0,0]
        elif r == 2: v_vec = [0,1,-1,0]
        elif r == 3: v_vec = [-1,1,0,0]
        elif r == 4: v_vec = [0,0,0,1]
        elif r == 5: v_vec = [1,1,0,0]
        else: v_vec = [0,0,1,1]
        
        for lj in range(2):
            for lk in range(2):
                if v_vec[lj*2+lk] != 0:
                    V[lj*p + lk, col] = v_vec[lj*2+lk]
                    
        # W_r
        if r == 0: w_vec = [1,0,0,1] # C00, C01, C10, C11 -> C00=1, C11=1
        elif r == 1: w_vec = [0,0,1,-1] # C10=1, C11=-1
        elif r == 2: w_vec = [0,1,0,1] # C01=1, C11=1
        elif r == 3: w_vec = [-1,0,1,0] # C00=-1, C10=1
        elif r == 4: w_vec = [1,1,0,0] # C00=1, C01=1
        elif r == 5: w_vec = [0,0,0,1] # C11=1
        else: w_vec = [-1,0,0,0] # C00=-1
        
        for li in range(2):
            for lk in range(2):
                if w_vec[li*2+lk] != 0:
                    W[lk*n + li, col] = w_vec[li*2+lk]
                    
        col += 1
        
    # 2. Schoolbook terms for the rest
    # Skip the 8 terms for i in 0,1 and j in 0,1 and k in 0,1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
