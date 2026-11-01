import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    
    Uses Strassen's algorithm for each of the 7 independent 2x2 blocks.
    Total rank = 49.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen forms for 2x2 multiplication
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A * B
    #
    # M1 = A00 * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = (A00 + A01) * B11
    # M4 = A11 * (B10 - B00)
    # M5 = A00 * (B01 - B11)
    # M6 = (A10 + A11) * (B00 + B01)
    # M7 = (A00 + A01) * (B10 - B11)
    #
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M7
    
    # U forms (coefficients for A)
    # Indices: 0:A00, 1:A01, 2:A10, 3:A11
    U_forms = [
        [1, 0, 0, 0],   # M1: A00
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 1, 0, 0],   # M3: A00 + A01
        [0, 0, 0, 1],   # M4: A11
        [1, 0, 0, 0],   # M5: A00
        [0, 0, 1, 1],   # M6: A10 + A11
        [1, 1, 0, 0]    # M7: A00 + A01
    ]
    
    # V forms (coefficients for B block)
    # Block B_k = [[B00, B01], [B10, B11]]
    # Indices in V for this block: 0:B00, 1:B01, 2:B10, 3:B11
    # Global indices: base + offset
    V_forms = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 0, 0, 1],   # M3: B11
        [-1, 0, 1, 0],  # M4: B10 - B00
        [0, 1, 0, -1],  # M5: B01 - B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, -1]   # M7: B10 - B11
    ]
    
    # W forms (coefficients for C block)
    # C = [[C00, C01], [C10, C11]]
    # Indices in W for this block: 0:C00, 1:C01, 2:C10, 3:C11
    W_forms = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, 1],   # M2: C10 + C11
        [0, 1, 0, -1],  # M3: C01 - C11
        [-1, 0, 1, 0],  # M4: C10 - C00
        [1, 1, 0, 0],   # M5: C00 + C01
        [1, 0, 0, 0],   # M6: C00
        [0, 1, 0, 1]    # M7: C01 + C11
    ]
    
    # Let's re-verify the W forms against the standard Strassen reconstruction.
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M7
    
    # W column for M1: C00 gets 1, C11 gets 1. -> [1, 0, 0, 1]
    # W column for M2: C10 gets 1, C11 gets 1. -> [0, 0, 1, 1]
    # W column for M3: C01 gets 1, C11 gets -1. -> [0, 1, 0, -1]
    # W column for M4: C00 gets -1, C10 gets 1. -> [-1, 0, 1, 0]
    # W column for M5: C00 gets 1, C01 gets 1. -> [1, 1, 0, 0]
    # W column for M6: C00 gets 1. -> [1, 0, 0, 0]
    # W column for M7: C01 gets 1, C11 gets 1. -> [0, 1, 0, 1]
    
    # The W_forms above match this.
    
    for r in range(num_blocks):
        if time.time() > deadline:
            break
            
        # V indices for block r
        # B is 2x14. Flattened row-major?
        # Task says: "output factor follows the official trace convention W[k*n+i,r]"
        # Usually inputs are flattened similarly.
        # A is 2x2. U is (n*m, rank) = (4, rank).
        # B is 2x14. V is (m*p, rank) = (28, rank).
        # C is 2x14. W is (p*n, rank) = (28, rank).
        
        # V indices:
        # Row 0 of B: B[0, 0..13] -> V[0..13]
        # Row 1 of B: B[1, 0..13] -> V[14..27]
        
        # For block r (cols 2r, 2r+1):
        # B[0, 2r] is V[2r]
        # B[0, 2r+1] is V[2r+1]
        # B[1, 2r] is V[14 + 2r]
        # B[1, 2r+1] is V[14 + 2r + 1]
        
        v_idx_B00 = 2 * r
        v_idx_B01 = 2 * r + 1
        v_idx_B10 = 14 + 2 * r
        v_idx_B11 = 14 + 2 * r + 1
        
        # W indices:
        # C is 2x14. Flattened row-major?
        # W[k*n+i, r] convention mentioned in prompt for W.
        # "W[k*n+i,r]" where k is column index? i is row index?
        # If C is p x n (14x2)? No, C is 2x14.
        # Standard trace convention for tensor decomposition of matrix mult:
        # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
        # Here i is row of C, j is col of C?
        # Let's check shapes.
        # U: (n*m, rank) -> i,j indices of A.
        # V: (m*p, rank) -> j,k indices of B? No, B is m x p.
        # If B is m x p, indices are j (row) and k (col).
        # V shape (m*p, rank).
        # W: (p*n, rank) -> k,i indices of C?
        # C is n x p. Indices i (row) and k (col).
        # So W index is k*n + i.
        
        # For block r (cols 2r, 2r+1 of C):
        # C[0, 2r] -> k=2r, i=0 -> W[2r*2 + 0] = W[4r]
        # C[0, 2r+1] -> k=2r+1, i=0 -> W[(2r+1)*2 + 0] = W[4r+2]
        # C[1, 2r] -> k=2r, i=1 -> W[2r*2 + 1] = W[4r+1]
        # C[1, 2r+1] -> k=2r+1, i=1 -> W[(2r+1)*2 + 1] = W[4r+3]
        
        w_idx_C00 = 4 * r
        w_idx_C01 = 4 * r + 2
        w_idx_C10 = 4 * r + 1
        w_idx_C11 = 4 * r + 3
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            U[0, c] = U_forms[s][0]
            U[1, c] = U_forms[s][1]
            U[2, c] = U_forms[s][2]
            U[3, c] = U_forms[s][3]
            
            # Fill V
            V[v_idx_B00, c] = V_forms[s][0]
            V[v_idx_B01, c] = V_forms[s][1]
            V[v_idx_B10, c] = V_forms[s][2]
            V[v_idx_B11, c] = V_forms[s][3]
            
            # Fill W
            W[w_idx_C00, c] = W_forms[s][0]
            W[w_idx_C01, c] = W_forms[s][1]
            W[w_idx_C10, c] = W_forms[s][2]
            W[w_idx_C11, c] = W_forms[s][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
