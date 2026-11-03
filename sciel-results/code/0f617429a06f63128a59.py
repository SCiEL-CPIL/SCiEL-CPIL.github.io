"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    # Random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Compute C from factors
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    C_comp = np.zeros((n, p))
    
    # Efficient computation:
    # C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Let S_A[i, r] = sum_j U[i*m+j, r] * A[i,j]
    # Let S_B[k, r] = sum_j V[j*p+k, r] * B[j,k]
    
    # Reshape U to (n, m, R)
    U_reshaped = U.reshape(n, m, R)
    # S_A[i, r] = sum_j U_reshaped[i, j, r] * A[i, j]
    S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
    
    # Reshape V to (m, p, R) -> V[j*p+k, r] is V_flat[j, k, r] if we reshape correctly?
    # V index is j*p+k. So V is (m, p, R) if we reshape to (m, p, R)?
    # V_flat[j*p+k, r]. If we reshape V to (m, p, R), then V_reshaped[j, k, r] = V[j*p+k, r].
    V_reshaped = V.reshape(m, p, R)
    # S_B[k, r] = sum_j V_reshaped[j, k, r] * B[j, k]
    S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
    
    # Reshape W to (p, n, R)
    # W index is k*n+i. So W_flat[k*n+i, r].
    # If we reshape W to (p, n, R), then W_reshaped[k, i, r] = W[k*n+i, r].
    W_reshaped = W.reshape(p, n, R)
    
    # C_ik = sum_r W_reshaped[k, i, r] * S_A[i, r] * S_B[k, r]
    # C_ik = sum_r W_reshaped[k, i, r] * S_A[i, r] * S_B[k, r]
    C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
    
    return np.allclose(C_true, C_comp, atol=tol)

# EVOLVE-BLOCK-START
def _strassen_2x2_block():
    """
    Returns U, V, W factor matrices for a 2x2x2 Strassen block.
    U: shape (4, 7) - for A_{i,j} where i,j in {0,1}
    V: shape (4, 7) - for B_{j,k} where j,k in {0,1}  
    W: shape (4, 7) - for C_{k,i} where k,i in {0,1}
    
    Strassen's 7 multiplications for 2x2 matrix multiply:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U factors: each row corresponds to A[i*m+j] = A[i*2+j] for i,j in {0,1}
    # U has shape (4, 7): rows are (A00, A01, A10, A11), cols are M1..M7
    U_block = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, -1],   # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=float)
    
    # V factors: each row corresponds to B[j*p+k] = B[j*2+k] for j,k in {0,1}
    # V has shape (4, 7): rows are (B00, B01, B10, B11), cols are M1..M7
    V_block = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [1, 0, 0, -1, 0, 1, 0],   # B01  -- wait, let me re-check
        [0, 0, 1, 0, 1, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ], dtype=float)
    
    # Let me re-derive carefully:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U[i,j] for A[i,j]:
    # A00: M1(1), M3(1), M5(1), M6(-1) -> [1,0,1,0,1,-1,0]
    # A01: M5(1), M7(1) -> [0,0,0,0,1,0,1]
    # A10: M2(1), M6(1) -> [0,1,0,0,0,1,0]
    # A11: M1(1), M2(1), M4(1), M7(-1) -> [1,1,0,1,0,0,-1]
    
    U_block = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=float)
    
    # V[j,k] for B[j,k]:
    # B00: M1(1), M2(1), M4(-1), M6(1) -> [1,1,0,-1,0,1,0]
    # B01: M4(1), M6(1) -> [0,0,0,1,0,1,0]
    # B10: M3(1), M5(1), M7(1) -> [0,0,1,0,1,0,1]
    # B11: M1(1), M3(-1), M5(1), M7(1) -> [1,0,-1,0,1,0,1]
    
    V_block = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 0, 1, 0, 1, 0],    # B01
        [0, 0, 1, 0, 1, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ], dtype=float)
    
    # W[k,i] for C[k,i]:
    # C00 = M1 + M4 - M5 + M7 -> [1,0,0,1,-1,0,1]
    # C01 = M3 + M5 -> [0,0,1,0,1,0,0]
    # C10 = M2 + M4 -> [0,1,0,1,0,0,0]
    # C11 = M1 - M2 + M3 + M6 -> [1,-1,1,0,0,1,0]
    
    W_block = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ], dtype=float)
    
    return U_block, V_block, W_block


def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Strategy: Use block decomposition with Strassen for 2x2 blocks
    # For 3x4 * 4x7, we can decompose into blocks:
    # 
    # A is 3x4, B is 4x7
    # We can view A as having 3 rows and 4 columns
    # B as having 4 rows and 7 columns
    # 
    # One approach: split the middle dimension (4) into two groups of 2
    # Then each "block" is 3x2 * 2x7
    # 
    # Alternatively, we can use a more sophisticated block structure.
    # 
    # Let's try: split A into top 2 rows and bottom 1 row
    # Split B into left 7 columns
    # 
    # Actually, let me think about this differently.
    # 
    # The schoolbook rank is 3*4*7 = 84.
    # Using Strassen-like blocking can reduce this.
    # 
    # For a 2x2x2 block, Strassen reduces rank from 8 to 7.
    # 
    # Let's construct a block decomposition:
    # 
    # Partition A (3x4) into:
    #   A_top (2x4), A_bottom (1x4)
    # 
    # Partition B (4x7) into:
    #   B_left (4x3), B_mid (4x1), B_right (4x3) -- this doesn't work well
    # 
    # Better: partition the middle dimension j (0..3) into j0,j1 (0,1) and j2,j3 (2,3)
    # 
    # So we have two "layers":
    # Layer 0: A[:, 0:2] (3x2) * B[0:2, :] (2x7)
    # Layer 1: A[:, 2:4] (3x2) * B[2:4, :] (2x7)
    # 
    # Each layer is 3x2 * 2x7.
    # 
    # For 3x2 * 2x7, we can further decompose.
    # 
    # Let's try a different approach: use the known result that rank(3x4x7) <= 80
    # via block methods.
    # 
    # I'll construct a specific decomposition using Strassen blocks where possible.
    
    # Let's try to build a rank-80 decomposition using block structure.
    # 
    # Partition:
    # i: 0,1,2 (rows of A, cols of C)
    # j: 0,1,2,3 (cols of A, rows of B)
    # k: 0,1,2,3,4,5,6 (cols of B, rows of C)
    # 
    # Group j into {0,1} and {2,3}
    # Group k into {0,1,2,3} and {4,5,6}
    # 
    # This gives 2 x 2 = 4 blocks of type (3 x 2) * (2 x 4) and (3 x 2) * (2 x 3)
    # 
    # Actually, let me just try to find a good decomposition by solving
    # the tensor equations for rank 80, 79, 78, etc.
    
    best_result = None
    
    # Try ranks from 80 down to 75
    for R in range(80, 74, -1):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            # Initialize with a structured guess
            np.random.seed(42 + attempt)
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            def residuals(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R :].reshape(p*n, R)
                
                U_r = U_flat.reshape(n, m, R)
                V_r = V_flat.reshape(m, p, R)
                W_r = W_flat.reshape(p, n, R)
                
                T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                return (T_comp - 1.0).flatten()
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = least_squares(residuals, x0, method='lm', max_nfev=500)
                if res.success and res.cost < 1e-10:
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    if _verify(U_sol, V_sol, W_sol):
                        return {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
            except Exception:
                pass

    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
