import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = 130
    
    start_time = time.time()
    
    # Build the true multiplication tensor
    # T[a_idx, b_idx, c_idx] = 1 if a_idx=(i,j), b_idx=(j,k), c_idx=(i,k)
    dim_a = n * m  # 8
    dim_b = m * p  # 16
    dim_c = p * n  # 8
    
    T_true = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * p + k
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    # First, get the rank-28 Strassen decomposition
    result_28 = _strassen_28()
    
    if time.time() > deadline:
        return result_28
    
    # Try to find a rank-27 decomposition
    # Strategy: Use the flattened tensor approach
    # Flatten T into a matrix of shape (dim_a * dim_b) x dim_c
    # Then look for sparse non-negative decompositions
    
    T_flat = T_true.reshape(dim_a * dim_b, dim_c)
    
    # Try rank 27
    for trial in range(20):
        if time.time() > deadline:
            break
        
        result_27 = _try_rank_27_nnls(T_flat, T_true, dim_a, dim_b, dim_c, trial)
        if result_27 is not None:
            return result_27
    
    # Try rank 26
    for trial in range(10):
        if time.time() > deadline:
            break
        
        result_26 = _try_rank_26_nnls(T_flat, T_true, dim_a, dim_b, dim_c, trial)
        if result_26 is not None:
            return result_26
    
    # Return the best we found (rank 28)
    return result_28

def _strassen_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Standard 2x2 Strassen blocks
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # M1 = (A00+A11)(B00+B11)
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # M2 = (A10+A11)B00
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # M3 = A00(B10-B11)
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # M4 = A11(B01-B10)
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # M5 = (A00+A01)B11
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # M6 = (A10-A00)(B00+B01)
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # M7 = (A01-A11)(B10+B11)
    ]
    
    def emit_block(a_cs, b_rs, b_cs, c_cs, off):
        for s in range(7):
            col = off + s
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c
    
    # Four 2x2 blocks
    emit_block(0, 0, 0, 0, 0)    # Top-left
    emit_block(0, 0, 2, 2, 7)    # Top-right
    emit_block(2, 2, 0, 0, 14)   # Bottom-left
    emit_block(2, 2, 2, 2, 21)   # Bottom-right
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    
    # Build the tensor from the decomposition
    T_decomp = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    
    for r in range(rank):
        u_col = np.array([U[i][r] for i in range(dim_a)], dtype=np.float64)
        v_col = np.array([V[j][r] for j in range(dim_b)], dtype=np.float64)
        w_col = np.array([W[k][r] for k in range(dim_c)], dtype=np.float64)
        
        T_decomp += np.einsum('a,b,c->abc', u_col, v_col, w_col)
    
    # Check against the true tensor
    T_true = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * p + k
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    if not np.allclose(T_decomp, T_true, atol=1e-8):
        return False
    
    # Check that entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-6:
                    return False
    
    return True

def _try_rank_27_nnls(T_flat, T_true, dim_a, dim_b, dim_c, trial):
    """Try to find a rank-27 decomposition using NNLS approach."""
    n, m, p = DIMENSIONS
    np.random.seed(42 + trial)
    
    # Strategy: Start from rank-28 and try to merge two terms
    # or use NNLS to find sparse decomposition
    
    # First approach: Try to find a rank-27 by solving a sparse problem
    # Flatten the tensor differently: for each c, we have a matrix M_c of shape (dim_a * dim_b)
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For fixed c: T[:, :, c] = sum_r W[c,r] * (U[:,r] * V[:,r])^T
    
    # Try a randomized approach: start from rank-28, try to eliminate one term
    U28 = np.array(_strassen_28()["U"], dtype=np.float64)
    V28 = np.array(_strassen_28()["V"], dtype=np.float64)
    W28 = np.array(_strassen_28()["W"], dtype=np.float64)
    
    # Try each column as a candidate to eliminate
    for col_to_elim in range(28):
        if time.time() > 130:
            return None
        
        # Create rank-27 by removing this column
        mask = [i for i in range(28) if i != col_to_elim]
        U27 = U28[:, mask].copy()
        V27 = V28[:, mask].copy()
        W27 = W28[:, mask].copy()
        
        # Add small random perturbations and try to fix
        for _ in range(50):
            if time.time() > 130:
                return None
            
            # Random perturbation
            U27 += np.random.normal(0, 0.001, U27.shape)
            V27 += np.random.normal(0, 0.001, V27.shape)
            W27 += np.random.normal(0, 0.001, W27.shape)
            
            # Check if valid
            if _verify_decomposition(U27.tolist(), V27.tolist(), W27.tolist()):
                return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Second approach: Use NNLS on the flattened tensor
    # For each c, try to decompose T[:, :, c] as sum_r W[c,r] * (U[:,r] * V[:,r])^T
    # This is equivalent to finding low-rank approximations
    
    # Try a different strategy: use the fact that the tensor has structure
    # and try to find a rank-27 decomposition by solving a system of equations
    
    # For now, return None if we couldn't find rank 27
    return None

def _try_rank_26_nnls(T_flat, T_true, dim_a, dim_b, dim_c, trial):
    """Try to find a rank-26 decomposition."""
    # More aggressive search for rank 26
    return None

# EVOLVE-BLOCK-END
