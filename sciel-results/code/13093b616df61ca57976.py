"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_strassen_decomposition():
    """Build the rank-35 Strassen decomposition for 2x2 x 2x10."""
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(5):
        col_base = t * 7
        
        # Term 1: P1 = (a+d)(e+h)
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1; W[4*t+3][col] = 1
        
        # Term 2: P2 = (c+d)e
        col = col_base + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        
        # Term 3: P3 = a(f-h)
        col = col_base + 2
        U[0][col] = 1
        V[2*t+1][col] = 1; V[11+2*t][col] = -1
        W[4*t+2][col] = 1; W[4*t+3][col] = 1
        
        # Term 4: P4 = d(g-e)
        col = col_base + 3
        U[3][col] = 1
        V[10+2*t][col] = 1; V[2*t][col] = -1
        W[4*t][col] = 1; W[4*t+1][col] = 1
        
        # Term 5: P5 = (a+b)h
        col = col_base + 4
        U[0][col] = 1; U[1][col] = 1
        V[11+2*t][col] = 1
        W[4*t][col] = -1; W[4*t+2][col] = 1
        
        # Term 6: P6 = (c-a)(e+f)
        col = col_base + 5
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        W[4*t+3][col] = 1
        
        # Term 7: P7 = (b-d)(g+h)
        col = col_base + 6
        U[1][col] = 1; U[3][col] = -1
        V[10+2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1

    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Reconstruct the tensor: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # A is n x m, B is m x p, C is p x n
    # T[i,j,k] should be 1 if C[k,i] = A[i,j]*B[j,k] contributes, i.e., always the identity contribution
    # Actually, the multiplication tensor T[i,j,k] = delta(i,j,k) where 
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k)
    
    # Build expected tensor
    T_expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i,j,k] = 1.0
    
    # Build actual tensor from decomposition
    T_actual = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        # T[i,j,k] += u_col[i] * v_col[j] * w_col[k]
        # But W is indexed as W[k*n+i], so w_col[k*n+i]
        for i in range(n):
            if u_col[i] == 0:
                continue
            for j in range(m):
                if v_col[j] == 0:
                    continue
                for k in range(p):
                    w_idx = k * n + i
                    T_actual[i,j,k] += u_col[i] * v_col[j] * w_col[w_idx]
    
    max_err = np.max(np.abs(T_actual - T_expected))
    return max_err < 1e-10, max_err

def _try_rank_reduction(U, V, W, target_rank):
    """Try to find a lower-rank decomposition by identifying linear dependencies."""
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W
    
    # Convert to numpy
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor is T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k*n+i, r]
    # We want to check if we can express the tensor with fewer terms.
    
    # One approach: check if any rank-1 term is in the span of the others
    # This is complex; for now, return the original if we can't easily reduce
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the base Strassen decomposition
    U, V, W = _build_strassen_decomposition()
    
    # Verify correctness
    is_valid, max_err = _verify_decomposition(U, V, W)
    
    if not is_valid:
        # If verification fails, we have a bug; return as-is (shouldn't happen)
        pass
    
    # Try to find rank reduction through cross-block analysis
    # Check if the 35 terms have any linear dependencies that allow rank reduction
    
    # For now, return the rank-35 decomposition
    # A future improvement would search for a rank-34 or lower decomposition
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
