"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check tensor identity: sum_r U_r[i*m+j] V_r[j*p+k] W_r[k*n+i] == 1
    # We can do this by contracting the tensors.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] == 1 for all i,j,k.
    
    # Efficient check:
    # Reshape U to (n, m, rank)
    # Reshape V to (m, p, rank)
    # Reshape W to (p, n, rank)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    # Note: W index is k*n+i. In W_reshaped, index is [k, i, r].
    
    # We can compute this using einsum or loops.
    # For small dimensions, loops are fine.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _is_nice_half_integer(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    return abs(doubled - rounded) < tol

def _build_strassen_444():
    """
    Build the Strassen decomposition for 4x4x4 matrix multiplication.
    Returns U, V, W as numpy arrays of shape (16, 49), (16, 49), (16, 49).
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Get 2x2x2 Strassen factors
    U2, V2, W2 = _get_strassen_222()
    
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            # U construction
            for I in range(2):
                for J in range(2):
                    coeff_u_block = U2[k, I * 2 + J]
                    if coeff_u_block == 0: continue
                    for x in range(2):
                        for y in range(2):
                            coeff_u_inner = U2[l, x * 2 + y]
                            global_idx = (2 * I + x) * 4 + (2 * J + y)
                            U[global_idx, col] += coeff_u_block * coeff_u_inner
                            
            # V construction
            for J in range(2):
                for K in range(2):
                    coeff_v_block = V2[k, J * 2 + K]
                    if coeff_v_block == 0: continue
                    for y in range(2):
                        for z in range(2):
                            coeff_v_inner = V2[l, y * 2 + z]
                            global_idx = (2 * J + y) * 4 + (2 * K + z)
                            V[global_idx, col] += coeff_v_block * coeff_v_inner
                            
            # W construction
            for I in range(2):
                for K in range(2):
                    coeff_w_block = W2[k, I * 2 + K]
                    if coeff_w_block == 0: continue
                    for x in range(2):
                        for z in range(2):
                            coeff_w_inner = W2[l, z * 2 + x]
                            global_idx = (2 * K + z) * 4 + (2 * I + x)
                            W[global_idx, col] += coeff_w_block * coeff_w_inner
                            
    return U, V, W

def _get_strassen_222():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, 0]
    
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, 0]
    
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 0, 1, -1]
    W[:, 3] = [0, 0, 1, 0]
    
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [0, 1, 0, 0]
    
    U[:, 5] = [0, 0, 1, -1]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _build_hybrid_447():
    """
    Build a hybrid decomposition for 4x4x7.
    Uses Strassen for the first 4 columns of B, and Schoolbook for the last 3.
    Rank = 49 + 48 = 97.
    """
    n, m, p = 4, 4, 7
    rank_s = 49
    rank_sb = 48
    rank_total = rank_s + rank_sb
    
    U_total = np.zeros((n * m, rank_total))
    V_total = np.zeros((m * p, rank_total))
    W_total = np.zeros((p * n, rank_total))
    
    # Part 1: Strassen for 4x4x4
    U_s, V_s, W_s = _build_strassen_444()
    
    # U_s is 16x49. U_total first 49 cols are U_s.
    U_total[:, :rank_s] = U_s
    
    # V_s is 16x49. V_total first 49 cols are V_s padded with 12 zeros.
    V_total[:16, :rank_s] = V_s
    
    # W_s is 16x49. W_total first 49 cols are W_s padded with 12 zeros.
    W_total[:16, :rank_s] = W_s
    
    # Part 2: Schoolbook for 4x4x3
    # This computes C[:, 4:7] = A @ B[:, 4:7]
    # B_sub is 4x3.
    # U_sb is 16x48.
    # V_sb is 12x48.
    # W_sb is 12x48.
    
    U_sb = np.zeros((16, 48))
    V_sb = np.zeros((12, 48))
    W_sb = np.zeros((12, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # U index: i*4+j
                U_sb[i * 4 + j, col] = 1
                # V index: j*3+k (within the 4x3 submatrix)
                # Global V index: j*7 + (4+k)
                V_global_idx = j * 7 + (4 + k)
                V_total[V_global_idx, rank_s + col] = 1
                # W index: k*4+i (within the 4x3 submatrix)
                # Global W index: (4+k)*4 + i
                W_global_idx = (4 + k) * 4 + i
                W_total[W_global_idx, rank_s + col] = 1
                col += 1
                
    # U_sb is standard schoolbook U for 4x4 input.
    # We can just fill U_total[:, rank_s:] with the same pattern as U_sb.
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_total[i * 4 + j, rank_s + col] = 1
                col += 1
                
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    # Try hybrid decomposition
    try:
        result = _build_hybrid_447()
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
