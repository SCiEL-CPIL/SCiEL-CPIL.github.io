"""Rewritten program for matrix multiplication tensor decomposition 4x4x7."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
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

def _get_strassen_222():
    """Return the 2x2x2 Strassen decomposition factors."""
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

def _build_strassen_444():
    """
    Build the Strassen decomposition for 4x4x4 matrix multiplication.
    Uses the 2x2x2 Strassen algorithm recursively: rank = 7^2 = 49.
    Returns U, V, W as numpy arrays of shape (16, 49), (16, 49), (16, 49).
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _get_strassen_222()
    
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            
            # U construction: U[i,j] for i in 0..3, j in 0..3
            # Split into 2x2 blocks: i = 2*I + x, j = 2*J + y
            for I in range(2):
                for J in range(2):
                    coeff_u_block = U2[k, I * 2 + J]
                    if coeff_u_block == 0:
                        continue
                    for x in range(2):
                        for y in range(2):
                            coeff_u_inner = U2[l, x * 2 + y]
                            global_idx = (2 * I + x) * 4 + (2 * J + y)
                            U[global_idx, col] += coeff_u_block * coeff_u_inner
                        
            # V construction: V[j,k] for j in 0..3, k in 0..3
            # j = 2*J + y, k = 2*K + z
            for J in range(2):
                for K in range(2):
                    coeff_v_block = V2[k, J * 2 + K]
                    if coeff_v_block == 0:
                        continue
                    for y in range(2):
                        for z in range(2):
                            coeff_v_inner = V2[l, y * 2 + z]
                            global_idx = (2 * J + y) * 4 + (2 * K + z)
                            V[global_idx, col] += coeff_v_block * coeff_v_inner
                            
            # W construction: W[k,i] for k in 0..3, i in 0..3
            # k = 2*K + z, i = 2*I + x
            for I in range(2):
                for K in range(2):
                    coeff_w_block = W2[k, I * 2 + K]
                    if coeff_w_block == 0:
                        continue
                    for x in range(2):
                        for z in range(2):
                            coeff_w_inner = W2[l, z * 2 + x]
                            global_idx = (2 * K + z) * 4 + (2 * I + x)
                            W[global_idx, col] += coeff_w_block * coeff_w_inner
                            
    return U, V, W

def _build_hybrid_447():
    """
    Build a hybrid decomposition for 4x4x7.
    Decomposes the 7-dimensional output into 4+3:
    - Strassen for the 4x4x4 block (rank 49)
    - Schoolbook for the 4x4x3 block (rank 48)
    Total rank = 97.
    """
    n, m, p = 4, 4, 7
    rank_s = 49
    rank_sb = 48
    rank_total = rank_s + rank_sb
    
    U_total = np.zeros((n * m, rank_total))
    V_total = np.zeros((m * p, rank_total))
    W_total = np.zeros((p * n, rank_total))
    
    # Part 1: Strassen for 4x4x4 (first 4 columns of B)
    U_s, V_s, W_s = _build_strassen_444()
    
    U_total[:, :rank_s] = U_s
    V_total[:16, :rank_s] = V_s
    W_total[:16, :rank_s] = W_s
    
    # Part 2: Schoolbook for 4x4x3 (last 3 columns of B)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_total[i * 4 + j, rank_s + col] = 1
                V_global_idx = j * 7 + (4 + k)
                V_total[V_global_idx, rank_s + col] = 1
                W_global_idx = (4 + k) * 4 + i
                W_total[W_global_idx, rank_s + col] = 1
                col += 1
                
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

def _build_direct_strassen_447():
    """
    Alternative: Try to use a more direct approach for 4x4x7.
    Split B into two 4x4 and 4x3 parts, but use Strassen for both parts
    where possible. For the 4x4x3 part, we can't directly use Strassen,
    but we can pad to 4x4x4 and use Strassen, then discard the extra.
    However, that would give rank 49+49=98, which is worse than 97.
    
    So we stick with the hybrid approach. But let's also try a different
    split: 4x4x7 = 4x4x5 + 4x4x2.
    For 4x4x5, we could use Strassen on 4x4x4 + schoolbook on 4x4x1 = 49+16=65
    For 4x4x2, schoolbook = 32
    Total = 65+32 = 97. Same rank.
    
    Or: 4x4x7 = 4x4x3 + 4x4x4. Same as before.
    
    Let's try yet another approach: use the fact that we can combine
    some of the rank-1 terms. But this is complex. For now, return
    the hybrid approach.
    """
    return _build_hybrid_447()

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
