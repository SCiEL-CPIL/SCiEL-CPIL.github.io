"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)

    # Try the structured 4-block rank-11 scheme (total rank 44)
    best = None
    best_rank = n * m * p

    if time.time() < deadline - 30:
        try:
            U1, V1, W1, r1 = _four_block_decomposition(n, m, p)
            if _verify(n, m, p, U1, V1, W1):
                best = (U1, V1, W1)
                best_rank = r1
        except Exception:
            pass

    # Try further refinement on the 44-rank solution (attempt rank 43, 42, ...)
    if best is not None and time.time() < deadline - 5:
        try:
            U2, V2, W2, r2 = _refine_decomposition(n, m, p, best[0], best[1], best[2], best_rank)
            if r2 < best_rank and _verify(n, m, p, U2, V2, W2):
                best = (U2, V2, W2)
                best_rank = r2
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _four_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition.
    Total rank: 44.
    
    The 2x3 x 3x2 tensor T has T[i,j,k] = delta_{j,k} for i in {0,1}, j in {0,1,2}, k in {0,1}.
    We decompose as:
    - Strassen-like 7 terms handling the 2x2 top-left block of A times top-right 2x2 of B
    - 4 terms for the bottom row of A (j=2) times B
    """
    n, m, p = 2, 3, 8
    block_size = p // 4  # 2
    
    rank_per_block = 11
    total_rank = rank_per_block * 4  # 44
    
    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44
    
    for b in range(4):
        k_offset = b * block_size  # 0, 2, 4, 6
        col_offset = b * rank_per_block
        
        # Strassen 7 terms for 2x2 * 2x2 part
        strassen_U = [
            {0: 1, 4: 1},      # M1
            {3: 1, 4: 1},      # M2
            {0: 1},            # M3
            {4: 1},            # M4
            {0: 1, 1: 1},      # M5
            {3: 1, 0: -1},     # M6
            {1: 1, 4: -1},     # M7
        ]
        
        strassen_V = [
            {0 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},  # M1
            {0 * p + k_offset + 0: 1},                           # M2
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: -1}, # M3
            {0 * p + k_offset + 1: 1, 0 * p + k_offset + 0: -1}, # M4
            {1 * p + k_offset + 1: 1},                           # M5
            {0 * p + k_offset + 0: 1, 0 * p + k_offset + 1: 1},  # M6
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},  # M7
        ]
        
        strassen_W = [
            {(k_offset + 0) * n + 0: 1, (k_offset + 1) * n + 1: 1},  # M1
            {(k_offset + 0) * n + 1: 1},                             # M2
            {(k_offset + 1) * n + 0: 1, (k_offset + 1) * n + 1: 1},  # M3
            {(k_offset + 0) * n + 0: 1, (k_offset + 0) * n + 1: 1},  # M4
            {(k_offset + 0) * n + 0: -1, (k_offset + 1) * n + 0: 1}, # M5
            {(k_offset + 1) * n + 1: 1},                             # M6
            {(k_offset + 0) * n + 0: 1},                             # M7
        ]
        
        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val
        
        # Outer product part: A_right(2x1) * B_bottom(1x2) - rank 4
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk  # 7,8,9,10
                U[ii * m + 2, col_offset + c] = 1
                V[2 * p + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * n + ii, col_offset + c] = 1
    
    return U, V, W, total_rank


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using the bilinear form:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k])
    Uses vectorized numpy for speed.
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    rank = U.shape[1]
    
    # Precompute U @ A for all i, c: shape (n, rank)
    # U is (n*m, rank), A is (n, m)
    # For each i: U[i*m:(i+1)*m, :] @ A[i, :]
    U_A = np.zeros((n, rank))
    for i in range(n):
        U_A[i] = U[i * m:(i + 1) * m, :] @ A[i, :]
    
    # Precompute V @ B for all k, c: shape (p, rank)
    # V is (m*p, rank), B is (m, p)
    # For each k: V[:, k] transposed... actually V[l*p+k, c]
    # V_T is (rank, m*p), reshape to (rank, m, p)
    V_T = V.T  # (rank, m*p)
    V_T = V_T.reshape(rank, m, p)  # (rank, m, p)
    # V_B[k, c] = sum_l V[l*p+k, c] * B[l, k]
    # = (V_T[:, :, k].T @ B[:, k]) for each k
    V_B = np.zeros((p, rank))
    for k in range(p):
        V_B[k] = V_T[:, :, k].T @ B[:, k]
    
    # C_comp[i, k] = sum_c W[k*n+i, c] * U_A[i, c] * V_B[k, c]
    # W is (p*n, rank)
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            C_comp[i, k] = np.dot(W[k * n + i, :], U_A[i, :] * V_B[k, :])
    
    if not np.allclose(C_comp, C_true, atol=1e-9):
        return False
    return True


def _refine_decomposition(n, m, p, U, V, W, current_rank):
    """
    Try to reduce the rank by finding linear dependencies among columns.
    Greedy removal: try removing each column and verify.
    Also try merging pairs of columns via linear combination.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, rank
    
    deadline = time.time() + 100  # safety margin
    
    # Phase 1: Greedy single-column removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for c_remove in range(rank - 1, -1, -1):
            if time.time() > deadline:
                break
            mask = [c for c in range(rank) if c != c_remove]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            if _verify(n, m, p, U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                changed = True
                break
        if rank <= 1:
            break
    
    # Phase 2: Try merging pairs - replace columns a,b with a single column
    # that is a linear combination. We search for alpha such that
    # col_a + alpha * col_b can be absorbed into remaining columns.
    # This is expensive, so limit attempts.
    if time.time() < deadline - 30:
        for c_remove in range(rank - 1, -1, -1):
            if time.time() > deadline:
                break
            for c_other in range(c_remove):
                if time.time() > deadline:
                    break
                # Try: can we remove c_remove and adjust c_other?
                # Simple check: just try removing c_remove
                mask = [c for c in range(rank) if c != c_remove]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if _verify(n, m, p, U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    rank -= 1
                    break
    
    return U, V, W, rank


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            # Round to nearest half-integer
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
