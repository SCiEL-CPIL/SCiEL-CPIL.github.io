"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """Build the rank-28 Strassen decomposition deterministically.
    
    A (2x4) = [A1 | A2] where A1, A2 are 2x2
    B (4x4) = [[B11, B12], [B21, B22]] where each Bi is 2x2
    C (2x4) = [C1 | C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    
    Each 2x2*2x2 product uses Strassen (rank 7), so total rank = 4*7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 8 x 28

    # Strassen 2x2x2 decomposition:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    col = 0
    for b_col_blk in range(2):  # B column block: 0 -> cols 0,1; 1 -> cols 2,3
        for a_blk in range(2):  # A column block / B row block: 0 -> cols/rows 0,1; 1 -> cols/rows 2,3
            for s in range(7):
                c = col
                j0 = 2 * a_blk
                j1 = 2 * a_blk + 1
                k0 = 2 * b_col_blk
                k1 = 2 * b_col_blk + 1
                
                # U contributions: A[i, j] where i in {0,1}, j in {j0, j1}
                # U index: i * 4 + j
                U_contrib = {
                    0: [(0, j0, 1), (1, j1, 1)],       # a11+a22
                    1: [(1, j0, 1), (1, j1, 1)],       # a21+a22
                    2: [(0, j0, 1)],                    # a11
                    3: [(1, j1, 1)],                    # a22
                    4: [(0, j0, 1), (0, j1, 1)],       # a11+a12
                    5: [(1, j0, 1), (0, j0, -1)],      # a21-a11
                    6: [(0, j1, 1), (1, j1, -1)],      # a12-a22
                }
                for (i, j, coeff) in U_contrib[s]:
                    U[i * 4 + j, c] += coeff

                # V contributions: B[j, k] where j in {j0, j1}, k in {k0, k1}
                # V index: j * 4 + k
                V_contrib = {
                    0: [(j0, k0, 1), (j1, k1, 1)],     # b11+b22
                    1: [(j0, k0, 1)],                   # b11
                    2: [(j0, k1, 1), (j1, k1, -1)],    # b12-b22
                    3: [(j1, k0, 1), (j0, k0, -1)],    # b21-b11
                    4: [(j1, k1, 1)],                   # b22
                    5: [(j0, k0, 1), (j0, k1, 1)],     # b11+b12
                    6: [(j1, k0, 1), (j1, k1, 1)],     # b21+b22
                }
                for (j, k, coeff) in V_contrib[s]:
                    V[j * 4 + k, c] += coeff

                # W contributions: C[i, k] where i in {0,1}, k in {k0, k1}
                # W index: k * 2 + i (trace convention W[k*n+i, r])
                W_contrib = {
                    0: {(0, k0): 1, (1, k1): 1},       # c11 gets m1, c22 gets m1
                    1: {(1, k0): 1, (1, k1): -1},      # c21 gets m2, c22 gets -m2
                    2: {(0, k1): 1, (1, k1): 1},       # c12 gets m3, c22 gets m3
                    3: {(0, k0): 1, (1, k0): 1},       # c11 gets m4, c21 gets m4
                    4: {(0, k0): -1, (0, k1): 1},      # c11 gets -m5, c12 gets m5
                    5: {(1, k1): 1},                   # c22 gets m6
                    6: {(0, k0): 1},                   # c11 gets m7
                }
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    W[k_loc * 2 + i_loc, c] += coeff

                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W correctly decompose the matmul tensor.
    
    T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1 for all (i,k,j).
    """
    rank = U.shape[1]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _attempt_rank_reduction(U, V, W, n, m, p):
    """
    Attempt to reduce the rank of the decomposition by checking for linear dependencies
    among the rank-1 tensor components.
    
    Returns (U_reduced, V_reduced, W_reduced) if rank is reduced, else original.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W

    size = n * m * p
    tensors = np.zeros((rank, size))
    
    for r in range(rank):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = i * (m * p) + k * p + j
                    tensors[r, idx] = U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]

    try:
        U_svd, s, Vt = np.linalg.svd(tensors, full_matrices=False)
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            null_space = Vt[num_rank:]
            
            best_col_to_remove = -1
            best_coeff = 0.0
            best_null_vec = None
            
            for vec in null_space:
                max_idx = np.argmax(np.abs(vec))
                max_val = np.abs(vec[max_idx])
                if max_val > best_coeff:
                    best_coeff = max_val
                    best_col_to_remove = max_idx
                    best_null_vec = vec
            
            if best_col_to_remove != -1:
                new_U = np.delete(U, best_col_to_remove, axis=1)
                new_V = np.delete(V, best_col_to_remove, axis=1)
                new_W = np.delete(W, best_col_to_remove, axis=1)
                
                for r_new in range(new_U.shape[1]):
                    r_old = r_new if r_new < best_col_to_remove else r_new + 1
                    factor = 1.0 - best_null_vec[r_old] / best_null_vec[best_col_to_remove]
                    new_U[:, r_new] *= factor
                    new_V[:, r_new] *= factor
                    new_W[:, r_new] *= factor
                
                return new_U, new_V, new_W
                
    except np.linalg.LinAlgError:
        pass
        
    return U, V, W


def _groebner_basis_check(U, V, W, n, m, p, max_time=10):
    """
    Use a symbolic approach to check for algebraic dependencies in the rank-28 decomposition.
    This is a heuristic: we check if the tensor terms are linearly independent over Q.
    If they are dependent, we can potentially reduce the rank.
    """
    start = time.time()
    rank = U.shape[1]
    size = n * m * p
    
    # Build the tensor matrix exactly using fractions for precision
    from fractions import Fraction
    
    tensor_matrix = np.zeros((rank, size), dtype=object)
    for r in range(rank):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = i * (m * p) + k * p + j
                    val = U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    tensor_matrix[r, idx] = Fraction(val).limit_denominator(1000000)
    
    # Check linear independence using exact Gaussian elimination
    # Convert to a list of lists for easier manipulation
    mat = [[tensor_matrix[r, c] for c in range(size)] for r in range(rank)]
    
    # Perform Gaussian elimination over Q
    row = 0
    for col in range(size):
        if time.time() - start > max_time:
            return False, "timeout"
        
        # Find pivot
        pivot = -1
        for r in range(row, rank):
            if mat[r][col] != 0:
                pivot = r
                break
        
        if pivot == -1:
            continue
        
        # Swap rows
        mat[row], mat[pivot] = mat[pivot], mat[row]
        
        # Normalize
        pivot_val = mat[row][col]
        for c in range(col, size):
            mat[row][c] = mat[row][c] / pivot_val
        
        # Eliminate below
        for r in range(rank):
            if r == row:
                continue
            factor = mat[r][col]
            if factor != 0:
                for c in range(col, size):
                    mat[r][c] = mat[r][c] - factor * mat[row][c]
        
        row += 1
        if row == rank:
            break
    
    # If we have zero rows, there are dependencies
    num_zero_rows = rank - row
    if num_zero_rows > 0:
        return True, num_zero_rows
    else:
        return False, 0


def _polish_decomposition(U, V, W, n, m, p, max_iterations=100, tol=1e-12):
    """
    Apply a final polish pass to ensure exact integer/half-integer entries.
    This helps with numerical stability and ensures the decomposition is exact.
    """
    rank = U.shape[1]
    
    # Try to snap values to nearest 0.5
    def snap_to_half(x):
        return round(x * 2) / 2
    
    U_polished = np.array([[snap_to_half(U[i, r]) for r in range(rank)] for i in range(U.shape[0])], dtype=float)
    V_polished = np.array([[snap_to_half(V[i, r]) for r in range(rank)] for i in range(V.shape[0])], dtype=float)
    W_polished = np.array([[snap_to_half(W[i, r]) for r in range(rank)] for i in range(W.shape[0])], dtype=float)
    
    # Verify the polished version
    if _verify_decomposition(U_polished, V_polished, W_polished, n, m, p, tol):
        return U_polished, V_polished, W_polished
    
    # If polishing broke it, return original
    return U, V, W


def construct():
    """
    Return a rank-minimal exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    Uses Strassen's algorithm on 2x2 blocks (rank 28) and attempts to reduce rank.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, keep margin before 120s
    
    # Build the base rank-28 Strassen decomposition
    U, V, W = _build_strassen_rank28()
    
    # Verify it's correct
    assert _verify_decomposition(U, V, W, n, m, p), "Base rank-28 decomposition failed verification"
    
    # Attempt to reduce rank via linear dependency check
    U, V, W = _attempt_rank_reduction(U, V, W, n, m, p)
    
    # If rank was reduced, verify
    original_U, original_V, original_W = _build_strassen_rank28()
    if U.shape[1] < original_U.shape[1]:
        if not _verify_decomposition(U, V, W, n, m, p):
            U, V, W = original_U, original_V, original_W
    
    # Apply polish pass to ensure exact entries
    if time.time() < deadline:
        U, V, W = _polish_decomposition(U, V, W, n, m, p)
        # Verify after polishing
        if not _verify_decomposition(U, V, W, n, m, p):
            U, V, W = original_U, original_V, original_W
    
    # Return the result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
