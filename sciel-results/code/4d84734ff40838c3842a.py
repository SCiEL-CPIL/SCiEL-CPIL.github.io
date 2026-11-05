"""Matrix multiplication 3x3 by 3x7 - rank minimization via structural factorization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0])
    rng = np.random.default_rng(42)
    for trial in range(8):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C_comp = np.zeros((n, p))
        for c in range(rank):
            a_form = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_form[i] += U[i * m + j][c] * A[i, j]
            b_form = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    b_form[k] += V[j * p + k][c] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i][c] * a_form[i] * b_form[k]
        if not np.allclose(C_true, C_comp, atol=tol, rtol=tol):
            return False
    return True

def _schoolbook(n, m, p):
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

def _check_tensor_equation(U, V, W, n, m, p, tol=1e-8):
    """
    Check if the tensor equation holds:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c] = delta(j,l)
    for all i, j, k, l.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    val = 0.0
                    for c in range(rank):
                        val += W_arr[k * n + i, c] * U_arr[i * m + j, c] * V_arr[l * p + k, c]
                    expected = 1.0 if j == l else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def _strassen_3x3(n, m):
    """
    Return Strassen decomposition for 3x3 * 3x3 multiplication.
    U: shape (n*m, 7), V: shape (m*3, 7), W: shape (3*n, 7)
    This computes C = A @ B where A is 3x3, B is 3x3.
    """
    # Strassen's 7 multiplications for 2x2, extended to 3x3 via block decomposition
    # Actually, let's use a known rank-7 decomposition for 3x3 matmul if it exists.
    # The minimal rank for 3x3 is not known to be less than 19, but Strassen gives 19 for 3x3.
    # Wait, Strassen is for 2x2. For 3x3, the schoolbook is 9.
    # Let's just use schoolbook for the 3x3 block to be safe, or find a better decomposition.
    # Actually, the task is 3x3 * 3x7. We can split B into B1 (3x3) and B2 (3x4).
    # C = A @ B = A @ B1 + A @ B2
    # Rank(A @ B1) <= rank(A) * rank(B1) is not the right way.
    # The rank of the tensor for A @ B1 is at most 9 (schoolbook).
    # The rank of the tensor for A @ B2 is at most 12 (schoolbook).
    # Total rank <= 9 + 12 = 21.
    # But we can potentially do better by sharing computations.
    # Let's try to construct a decomposition of rank <= 21.
    
    # For now, return the schoolbook decomposition for 3x3 * 3x3
    rank = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * 3)]
    W = [[0] * rank for _ in range(3 * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _schoolbook_block(n, m, p_block, offset_k):
    """
    Return schoolbook decomposition for 3x3 * 3xp_block multiplication.
    U: shape (n*m, p_block), V: shape (m*p_block, p_block), W: shape (p_block*n, p_block)
    offset_k is the starting column index in the full B matrix.
    """
    rank = p_block
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p_block)]
    W = [[0] * rank for _ in range(p_block * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p_block):
                U[i * m + j][col] = 1
                V[j * p_block + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 110.0  # Leave 10s safety margin
    n, m, p = DIMENSIONS
    
    # Schoolbook is the guaranteed valid solution with rank n*m*p = 63
    candidate = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        # Should not happen, but just in case
        return candidate
    
    # Try to find lower rank decompositions
    # Strategy: Split B into B1 (3x3) and B2 (3x4)
    # C = A @ B1 + A @ B2
    # Use schoolbook for both blocks: rank = 9 + 12 = 21
    # Then try to eliminate linearly dependent terms.
    
    # Construct the combined decomposition
    p1 = 3
    p2 = 4
    rank1 = n * m * p1  # 9
    rank2 = n * m * p2  # 12
    total_rank = rank1 + rank2  # 21
    
    U_combined = [[0] * total_rank for _ in range(n * m)]
    V_combined = [[0] * total_rank for _ in range(m * p)]
    W_combined = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in the first block (columns 0-2 of B)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                U_combined[i * m + j][col] = 1
                V_combined[j * p + k][col] = 1
                W_combined[k * n + i][col] = 1
                col += 1
    
    # Fill in the second block (columns 3-6 of B)
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                U_combined[i * m + j][col] = 1
                V_combined[j * p + (p1 + k)][col] = 1
                W_combined[(p1 + k) * n + i][col] = 1
                col += 1
    
    # Verify the combined decomposition
    if _verify(U_combined, V_combined, W_combined, n, m, p):
        best_rank = min(best_rank, total_rank)
        candidate = {"U": U_combined, "V": V_combined, "W": W_combined}
    
    # Now try to eliminate linearly dependent terms using Gaussian elimination
    # The tensor equation is: sum_c W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c] = delta(j,l)
    # We can view each column c as a term in the sum.
    # If two columns are linearly dependent in the sense that one is a scalar multiple of the other,
    # we can eliminate one.
    
    # Convert to numpy arrays for easier manipulation
    U_arr = np.array(U_combined, dtype=float)
    V_arr = np.array(V_combined, dtype=float)
    W_arr = np.array(W_combined, dtype=float)
    
    # For each (i, k), the vectors u_i = [U[i*m+j, c] for j in [0,m)] 
    # and v_k = [V[l*p+k, c] for l in [0,m)] and w_ik = W[k*n+i, c] must satisfy:
    # sum_c w_ik[c] * u_i[j,c] * v_k[l,c] = delta(j,l)
    
    # This is a system of equations. We can try to find a basis of the solution space.
    # But for now, let's just check if the rank can be reduced by looking at linear dependencies.
    
    # Build the coefficient matrix for the tensor equation
    # For each (i, j, k, l), we have:
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c] = delta(j,l)
    
    # This is a linear system in the variables x_c = W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c]
    # But x_c depends on (i, j, k, l), so it's not a simple linear system.
    
    # Instead, let's try a different approach: use random linear combinations of columns
    # to see if we can reduce the rank.
    
    # For now, return the best result found so far
    if time.time() > deadline - 5.0:
        return candidate
    
    # Try to find a lower rank decomposition by searching over random linear combinations
    # This is a heuristic approach that may or may not work.
    
    # For the 3x3 * 3x7 case, the minimal rank is believed to be 21 based on the
    # independence of the 7 column multiplications, each requiring rank 3.
    # Wait, that's not right. Each column of C requires 3 multiplications (one for each row of A).
    # So the total rank is at least 7 * 3 = 21.
    # And the schoolbook gives 63, so there's a lot of room for improvement.
    
    # Let's try to construct a rank-21 decomposition directly.
    # For each column k of B, we need to compute C[:, k] = A @ B[:, k]
    # This requires 3 multiplications: C[i, k] = sum_j A[i, j] * B[j, k]
    # So for each k, we can use 3 terms in the decomposition.
    # Total rank = 7 * 3 = 21.
    
    # Let's construct this directly.
    rank_direct = 21
    U_direct = [[0] * rank_direct for _ in range(n * m)]
    V_direct = [[0] * rank_direct for _ in range(m * p)]
    W_direct = [[0] * rank_direct for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                U_direct[i * m + j][col] = 1
                V_direct[j * p + k][col] = 1
                W_direct[k * n + i][col] = 1
                col += 1
    
    # Verify the direct construction
    if _verify(U_direct, V_direct, W_direct, n, m, p):
        best_rank = min(best_rank, rank_direct)
        candidate = {"U": U_direct, "V": V_direct, "W": W_direct}
    
    return candidate
# EVOLVE-BLOCK-END
