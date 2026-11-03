"""Matrix multiplication 4,4,7 tensor decomposition - deterministic algebraic block composition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
SEED = 42

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2():
    """
    Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen).
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    
    Strassen formulas for C = A*B:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B21-B22)
      M4 = A22(B12-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
      
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_block_composition(n, m, p):
    """
    Build a rank-93 decomposition for 4x4 * 4x7.
    
    Split B (4x7) into B_left (4x4) and B_right (4x3).
    
    C_left (4x4) = A * B_left: Use Strassen on 2x2 blocks.
      A = [[A11, A12], [A21, A22]] (each 2x2)
      B_left = [[B11, B12], [B21, B22]] (each 2x2)
      C_left = [[C11, C12], [C21, C22]] (each 2x2)
      
      Each 2x2*2x2 sub-product uses Strassen rank 7.
      Total for C_left: 4 * 7 = 28 rank terms.
      
    C_right (4x3) = A * B_right: Schoolbook.
      B_right = [[BR11, BR12, BR13], [BR21, BR22, BR23]] (2x3 each)
      C_right = [[CR11, CR12, CR13], [CR21, CR22, CR23]] (2x3 each)
      
      Each 2x2 * 2x3 sub-product: rank 2*2*3 = 12 schoolbook terms.
      Total for C_right: 4 * 12 = 48 rank terms.
      
    Total rank: 28 + 48 = 76? Wait, let me recount.
    
    Actually for C_right (4x3), we have 4 output blocks each 2x3.
    Each 2x2 * 2x3 product has rank 2*2*3 = 12.
    4 blocks * 12 = 48.
    
    Total: 28 + 48 = 76.
    
    Hmm, but the inspiration said 93. Let me re-examine.
    
    Wait - the Strassen decomposition for 2x2*2x2 gives rank 7 for each 2x2*2x2 block.
    For C_left (4x4), we need 4 such blocks, so 4*7 = 28.
    
    For C_right (4x3), each 2x2*2x3 block needs rank 12 (schoolbook).
    4 blocks * 12 = 48.
    
    Total: 28 + 48 = 76.
    
    But wait - we need to be careful about how the indices work.
    Let me just build it directly.
    """
    n, m, p = 4, 4, 7
    
    # We'll build U (n*m=16, rank), V (m*p=28, rank), W (p*n=28, rank)
    
    # Part 1: C_left (columns 0-3 of output) = A * B[:, 0:4]
    # This is 4x4 * 4x4 = 4x4
    # Use Strassen on 2x2 blocks: 4 blocks, each rank 7 -> total 28
    
    # Part 2: C_right (columns 4-6 of output) = A * B[:, 4:7]
    # This is 4x4 * 4x3 = 4x3
    # Schoolbook: each 2x2*2x3 block has rank 12, 4 blocks -> total 48
    
    total_rank = 28 + 48
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: Strassen for left 4x4 output ---
    # A rows 0-3, B cols 0-3, C cols 0-3
    # Block (bi, bj) where bi in {0,1} (row block of A), bj in {0,1} (col block of B/C)
    # A block (bi, bj): rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    #   A index for (i,j) = i*m + j = i*4 + j
    # B block (bj, bk): rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
    #   B index for (j,k) = j*p + k = j*7 + k
    # C block (bi, bk): rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
    #   C index for (k,i) = k*n + i = k*4 + i
    
    col_offset = 0
    for bi in range(2):  # row block of A / C
        for bj in range(2):  # col block of A / row block of B
            for bk in range(2):  # col block of B / C
                # Get Strassen factors for this 2x2*2x2 block
                sU, sV, sW = _strassen_2x2_2x2()
                
                for r in range(7):
                    # A indices: local (li, lj) -> global (2*bi+li, 2*bj+lj)
                    # A global index = (2*bi+li)*4 + (2*bj+lj)
                    for li in range(2):
                        for lj in range(2):
                            if sU[li * 2 + lj, r] != 0:
                                a_idx = (2 * bi + li) * 4 + (2 * bj + lj)
                                U[a_idx, col_offset + r] += sU[li * 2 + lj, r]
                    
                    # B indices: local (lj, lk) -> global (2*bj+lj, 2*bk+lk)
                    # B global index = (2*bj+lj)*7 + (2*bk+lk)
                    for lj in range(2):
                        for lk in range(2):
                            if sV[lj * 2 + lk, r] != 0:
                                b_idx = (2 * bj + lj) * 7 + (2 * bk + lk)
                                V[b_idx, col_offset + r] += sV[lj * 2 + lk, r]
                    
                    # C indices: local (lk, li) -> global (2*bk+lk, 2*bi+li)
                    # C global index = (2*bk+lk)*4 + (2*bi+li)
                    for lk in range(2):
                        for li in range(2):
                            if sW[lk * 2 + li, r] != 0:
                                c_idx = (2 * bk + lk) * 4 + (2 * bi + li)
                                W[c_idx, col_offset + r] += sW[lk * 2 + li, r]
                
                col_offset += 7
    
    # --- Part 2: Schoolbook for right 4x3 output ---
    # A rows 0-3, B cols 4-6, C cols 4-6
    # For each (i, j, k) where i in 0-3, j in 0-3, k in 4-6:
    #   U[i*4+j, col] = 1
    #   V[j*7+k, col] = 1
    #   W[k*4+i, col] = 1
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j, col_offset] = 1
                V[j * 7 + k, col_offset] = 1
                W[k * 4 + i, col_offset] = 1
                col_offset += 1
    
    assert col_offset == total_rank, f"Rank mismatch: {col_offset} vs {total_rank}"
    
    return U, V, W


def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification of tensor decomposition."""
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    if U_mat.shape[1] != V_mat.shape[1] or V_mat.shape[1] != W_mat.shape[1]:
        return False
    if U_mat.shape[0] != n * m or V_mat.shape[0] != m * p or W_mat.shape[0] != p * n:
        return False
    
    rank = U_mat.shape[1]
    
    # Check all target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check all non-target entries
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_set.add((i * m + j, j * p + k, k * n + i))
    
    for alpha in range(n * m):
        for beta in range(m * p):
            for gamma in range(p * n):
                if (alpha, beta, gamma) not in target_set:
                    val = np.sum(U_mat[alpha, :] * V_mat[beta, :] * W_mat[gamma, :])
                    if abs(val) > tol:
                        return False
    
    return True


def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    Use linear algebra: if we have R terms, check if any can be expressed as 
    linear combination of others by solving a linear system.
    
    This is a heuristic - we look for columns that are linearly dependent 
    in the combined tensor space.
    """
    if time.time() > deadline:
        return None
    
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return (U, V, W)
    
    # Try to find a linear combination of rank-1 tensors that sums to zero.
    # This is equivalent to finding a vector c in R^R such that:
    #   sum_r c_r * U[:,r] * V[:,r] * W[:,r] = 0 (as a tensor)
    # which means for all (alpha, beta, gamma):
    #   sum_r c_r * U[alpha,r] * V[beta,r] * W[gamma,r] = 0
    
    # Build the linear system: for each (alpha, beta, gamma), we get one equation.
    # Number of unknowns: current_rank
    # Number of equations: (n*m)*(m*p)*(p*n) = 16*28*28 = 12544
    
    # But we only need the non-target entries to be zero, and target entries to equal 1.
    # For rank reduction, we need: sum_r c_r * (tensor_r) = 0
    
    # Let's build the matrix A where A[eq, r] = U[alpha,r]*V[beta,r]*W[gamma,r]
    # and solve A * c = 0 for non-trivial c.
    
    # This is a large system. Let's use SVD to find the null space.
    # But 12544 x R might be too large. Let's sample.
    
    # Alternative: just try to merge pairs of terms by checking if their 
    # "direction" in the tensor space is linearly dependent.
    
    # Simpler heuristic: try removing one term at a time and see if the rest
    # can still satisfy the constraints (via least squares).
    
    # For now, this is a placeholder - the deterministic construction already gives rank 76.
    # Further reduction would require deeper algebraic insight.
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s limit (using 115 for extra safety)
    
    # Build the deterministic block composition
    U, V, W = _build_block_composition(n, m, p)
    
    # Verify
    if _verify_full(U, V, W, n, m, p, tol=1e-9):
        best_rank = U.shape[1]
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_s = np.zeros((n * m, rank))
        V_s = np.zeros((m * p, rank))
        W_s = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        best_rank = rank
        best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Try to reduce rank further if time permits
    if time.time() < deadline - 10 and best_rank > 76:
        reduced = _try_reduce_rank(
            np.array(best_result["U"]), np.array(best_result["V"]), np.array(best_result["W"]),
            n, m, p, best_rank - 1, deadline
        )
        if reduced is not None:
            rU, rV, rW = reduced
            if _verify_full(rU, rV, rW, n, m, p, tol=1e-9):
                best_rank = rU.shape[1]
                best_result = {"U": rU.tolist(), "V": rV.tolist(), "W": rW.tolist()}
    
    return best_result

# EVOLVE-BLOCK-END
