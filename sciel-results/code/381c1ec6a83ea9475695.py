"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Structural idea:
  Split A (2x3) into A0 (2x1, j=0) and A2 (2x2, j=1,2).
  - A0 * B0 (2x1 * 1x10): naive, 2*1*10 = 20 multiplications.
  - A2 * B2 (2x2 * 2x10): split B2 into 5 blocks of 2x2, apply Strassen (7 mults each) = 35.
  Total rank = 55.

Refinement:
  Use systematic algebraic analysis to check if rank < 55 is possible
  by exploiting the structure of the multiplication tensor and searching
  for linear dependencies among rank-1 tensor components.
"""

import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    # Build expected tensor
    T_expected = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i * m + j, j * p + k, k * n + i] = 1.0

    # Reconstruct tensor from decomposition
    T_reconstructed = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    # Check all entries match
    if not np.allclose(T_reconstructed, T_expected, atol=1e-9):
        return False

    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.

    A is 2x3: columns j=0,1,2.
    B is 3x10: rows j=0,1,2.
    C is 2x10: C[i,k] = sum_j A[i,j]*B[j,k].

    Part A: j=0 -> A[i,0]*B[0,k] for i in {0,1}, k in 0..9.  20 terms.
    Part B: j=1,2 -> A2 (2x2) * B2 (2x10). Split B2 into 5 blocks of 2x2.
             Each 2x2 * 2x2 block uses Strassen: 7 multiplications. 5*7 = 35.
    Total: 55.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10. Split B2 into 5 blocks of 2x2.
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen indices for U (A2 entries)
        u_11 = 0 * m + 1  # A[0,1]
        u_12 = 0 * m + 2  # A[0,2]
        u_21 = 1 * m + 1  # A[1,1]
        u_22 = 1 * m + 2  # A[1,2]

        # Strassen indices for V (B2 block entries)
        v_11 = 1 * p + k0  # B[1,k0]
        v_12 = 1 * p + k1  # B[1,k1]
        v_21 = 2 * p + k0  # B[2,k0]
        v_22 = 2 * p + k1  # B[2,k1]

        # W indices for outputs
        w_00 = k0 * n + 0  # C[0,k0]
        w_01 = k1 * n + 0  # C[0,k1]
        w_10 = k0 * n + 1  # C[1,k0]
        w_11 = k1 * n + 1  # C[1,k1]

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W


def _find_linear_dependencies(U, V, W, n, m, p, deadline):
    """
    Find linear dependencies among the rank-1 tensors in the decomposition.
    If dependencies exist, we might be able to reduce the rank.
    
    Returns: (U_new, V_new, W_new) if a lower-rank decomposition is found, else None
    """
    start_time = time.time()
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Flatten each rank-1 tensor
    rank1_flat = []
    for r in range(rank):
        t = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
        rank1_flat.append(t)
    
    A_matrix = np.column_stack(rank1_flat)
    
    # Compute rank and find dependencies
    try:
        U_svd, s, Vt = np.linalg.svd(A_matrix, full_matrices=False)
        matrix_rank = int(np.sum(s > 1e-10 * s[0]))
    except Exception:
        return None
    
    if matrix_rank >= rank:
        return None
    
    # There are dependencies. Try to find a minimal generating set.
    # Use greedy algorithm: start with empty set, add tensors that increase rank
    selected = []
    current_rank = 0
    
    for r in range(rank):
        if time.time() > deadline:
            break
        
        # Check if adding tensor r increases the rank
        if not selected:
            selected.append(r)
            current_rank = 1
        else:
            # Build matrix with selected + r
            indices = selected + [r]
            sub_matrix = A_matrix[:, indices]
            try:
                s_sub = np.linalg.svd(sub_matrix, compute_uv=False)
                new_rank = int(np.sum(s_sub > 1e-10 * s_sub[0]))
                if new_rank > current_rank:
                    selected.append(r)
                    current_rank = new_rank
            except Exception:
                continue
    
    # If we found a smaller set, verify it works
    if len(selected) < rank:
        U_sub = U[:, selected]
        V_sub = V[:, selected]
        W_sub = W[:, selected]
        
        # Verify
        T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for r in range(len(selected)):
            T_recon += np.einsum('a,b,c->abc', U_sub[:, r], V_sub[:, r], W_sub[:, r])
        
        if np.allclose(T_recon, T_target, atol=1e-9):
            # Check entries are valid
            if _is_valid_entries(U_sub.tolist(), V_sub.tolist(), W_sub.tolist()):
                return U_sub.tolist(), V_sub.tolist(), W_sub.tolist()
    
    return None


def _structured_rank_reduction(n, m, p, deadline):
    """
    Systematically try to reduce the rank below 55 by exploiting the block structure.
    
    Key insight: The j=0 part (20 terms) and j=1,2 part (35 Strassen terms) may
    have cross-block linear dependencies if we allow the rank-1 tensors to mix
    indices from both blocks.
    
    Strategy:
    1. Start with rank-55 decomposition
    2. Try to find linear combinations of the 55 rank-1 tensors that give
       fewer independent tensors
    3. If dependencies found, construct the lower-rank decomposition
    """
    start_time = time.time()
    
    U55, V55, W55 = _build_strassen55(n, m, p)
    
    # First, check for linear dependencies in the current decomposition
    result = _find_linear_dependencies(U55, V55, W55, n, m, p, deadline - 10)
    if result is not None:
        return result
    
    # If no dependencies in the 55, try a different approach:
    # Construct a new decomposition that mixes the j=0 and j=1,2 blocks
    
    # The key idea: for each k, the output C[:,k] depends on all of A[:,0], A[:,1], A[:,2]
    # and B[0,k], B[1,k], B[2,k]. Maybe we can use a more efficient algorithm
    # that doesn't separate j=0 from j=1,2.
    
    # Try: treat the whole thing as 2x3 * 3x10 and use a block Strassen approach
    # where we group columns of A and rows of B differently.
    
    # Alternative: use the fact that n=2 is small. For each pair of rows in C,
    # we need to compute 2 outputs. Maybe we can use a 2x3x10 specific algorithm.
    
    # For now, return None if we can't find a reduction
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave margin for the evaluator

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify it
    if not (_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        # Fallback to schoolbook
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try to find linear dependencies to reduce rank
    if time.time() < deadline - 20:
        result = _structured_rank_reduction(n, m, p, deadline - 10)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = len(U_best[0])

    return {"U": U_best, "V": V_best, "W": W_best}
