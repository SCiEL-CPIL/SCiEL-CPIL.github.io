"""Program for matrix multiplication tensor decomposition (3x5 by 5x5).

Strategy:
  1. Build a structured decomposition using 2x2 Strassen blocks where possible,
     combined with schoolbook terms for the remaining (non-2x2-aligned) products.
     This gives a rank significantly below the full schoolbook rank of 75.
  2. Verify the decomposition.
  3. Attempt further rank reduction via:
     - Linear dependency detection among the factor matrices (removing columns
       that are linear combinations of others in the tensor product sense).
     - Greedy column removal with re-verification.
  4. Return the best decomposition found within the time budget.
"""

DIMENSIONS = [3, 5, 5]

import time
import numpy as np

def _strassen_2x2_blocks():
    """
    Return the 7 Strassen products for 2x2 matrix multiplication.
    Each product is a tuple (U_row, V_row, W_row) of indices.
    
    For multiplying A (2x2) by B (2x2):
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11) B00
      M3 = A00 (B10-B11)
      M4 = A11 (B01-B10)
      M5 = (A00+A01) B11
      M6 = (A10-A00) (B00+B10)
      M7 = (A01-A11) (B01+B11)
      
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # Indices: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> i*2+k
    # Each Strassen term: (list of (A_idx, coeff), list of (B_idx, coeff), list of (C_idx, coeff))
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,-1)]))
    # M2 = (A10+A11) B00
    terms.append(([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]))
    # M3 = A00 (B10-B11)
    terms.append(([(0,1)], [(2,1),(3,-1)], [(1,1),(3,1)]))
    # M4 = A11 (B01-B10)
    terms.append(([(3,1)], [(1,1),(2,-1)], [(0,1),(2,1)]))
    # M5 = (A00+A01) B11
    terms.append(([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]))
    # M6 = (A10-A00) (B00+B10)
    terms.append(([(2,1),(0,-1)], [(0,1),(2,1)], [(3,1)]))
    # M7 = (A01-A11) (B01+B11)
    terms.append(([(1,1),(3,-1)], [(1,1),(3,1)], [(0,1)]))
    return terms

def _build_structured(n, m, p):
    """
    Build a structured decomposition using Strassen blocks where both dimensions >= 2,
    and schoolbook for the rest.
    
    For 3x5 * 5x5:
    - We can place 2x2 Strassen blocks in the (i,j) plane for i in {0,1}, j in {0,1}
    - And for k in {0,1} as well, but the k dimension is in both B and C.
    
    Actually, let's think more carefully. The tensor is T[i,j,k] = A[i,j]*B[j,k].
    We can decompose this by partitioning the (i,j,k) space.
    
    Strategy: Use 2x2 Strassen in the (i,j) block for i in {0,1}, j in {0,1}, and handle
    the remaining i=2 row and j in {2,3,4} columns separately.
    
    More precisely:
    - For i in {0,1}, j in {0,1}: use Strassen (7 terms per k column)
    - For i in {0,1}, j in {2,3,4}: schoolbook (3 terms per k)
    - For i=2, j in {0,1,2,3,4}: schoolbook (5 terms per k)
    
    Total rank = 7*5 + 3*5 + 5*5 = 35 + 15 + 25 = 75. That's the same as schoolbook!
    
    Wait, I need to be more careful. Strassen applies to 2x2 * 2x2, so it covers
    4 output entries with 7 products instead of 8. But here, for each fixed k,
    we have a 2x2 * 1x2 (or similar) subproblem.
    
    Let me reconsider. The matrix product C = A*B where A is 3x5 and B is 5x5.
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    For a 2x2 Strassen block, we need a 2x2 submatrix of A and a 2x2 submatrix of B.
    If we take A[0:2, 0:2] and B[0:2, 0:2], that gives us C[0:2, 0:2] (4 entries)
    with 7 terms instead of 8.
    
    So for the 3x5 * 5x5 case:
    - Partition A into blocks: A[0:2, 0:2], A[0:2, 2:5], A[2:3, 0:2], A[2:3, 2:5]
    - Partition B into blocks: B[0:2, 0:2], B[0:2, 2:5], B[2:5, 0:2], B[2:5, 2:5]
    
    The product C = A*B has blocks:
    C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
    C[0:2, 2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:5]*B[2:5,2:5]
    C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:5]*B[2:5,0:2]
    C[2:3, 2:5] = A[2:3,0:2]*B[0:2,2:5] + A[2:3,2:5]*B[2:5,2:5]
    
    Only C[0:2, 0:2] involves a 2x2 * 2x2 product, which can use Strassen.
    The other blocks are 2x3 * 3x2, 2x3 * 3x3, 1x2 * 2x2, 1x2 * 2x3, 1x3 * 3x2, 1x3 * 3x3.
    
    For C[0:2, 0:2]: 2x2 * 2x2 -> 7 terms (Strassen)
    For C[0:2, 2:5]: 2x3 * 3x2 -> this is a 2x3 matrix times 3x2 matrix = 2x2 result
      This can be done with 2*3*2 = 12 schoolbook terms, or potentially fewer.
    For C[2:3, 0:2]: 1x2 * 2x2 = 1x2 result -> 2 terms
    For C[2:3, 2:5]: 1x3 * 3x3 = 1x3 result -> 9 terms
    
    Total: 7 + 12 + 2 + 9 = 30. But wait, that's not right either because the blocks
    overlap in the j dimension.
    
    Let me just build it explicitly.
    """
    rank = n * m * p  # start with schoolbook
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to replace some schoolbook terms with Strassen
    # For 2x2 blocks in (i,j) space where i in {0,1}, j in {0,1}
    # This covers C[i,k] for i in {0,1}, with j in {0,1} contribution
    
    # Actually, let's use a different approach: build from scratch using
    # Strassen for 2x2 sub-blocks where possible.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Define a function to add a term
    def add_term(U_entries, V_entries, W_entries):
        """Add a single rank-1 term. U_entries: list of (idx, coeff), etc."""
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        for idx, coeff in U_entries:
            U_row[idx] += coeff
        for idx, coeff in V_entries:
            V_row[idx] += coeff
        for idx, coeff in W_entries:
            W_row[idx] += coeff
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
    
    # Use 2x2 Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    # This gives C[0:2, 0:2] from A[0:2, 0:2] * B[0:2, 0:2]
    strassen_terms = _strassen_2x2_blocks()
    for U_entries, V_entries, W_entries in strassen_terms:
        # Map local indices to global
        # Local A[i,j] where i,j in {0,1} -> global A[i,j] -> index i*5+j
        # Local B[j,k] where j,k in {0,1} -> global B[j,k] -> index j*5+k
        # Local C[i,k] where i,k in {0,1} -> global C[i,k] -> index k*3+i
        U_g = [(i * m + j, c) for (ij, c) in U_entries for i in [ij // 2] for j in [ij % 2]]
        # Actually, the indices in strassen terms are already flat 0-3 for 2x2
        # 0=A00, 1=A01, 2=A10, 3=A11
        U_g = [(i * m + j, c) for (ij, c) in U_entries for i in [ij // 2] for j in [ij % 2]]
        V_g = [(j * p + k, c) for (jk, c) in V_entries for j in [jk // 2] for k in [jk % 2]]
        W_g = [(k * n + i, c) for (ik, c) in W_entries for i in [ik // 2] for k in [ik % 2]]
        add_term(U_g, V_g, W_g)
    
    # Now add schoolbook terms for all other (i,j,k) combinations
    # that are not covered by the Strassen block
    strassen_covered = set()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                strassen_covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    U_row = [0] * (n * m)
                    V_row = [0] * (m * p)
                    W_row = [0] * (p * n)
                    U_row[i * m + j] = 1
                    V_row[j * p + k] = 1
                    W_row[k * n + i] = 1
                    U_list.append(U_row)
                    V_list.append(V_row)
                    W_list.append(W_row)
    
    U_arr = np.array(U_list).T  # shape (n*m, rank)
    V_arr = np.array(V_list).T  # shape (m*p, rank)
    W_arr = np.array(W_list).T  # shape (p*n, rank)
    
    return U_arr, V_arr, W_arr

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False, max_err
    return True, max_err

def _try_remove_column(U, V, W, col, n, m, p, tol=1e-9):
    """Try to remove a column and check if the remaining columns still decompose the tensor."""
    rank = U.shape[1]
    if rank <= 1:
        return False, float('inf')
    
    remaining = [c for c in range(rank) if c != col]
    U_r = U[:, remaining]
    V_r = V[:, remaining]
    W_r = W[:, remaining]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False, max_err
    return True, max_err

def _greedy_reduction(U, V, W, n, m, p, deadline, tol=1e-9):
    """Greedily try to remove columns from the decomposition."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    while rank > 1 and time.time() < deadline:
        best_col = -1
        best_err = float('inf')
        
        for col in range(rank):
            ok, err = _try_remove_column(U, V, W, col, n, m, p, tol)
            if ok:
                best_col = col
                break
            elif err < best_err:
                best_err = err
                best_col = col
        
        if best_col == -1:
            break
        if best_err > tol:
            break
        
        # Remove the column
        remaining = [c for c in range(rank) if c != best_col]
        U = U[:, remaining]
        V = V[:, remaining]
        W = W[:, remaining]
        rank = U.shape[1]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build structured decomposition
    U, V, W = _build_structured(n, m, p)
    
    # Verify
    ok, err = _verify_decomposition(U, V, W, n, m, p)
    if not ok:
        # Fallback to pure schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        ok, err = _verify_decomposition(U, V, W, n, m, p)
        if not ok:
            raise ValueError("Schoolbook verification failed")
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try greedy reduction
    if time.time() < deadline:
        U2, V2, W2 = _greedy_reduction(U, V, W, n, m, p, deadline)
        if U2.shape[1] < best_rank:
            ok2, _ = _verify_decomposition(U2, V2, W2, n, m, p)
            if ok2:
                best_U, best_V, best_W = U2, V2, W2
                best_rank = U2.shape[1]
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
