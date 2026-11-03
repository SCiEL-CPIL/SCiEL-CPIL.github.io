"""Rank-66 exact tensor decomposition for 3x4 by 4x6 matrix multiplication.

Partition strategy:
  A (3x4): rows split 1+2, cols split 2+2
  B (4x6): rows split 2+2, cols split 2+2+2
  C (3x6): each block row uses two column blocks of A times two row blocks of B

Block ranks:
  1x2 * 2x2  -> rank 4 (schoolbook)
  2x2 * 2x2  -> rank 7 (Strassen)

Total: 6 * (rank 4) + 6 * (rank 7) = 24 + 42 = 66

Additionally, we attempt to reduce rank by finding linear dependencies
among the rank-1 terms that allow combining them.
"""
import time
import numpy as np
from fractions import Fraction

def construct():
    _t0 = time.time()
    _deadline = 145.0  # safety margin before 150s kill

    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m * p

    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]  # (12, 66)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 66)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 66)

    col = 0

    # Indexing:
    #   A_{i,k}  -> U row  i * m + k       (i in [0,n), k in [0,m))
    #   B_{k,j}  -> V row  k * p + j       (k in [0,m), j in [0,p))
    #   C_{i,j}  -> W row  j * n + i       (j in [0,p), i in [0,n))  (trace convention)

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """1x2 * 2x2 block, rank 4 (schoolbook).
        A[ar:ar+1, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr, cc], C[cr, cc+1]
        """
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        # M1 = A0 * B00 -> C0
        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1
        col += 1
        # M2 = A1 * B10 -> C0
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1
        col += 1
        # M3 = A0 * B01 -> C1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1
        col += 1
        # M4 = A1 * B11 -> C1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1
        col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """2x2 * 2x2 block, rank 7 (Strassen).
        A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr:cr+2, cc:cc+2]
        Standard Strassen:
          M1 = (A00+A11)(B00+B11)   -> contributes to C00, C11
          M2 = (A10+A11)B00         -> C10 += M2, C11 -= M2
          M3 = A00(B01-B11)         -> C01 += M3, C11 += M3
          M4 = A11(B10-B00)         -> C00 += M4, C10 += M4
          M5 = (A00+A01)B11         -> C00 -= M5, C01 += M5
          M6 = (A10-A00)(B00+B01)   -> C11 += M6
          M7 = (A01-A11)(B10+B11)   -> C00 += M7
        """
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # ---- Build the 66-term decomposition ----
    # C has 3 rows × 6 columns.
    # Split A rows: row 0 (1 row), rows 1-2 (2 rows)
    # Split A cols: cols 0-1 (2 cols), cols 2-3 (2 cols)
    # Split B rows: rows 0-1 (2 rows), rows 2-3 (2 rows)
    # Split B cols: cols 0-1, 2-3, 4-5 (three blocks of 2)
    #
    # For each C column-block j (0,1,2) and each A-row-group r,
    # the contribution to C is:
    #   sum over A-col-group ac of sum over B-row-group br:
    #     A_rowgroup_r, colgroup_ac  *  B_rowgroup_br, colgroup_j
    #
    # A row group 0 (row 0) -> 1x2 * 2x2 blocks (rank 4)
    # A row group 1 (rows 1-2) -> 2x2 * 2x2 blocks (rank 7)

    for j in range(3):  # C column block index (B col offset 2*j)
        bc = 2 * j  # B column start

        # --- Row 0 of C (A row 0, 1x2 blocks) ---
        # A[0, 0:2] * B[0:2, bc:bc+2]  (1x2 * 2x2, rank 4)
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        # A[0, 2:4] * B[2:4, bc:bc+2]  (1x2 * 2x2, rank 4)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)

        # --- Rows 1-2 of C (A rows 1:3, 2x2 blocks) ---
        # A[1:3, 0:2] * B[0:2, bc:bc+2]  (2x2 * 2x2, rank 7, Strassen)
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        # A[1:3, 2:4] * B[2:4, bc:bc+2]  (2x2 * 2x2, rank 7, Strassen)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    # ---- Attempt rank reduction via GF(2) analysis ----
    # Check if any two rank-1 terms in the decomposition have the same
    # "signature" over GF(2) in a way that allows combining them.
    # Specifically, if we can find a subset S of terms such that the XOR
    # of their rank-1 tensors equals another rank-1 tensor T, then we can
    # replace S ∪ {T} with just the remaining terms, reducing rank.
    #
    # More practically: check if the 66 rank-1 tensors are linearly
    # dependent over GF(2). If the rank of the set {tensor_i mod 2} is
    # less than 66, there's a dependency. However, a GF(2) dependency
    # doesn't directly give a rank reduction over Q unless we can
    # construct a valid decomposition.
    #
    # Alternative approach: try to find if removing one term and adjusting
    # coefficients (allowing half-integers) still works.
    #
    # For now, return the rank-66 decomposition.
    # A full GF(2) SAT search for rank 65 would be too slow.

    # ---- Attempt rank reduction via linear dependency search ----
    # Convert U, V, W to numpy arrays for efficient computation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T = sum_k U[:,k] ⊗ V[:,k] ⊗ W[:,k]
    # We want to find if any term can be expressed as a linear combination of others.
    # 
    # A more practical approach: try to find a non-trivial linear combination
    # of the rank-1 tensors that equals zero. If such a combination exists with
    # all coefficients non-zero, we can potentially restructure.
    #
    # For rank reduction by 1, we need to find coefficients c_0, ..., c_65 such that
    # sum_k c_k * (U[:,k] ⊗ V[:,k] ⊗ W[:,k]) = 0
    # and then we can drop one term if the others span it.
    #
    # This is a 66-variable system. Let's set up the equations:
    # For each (i,j,k) in the tensor space, we have:
    # sum_l c_l * U[i,l] * V[j,l] * W[k,l] = 0
    #
    # This is a homogeneous linear system in the c_l variables.
    # The number of equations is n*m*p = 3*4*6 = 72.
    # We have 66 unknowns. If the system has a non-trivial solution,
    # we can potentially reduce rank.
    
    # Build the constraint matrix: each row corresponds to a tensor element (i,j,k),
    # each column corresponds to a term l. Entry = U[i,l] * V[j,l] * W[k,l]
    
    # Flatten indices: (i,j,k) -> i * (m*p) + j * p + k
    # But our W uses trace convention: W[j*n+i, l]
    # So for tensor element C_{i,j}, the W index is j*n + i
    
    # Let's build the matrix properly
    # Tensor element (i, j, k) where i is A row, k is A col, j is B col
    # Wait, let me re-clarify the indices:
    # A is n x m: A[i,k] for i in [0,n), k in [0,m)
    # B is m x p: B[k,j] for k in [0,m), j in [0,p)
    # C is n x p: C[i,j] for i in [0,n), j in [0,p)
    # 
    # The multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # In the decomposition: C[i,j] = sum_l U[i*m+k, l] * V[k*p+j, l] * W[j*n+i, l]
    # Wait, that's not right. Let me re-examine.
    # 
    # Actually, the standard decomposition is:
    # C[i,j] = sum_l U[i,k,l] * V[k,j,l] * W[i,j,l]  -- but this is 3D
    # 
    # In our 2D representation:
    # U has shape (n*m, rank): U[i*m+k, l] represents the A coefficient
    # V has shape (m*p, rank): V[k*p+j, l] represents the B coefficient  
    # W has shape (p*n, rank): W[j*n+i, l] represents the C coefficient
    # 
    # The reconstruction: C[i,j] = sum_l sum_k U[i*m+k, l] * V[k*p+j, l] * W[j*n+i, l]
    # 
    # Wait, that doesn't look right either. Let me think again.
    # 
    # The correct form is: C[i,j] = sum_l (sum_k A[i,k] * B[k,j])_l
    # where each term l contributes: (U[:,l] as A) * (V[:,l] as B) -> (W[:,l] as C)
    # 
    # So: C[i,j] = sum_l W[j*n+i, l] * (sum_k U[i*m+k, l] * V[k*p+j, l])
    # 
    # This is a bilinear form in U and V, multiplied by W.
    # 
    # For the linear dependency: we want sum_l c_l * T_l = 0 where T_l is the rank-1 tensor
    # T_l[i,k,j] = U[i*m+k, l] * V[k*p+j, l] * W[j*n+i, l]
    # 
    # So the constraint for each (i,k,j) is:
    # sum_l c_l * U[i*m+k, l] * V[k*p+j, l] * W[j*n+i, l] = 0
    
    # Build the matrix M where M[(i,k,j), l] = U[i*m+k, l] * V[k*p+j, l] * W[j*n+i, l]
    # Number of (i,k,j) tuples: n * m * p = 3 * 4 * 6 = 72
    # Number of l: 66
    
    M = np.zeros((n * m * p, RANK))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                idx = i * m * p + k * p + j
                M[idx] = U_arr[i * m + k] * V_arr[k * p + j] * W_arr[j * n + i]
    
    # Find the null space of M (transposed, since we want M @ c = 0)
    # M is 72 x 66, so we want the null space of M^T which is 66 x 72
    # Actually, we want vectors c such that M @ c = 0
    # This is the null space of M
    
    # Use SVD to find the null space
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=True)
        # The null space is spanned by the right singular vectors corresponding to zero singular values
        # Find the rank of M
        tol = 1e-10
        rank_M = np.sum(s > tol)
        nullity = RANK - rank_M
        
        if nullity > 0:
            # There's a non-trivial null space
            # The null space vectors are the last `nullity` columns of Vt
            null_space = Vt[RANK - nullity:].T  # shape (RANK, nullity)
            
            # For each null space vector, check if it has a sparse structure
            # that allows rank reduction
            for ns_idx in range(nullity):
                c = null_space[:, ns_idx]
                # Find which terms have non-zero coefficients
                nonzero = np.where(np.abs(c) > 1e-8)[0]
                if len(nonzero) >= 2:
                    # We have a dependency involving these terms
                    # Try to see if we can drop one term and adjust others
                    # This is complex; for now, just note the dependency exists
                    pass
        
        # If nullity > 0, we might be able to reduce rank
        # But the actual reduction requires careful analysis
        # For now, return the rank-66 decomposition
    except Exception:
        pass
    
    if time.time() > _deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
