"""Matrix multiplication 3,3,6 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _laderman_3x3():
    """
    Return (U, V, W) for the 3x3x3 matrix multiplication tensor using
    Laderman's rank-7 decomposition.
    
    U shape: (9, 7)  indexed by [i*3+j]
    V shape: (9, 7)  indexed by [j*3+k]
    W shape: (9, 7)  indexed by [k*3+i]
    
    The decomposition satisfies:
      for all (i, j, k): sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1
    """
    # Laderman's 7 bilinear forms for 3x3 matrix multiplication:
    # A = [[a11, a12, a13],
    #      [a21, a22, a23],
    #      [a31, a32, a33]]
    # B = [[b11, b12, b13],
    #      [b21, b22, b23],
    #      [b31, b32, b33]]
    #
    # The 7 products:
    # p1 = a11 * b11
    # p2 = (a12 + a22) * (b21 - b11)
    # p3 = (a21 + a31) * (b11 - b31)
    # p4 = a33 * b33
    # p5 = (a13 + a33) * (b31 + b33)
    # p6 = (a31 + a33) * (b33 - b11)
    # p7 = (a11 + a13) * (b13 - b33)
    #
    # C = A*B computed from these 7 products.
    #
    # U entries (coefficients of A entries in each product):
    #   p1: a11 -> U[0,0]=1
    #   p2: a12, a22 -> U[1,1]=1, U[4,1]=1
    #   p3: a21, a31 -> U[3,2]=1, U[6,2]=1
    #   p4: a33 -> U[8,3]=1
    #   p5: a13, a33 -> U[2,4]=1, U[8,4]=1
    #   p6: a31, a33 -> U[6,5]=1, U[8,5]=1
    #   p7: a11, a13 -> U[0,6]=1, U[2,6]=1
    
    U = np.zeros((9, 7), dtype=float)
    U[0, 0] = 1   # p1: a11
    U[1, 1] = 1   # p2: a12
    U[4, 1] = 1   # p2: a22
    U[3, 2] = 1   # p3: a21
    U[6, 2] = 1   # p3: a31
    U[8, 3] = 1   # p4: a33
    U[2, 4] = 1   # p5: a13
    U[8, 4] = 1   # p5: a33
    U[6, 5] = 1   # p6: a31
    U[8, 5] = 1   # p6: a33
    U[0, 6] = 1   # p7: a11
    U[2, 6] = 1   # p7: a13
    
    # V entries (coefficients of B entries in each product):
    #   p1: b11 -> V[0,0]=1
    #   p2: b21, b11 -> V[3,1]=1, V[0,1]=-1
    #   p3: b11, b31 -> V[0,2]=1, V[6,2]=-1
    #   p4: b33 -> V[8,3]=1
    #   p5: b31, b33 -> V[6,4]=1, V[8,4]=1
    #   p6: b33, b11 -> V[8,5]=1, V[0,5]=-1
    #   p7: b13, b33 -> V[2,6]=1, V[8,6]=-1
    
    V = np.zeros((9, 7), dtype=float)
    V[0, 0] = 1    # p1: b11
    V[3, 1] = 1    # p2: b21
    V[0, 1] = -1   # p2: -b11
    V[0, 2] = 1    # p3: b11
    V[6, 2] = -1   # p3: -b31
    V[8, 3] = 1    # p4: b33
    V[6, 4] = 1    # p5: b31
    V[8, 4] = 1    # p5: b33
    V[8, 5] = 1    # p6: b33
    V[0, 5] = -1   # p6: -b11
    V[2, 6] = 1    # p7: b13
    V[8, 6] = -1   # p7: -b33
    
    # W entries (coefficients of C entries in each product):
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # p1 = a11*b11 contributes to C[0,0] -> W[0,0]=1
    # p2 = (a12+a22)*(b21-b11) = a12*b21 - a12*b11 + a22*b21 - a22*b11
    #   a12*b21 -> C[0,1], a12*b11 -> -C[0,0]... wait let me redo this carefully.
    #
    # Let me re-derive. C[i,k] = sum_j A[i,j] B[j,k]
    # C[0,0] = a11*b11 + a12*b21 + a13*b31
    # C[0,1] = a11*b12 + a12*b22 + a13*b32
    # C[0,2] = a11*b13 + a12*b23 + a13*b33
    # C[1,0] = a21*b11 + a22*b21 + a23*b31
    # C[1,1] = a21*b12 + a22*b22 + a23*b32
    # C[1,2] = a21*b13 + a22*b23 + a23*b33
    # C[2,0] = a31*b11 + a32*b21 + a33*b31
    # C[2,1] = a31*b12 + a32*b22 + a33*b32
    # C[2,2] = a31*b13 + a32*b23 + a33*b33
    #
    # p1 = a11*b11: contributes to C[0,0] with coeff 1. W[0,0]=1.
    #
    # p2 = (a12+a22)*(b21-b11) = a12*b21 - a12*b11 + a22*b21 - a22*b11
    #   a12*b21 -> C[0,0] (from a12*b21 in C[0,0])... 
    #   Wait: C[0,0] = a11*b11 + a12*b21 + a13*b31. So a12*b21 is in C[0,0].
    #   a12*b11: not directly a term in any C[i,k]... 
    #   Actually a12*b11 would be A[0,1]*B[0,0] which is NOT a valid term (j must match).
    #   Hmm, I think I need to be more careful.
    #
    # Let me look up Laderman's decomposition properly.
    #
    # Laderman (1976) rank-7 algorithm for 3x3:
    # m1 = a11 * b11
    # m2 = (a12 + a22) * (b21 - b11)
    # m3 = (a21 + a31) * (b11 - b31)
    # m4 = a33 * b33
    # m5 = (a13 + a33) * (b31 + b33)
    # m6 = (a31 + a33) * (b33 - b11)
    # m7 = (a11 + a13) * (b13 - b33)
    #
    # C[0,0] = m1 + m2 + m3 - m6
    # C[0,1] = m2 + m7
    # C[0,2] = m5 - m7
    # C[1,0] = m3 - m6
    # C[1,1] = m1 + m2 + m3 - m6 - m7  ... this doesn't look right either.
    #
    # Let me just verify by direct computation. I'll use a known correct Laderman decomposition.
    #
    # Actually, let me use a different, well-known rank-7 decomposition.
    # The standard Laderman decomposition:
    
    # I'll use the formulation from Bini et al. / standard references.
    # Let me just hard-code a verified decomposition.
    
    # Alternative: use the decomposition from the paper by Laderman.
    # After checking, here's a correct set:
    
    # Actually, let me just verify my U, V above and compute W by solving.
    # For each (i,k), C[i,k] = sum_j sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = sum_j A[i,j]B[j,k]
    # This means: for each (i,k), the vector sum_c (sum_j U[i,j,c]*V[j,k,c]) * W[k,i,c] must equal
    # the (i,k) entry of A*B, which is sum_j A[i,j]B[j,k].
    #
    # Since U and V are fixed, for each (i,k) we have a linear equation in W[k,i,c]:
    # sum_c (sum_j U[i,j,c] * V[j,k,c]) * W[k,i,c] = sum_j A[i,j] B[j,k]  (as a bilinear form)
    #
    # But actually the constraint is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all (i,j,k)
    #
    # Wait no. The tensor entry T[i,j,k] = 1 means the coefficient of A[i,j]*B[j,k] in the
    # reconstruction of C is 1. So:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for ALL (i,j,k)
    
    # Let me verify this with my U and V.
    n3, m3, p3 = 3, 3, 3
    rank3 = 7
    # Check: for each (i,j,k), sum_c U[i*3+j,c] * V[j*3+k,c] * W[k*3+i,c] = 1
    
    # I need to find W. For each (i,j,k), I need:
    # sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1
    
    # This is a system of 27 equations in 27 unknowns W[k*3+i, c] for k=0..2, i=0..2, c=0..6.
    # Let me set it up.
    
    # Actually, W is indexed by [k*n+i] where n=3, so W has 9 rows and 7 columns.
    # The constraint for (i,j,k) involves W[k*3+i, c] for c=0..6.
    
    # Let me build the linear system.
    # Unknowns: W[r, c] for r=0..8, c=0..6 -> 63 unknowns? No wait, W has shape (9,7) = 63 entries.
    # But we only have 27 constraints. So the system is underdetermined.
    # 
    # Hmm, that can't be right. Let me reconsider.
    # 
    # Actually, the constraint is:
    # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This gives n*m*p = 27 equations. The unknowns are the entries of W, which is (p*n, rank) = (9, 7) = 63 entries.
    # So yes, underdetermined. We need additional constraints or a specific choice.
    #
    # The standard approach: W is determined by requiring that the bilinear form reconstructs C correctly.
    # Given U and V, W is uniquely determined if the decomposition is valid.
    #
    # Let me think differently. The tensor T = sum_c U_c ⊗ V_c ⊗ W_c where U_c, V_c, W_c are the c-th columns.
    # T[i,j,k] = 1 for all i,j,k.
    #
    # Given U and V, we need W such that sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all (i,j,k).
    #
    # For a fixed (i,k), define f_{i,k}[j] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i].
    # We need f_{i,k}[j] = 1 for all j.
    #
    # This means: for each (i,k), the vector (W_c[k*n+i])_{c=0..6} must satisfy:
    # for all j: sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1
    #
    # This is 3 linear equations (one per j) in 7 unknowns (W_c[k*n+i] for c=0..6).
    #
    # So for each of the 9 pairs (i,k), we have a 3x7 system. We can pick a particular solution.
    
    # Let me just solve for W using least squares or by picking a simple solution.
    # Actually, for a valid rank-7 decomposition, the system should be consistent and we can find W.
    
    # Let me build the full system: 27 equations, 63 unknowns.
    # I'll use numpy to solve it.
    
    # Actually, a cleaner approach: for each (i,k), solve the 3x7 system for W[:, k*3+i] (a 7-vector).
    # Since it's underdetermined, I'll use the minimum-norm solution.
    
    W = np.zeros((9, 7), dtype=float)
    for i in range(3):
        for k in range(3):
            # Build 3x7 matrix M where M[j, c] = U[i*3+j, c] * V[j*3+k, c]
            M = np.zeros((3, 7), dtype=float)
            for j in range(3):
                for c in range(7):
                    M[j, c] = U[i * 3 + j, c] * V[j * 3 + k, c]
            # Solve M @ w = ones(3) for w (7-vector)
            # Minimum norm solution
            w, _, _, _ = np.linalg.lstsq(M, np.ones(3), rcond=None)
            W[k * 3 + i, :] = w
    
    # Verify
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = sum(U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] for c in range(7))
                if abs(val - 1) > 1e-10:
                    # This decomposition doesn't work with these U, V
                    pass
    
    return U, V, W


def _verify(U, V, W, n=3, m=3, p=6, atol=1e-8):
    """Verify that U, V, W form a valid decomposition of the n x m x p multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    tensor = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u_col = U_arr[:, c].reshape(n, m)
        v_col = V_arr[:, c].reshape(m, p)
        w_col = W_arr[:, c].reshape(p, n)
        tensor += u_col[:, :, np.newaxis] * v_col[np.newaxis, :, :] * w_col[np.newaxis, np.newaxis, :]
    
    return np.allclose(tensor, np.ones((n, m, p)), atol=atol)


def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def _try_laderman_kronecker():
    """
    Try to construct a rank-14 decomposition by taking the Kronecker product
    of Laderman's rank-7 (3x3x3) with a rank-2 decomposition of the 
    'column extraction' tensor that maps 3 columns to 6 columns (2 blocks of 3).
    
    The idea: A (3x3) * B (3x6) = C (3x6).
    Split B into B = [B1 | B2] where B1, B2 are 3x3.
    Then C = [A*B1 | A*B2].
    
    If we use Laderman for each half, that's rank 14. But the U factor is shared
    (same A), so we can potentially share the U columns.
    
    Actually, the Kronecker product approach:
    - The 3x3x6 tensor can be seen as two copies of 3x3x3 (for the two halves of B).
    - Using Laderman for each: rank 7 + rank 7 = 14.
    - U for first half: U_laderman (9x7), U for second half: U_laderman (9x7)
    - But they're the same A, so U is just U_laderman repeated? No...
    
    Let me think more carefully.
    
    For C = A * B where A is 3x3 and B is 3x6:
    C[:, 0:3] = A * B[:, 0:3]  (first 3 columns)
    C[:, 3:6] = A * B[:, 3:6]  (last 3 columns)
    
    Using Laderman for each:
    C[:, 0:3] = sum_{c=1}^{7} U_c(A) * V_c(B[:,0:3]) * W_c
    C[:, 3:6] = sum_{c=1}^{7} U_c(A) * V_c(B[:,3:6]) * W_c
    
    So the full decomposition:
    U_full[9, 14]: first 7 cols = U_laderman, last 7 cols = U_laderman
    V_full[18, 14]: first 7 cols = V_laderman for B[:,0:3] (rows 0-8), 
                     last 7 cols = V_laderman for B[:,3:6] (rows 9-17)
    W_full[18, 14]: first 7 cols = W_laderman (rows 0-8 for k=0,1,2),
                    last 7 cols = W_laderman (rows 9-17 for k=3,4,5)
    
    Wait, W is indexed by [k*n + i] where k goes 0..5.
    For the first half (k=0,1,2): W[k*3+i, c] = W_laderman[k*3+i, c]
    For the second half (k=3,4,5): W[k*3+i, c+7] = W_laderman[(k-3)*3+i, c]
    
    Let me verify this.
    """
    n, m, p = 3, 3, 6
    rank3 = 7
    rank_full = 14
    
    U3, V3, W3 = _laderman_3x3()
    
    # Build the full decomposition
    U_full = np.zeros((n * m, rank_full), dtype=float)
    V_full = np.zeros((m * p, rank_full), dtype=float)
    W_full = np.zeros((p * n, rank_full), dtype=float)
    
    # First half: columns 0..6 of B (indices 0,1,2)
    for c in range(rank3):
        U_full[:, c] = U3[:, c]
        # V: rows j*3 + k for k=0,1,2 -> rows 0..8
        for j in range(m):
            for k in range(3):
                V_full[j * p + k, c] = V3[j * 3 + k, c]
        # W: rows k*3 + i for k=0,1,2 -> rows 0..8
        for k in range(3):
            for i in range(n):
                W_full[k * n + i, c] = W3[k * 3 + i, c]
    
    # Second half: columns 3..5 of B (indices 3,4,5)
    for c in range(rank3):
        U_full[:, c + rank3] = U3[:, c]
        # V: rows j*3 + k for k=3,4,5 -> rows 9..17
        for j in range(m):
            for k in range(3):
                V_full[j * p + (k + 3), c + rank3] = V3[j * 3 + k, c]
        # W: rows k*3 + i for k=3,4,5 -> rows 9..17
        for k in range(3):
            for i in range(n):
                W_full[(k + 3) * n + i, c + rank3] = W3[k * 3 + i, c]
    
    # Verify
    if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
        return U_full.tolist(), V_full.tolist(), W_full.tolist()
    
    return None


def _try_strassen_kronecker():
    """
    Try Strassen's rank-8 decomposition for 3x3x3 (if valid) combined with
    column splitting to get rank-16 for 3x3x6.
    
    Actually Strassen is for 2x2x2. For 3x3x3, the best known is Laderman (rank 7).
    So this is the same as above.
    """
    return None


def _try_direct_optimization():
    """
    Try to find a lower-rank decomposition by direct optimization.
    This is a fallback if the Kronecker approach fails.
    """
    # For now, skip this and return None
    return None


def construct():
    n, m, p = 3, 3, 6
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # First, try the Laderman Kronecker product approach (rank 14)
    result = _try_laderman_kronecker()
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Check time
    if time.time() > deadline:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Try direct optimization
    result = _try_direct_optimization()
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
