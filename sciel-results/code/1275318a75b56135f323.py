import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14

    deadline = time.time() + 110  # safety margin before 120s

    # ------------------------------------------------------------------
    # Strategy: Structured convolution-based decomposition
    #
    # Key insight: C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # For each row i, this is a length-4 convolution of A[i,:] with each column of B.
    #
    # We model the tensor decomposition as:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    #
    # Approach: Use the algebraic structure of 1x4x7 matmul (convolution)
    # to build a decomposition. We try several structured families:
    #
    # Family A: Direct rank-28 (one per (j,k) pair) - schoolbook per row
    # Family B: Winograd/FFT-like structured rank for the 1x4x7 convolution
    # Family C: General search for small-rank exact decompositions
    #
    # The 1x4x7 tensor (convolution) has known rank bounds.
    # For a 1x4x7 convolution, we can use:
    #   - Rank 7 (schoolbook: 4*7/4 = 7 per output? No, it's 4*7=28 schoolbook)
    #   - Actually 1x4x7: C[k] = sum_j A[j]*B[j,k], rank at most 28
    #   - With structure: if we use DFT-like transforms, rank can be reduced
    #
    # For 2x4x7: two independent rows, but they share the B matrix.
    # The key coupling is through V (which depends on j and k).
    #
    # Let's try: rank = 2 * 7 = 14 using per-row convolution structure
    # where each row uses 7 "convolution points" (like FFT evaluation points).
    #
    # Actually, let's think more carefully. The tensor T[i,j,k]=1 for all i,j,k
    # with i in {0,1}, j in {0,1,2,3}, k in {0,...,6}.
    #
    # A rank-1 term is U[i,j] * V[j,k] * W[k,i].
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for ALL (i,j,k).
    #
    # This is a 2x4x7 all-ones tensor. Its tensor rank is what we seek.
    #
    # Lower bound: max(n*m, m*p, p*n) / max(n,m,p) ... no.
    # The tensor rank of an a x b x c all-ones tensor is at most min(ab, bc, ca).
    # But it can be much less. For 2x4x7: min(8, 28, 14) = 8.
    #
    # Can we achieve rank 8? We need 8 rank-1 tensors whose sum is all-ones 2x4x7.
    #
    # Let's try a systematic approach: use the fact that for the all-ones tensor,
    # we can write 1 = sum_r u_r(i,j) * v_r(j,k) * w_r(k,i).
    #
    # If we fix the "middle" dimension j (size 4), we can think of this as:
    # For each j, the 2x7 matrix of ones must be expressed as sum of rank-1
    # matrices u_r(i) * w_r(k) weighted by v_r(j).
    #
    # Actually, a cleaner approach: use the "slice" method.
    # Fix j. Then we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,k.
    # Let a_r(i) = U[i,j,r], b_r(k) = V[j,k,r] * W[k,i,r]... no, W depends on i too.
    #
    # Let me just do a direct search for small ranks using the structure.
    #
    # Key structural observation: The all-ones tensor of size 2x4x7 can be
    # decomposed as follows. Note that:
    #   1 = (1/4) * sum_{j=0}^{3} 1  (trivially)
    # But we need rank-1 tensor terms, not just scalar sums.
    #
    # Let's try rank = 7 (one per k-value, exploiting that the tensor is
    # constant in i and j):
    #   Term r (r=0..6): U[i,j,r] = 1, V[j,k,r] = delta_{k,r}, W[k,i,r] = 1
    #   Sum: sum_r 1 * delta_{k,r} * 1 = 1. ✓ But rank = 7.
    #   Wait, that works! For each k, we have one term that "selects" that k.
    #   U[i*m+j, r] = 1 for all i,j when r = k.
    #   V[j*p+k, r] = 1 if k == r, else 0.
    #   W[k*n+i, r] = 1 for all k,i when r = k.
    #   Check: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_r 1 * delta_{k,r} * 1 = 1. ✓
    #
    # So rank 7 works! Let me verify this is correct.
    # U is 8x7, V is 28x7, W is 14x7.
    # U[i*4+j, r] = 1 for all i,j,r (all ones).
    # V[j*7+k, r] = 1 if k==r else 0.
    # W[k*2+i, r] = 1 for all k,i,r (all ones).
    #
    # sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*2+i, r]
    # = sum_r 1 * delta_{k,r} * 1 = 1. ✓✓✓
    #
    # Rank = 7! This is much better than 46.
    #
    # Can we do even better? Lower bound: the tensor is 2x4x7 all-ones.
    # The matrix flattening (i,j) vs k gives an 8x7 all-ones matrix, rank 1.
    # The flattening (i,k) vs j gives a 14x4 all-ones matrix, rank 1.
    # The flattening (j,k) vs i gives a 28x2 all-ones matrix, rank 1.
    # So matrix rank lower bound is 1, which is not helpful.
    #
    # The true tensor rank lower bound for all-ones a×b×c is more subtle.
    # For 2×4×7, can we do rank < 7?
    #
    # Try rank 4: We need 4 terms. Each term is u_r ⊗ v_r ⊗ w_r.
    # sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k.
    #
    # This is a system of 2*4*7 = 56 equations in 4*(8+28+14) = 192 unknowns.
    # Over-determined? No, 192 > 56, so potentially solvable.
    # But the equations are nonlinear (products of variables).
    #
    # Let's try rank 4 with a structured ansatz:
    # Use 4 "evaluation points" like in FFT. For the 1x4x7 convolution:
    # C[k] = sum_j A[j] B[j,k]
    # If we evaluate A at 4 points and B at 4 points... but B has 7 columns.
    #
    # Actually, let's just try rank 7 first (which we know works) and then
    # attempt to reduce further with a search.
    # ------------------------------------------------------------------

    # Rank 7 construction (verified above):
    U = np.ones((nm, 7), dtype=int)
    V = np.zeros((mp, 7), dtype=int)
    W = np.ones((pn, 7), dtype=int)
    for j in range(m):
        for k in range(p):
            V[j * p + k, k] = 1  # V[j*7+k, k] = 1

    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ------------------------------------------------------------------
    # Try to find rank < 7 via structured search
    #
    # For rank r, we need integer/half-integer solutions to:
    #   sum_{s=0}^{r-1} U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    # for all i in {0,1}, j in {0,1,2,3}, k in {0,...,6}
    #
    # This is a system of 56 trilinear equations in r*(8+28+14) = 50r unknowns.
    #
    # Strategy: Use the symmetry. The tensor is all-ones, so we can try
    # symmetric decompositions. For rank 4, try:
    #   U[i,j,s] = a_s (independent of i,j) -- no, that's too restrictive
    #
    # Better: exploit that the tensor factors as (ones_2) ⊗ (ones_4) ⊗ (ones_7)
    # and try to decompose using the Kronecker structure.
    #
    # Actually, the simplest improvement: can we do rank 6?
    # Try: for k=0..5, use the same construction as above (V selects k),
    # and for k=6, combine with another column.
    #
    # Let's try a more general search for rank r using random initialization
    # and iterative refinement (like tensor ring / CP decomposition with
    # integer constraints).
    #
    # For small r, we can try exhaustive structured searches.
    #
    # Key insight for the all-ones tensor:
    # If we can find vectors u_s (in R^8), v_s (in R^28), w_s (in R^14)
    # such that the outer product sum equals the all-ones tensor, we're done.
    #
    # The all-ones tensor T = 1_2 ⊗ 1_4 ⊗ 1_7 (Kronecker product of all-ones vectors).
    #
    # For rank 1: T = u ⊗ v ⊗ w requires u[i,j]=1, v[j,k]=1, w[k,i]=1,
    # which gives u[i,j]*v[j,k]*w[k,i] = 1*1*1 = 1. ✓
    # Wait! Rank 1 works?!
    #
    # U[i*4+j, 0] = 1 for all i,j
    # V[j*7+k, 0] = 1 for all j,k
    # W[k*2+i, 0] = 1 for all k,i
    # sum_0 U[i,j,0]*V[j,k,0]*W[k,i,0] = 1*1*1 = 1. ✓
    #
    # RANK 1! The all-ones tensor has rank 1!
    # ------------------------------------------------------------------

    # Rank 1 construction:
    U1 = np.ones((nm, 1), dtype=int)
    V1 = np.ones((mp, 1), dtype=int)
    W1 = np.ones((pn, 1), dtype=int)

    if _verify(U1, V1, W1, n, m, p):
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}

    # ------------------------------------------------------------------
    # Wait - I need to re-examine the problem formulation.
    #
    # The task says: "exactly decompose the matrix multiplication tensor"
    # with the convention W[k*n+i, r].
    #
    # For matrix multiplication C = AB where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # The bilinear algorithm computes:
    #   M_r = sum_{i,j} U[i*m+j, r] * A[i,j]
    #   N_r = sum_{j,k} V[j*p+k, r] * B[j,k]
    #   C[i,k] = sum_r W[k*n+i, r] * M_r * N_r
    #
    # Expanding: C[i,k] = sum_r W[k*n+i, r] * (sum_{j1} U[i*m+j1, r] A[i,j1]) * (sum_{j2} V[j2*p+k, r] B[j2,k])
    #
    # = sum_r sum_{j1,j2} W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r] A[i,j1] B[j2,k]
    #
    # For this to equal sum_j A[i,j] B[j,k] for ALL A and B, we need:
    #   sum_r W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r] = delta_{j1,j2}
    #
    # NOT 1 for all (i,j,k)! It's the Kronecker delta delta_{j1,j2}!
    #
    # This is the KEY distinction. The tensor is NOT all-ones. It's:
    #   T[i, j1, j2, k] = delta_{j1,j2}
    #
    # Or equivalently, the "multiplication tensor" has the property that
    # for each (i,k), the m×m matrix T_{(i,k)} is the identity matrix.
    #
    # So the condition is:
    #   sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] = delta_{j1,j2}
    #   for all i, k, j1, j2.
    #
    # This is 2 * 7 * 4 * 4 = 224 equations!
    # And the tensor rank is the rank of the matmul tensor, which is
    # NOT the rank of the all-ones tensor.
    #
    # The all-ones rank-1 construction does NOT work here because
    # sum_r 1*1*1 = 1 ≠ delta_{j1,j2} when j1 ≠ j2.
    #
    # So we're back to the original problem. The Strassen-based rank 46
    # is the correct approach. Let me just use that.
    # ------------------------------------------------------------------

    # Use the Strassen-based construction (rank 46):
    U_s, V_s, W_s = _build_strassen_based(n, m, p)
    if _verify(U_s, V_s, W_s, n, m, p):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Fallback: schoolbook (rank 56)
    U_f, V_f, W_f = _build_schoolbook(n, m, p)
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}


def _build_strassen_based(n, m, p):
    """Build a Strassen-based decomposition for 2x4x7 matmul.

    Strategy:
    - A (2x4) = [A1 | A2] where A1, A2 are 2x2
    - B (4x7) = [B1; B2] where B1, B2 are 2x7
    - C = A1*B1 + A2*B2
    - For each 2x2 * 2x7 product, split 7 cols into 2+2+2+1
    - 2x2*2x2 uses Strassen (rank 7), 2x2*2x1 uses schoolbook (rank 2)
    - Total: 2 * (3*7 + 1*2) = 46
    """
    # Strassen 2x2x2
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)

    # M1 = (A00+A11)(B01+B10)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[1, 0] = 1; V_s[2, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    # M2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    # M3 = A00(B00-B10)
    U_s[0, 2] = 1
    V_s[0, 2] = 1; V_s[2, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    # M4 = A11(B11-B01)
    U_s[3, 3] = 1
    V_s[3, 3] = 1; V_s[1, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    # M5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # Schoolbook 2x2x1 (rank 2)
    U_1 = np.zeros((4, 2), dtype=int)
    V_1 = np.zeros((2, 2), dtype=int)
    W_1 = np.zeros((2, 2), dtype=int)
    for j in range(2):
        for i in range(2):
            U_1[i * 2 + j, j] = 1
        V_1[j, j] = 1
        for i in range(2):
            W_1[i, j] = 1

    total_rank = 46
    U = np.zeros((8, total_rank), dtype=int)
    V = np.zeros((28, total_rank), dtype=int)
    W = np.zeros((14, total_rank), dtype=int)

    col = 0
    for block in range(2):
        # A cols: 2*block, 2*block+1
        # B rows: 2*block, 2*block+1
        # B cols: 0..6, split as 2,2,2,1

        # Three 2x2x2 Strassen blocks
        for sub in range(3):
            for r in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + (2 * block + j), col] += U_s[i * 2 + j, r]
                # V
                for j in range(2):
                    for k in range(2):
                        V[(2 * block + j) * 7 + (sub * 2 + k), col] += V_s[j * 2 + k, r]
                # W
                for k in range(2):
                    for i in range(2):
                        W[(sub * 2 + k) * 2 + i, col] += W_s[k * 2 + i, r]
                col += 1

        # One 2x2x1 schoolbook block (col 6)
        for r in range(2):
            for i in range(2):
                for j in range(2):
                    U[i * 4 + (2 * block + j), col] += U_1[i * 2 + j, r]
            for j in range(2):
                V[(2 * block + j) * 7 + 6, col] += V_1[j, r]
            for i in range(2):
                W[6 * 2 + i, col] += W_1[i, r]
            col += 1

    return U, V, W


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j1,r] V[j2*p+k,r] W[k*n+i,r] = delta_{j1,j2}
    for all i, k, j1, j2."""
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j1, :], V[j2 * p + k, :] * W[k * n + i, :])
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True
