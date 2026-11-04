"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    schoolbook_rank = n * m * p  # 96

    # ------------------------------------------------------------------
    # Build the schoolbook solution (rank 96)
    # ------------------------------------------------------------------
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]
        T_recon = np.zeros((n, m, p))
        for c in range(rank):
            T_recon += np.einsum('a,b,c->abc',
                                 U_arr.reshape(n, m, rank),
                                 V_arr.reshape(m, p, rank),
                                 W_arr.reshape(p, n, rank))
        return bool(np.allclose(T_recon, 1.0, atol=tol))

    # ------------------------------------------------------------------
    # Structural Strassen decomposition on the m=4 dimension.
    #
    # Idea: split j in {0,1,2,3} into two 2-vectors:
    #   jA = (j0, j1), jB = (j2, j3)
    # For a fixed i and a fixed 2-column k-block kC = (k0, k1), the
    # 2x2 multiplication jA * kC (and jB * kC) can be done with 7
    # Strassen multiplications instead of 4.  The 7 Strassen intermediates
    # depend only on (i, k-block) and the *sums/differences* of the
    # corresponding j-entries, so they can be SHARED across the two
    # j-halves when we process them together.
    #
    # Concretely, for each i (3) and each k-block (4 blocks of 2), we
    # replace the 4 schoolbook terms (j=0..3, one per j) by 7 Strassen
    # terms.  That is 3*4 = 12 groups, each replacing 4 terms with 7,
    # i.e. +3 per group -> rank = 96 - 12*... wait, 4->7 is an INCREASE.
    #
    # The real saving comes from sharing: the 7 Strassen terms for the
    # jA-half and the 7 for the jB-half share the same 7 "W" factors
    # (the k-side and i-side are identical), so we only need 7 columns
    # per (i, k-block) instead of 4+4=8.  Net: 8 -> 7 per (i, k-block),
    # saving 1 per group, 12 groups -> rank 96 - 12 = 84?  Let's just
    # build it and verify; if it doesn't verify we fall back.
    # ------------------------------------------------------------------

    def build_strassen_decomposition():
        """Build a Strassen-based decomposition.  Returns (U_list, V_list, W_list, rank)."""
        # Strassen 2x2: C = A*B, A,B 2x2.
        # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b10)
        # m7 = (a01-a11)*(b01+b11)
        # C00 = m1+m4-m5+m7
        # C01 = m3+m5
        # C10 = m2+m4
        # C11 = m1-m2+m3+m6
        #
        # In our tensor, for fixed i and fixed k-block (k0,k1):
        #   The "A" matrix entries are the U-factors for j in the j-half
        #   and the "B" matrix entries are the V-factors for k in the k-block.
        #   The "C" entries are the W-factors for (i, k).
        #
        # We split j into jA={0,1} and jB={2,3}.  For each (i, k-block),
        # we create 7 columns.  In column c (Strassen step s):
        #   U[i*m + j, c] = coefficient of j in step s  (for j in the half)
        #   V[j*p + k, c] = coefficient of k in step s  (for k in the block)
        #   W[k*n + i, c] = coefficient of (i,k) in step s
        #
        # The tricky part: U and V factors must be consistent.  In the
        # schoolbook, U[i*m+j, c]=1 and V[j*p+k, c]=1 for the matching
        # (j,k).  In Strassen, we combine multiple (j,k) pairs.  The
        # tensor entry T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
        #
        # For a fixed (i, k-block), the 7 Strassen columns each have:
        #   U-factors: a 2-vector over jA (or jB)
        #   V-factors: a 2-vector over the k-block
        #   W-factor: a scalar for each k in the block (and i)
        #
        # But U[i*m+j,c] is a single number per (j,c), and V[j*p+k,c] is
        # a single number per (j,k,c).  For the tensor to work, for a
        # fixed c, the product U[i*m+j,c]*V[j*p+k,c] must equal the
        # Strassen "A_j * B_k" product for the relevant (j,k).  This is
        # only possible if U and V factor the A*B product, which in
        # general they don't (A*B is a 2x2 matrix, not a rank-1 matrix).
        #
        # So the simple Strassen sharing does NOT directly give a valid
        # tensor decomposition.  We need a different approach.
        #
        # Alternative: use the fact that the target tensor T[i,j,k]=1 is
        # a rank-1 tensor in (i,j,k) when viewed as T = 1_i * 1_j * 1_k.
        # The matrix multiplication tensor is NOT rank 1; it's the
        # identity-like tensor.  So we can't use rank-1 tricks.
        #
        # Given the difficulty, let's fall back to a careful greedy
        # reduction with linear-solve compensation.
        return None

    # ------------------------------------------------------------------
    # Greedy reduction with linear-solve compensation.
    #
    # To remove term c0, we need to find adjustments to the remaining
    # terms such that the tensor is still reconstructed.  This is a
    # nonlinear problem, but we can linearize: fix U and V, solve for W,
    # or use a first-order perturbation.
    #
    # A simpler approach: try removing term c0 and check if the remaining
    # terms can be *scaled* to compensate.  This is a linear feasibility
    # problem in the W-factors (fixing U and V).
    # ------------------------------------------------------------------

    def try_greedy_reduction(U_arr, V_arr, W_arr, current_rank, max_iterations=30):
        """Try to reduce rank by greedily removing terms and solving for W."""
        U_curr = U_arr.copy()
        V_curr = V_arr.copy()
        W_curr = W_arr.copy()
        r = current_rank

        for iteration in range(max_iterations):
            if time.time() > deadline - 15.0:
                break

            best_remove_idx = -1
            best_new_W = None

            for c_remove in range(r):
                if time.time() > deadline - 15.0:
                    break

                # Remove column c_remove
                mask = [c for c in range(r) if c != c_remove]
                if not mask:
                    break

                U_red = U_curr[:, mask]
                V_red = V_curr[:, mask]
                W_red = W_curr[:, mask].copy()

                # Build the linear system: for each (i,j,k),
                #   sum_{c in mask} U_red[i*m+j,c] * V_red[j*p+k,c] * W_red[k*n+i,c] = 1
                # This is linear in W_red.  Build the matrix A where
                # A[(i,j,k), (k',c)] = U_red[i*m+j,c] * V_red[j*p+k,c] * delta(k'=k')
                # Actually, for a fixed (i,j,k), the equation is:
                #   sum_c U_red[i*m+j,c] * V_red[j*p+k,c] * W_red[k*n+i,c] = 1
                # The unknowns are W_red[k'*n+i, c] for all k', c.
                # For a fixed (i,j,k), only W_red[k*n+i, c] appears.
                # So the equation involves only the W-entries with that specific (k,i).
                #
                # This means the system is block-diagonal by (k,i)!  For each
                # (k,i), the equations are over (j) and (c):
                #   sum_c U_red[i*m+j,c] * V_red[j*p+k,c] * W_red[k*n+i,c] = 1  for all j
                # This is a linear system in W_red[k*n+i, c] with m=4 equations
                # and r-1 unknowns.  We can solve this least-squares.

                feasible = True
                for i in range(n):
                    for k in range(p):
                        # Build A: m x (r-1), A[j, c_idx] = U_red[i*m+j,c] * V_red[j*p+k,c]
                        A = np.zeros((m, len(mask)))
                        for j in range(m):
                            for c_idx, c in enumerate(mask):
                                A[j, c_idx] = U_red[i*m+j, c] * V_red[j*p+k, c]
                        b = np.ones(m)
                        # Solve A @ x = b in least squares
                        x, residuals, rank_l, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if the solution is exact
                        if np.linalg.norm(A @ x - b) > 1e-8:
                            feasible = False
                            break
                        W_red[k*n+i, :] = x
                    if not feasible:
                        break

                if feasible:
                    # Verify the full decomposition
                    if verify_decomposition(U_red, V_red, W_red, tol=1e-8):
                        best_remove_idx = c_remove
                        best_new_U = U_red
                        best_new_V = V_red
                        best_new_W = W_red
                        break

            if best_remove_idx >= 0:
                U_curr = best_new_U
                V_curr = best_new_V
                W_curr = best_new_W
                r -= 1
            else:
                break

        return U_curr, V_curr, W_curr, r

    # Try greedy reduction
    if time.time() < deadline - 30.0:
        U_reduced, V_reduced, W_reduced, new_rank = try_greedy_reduction(
            U_np, V_np, W_np, schoolbook_rank, max_iterations=30
        )
        if new_rank < best_rank and verify_decomposition(U_reduced, V_reduced, W_reduced, tol=1e-9):
            best_result = {
                "U": U_reduced.tolist(),
                "V": V_reduced.tolist(),
                "W": W_reduced.tolist()
            }
            best_rank = new_rank

    return best_result
# EVOLVE-BLOCK-END
