"""Strategy: direct rank-minimization via structured tensor decomposition search.
Instead of composing known algorithms (Strassen/schoolbook) and merging, we
treat the matrix-multiplication tensor as a 3D array and search for its
rank-minimal decomposition using:
1. A fast structured baseline (Strassen blocks) to get rank ~64.
2. Iterative 'term replacement' search: try replacing small groups of rank-1
   terms with a smaller group that spans the same tensor contribution.
3. Randomized perturbation + rounding to half-integers to escape local minima.
4. Linear-algebraic pruning: identify linearly dependent rank-1 terms and
   eliminate them.
"""
import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 20 x rank
    # W: (p*n) x rank = 20 x rank
    # Convention: W[k*n+i, r]

    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s kill

    def verify(U, V, W):
        """Check if the decomposition is exact using numpy for speed."""
        rank = len(U)
        if rank == 0:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # For each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def verify_fast(U, V, W):
        """Vectorized verification."""
        rank = len(U)
        if rank == 0:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Build tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # Use einsum
        # U_arr shape: (16, rank), V_arr: (20, rank), W_arr: (20, rank)
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a 3-tensor contraction
        # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        T = np.einsum('ijr,jkr,ki r->ijk'.replace(' ', ''), U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)

    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition as numpy arrays.
        U: 4x7, V: 4x7, W: 4x7
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # r=0: M1 = (A11+A22)*(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # r=1: M2 = (A21+A22)*B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1

        # r=2: M3 = A11*(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1

        # r=3: M4 = A22*(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1

        # r=4: M5 = (A11+A12)*B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1

        # r=5: M6 = (A21-A11)*(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # r=6: M7 = (A12-A22)*(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1

        return U, V, W

    def build_strassen_block():
        """Build 4x4*4x5 using Strassen for 2x2*2x2 and schoolbook for 2x2*2x3."""
        sU, sV, sW = strassen_2x2()
        U_list = []
        V_list = []
        W_list = []

        def add_strassen(a_map, b_map, c_map):
            for r in range(7):
                U_new = np.zeros(16)
                V_new = np.zeros(20)
                W_new = np.zeros(20)
                for si, val in enumerate(sU[:, r]):
                    if val != 0:
                        U_new[a_map[si]] = val
                for sj, val in enumerate(sV[:, r]):
                    if val != 0:
                        V_new[b_map[sj]] = val
                for sk, val in enumerate(sW[:, r]):
                    if val != 0:
                        W_new[c_map[sk]] = val
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)

        def add_schoolbook_2x2x2x3(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = np.zeros(16)
                        V_new = np.zeros(20)
                        W_new = np.zeros(20)
                        U_new[(a_row_start+i)*4 + (a_col_start+j)] = 1
                        V_new[(b_row_start+j)*5 + (b_col_start+k)] = 1
                        W_new[(c_col_start+k)*4 + (c_row_start+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

        # Block (1,1): C11 += A11*B11
        add_strassen([0, 1, 4, 5], [0, 1, 5, 6], [0, 1, 4, 5])
        # Block (1,2): C12 += A11*B12
        add_schoolbook_2x2x2x3(0, 0, 0, 2, 0, 2)
        # Block (2,1): C21 += A21*B11
        add_strassen([8, 9, 12, 13], [0, 1, 5, 6], [2, 3, 6, 7])
        # Block (2,2): C22 += A21*B12
        add_schoolbook_2x2x2x3(2, 0, 0, 2, 2, 2)
        # Block (3,1): C11 += A12*B21
        add_strassen([2, 3, 6, 7], [10, 11, 15, 16], [0, 1, 4, 5])
        # Block (3,2): C12 += A12*B22
        add_schoolbook_2x2x2x3(0, 2, 2, 2, 0, 2)
        # Block (4,1): C21 += A22*B21
        add_strassen([10, 11, 14, 15], [10, 11, 15, 16], [2, 3, 6, 7])
        # Block (4,2): C22 += A22*B22
        add_schoolbook_2x2x2x3(2, 2, 2, 2, 2, 2)

        return np.array(U_list), np.array(V_list), np.array(W_list)

    # Start with Strassen block decomposition
    U, V, W = build_strassen_block()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = len(U)

    if not verify_fast(best_U, best_V, best_W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((rank, 16))
        V = np.zeros((rank, 20))
        W = np.zeros((rank, 20))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[col, i*m+j] = 1
                    V[col, j*p+k] = 1
                    W[col, k*n+i] = 1
                    col += 1
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank

    # Strategy: iterative term elimination and replacement
    # Key insight: if we have a set of rank-1 tensors that sum to the matmul tensor,
    # we can try to find linear dependencies among them and eliminate redundant terms.

    def get_tensor(U_arr, V_arr, W_arr):
        """Compute the full 3D tensor from factor matrices."""
        rank = len(U_arr)
        U3 = U_arr.T.reshape(n, m, rank)  # (n, m, rank)
        V3 = V_arr.T.reshape(m, p, rank)  # (m, p, rank)
        W3 = W_arr.T.reshape(p, n, rank)  # (p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return T

    def try_eliminate_dependent_terms(U_arr, V_arr, W_arr, max_rounds=10):
        """Try to eliminate linearly dependent rank-1 terms."""
        U_curr = U_arr.copy()
        V_curr = V_arr.copy()
        W_curr = W_arr.copy()
        rank = len(U_curr)
        if rank < 2:
            return U_curr, V_curr, W_curr

        for round_num in range(max_rounds):
            if time.time() > start_time + deadline:
                break

            # Build the tensor
            T = get_tensor(U_curr, V_curr, W_curr)

            # For each term r, check if it can be expressed as a linear combination
            # of the other terms. This is equivalent to checking if the rank-1 tensor
            # u_r v_r^T w_r^T (as a 3D array) lies in the span of the others.
            # A cheaper heuristic: check if removing term r and adjusting others
            # can still reproduce T.

            # Build the "term tensor" for each r: T_r[i,j,k] = U_curr[r,i]*V_curr[r,j]*W_curr[r,k]
            # We want to find a subset of terms whose sum equals T.

            # Use a greedy approach: try removing each term and see if the residual
            # can be absorbed by scaling others.

            removed_any = False
            for r_remove in range(len(U_curr)):
                if time.time() > start_time + deadline:
                    break

                U_test = np.delete(U_curr, r_remove, axis=0)
                V_test = np.delete(V_curr, r_remove, axis=0)
                W_test = np.delete(W_curr, r_remove, axis=0)

                if len(U_test) == 0:
                    continue

                T_test = get_tensor(U_test, V_test, W_test)
                residual = T - T_test

                # The residual should be the removed term: U_curr[r_remove] outer V_curr[r_remove] outer W_curr[r_remove]
                # Check if residual is zero (meaning the term was redundant)
                if np.allclose(residual, 0, atol=1e-9):
                    # Term is redundant, remove it
                    U_curr = U_test
                    V_curr = V_test
                    W_curr = W_test
                    removed_any = True
                    break

            if not removed_any:
                break

        return U_curr, V_curr, W_curr

    # Try elimination
    U_elim, V_elim, W_elim = try_eliminate_dependent_terms(best_U, best_V, best_W)
    if len(U_elim) < best_rank and verify_fast(U_elim, V_elim, W_elim):
        best_U, best_V, best_W = U_elim, V_elim, W_elim
        best_rank = len(U_elim)

    # Strategy 2: Randomized perturbation + rounding
    # Perturb the factor matrices slightly, then round to half-integers, and verify.
    # If verification passes, we may have found a different valid decomposition.

    def random_perturbation_round(U_arr, V_arr, W_arr, n_trials=100):
        """Try random perturbations and round to half-integers."""
        best_local = (U_arr.copy(), V_arr.copy(), W_arr.copy(), len(U_arr))

        for trial in range(n_trials):
            if time.time() > start_time + deadline:
                break

            # Random perturbation
            scale = 0.1 * (1 + np.random.rand())
            U_pert = U_arr + scale * np.random.randn(*U_arr.shape)
            V_pert = V_arr + scale * np.random.randn(*V_arr.shape)
            W_pert = W_arr + scale * np.random.randn(*W_arr.shape)

            # Round to half-integers
            U_round = np.round(U_pert * 2) / 2
            V_round = np.round(V_pert * 2) / 2
            W_round = np.round(W_pert * 2) / 2

            # Try to eliminate zero rows (terms that became all-zero after rounding)
            mask_u = np.any(U_round != 0, axis=1)
            mask_v = np.any(V_round != 0, axis=1)
            mask_w = np.any(W_round != 0, axis=1)
            mask = mask_u & mask_v & mask_w
            U_round = U_round[mask]
            V_round = V_round[mask]
            W_round = W_round[mask]

            if len(U_round) < best_local[3] and verify_fast(U_round, V_round, W_round):
                best_local = (U_round.copy(), V_round.copy(), W_round.copy(), len(U_round))

        return best_local

    if best_rank > 40:
        U_pert, V_pert, W_pert, rank_pert = random_perturbation_round(best_U, best_V, best_W)
        if rank_pert < best_rank:
            best_U, best_V, best_W = U_pert, V_pert, W_pert
            best_rank = rank_pert

    # Strategy 3: Structured search for lower-rank decompositions
    # Use the fact that the matmul tensor has a known structure.
    # Try to find a decomposition with rank < best_rank by solving a system of equations.

    def structured_search(target_rank, U_init=None, V_init=None, W_init=None):
        """Try to find a decomposition of given rank using structured initialization."""
        if target_rank >= best_rank:
            return None

        # Initialize with random or structured factors
        if U_init is not None:
            U_s = U_init[:target_rank].copy() if len(U_init) >= target_rank else U_init.copy()
            V_s = V_init[:target_rank].copy() if len(V_init) >= target_rank else V_init.copy()
            W_s = W_init[:target_rank].copy() if len(W_init) >= target_rank else W_init.copy()
        else:
            U_s = np.random.randn(target_rank, 16) * 0.1
            V_s = np.random.randn(target_rank, 20) * 0.1
            W_s = np.random.randn(target_rank, 20) * 0.1

        # Use ALS-like optimization: fix two factors, solve for the third
        # For each term r, the contribution to T[i,j,k] is U[r,i]*V[r,j]*W[r,k]
        # We want sum_r U[r,i]*V[r,j]*W[r,k] = 1 for all (i,j,k)

        # This is a non-convex problem, so we use a heuristic:
        # Alternate between fixing U,V and solving for W, etc.

        for iteration in range(50):
            if time.time() > start_time + deadline:
                break

            # Fix U and V, solve for W
            # For each (k,i), W[r,k*n+i] should satisfy:
            # sum_r U[r,i*m+j] * V[r,j*p+k] * W[r,k*n+i] = 1 for all j
            # This is a linear system in W for fixed (k,i)

            # Build the linear system:
            # For fixed (k,i): sum_j sum_r U[r,i*m+j]*V[r,j*p+k]*W[r,k*n+i] = 1
            # Let A[j,r] = U[r,i*m+j]*V[r,j*p+k], then sum_r A[j,r]*W[r,k*n+i] = 1 for all j
            # This gives m equations in rank unknowns (W[r,k*n+i] for r=0..rank-1)

            # Actually, it's: for each j, sum_r A[j,r] * W[r,k*n+i] = 1
            # So we have a system: A @ W_col = ones(m)
            # where A is m x rank

            W_new = W_s.copy()
            for k in range(p):
                for i in range(n):
                    # Build A matrix: m x rank
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            A_mat[j, r] = U_s[r, i*m+j] * V_s[r, j*p+k]

                    # Solve A_mat @ W_col = ones(m)
                    try:
                        W_col, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        W_s[:, k*n+i] = W_col
                    except:
                        pass

            # Fix V and W, solve for U
            U_new = U_s.copy()
            for i in range(n):
                for j in range(m):
                    # For fixed (i,j): sum_k sum_r U[r,i*m+j]*V[r,j*p+k]*W[r,k*n+i] = 1
                    # Let B[k,r] = V[r,j*p+k]*W[r,k*n+i], then sum_r B[k,r]*U[r,i*m+j] = 1 for all k
                    B_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        for r in range(target_rank):
                            B_mat[k, r] = V_s[r, j*p+k] * W_s[r, k*n+i]
                    try:
                        U_col, residuals, rank_sol, sv = np.linalg.lstsq(B_mat, np.ones(p), rcond=None)
                        U_s[:, i*m+j] = U_col
                    except:
                        pass

            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k): sum_i sum_r U[r,i*m+j]*V[r,j*p+k]*W[r,k*n+i] = 1
                    C_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        for r in range(target_rank):
                            C_mat[i, r] = U_s[r, i*m+j] * W_s[r, k*n+i]
                    try:
                        V_col, residuals, rank_sol, sv = np.linalg.lstsq(C_mat, np.ones(n), rcond=None)
                        V_s[:, j*p+k] = V_col
                    except:
                        pass

        # Round to half-integers
        U_round = np.round(U_s * 2) / 2
        V_round = np.round(V_s * 2) / 2
        W_round = np.round(W_s * 2) / 2

        # Remove zero rows
        mask_u = np.any(U_round != 0, axis=1)
        mask_v = np.any(V_round != 0, axis=1)
        mask_w = np.any(W_round != 0, axis=1)
        mask = mask_u & mask_v & mask_w
        U_round = U_round[mask]
        V_round = V_round[mask]
        W_round = W_round[mask]

        if len(U_round) < best_rank and verify_fast(U_round, V_round, W_round):
            return U_round, V_round, W_round

        return None

    # Try structured search for lower ranks
    for target_rank in range(best_rank - 1, max(best_rank - 10, 30), -1):
        if time.time() > start_time + deadline:
            break

        result = structured_search(target_rank, best_U, best_V, best_W)
        if result is not None:
            U_r, V_r, W_r = result
            if len(U_r) < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = len(U_r)
                break

    # Final verification and conversion to list format
    if not verify_fast(best_U, best_V, best_W):
        # Fallback
        rank = n * m * p
        U = np.zeros((rank, 16))
        V = np.zeros((rank, 20))
        W = np.zeros((rank, 20))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[col, i*m+j] = 1
                    V[col, j*p+k] = 1
                    W[col, k*n+i] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    # Convert to list format with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in best_U.T]
    V_list = [[float(x) for x in row] for row in best_V.T]
    W_list = [[float(x) for x in row] for row in best_W.T]

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
