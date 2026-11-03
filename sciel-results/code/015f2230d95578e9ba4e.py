"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    schoolbook_rank = n * m * p  # 80

    # Build schoolbook as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank

    deadline = time.time() + 140

    def unflatten_factors(x, n, m, p, rank):
        u_size = n * m * rank
        v_size = m * p * rank
        U = x[:u_size].reshape(n * m, rank)
        V = x[u_size:u_size + v_size].reshape(m * p, rank)
        W = x[u_size + v_size:].reshape(p * n, rank)
        return U, V, W

    def flatten_factors(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def objective_fast(x, n, m, p, rank):
        u_size = n * m * rank
        v_size = m * p * rank
        U = x[:u_size].reshape(n * m, rank)
        V = x[u_size:u_size + v_size].reshape(m * p, rank)
        W = x[u_size + v_size:].reshape(p * n, rank)

        error = 0.0
        for i in range(n):
            for j in range(m):
                ui = U[i * m + j]
                for k in range(p):
                    vk = V[j * p + k]
                    wi = W[k * n + i]
                    val = np.sum(ui * vk * wi)
                    error += (val - 1.0) ** 2
        return error

    def verify(U, V, W, n, m, p, tol=1e-6):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err += (val - 1.0) ** 2
        return err < tol

    def round_to_half(arr):
        return np.round(arr * 2) / 2

    def to_list_with_ints(arr):
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_int) < 1e-6:
                    row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-6:
                    row.append(nearest_half)
                else:
                    row.append(round(val * 2) / 2)
            result.append(row)
        return result

    # Structural change: syndrome-based parity-check strategy.
    # Model the 80 bilinear monomials as a linear code over half-integers.
    # The "syndrome" of a decomposition at rank r is the residual tensor
    # T - sum_c U_c ⊗ V_c ⊗ W_c. We seek rank r where the syndrome vanishes.
    #
    # Key insight: the matmul tensor T[i,j,k]=1 has massive linear structure.
    # Specifically, for fixed j, the slice T[:,j,:] is all-ones (n×p matrix).
    # This means we can exploit the fact that the tensor lives in a low-
    # dimensional subspace when viewed along certain contractions.
    #
    # Strategy: Build a "parity-check" matrix H from the linear dependencies
    # among the 80 standard basis vectors of the tensor space. Then search
    # for a generator matrix G (the factor matrices) such that H @ G = 0
    # (in the appropriate tensor sense), yielding a valid decomposition.
    #
    # Concretely: we use a Kronecker-embedded Strassen-like construction
    # as the initial point, then refine via ALS (alternating least squares)
    # which is much more effective than generic nonlinear optimization
    # for tensor decompositions.

    # --- Build a Kronecker-embedded Strassen baseline ---
    # For 4x4 * 4x5: split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x2 + 2x1
    # Use Strassen (rank 7) for the 2x2*2x2 part, and schoolbook for the remainder.
    #
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where B00,B10 are 2x2, B01,B11 are 2x2, B02,B12 are 2x1
    # C = A*B, each C block is 2x5
    #
    # C00 = A00*B00 + A01*B10  (2x2 * 2x2 + 2x2 * 2x2)
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12  (2x2 * 2x1 + 2x2 * 2x1)
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12

    # For each 2x2*2x2 product, Strassen uses 7 multiplications.
    # There are 4 such products (C00, C01, C10, C11 each involve two 2x2*2x2 terms).
    # But Strassen computes M1..M7 from combined inputs, so we can share.
    #
    # Actually, let's use a simpler approach: embed Strassen for the 4x4*4x4 part
    # and handle the 4x4*4x1 part with schoolbook (4 multiplications).
    # Strassen 4x4*4x4 has rank 49 (7^2 from 2-level Strassen, or 28 from 1-level on 2x2 blocks).
    # 1-level Strassen on 2x2 blocks: 7 multiplications of 2x2 matrices = 7*4 = 28 rank-1 terms.
    # Plus 4x4*4x1: 4 rank-1 terms. Total: 32.
    #
    # Wait, that's too optimistic. Let me be more careful.
    #
    # 4x4 * 4x4 via 1-level Strassen (2x2 blocks):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # Each Mi is a 2x2 matrix product, requiring 4 rank-1 terms.
    # Total: 7*4 = 28 rank-1 terms for 4x4*4x4.
    #
    # For 4x4*4x1: schoolbook, 4 rank-1 terms.
    # Total: 32 rank-1 terms.
    #
    # But this requires the B matrix to be partitioned as 4x4 + 4x1, and the
    # Strassen formulas assume specific block structures. Let's construct it.

    # Actually, let's try a different structural approach that's cleaner:
    # Use the "syndrome" idea directly. The tensor T[i,j,k]=1 satisfies:
    # For any fixed (i,k), T[i,j,k] is constant in j. This means the tensor
    # has rank at most n*p = 20 in the "i-k" slice sense. But the full
    # tensor rank is higher.
    #
    # Let's just try ALS from multiple structured initializations.

    def als_decompose(n, m, p, rank, U0, V0, W0, max_iter=50, tol=1e-10, deadline=None):
        """Alternating least squares for tensor decomposition."""
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()

        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break

            # Update U: for each row i*m+j, solve least squares
            for i in range(n):
                for j in range(m):
                    # U[i*m+j, c] * sum_k V[j*p+k,c] * W[k*n+i,c] = 1
                    # This is not linear in U alone. Instead:
                    # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for each k
                    # Let's define for fixed (i,j): we need sum_c U[i*m+j,c] * X[c] = 1
                    # where X[c] = V[j*p+k,c] * W[k*n+i,c]... but this depends on k.
                    #
                    # Actually, ALS for tensor decomposition is:
                    # Fix V, W, update U: U[i*m+j, :] should satisfy
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # This is a system of p equations in rank unknowns for each (i,j).
                    # Let A_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Then A_ij @ U[i*m+j, :] = ones(p)

                    A_ij = np.zeros((p, rank))
                    for k in range(p):
                        A_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A_ij @ u = ones(p)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                        U[i * m + j, :] = u_sol
                    except:
                        pass

            # Update V: for each row j*p+k
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                    # A_jk[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    A_jk = np.zeros((n, rank))
                    for i in range(n):
                        A_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A_jk, np.ones(n), rcond=None)
                        V[j * p + k, :] = v_sol
                    except:
                        pass

            # Update W: for each row k*n+i
            for k in range(p):
                for i in range(n):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                    A_ki = np.zeros((m, rank))
                    for j in range(m):
                        A_ki[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W[k * n + i, :] = w_sol
                    except:
                        pass

            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err += (val - 1.0) ** 2
            if err < tol:
                break

        return U, V, W, err

    # Try ranks from a reasonable starting point downward
    # Lower bound: max(n*m, m*p, n*p) = max(16, 20, 20) = 20
    # Try from 35 down to 25, with multiple random restarts

    for target_rank in range(35, 24, -1):
        if time.time() > deadline - 30:
            break

        found = False
        n_attempts = 5 if target_rank >= 30 else 3

        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break

            np.random.seed(attempt * 1000 + target_rank)

            # Random initialization with small values
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1

            # Also try initialization from truncated schoolbook
            if target_rank <= schoolbook_rank:
                U_init2 = U_school[:, :target_rank].copy()
                V_init2 = V_school[:, :target_rank].copy()
                W_init2 = W_school[:, :target_rank].copy()
                U_init2 += np.random.randn(n * m, target_rank) * 0.05
                V_init2 += np.random.randn(m * p, target_rank) * 0.05
                W_init2 += np.random.randn(p * n, target_rank) * 0.05

                U_a, V_a, W_a, err_a = als_decompose(n, m, p, target_rank, U_init2, V_init2, W_init2, max_iter=80, deadline=deadline - 15)
                if err_a < 1e-8:
                    U_r = round_to_half(U_a)
                    V_r = round_to_half(V_a)
                    W_r = round_to_half(W_a)
                    if verify(U_r, V_r, W_r, n, m, p):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break

            U_b, V_b, W_b, err_b = als_decompose(n, m, p, target_rank, U_init, V_init, W_init, max_iter=80, deadline=deadline - 15)
            if err_b < 1e-8:
                U_r = round_to_half(U_b)
                V_r = round_to_half(V_b)
                W_r = round_to_half(W_b)
                if verify(U_r, V_r, W_r, n, m, p):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break

        if found:
            break

    # Final fallback: try a few more ranks with generic optimization if ALS didn't find anything
    if best_rank == schoolbook_rank:
        for target_rank in [40, 38, 36]:
            if time.time() > deadline - 20:
                break

            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                np.random.seed(attempt * 777 + target_rank)
                U_init = np.random.randn(n * m, target_rank) * 0.1
                V_init = np.random.randn(m * p, target_rank) * 0.1
                W_init = np.random.randn(p * n, target_rank) * 0.1
                x_init = flatten_factors(U_init, V_init, W_init)

                try:
                    result = minimize(
                        objective_fast, x_init,
                        args=(n, m, p, target_rank),
                        method='BFGS',
                        options={'maxiter': 300, 'gtol': 1e-8}
                    )
                    if result.fun < 1e-8:
                        U_f, V_f, W_f = unflatten_factors(result.x, n, m, p, target_rank)
                        U_r = round_to_half(U_f)
                        V_r = round_to_half(V_f)
                        W_r = round_to_half(W_f)
                        if verify(U_r, V_r, W_r, n, m, p):
                            best_U, best_V, best_W = U_r, V_r, W_r
                            best_rank = target_rank
                            break
                except:
                    pass
            if best_rank < schoolbook_rank:
                break

    # Convert to lists
    U_list = to_list_with_ints(best_U)
    V_list = to_list_with_ints(best_V)
    W_list = to_list_with_ints(best_W)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
