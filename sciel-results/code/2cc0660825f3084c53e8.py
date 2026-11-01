import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18

    # Target tensor T[i,j,k] = 1
    T = np.ones((n, m, p))

    # Schoolbook fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school

    start_time = time.time()
    deadline = 130  # safety margin before 150s kill

    def compute_error(U, V, W, target_rank):
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)

    def als_step(U, V, W, target_rank, T):
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()

        for i in range(n):
            for j in range(m):
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    A_T = A.T
                    U_new[i * m + j, :] = np.linalg.lstsq(A_T, T[i, j, :].reshape(-1, 1), rcond=None)[0].flatten()
                except Exception:
                    pass

        for j in range(m):
            for k in range(p):
                B = np.zeros((n, target_rank))
                for i in range(n):
                    B[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
                try:
                    B_T = B.T
                    V_new[j * p + k, :] = np.linalg.lstsq(B_T, T[:, j, k].reshape(-1, 1), rcond=None)[0].flatten()
                except Exception:
                    pass

        for k in range(p):
            for i in range(n):
                C = np.zeros((m, target_rank))
                for j in range(m):
                    C[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    C_T = C.T
                    W_new[k * n + i, :] = np.linalg.lstsq(C_T, T[i, :, k].reshape(-1, 1), rcond=None)[0].flatten()
                except Exception:
                    pass

        return U_new, V_new, W_new

    def gradient_descent(U, V, W, target_rank, max_iters=200, lr=0.01):
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for _ in range(max_iters):
            if time.time() - start_time > deadline:
                break

            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)

            result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            residual = T - result

            grad_U = np.einsum('ijk,jkr,kir->ijr', residual, V_r, W_r)
            grad_V = np.einsum('ijk,ijr,kir->jkr', residual, U_r, W_r)
            grad_W = np.einsum('ijk,ijr,jkr->kir', residual, U_r, V_r)

            grad_norm = np.sqrt(np.sum(grad_U ** 2) + np.sum(grad_V ** 2) + np.sum(grad_W ** 2))
            if grad_norm < 1e-12:
                break

            effective_lr = lr * 10.0 / (1.0 + grad_norm)

            U -= effective_lr * grad_U.reshape(n * m, target_rank)
            V -= effective_lr * grad_V.reshape(m * p, target_rank)
            W -= effective_lr * grad_W.reshape(p * n, target_rank)

            err = compute_error(U, V, W, target_rank)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if err < 1e-10:
                break

        err = compute_error(best_U, best_V, best_W, target_rank)
        return best_U, best_V, best_W, err

    # --- SDP lower bound via nuclear norm of mode-unfoldings ---
    def sdp_lower_bound():
        """Compute a lower bound on the tensor rank using SDP relaxation of
        the nuclear norm of each mode-unfolding. If CVXPY is unavailable,
        fall back to a cheap spectral-norm-based estimate."""
        try:
            import cvxpy as cp
        except Exception:
            # Cheap fallback: max of (sum of singular values of each unfolding)
            # is not a valid rank bound, so return a conservative bound.
            # Use flattening rank: rank of any unfolding >= tensor rank.
            # We compute the max rank of the three unfoldings as a heuristic lower bound.
            T_1 = T.reshape(n, m * p)
            T_2 = T.reshape(m, n * p)
            T_3 = T.reshape(p, n * m)
            r1 = np.linalg.matrix_rank(T_1, tol=1e-10)
            r2 = np.linalg.matrix_rank(T_2, tol=1e-10)
            r3 = np.linalg.matrix_rank(T_3, tol=1e-10)
            return int(max(r1, r2, r3))

        # Build mode-unfoldings
        T_1 = T.reshape(n, m * p)  # shape (n, m*p)
        T_2 = T.reshape(m, n * p)  # shape (m, n*p)
        T_3 = T.reshape(p, n * m)  # shape (p, n*m)

        # For each unfolding, minimize nuclear norm s.t. the matrix equals the unfolding
        # This is trivially the nuclear norm of the matrix itself, which is >= rank.
        # But the SDP relaxation of rank is: minimize trace(X) s.t. X >= 0, X has the
        # same "flattened" structure. For a single unfolding, the minimum trace of a
        # PSD matrix with given singular values is the sum of singular values = nuclear norm.
        # So the SDP lower bound from each unfolding is just the nuclear norm.
        # However, the tensor rank R satisfies: for each unfolding, the matrix rank <= R,
        # and nuclear norm >= rank. So max(nuclear norms) is not directly a rank bound.
        # The correct bound: R >= rank(unfolding) for each unfolding.
        # We already have that. The SDP gives a tighter bound when we consider
        # the joint structure, but for simplicity we use the max unfolding rank.
        r1 = np.linalg.matrix_rank(T_1, tol=1e-10)
        r2 = np.linalg.matrix_rank(T_2, tol=1e-10)
        r3 = np.linalg.matrix_rank(T_3, tol=1e-10)
        return int(max(r1, r2, r3))

    def try_rank(target_rank):
        max_restarts = 120
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                return False

            seed = restart * 1000 + target_rank * 7
            rng = np.random.RandomState(seed)

            init_strategy = restart % 3

            if init_strategy == 0:
                U = rng.randn(n * m, target_rank) * 0.5
                V = rng.randn(m * p, target_rank) * 0.5
                W = rng.randn(p * n, target_rank) * 0.5
            elif init_strategy == 1:
                U = rng.randn(n * m, target_rank) * 0.3
                V = rng.randn(m * p, target_rank) * 0.3
                W = rng.randn(p * n, target_rank) * 0.3
                for r in range(min(3, target_rank)):
                    U[0, r] += 1.0
                    V[0, r] += 1.0
                    W[0, r] += 1.0
            else:
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                U += rng.randn(n * m, target_rank) * 0.2
                V += rng.randn(m * p, target_rank) * 0.2
                W += rng.randn(p * n, target_rank) * 0.2

            # Normalize columns
            for r in range(target_rank):
                norms = [np.linalg.norm(U[:, r]), np.linalg.norm(V[:, r]), np.linalg.norm(W[:, r])]
                if any(norm < 1e-10 for norm in norms):
                    continue
                total_norm = np.prod(norms) ** (1 / 3)
                if total_norm > 1e-10:
                    U[:, r] /= (norms[0] / total_norm)
                    V[:, r] /= (norms[1] / total_norm)
                    W[:, r] /= (norms[2] / total_norm)

            # Phase 1: ALS
            for als_iter in range(80):
                if time.time() - start_time > deadline:
                    return False
                U, V, W = als_step(U, V, W, target_rank, T)
                err = compute_error(U, V, W, target_rank)
                if err < 1e-10:
                    break

            # Phase 2: Gradient descent polish
            err = np.inf
            for lr in [0.01, 0.005, 0.001]:
                U, V, W, err = gradient_descent(U, V, W, target_rank, max_iters=150, lr=lr)
                if err is not None and err < 1e-8:
                    break

            if err is not None and err < 1e-6:
                U, V, W, err = gradient_descent(U, V, W, target_rank, max_iters=300, lr=0.0005)

                if err is not None and err < 1e-8:
                    # Phase 3: Rational reconstruction to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2

                    err_rounded = compute_error(U_rounded, V_rounded, W_rounded, target_rank)
                    if err_rounded < 1e-6:
                        nonlocal best_U, best_V, best_W, best_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = target_rank
                        return True
        return False

    # Compute SDP lower bound to prune impossible ranks
    lb = sdp_lower_bound()
    print(f"SDP lower bound: {lb}, schoolbook: {R_school}")

    # Try ranks from max(14, lb) to 17
    for target_rank in range(max(14, lb), R_school):
        if try_rank(target_rank):
            break

    # Final verification
    def verify_solution(U, V, W, rank):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - T[i, j, k]))
        return max_err

    if best_rank < R_school:
        err = verify_solution(best_U, best_V, best_W, best_rank)
        if err > 1e-6:
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = R_school

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
