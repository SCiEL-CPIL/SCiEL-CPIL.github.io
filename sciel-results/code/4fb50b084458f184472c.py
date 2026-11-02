"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42

    def schoolbook():
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    U0, V0, W0 = schoolbook()

    def verify_exact(U, V, W, rank):
        """Verify that the decomposition exactly reproduces the matmul tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m + j, :rank]
                    v_val = V[j*p + k, :rank]
                    w_val = W[k*n + i, :rank]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_integer(U, V, W, rank):
        """Check all entries are integer or half-integer."""
        for mat in [U, V, W]:
            for r in range(rank):
                col = mat[:, r]
                if not np.allclose(col * 2, np.round(col * 2), atol=1e-9):
                    return False
        return True

    def project_to_half_int(arr, rank):
        """Project values to nearest half-integer."""
        out = np.zeros_like(arr)
        for r in range(rank):
            out[:, r] = np.round(arr[:, r] * 2) / 2.0
        return out

    def find_best_rank1_residual(R, max_iter=2000, tol=1e-14, seed=42):
        """Find best rank-1 approximation of 3-tensor R[i,j,k] via alternating minimization.
        
        The rank-1 term is: U[i*m+j] * V[j*p+k] * W[k*n+i]
        We maximize |sum_{i,j,k} R[i,j,k] * U[i*m+j] * V[j*p+k] * W[k*n+i]|
        
        Returns (U_vec, V_vec, W_vec, scalar)
        """
        np.random.seed(seed)
        Uv = np.random.randn(n*m)
        Vv = np.random.randn(m*p)
        Wv = np.random.randn(p*n)
        Uv /= np.linalg.norm(Uv)
        Vv /= np.linalg.norm(Vv)
        Wv /= np.linalg.norm(Wv)

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break

            # Update U: fix V, W, maximize sum_{i,j,k} R[i,j,k]*U[i*m+j]*V[j*p+k]*W[k*n+i]
            U_new = np.zeros(n*m)
            for i in range(n):
                for j in range(m):
                    s = 0.0
                    for k in range(p):
                        s += R[i,j,k] * Vv[j*p+k] * Wv[k*n+i]
                    U_new[i*m+j] = s
            nu = np.linalg.norm(U_new)
            if nu < 1e-15:
                return Uv, Vv, Wv, 0.0
            U_new /= nu

            # Update V: fix U, W
            V_new = np.zeros(m*p)
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for i in range(n):
                        s += R[i,j,k] * Uv[i*m+j] * Wv[k*n+i]
                    V_new[j*p+k] = s
            nv = np.linalg.norm(V_new)
            if nv < 1e-15:
                return Uv, Vv, Wv, 0.0
            V_new /= nv

            # Update W: fix U, V
            W_new = np.zeros(p*n)
            for k in range(p):
                for i in range(n):
                    s = 0.0
                    for j in range(m):
                        s += R[i,j,k] * Uv[i*m+j] * Vv[j*p+k]
                    W_new[k*n+i] = s
            nw = np.linalg.norm(W_new)
            if nw < 1e-15:
                return Uv, Vv, Wv, 0.0
            W_new /= nw

            # Check convergence
            if (np.linalg.norm(U_new - Uv) < tol and
                np.linalg.norm(V_new - Vv) < tol and
                np.linalg.norm(W_new - Wv) < tol):
                Uv, Vv, Wv = U_new, V_new, W_new
                break

            Uv, Vv, Wv = U_new, V_new, W_new

        # Compute the scalar (the value of the trilinear form)
        scalar = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    scalar += R[i,j,k] * Uv[i*m+j] * Vv[j*p+k] * Wv[k*n+i]
        
        return Uv, Vv, Wv, scalar

    def greedy_peel(target_rank, max_restarts=5):
        """Try to find a rank-r decomposition via greedy rank-1 peeling with multiple restarts."""
        if time.time() > deadline - 20:
            return None

        best_result = None
        best_score = -1

        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break

            # Target tensor: T[i,j,k] = 1 for all i,j,k
            T = np.ones((n, m, p), dtype=float)
            R = T.copy()

            U_cols = []
            V_cols = []
            W_cols = []

            success = True
            for r in range(target_rank):
                if time.time() > deadline - 25:
                    success = False
                    break

                # Try multiple random initializations for this step
                best_u, best_v, best_w, best_scalar = None, None, None, 0.0
                np.random.seed(42 + restart * 100 + r)
                
                for init in range(3):
                    u, v, w, scalar = find_best_rank1_residual(R, max_iter=500)
                    if abs(scalar) > abs(best_scalar):
                        best_u, best_v, best_w, best_scalar = u, v, w, scalar

                if abs(best_scalar) < 1e-12:
                    success = False
                    break

                # Subtract the rank-1 term from residual
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            R[i,j,k] -= best_scalar * best_u[i*m+j] * best_v[j*p+k] * best_w[k*n+i]

                U_cols.append(best_u)
                V_cols.append(best_v)
                W_cols.append(best_w)

            if not success:
                continue

            # Check if residual is close to zero
            max_res = np.max(np.abs(R))
            if max_res < 1e-8:
                # Build the full matrices
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    U[:, r] = U_cols[r]
                    V[:, r] = V_cols[r]
                    W[:, r] = W_cols[r]

                # Project to half-integers
                U = project_to_half_int(U, target_rank)
                V = project_to_half_int(V, target_rank)
                W = project_to_half_int(W, target_rank)

                # Verify exactness
                if verify_exact(U, V, W, target_rank) and verify_integer(U, V, W, target_rank):
                    return U, V, W

        return None

    # Try to find a lower-rank decomposition
    # Start from reasonable ranks and go up
    for target_rank in [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]:
        if time.time() > deadline - 30:
            break

        result = greedy_peel(target_rank)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    if not verify_exact(U0, V0, W0, rank_schoolbook):
        raise ValueError("Schoolbook decomposition failed verification")

    if not verify_integer(U0, V0, W0, rank_schoolbook):
        raise ValueError("Schoolbook decomposition has non-integer entries")

    return {
        "U": U0.tolist(),
        "V": V0.tolist(),
        "W": W0.tolist()
    }
# EVOLVE-BLOCK-END
