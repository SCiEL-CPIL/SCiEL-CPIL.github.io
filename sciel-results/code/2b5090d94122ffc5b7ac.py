import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 135

    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_exact(U, V, W):
        """Verify with Fraction for exactness."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(U.shape[1]):
                        total += Fraction(U[i*m+j, r]) * Fraction(V[j*p+k, r]) * Fraction(W[k*n+i, r])
                    if total != 1:
                        return False
        return True

    def try_hardcoded_decompositions():
        """Try known or easily constructible low-rank decompositions."""
        
        # Strategy: exploit the structure. p=6, n=2, m=3.
        # Tensor T_{i,j,k} = 1 if k < 6.
        # Actually T_{i,j,k} is the identity for each (i,j) pair? No.
        # T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'}? No, matrix multiplication is (AB)_{ik} = sum_j A_{ij} B_{jk}.
        # So the tensor is T_{i,j,k} = 1 for all i,j,k? No.
        # The tensor M_{i,j,k} such that (AB)_{ik} = sum_j M_{i,j,k} A_{ij} B_{jk}.
        # So M_{i,j,k} = 1 for all i,j,k.
        # Wait, the standard matrix multiplication tensor T_{i,j,k} has T_{i,j,k} = 1 if the indices match the multiplication rule.
        # For (AB)_{ik} = sum_j A_{ij} B_{jk}, the tensor is T_{i,j,k} = 1.
        # Yes, T_{i,j,k} = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # So we need to decompose the all-ones tensor of size 2x3x6.
        # Rank of all-ones tensor of size n x m x p is max(n, m, p) if we allow arbitrary entries?
        # No, the rank of the all-ones tensor is known to be small.
        # For 2x2x2, rank is 7 (Strassen).
        # For 2x3x6, let's try to find a low rank decomposition.
        
        # Let's try rank 12.
        # If we can find a rank 12 decomposition, that's much better than 36.
        
        # Let's use the fact that 6 = 2*3.
        # Maybe we can decompose it into two 2x3x3 blocks?
        # T_{i,j,k} for k=0,1,2 and k=3,4,5.
        # If we find a rank r decomposition for 2x3x3, then rank 2r for 2x3x6.
        # Rank of 2x3x3 all-ones tensor?
        # Let's try to find a rank 12 decomposition for 2x3x3.
        # If we find rank 12 for 2x3x3, then rank 24 for 2x3x6.
        # If we find rank 6 for 2x3x3, then rank 12 for 2x3x6.
        
        # Let's try to construct a rank 12 decomposition for 2x3x6 directly.
        # Or maybe rank 18?
        
        # Let's try a simple construction.
        # U_{i,j,r}, V_{j,k,r}, W_{k,i,r}.
        # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1.
        
        # Let's try to use the structure.
        # Let r = 12.
        # Let's try to find a solution using linear algebra.
        # Fix V and W, solve for U.
        
        return None

    def solve_U_given_VW(V, W, r):
        """Solve for U given V and W. Returns (U, residual_norm)."""
        U = np.zeros((n * m, r), dtype=np.float64)
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, r), dtype=np.float64)
                for k in range(p):
                    A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b_vec = np.ones(p, dtype=np.float64)
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[i * m + j, :] = sol
                    res = A_mat @ sol - b_vec
                    max_res = max(max_res, float(np.max(np.abs(res))))
                except np.linalg.LinAlgError:
                    U[i * m + j, :] = 0.0
                    max_res = float('inf')
        return U, max_res

    def compute_residual_vec(U, V, W, r):
        """Compute the residual vector for all (i,j,k)."""
        res_vec = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res_vec[i, j, k] = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])) - 1.0
        return res_vec

    def try_rank_optimization(r, max_iter=1000, tol=1e-10, n_restarts=20):
        """
        Optimize V and W to minimize residual, with U solved exactly each step.
        Uses gradient descent with adaptive learning rate.
        """
        if r >= best_rank:
            return None

        for init_seed in range(n_restarts):
            if time.time() > deadline:
                return None

            np.random.seed(10000 + r * 1000 + init_seed * 137)
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5

            U, res = solve_U_given_VW(V, W, r)
            if res < tol:
                break

            lr = 0.05
            lr_decay = 0.995
            best_local_res = res
            stuck_count = 0

            for it in range(max_iter):
                if time.time() > deadline:
                    return None

                U, res = solve_U_given_VW(V, W, r)
                if res < tol:
                    break

                if res < best_local_res:
                    best_local_res = res
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count > 50:
                        V += np.random.randn(m * p, r) * 0.1
                        W += np.random.randn(p * n, r) * 0.1
                        stuck_count = 0
                        lr = 0.05
                        continue

                res_vec = compute_residual_vec(U, V, W, r)

                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            df = 2.0 * res_vec[i, j, k]
                            for r_idx in range(r):
                                grad_V[j * p + k, r_idx] += df * U[i * m + j, r_idx] * W[k * n + i, r_idx]
                                grad_W[k * n + i, r_idx] += df * U[i * m + j, r_idx] * V[j * p + k, r_idx]

                grad_norm_V = np.linalg.norm(grad_V)
                grad_norm_W = np.linalg.norm(grad_W)
                max_grad = max(grad_norm_V, grad_norm_W, 1e-10)
                effective_lr = lr * min(1.0, 1.0 / max_grad)

                V -= effective_lr * grad_V
                W -= effective_lr * grad_W

                lr *= lr_decay

                if it % 100 == 0 and it > 0:
                    V_norm = np.linalg.norm(V)
                    W_norm = np.linalg.norm(W)
                    if V_norm > 0 and W_norm > 0:
                        V *= (1.0 / V_norm) * np.sqrt(m * p)
                        W *= (1.0 / W_norm) * np.sqrt(p * n)

            U, res = solve_U_given_VW(V, W, r)
            if res < tol:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if verify(U_rounded, V_rounded, W_rounded, 1e-9):
                    return U_rounded, V_rounded, W_rounded
                U_rounded = np.round(U)
                V_rounded = np.round(V)
                W_rounded = np.round(W)
                if verify(U_rounded, V_rounded, W_rounded, 1e-9):
                    return U_rounded, V_rounded, W_rounded
                if verify(U, V, W, 1e-8):
                    return U, V, W

        return None

    # Try ranks from 35 down to 12
    candidate_ranks = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]

    for r in candidate_ranks:
        if r >= best_rank:
            continue
        if time.time() > deadline:
            break
        result = try_rank_optimization(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
