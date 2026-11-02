"""Rewritten program for 3x3 x 3x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18

    # Build target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-7):
        """Check if U, V, W reconstruct T_target exactly."""
        r = U.shape[1]
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_recon, T_target, atol=tol, rtol=0)

    def schoolbook_decomposition():
        rank = n * m * p  # 54
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Strassen-based initialization for 3x3 blocks
    def strassen_3x3_decomposition():
        """
        Use Strassen's 7-multiplication algorithm for 3x3 matrices.
        Split the 3x3 into quadrants and use 7 multiplications per block.
        For 3x3 x 3x6, we can split into two 3x3 blocks in the output.
        """
        # Strassen's algorithm for 2x2 matrices uses 7 multiplications
        # For 3x3, we can pad to 4x4 and use Strassen, or use specific 3x3 decompositions
        # Known: 3x3 matrix multiplication can be done with 14 multiplications (2 Strassen blocks)
        
        # Let's use a direct approach: split the 3x6 into two 3x3 blocks
        # C[:, 0:3] = A * B[:, 0:3] and C[:, 3:6] = A * B[:, 3:6]
        
        # For each 3x3 x 3x3 multiplication, use 7 multiplications (Strassen-like)
        # Total: 14 multiplications for the 3x3 x 3x6 case
        
        # Strassen's 7 multiplications for 2x2:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # For 3x3, we need a different approach. Let's use the fact that
        # the tensor rank for 3x3 x 3x3 is at most 14 (2*7 for Strassen blocks)
        
        # Actually, let's just use random initialization with structure
        rank = 14
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Initialize with random values but with some structure
        np.random.seed(42)
        for t in range(rank):
            # Each factor vector
            U[:, t] = np.random.randn(nm) * 0.5
            V[:, t] = np.random.randn(mp) * 0.5
            W[:, t] = np.random.randn(pn) * 0.5
        
        return U, V, W, rank

    U_sb, V_sb, W_sb, sb_rank = schoolbook_decomposition()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), sb_rank

    def try_rank_decomposition(target_rank, num_trials=10, max_iter=3000, maxfun=15000):
        """
        Try to find a rank-target_rank decomposition using L-BFGS-B.
        """
        if target_rank >= best_rank:
            return None
        if time.time() > deadline:
            return None

        from scipy.optimize import minimize

        for trial in range(num_trials):
            if time.time() > deadline:
                return None

            # Use different initialization strategies
            if trial % 3 == 0:
                # Random initialization
                np.random.seed(trial * 7919 + 41)
                U_t = np.random.randn(nm, target_rank) * 0.3
                V_t = np.random.randn(mp, target_rank) * 0.3
                W_t = np.random.randn(pn, target_rank) * 0.3
            elif trial % 3 == 1:
                # Strassen-like initialization
                np.random.seed(trial * 3571 + 13)
                U_t = np.random.randn(nm, target_rank) * 0.5
                V_t = np.random.randn(mp, target_rank) * 0.5
                W_t = np.random.randn(pn, target_rank) * 0.5
            else:
                # Sparse initialization
                np.random.seed(trial * 5233 + 29)
                U_t = np.zeros((nm, target_rank))
                V_t = np.zeros((mp, target_rank))
                W_t = np.zeros((pn, target_rank))
                for t in range(target_rank):
                    # Random sparse vectors
                    idx_u = np.random.choice(nm, size=max(1, nm // 3), replace=False)
                    idx_v = np.random.choice(mp, size=max(1, mp // 3), replace=False)
                    idx_w = np.random.choice(pn, size=max(1, pn // 3), replace=False)
                    U_t[idx_u, t] = np.random.choice([-1, 1], size=len(idx_u))
                    V_t[idx_v, t] = np.random.choice([-1, 1], size=len(idx_v))
                    W_t[idx_w, t] = np.random.choice([-1, 1], size=len(idx_w))

            x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])

            def objective(x):
                if time.time() > deadline:
                    return 1e10
                U_f = x[:nm * target_rank].reshape(nm, target_rank)
                V_f = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_f = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_target - T_recon
                return np.sum(error ** 2)

            def gradient(x):
                if time.time() > deadline:
                    return np.zeros_like(x)
                U_f = x[:nm * target_rank].reshape(nm, target_rank)
                V_f = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_f = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_target - T_recon

                dU = -2.0 * np.einsum('abc,bt,ct->at', error, V_f, W_f)
                dV = -2.0 * np.einsum('abc,at,ct->bt', error, U_f, W_f)
                dW = -2.0 * np.einsum('abc,at,bt->ct', error, U_f, V_f)

                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                  options={'maxiter': max_iter, 'gtol': 1e-14, 'maxfun': maxfun,
                                           'ftol': 1e-16, 'maxls': 80})
            except Exception:
                continue

            if time.time() > deadline:
                return None

            U_final = result.x[:nm * target_rank].reshape(nm, target_rank)
            V_final = result.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
            W_final = result.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

            if check_decomposition(U_final, V_final, W_final, tol=1e-7):
                # High-precision polish pass
                if time.time() < deadline - 10:
                    x_polish = np.concatenate([U_final.flatten(), V_final.flatten(), W_final.flatten()])
                    try:
                        result_polish = minimize(objective, x_polish, jac=gradient, method='L-BFGS-B',
                                                options={'maxiter': 1000, 'gtol': 1e-16, 'maxfun': 5000,
                                                         'ftol': 1e-18, 'maxls': 50})
                        U_polish = result_polish.x[:nm * target_rank].reshape(nm, target_rank)
                        V_polish = result_polish.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                        W_polish = result_polish.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                        if check_decomposition(U_polish, V_polish, W_polish, tol=1e-8):
                            return U_polish, V_polish, W_polish, target_rank
                    except Exception:
                        pass
                return U_final, V_final, W_final, target_rank

        return None

    # Try progressively lower ranks starting from more promising values
    # Theoretical lower bound: max(nm, mp, pn) = 18
    # Strassen-based upper bound: ~14 (7+7 for two 3x3 blocks)
    for target_rank in [14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue

        # Use more trials and iterations for lower ranks (harder)
        if target_rank <= 16:
            result = try_rank_decomposition(target_rank, num_trials=8, max_iter=2500, maxfun=12000)
        elif target_rank <= 20:
            result = try_rank_decomposition(target_rank, num_trials=5, max_iter=2000, maxfun=10000)
        else:
            result = try_rank_decomposition(target_rank, num_trials=3, max_iter=1500, maxfun=8000)

        if result is not None:
            U_t, V_t, W_t, r = result
            best_U, best_V, best_W, best_rank = U_t, V_t, W_t, r
            # If we found a good rank, try to go even lower
            if r <= 16:
                # Try a few more lower ranks
                for lower_rank in range(r - 1, max(13, r - 5) - 1, -1):
                    if time.time() > deadline:
                        break
                    result2 = try_rank_decomposition(lower_rank, num_trials=5, max_iter=2000, maxfun=10000)
                    if result2 is not None:
                        U_t2, V_t2, W_t2, r2 = result2
                        best_U, best_V, best_W, best_rank = U_t2, V_t2, W_t2, r2
                        break
                break
            elif r <= 20:
                for lower_rank in range(r - 1, 13, -1):
                    if time.time() > deadline:
                        break
                    result2 = try_rank_decomposition(lower_rank, num_trials=4, max_iter=2000, maxfun=10000)
                    if result2 is not None:
                        U_t2, V_t2, W_t2, r2 = result2
                        best_U, best_V, best_W, best_rank = U_t2, V_t2, W_t2, r2
                        break
                break

    # Final polish: if we have a solution, try to verify and clean up
    if best_rank < sb_rank:
        # Verify the solution
        if not check_decomposition(best_U, best_V, best_W, tol=1e-6):
            # Fall back to schoolbook if verification fails
            best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), sb_rank

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                frac = Fraction(val).limit_denominator(10000)
                if abs(float(frac) - val) < 1e-9:
                    if frac.denominator == 1:
                        row_list.append(frac.numerator)
                    elif frac.denominator == 2:
                        row_list.append(frac.numerator / 2.0)
                    else:
                        row_list.append(float(frac))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
