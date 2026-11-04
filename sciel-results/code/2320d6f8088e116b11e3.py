"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Construct the target tensor T of shape (nm, mp, pn)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # T[u, v, w] = 0 otherwise
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def residual(params, rank):
        """Vectorized residual computation for the full tensor decomposition."""
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)

        # Compute the decomposed tensor D[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        # Using einsum for efficiency
        D = np.einsum('ur,vr,wr->uvw', U_flat, V_flat, W_flat)
        
        # Residual is D - T
        res = D - T
        return res.flatten()

    def check_decomposition(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition is exact for all entries."""
        D = np.einsum('ur,vr,wr->uvw', U, V, W)
        max_err = np.max(np.abs(D - T))
        return max_err < tol, max_err

    def to_list(arr):
        """Convert numpy array to nested list with half-integer rounding."""
        lst = []
        for row in arr:
            row_lst = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    row_lst.append(rounded)
                else:
                    row_lst.append(float(val))
            lst.append(row_lst)
        return lst

    def naive_decomposition(rank=None):
        """Return the naive schoolbook decomposition of rank n*m*p."""
        if rank is None:
            rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * (m * p) + j * p + k
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
        return {"U": U, "V": V, "W": W}

    # Try ranks from low to high, with multiple random restarts
    # Known lower bound is max(nm, mp, pn) = 12.
    # Known rank for 2x3x4 is 18.
    for rank in range(12, 25):
        if time.time() > deadline - 15:
            break

        remaining = deadline - 15 - time.time()
        # Allocate more time per attempt for lower ranks
        max_attempts = max(2, int(remaining / 5))

        for attempt in range(max_attempts):
            if time.time() > deadline - 8:
                break

            np.random.seed(rank * 10000 + attempt * 7 + 42)
            scale = 1.0 / (rank ** 0.5)
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                # First pass: coarse optimization
                # Using 'lm' (Levenberg-Marquardt) for unconstrained least squares
                result = least_squares(residual, x0, args=(rank,), method='lm', max_nfev=200)
                if result.cost < 1e-8:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)

                    valid, max_err = check_decomposition(U_sol, V_sol, W_sol, rank)

                    if valid:
                        # Polish pass with tighter tolerance
                        x_polish = np.concatenate([U_sol.flatten(), V_sol.flatten(), W_sol.flatten()])
                        result2 = least_squares(residual, x_polish, args=(rank,), method='lm', max_nfev=100)
                        if result2.cost < 1e-12:
                            U_pol = result2.x[:nm * rank].reshape(nm, rank)
                            V_pol = result2.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                            W_pol = result2.x[nm * rank + mp * rank:].reshape(pn, rank)
                            valid2, _ = check_decomposition(U_pol, V_pol, W_pol, rank, tol=1e-10)
                            if valid2:
                                return {"U": to_list(U_pol), "V": to_list(V_pol), "W": to_list(W_pol)}
                        # If polish didn't help, return the first solution
                        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
            except Exception:
                continue

    # Fallback: naive decomposition
    return naive_decomposition()
# EVOLVE-BLOCK-END
