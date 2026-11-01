import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Target tensor T[i, j, j_prime, k] = delta_{j, j_prime}
    # Shape: (n, m, m, p)
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k] = 1.0

    def compute_M(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)

        # M[i, j, a, k] = sum_r U_r[i, j, r] * V_r[a, k, r] * W_r[k, i, r]
        # Note: W_r is (p, n, R) -> indices k, i, r
        # V_r is (m, p, R) -> indices a, k, r (a is j')
        # U_r is (n, m, R) -> indices i, j, r
        # Result: (n, m, m, p) -> i, j, a, k
        M = np.einsum('ijr,akr,kr->ijak', U_r, V_r, W_r)
        return M

    def residuals(params, R):
        # params: flattened U, V, W
        size_U = nm * R
        size_V = mp * R
        U = params[:size_U].reshape(nm, R)
        V = params[size_U:size_U + size_V].reshape(mp, R)
        W = params[size_U + size_V:].reshape(pn, R)

        M = compute_M(U, V, W, R)
        return (M - T).flatten()

    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Target ranks: 9 is the theoretical minimum for 2x3x4.
    # We try 9, then 10, 11, etc.
    # We use random restarts to escape local minima.

    target_ranks = [9, 10, 11, 12, 13, 14, 15]

    for R in target_ranks:
        if time.time() > deadline:
            break

        # Try multiple random initializations
        # The number of restarts depends on time remaining
        max_restarts = 50
        found = False

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Random initialization
            # Scale: entries should be around 1.
            # Using normal distribution with small std might help.
            # Or random integers?
            # Let's try random normal.
            np.random.seed(42 + restart) # Reproducibility for debugging, but maybe random is better?
            # Actually, let's use a fixed seed for the first few, then random.

            U0 = np.random.randn(nm, R)
            V0 = np.random.randn(mp, R)
            W0 = np.random.randn(pn, R)

            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            try:
                # Use 'lm' for speed, but it requires no bounds.
                # 'trf' is more robust but slower.
                # Given the non-convexity, 'lm' is often used for tensor decomposition.
                result = least_squares(residuals, params0, args=(R,), method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)

                if result.cost < 1e-8:
                    # Check if it's actually valid
                    U_opt = result.x[:nm * R].reshape(nm, R)
                    V_opt = result.x[nm * R: nm * R + mp * R].reshape(mp, R)
                    W_opt = result.x[nm * R + mp * R:].reshape(pn, R)

                    M_opt = compute_M(U_opt, V_opt, W_opt, R)
                    err = np.linalg.norm(M_opt - T)

                    if err < 1e-6:
                        # Found a valid decomposition of rank R
                        # Check for rational/half-integer structure
                        # If the solution is close to half-integers, snap it.
                        U_snap = np.round(U_opt * 2) / 2.0
                        V_snap = np.round(V_opt * 2) / 2.0
                        W_snap = np.round(W_opt * 2) / 2.0

                        M_snap = compute_M(U_snap, V_snap, W_snap, R)
                        if np.allclose(M_snap, T, atol=1e-9):
                            best_rank = R
                            best_U = U_snap
                            best_V = V_snap
                            best_W = W_snap
                            found = True
                            break # Found for this R, try lower R?
                            # Since we iterate R ascending, if we find 9, we are done.
                            # If we find 10, we might still find 9?
                            # We iterate 9, 10, 11... so if we find 9, we break the outer loop?
                            # No, we break the restart loop, then the outer loop continues?
                            # We should break the outer loop if we find the minimal known rank (9).

            except Exception:
                pass

        if found and R == 9:
            break

        # If we didn't find 9, we continue to 10, etc.
        # But if we found 10, we should keep it as best, but try 9?
        # The loop order is 9, 10, 11...
        # So if we fail 9, we try 10. If we succeed 10, best_rank becomes 10.
        # We should then break if we are satisfied or if time is up.
        # Given the difficulty of 9, finding 10 is a good improvement over 24.

    # Convert best solution to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
