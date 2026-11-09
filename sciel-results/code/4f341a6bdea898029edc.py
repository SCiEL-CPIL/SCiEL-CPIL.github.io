"""Matrix multiplication 2,3,4 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]


def construct():
    deadline = time.time() + 140  # safety margin
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p

    # Build the schoolbook decomposition as a fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Try to find a lower-rank decomposition
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k (since (AB)_{ik} = sum_j A_{ij}B_{jk})
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # Try random search with linear solve for U
    # For a given V, W, the system is linear in U:
    # For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
    # This is a system of p equations in R unknowns for each (i,j) pair

    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Try ranks from 15 up to 23
    for rank in range(15, min(24, rank_schoolbook)):
        if time.time() > deadline:
            break

        attempts = 0
        max_attempts = 500

        while attempts < max_attempts and time.time() < deadline:
            attempts += 1

            # Generate random V and W with entries from a small set
            # Use half-integers: values in {-1, -0.5, 0, 0.5, 1}
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            V = np.random.choice(vals, size=(m * p, rank))
            W = np.random.choice(vals, size=(p * n, rank))

            # For each (i, j), solve for U[i*m+j, :]
            U = np.zeros((n * m, rank))
            valid = True
            for i in range(n):
                for j in range(m):
                    # Build matrix A where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A @ U[i*m+j, :] = ones(p)
                    # Use least squares
                    try:
                        sol, residuals, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    # Check if residual is small
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-8:
                            valid = False
                            break
                    else:
                        # Check directly
                        if np.max(np.abs(A @ sol - np.ones(p))) > 1e-8:
                            valid = False
                            break
                    U[i * m + j, :] = sol

            if not valid:
                continue

            # Check if U entries are close to half-integers
            U_rounded = np.round(U * 2) / 2.0
            if np.max(np.abs(U - U_rounded)) < 1e-8:
                # Verify the full tensor
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i * m + j, :] * V[:, j * p + k] * W[:, k * n + i])
                            if abs(val - 1.0) > 1e-8:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V
                    best_W = W
                    break  # Found minimal rank for this rank value, try lower

    if best_rank < rank_schoolbook:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
