"""Matrix multiplication 4,4,5 - search for low-rank exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]
DEADLINE = 145.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    def check_valid(U, V, W, n, m, p, tol=1e-9):
        """Check if decomposition is valid."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Strategy: For each candidate rank r, try to find a solution.
    # Use the fact that we can fix some variables and solve for others.
    #
    # Key insight: If we fix V and W, then for each (i,j,k):
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in U! So we can set up a linear system for U.
    #
    # For fixed V, W: the equation for (i,j,k) is:
    #   sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
    # Let A_{(i,j,k), (i',j',c)} = delta(i,i') * delta(j,j') * V[j*p+k,c] * W[k*n+i,c]
    # Then A * U_vec = 1_vec
    #
    # This is an 80 x (16*r) system. For r >= 20, it's overdetermined but may have solutions.
    #
    # Better: Fix V and W randomly (sparse), then solve for U via least squares,
    # then check if the solution is exact.

    # Try ranks from 20 upward
    for r in range(20, 80):
        if time.time() - start_time > DEADLINE - 15:
            break

        found = False
        num_attempts = 100 if r <= 30 else 50

        for attempt in range(num_attempts):
            if time.time() - start_time > DEADLINE - 15:
                break

            # Random sparse V and W
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))

            for c in range(r):
                # V column: sparse
                nnz_v = random.randint(2, 5)
                v_idx = random.sample(range(m*p), min(nnz_v, m*p))
                for idx in v_idx:
                    V[idx, c] = random.choice([1.0, -1.0])

                # W column: sparse
                nnz_w = random.randint(2, 5)
                w_idx = random.sample(range(p*n), min(nnz_w, p*n))
                for idx in w_idx:
                    W[idx, c] = random.choice([1.0, -1.0])

            # Now solve for U: for each (i,j,k),
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            #
            # Group by (i,j): for fixed (i,j), we have p equations in r unknowns U[i*m+j, :]
            # For (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for k=0..p-1
            #
            # This is a p x r system: B_{i,j} * U_{i,j}^T = 1
            # where B_{i,j}[k,c] = V[j*p+k,c] * W[k*n+i,c]

            U = np.zeros((n*m, r))
            valid = True
            for i in range(n):
                for j in range(m):
                    # Build p x r matrix B
                    B = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            B[k, c] = V[j*p+k, c] * W[k*n+i, c]

                    # Solve B * u = 1 where u = U[i*m+j, :]
                    # Use least squares
                    try:
                        u, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        # Check if exact
                        if np.allclose(B @ u, np.ones(p), atol=1e-9):
                            U[i*m+j, :] = u
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break

                if not valid:
                    break

            if valid:
                # Final verification
                if check_valid(U, V, W, n, m, p):
                    found = True
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    break

        if found:
            # Found at rank r, try to go lower? No, we're going up from 20.
            # Actually we want the minimum, so once found, we can stop.
            break

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
