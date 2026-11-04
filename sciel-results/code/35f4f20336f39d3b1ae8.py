"""Combined program: schoolbook fallback + principled linear-in-W search with deadline guard."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    rank = U_np.shape[1]
    # Vectorized: build for each (i,j,k)
    # U rows indexed by i*m+j, V by j*p+k, W by k*n+i
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j, :], V_np[j*p+k, :])  # placeholder
                # correct: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                val = np.einsum('c,c,c->', U_np[i*m+j, :], V_np[j*p+k, :], W_np[k*n+i, :])
                if abs(val - 1.0) > tol:
                    ok = False
    return ok


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _try_rank(rank, n, m, p, deadline, max_attempts):
    """Fix random U,V; solve linear system for W per (k,i) block. Return (U,V,W) or None."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        # Random U,V with small integer/half-integer-ish values; use {-1,0,1} for U,V
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        for c in range(rank):
            U[:, c] = np.random.choice([-1, 0, 1], size=n*m)
            V[:, c] = np.random.choice([-1, 0, 1], size=m*p)
        # For each (k,i), solve for W[k*n+i, :] of length rank
        # Equations over j: sum_c U[i*m+j,c]*V[j*p+k,c]*Wc[c] = 1, j=0..m-1
        W = np.zeros((p*n, rank))
        feasible = True
        for k in range(p):
            for i in range(n):
                # Build m x rank matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                rhs = np.ones(m)
                # Solve A @ Wc = rhs (least squares), then check residual
                try:
                    Wc, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    feasible = False
                    break
                # Check consistency
                if np.max(np.abs(A @ Wc - rhs)) > 1e-7:
                    feasible = False
                    break
                # Round to nearest half-integer to satisfy constraint
                Wc_rounded = 0.5 * np.round(2.0 * Wc)
                # Check if rounding introduced significant error
                if np.max(np.abs(A @ Wc_rounded - rhs)) > 1e-7:
                    feasible = False
                    break
                W[k*n+i, :] = Wc_rounded
        if feasible:
            # Verify fully with rounded values
            if _verify(U, V, W, n, m, p):
                return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0  # safety margin below 150s kill

    # Best-so-far: schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    # Convert to list of floats to ensure type consistency
    U0_list = [[float(x) for x in row] for row in U0]
    V0_list = [[float(x) for x in row] for row in V0]
    W0_list = [[float(x) for x in row] for row in W0]
    best = {"U": U0_list, "V": V0_list, "W": W0_list}
    best_rank = n * m * p

    # Try decreasing ranks; for each, attempt the linear-in-W search
    for rank in range(best_rank - 1, 4, -1):
        if time.time() > deadline - 5.0:
            break
        # Budget per rank: more attempts for larger ranks (easier), fewer for small
        remaining = deadline - 5.0 - time.time()
        attempts = max(20, int(remaining * 4))
        result = _try_rank(rank, n, m, p, deadline - 5.0, attempts)
        if result is not None:
            U, V, W = result
            # Ensure U, V, W are lists of floats that are exactly int or half-int
            # U and V are already int (-1,0,1), but convert to list of floats for consistency
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            break  # found lowest-rank feasible in this sweep; stop

    return best
# EVOLVE-BLOCK-END
