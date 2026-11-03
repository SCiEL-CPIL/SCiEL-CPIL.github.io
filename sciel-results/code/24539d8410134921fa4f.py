"""Rewritten: block decomposition + elimination + local pruning for 3x3x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
_DEADLINE = 150.0  # evaluator kills at 150s; return well before


def _build_multiplication_tensor(n, m, p):
    """Return the n*m x m*p matrix whose (i*m+j, j'*p+k) entry is 1 iff j==j'.
    This is the 'multiplication tensor' flattened: M = kron(I_m, ones_{n x p})."""
    M = np.zeros((n * m, m * p))
    for j in range(m):
        for i in range(n):
            for k in range(p):
                M[i * m + j, j * p + k] = 1.0
    return M


def _schoolbook_decomposition(n, m, p):
    """Return (U, V, W) of rank n*m*p with integer entries that exactly decompose the
    multiplication tensor. U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for factor (i,j,k)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _decomposition_valid(U, V, W, n, m, p, tol=1e-8):
    """Check that the decomposition is valid.
    Condition: For all i in [0,n), k in [0,p), j in [0,m), j' in [0,m):
    sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] == delta_{j,j'}
    """
    for i in range(n):
        for k in range(p):
            w_idx = k * n + i
            for j in range(m):
                for jp in range(m):
                    # Compute sum_r W[w_idx, r] * U[i*m+j, r] * V[jp*p+k, r]
                    # U_row = U[i*m+j, :]
                    # V_row = V[jp*p+k, :]
                    # term = W[w_idx, :] * U_row * V_row
                    # val = sum(term)
                    val = np.dot(W[w_idx, :], U[i * m + j, :] * V[jp * p + k, :])
                    target = 1.0 if j == jp else 0.0
                    if abs(val - target) > tol:
                        return False
    return True


def _rank_of_M(M):
    """Numerical rank of M."""
    try:
        r = np.linalg.matrix_rank(M, tol=1e-10)
    except np.linalg.LinAlgError:
        r = M.shape[0]
    return int(r)


def _solve_for_W(U, V, n, m, p, rank, tol=1e-8):
    """Given U and V, solve for W such that the tensor decomposition holds.
    Returns W if a solution exists (least squares), else None.
    """
    W = np.zeros((p * n, rank))
    for i in range(n):
        for k in range(p):
            w_idx = k * n + i
            # Build linear system A w = b
            # A is m^2 x rank
            # b is m^2 x 1
            A = np.zeros((m * m, rank))
            b = np.zeros(m * m)
            idx = 0
            for j in range(m):
                for jp in range(m):
                    # Coefficient for W[w_idx, r] is sum_r? No, W[w_idx, r] is the variable.
                    # Equation: sum_r W[w_idx, r] * (sum_j? No, U and V are fixed)
                    # The term is U[i*m+j, r] * V[jp*p+k, r]
                    # So A[idx, r] = U[i*m+j, r] * V[jp*p+k, r]
                    A[idx, :] = U[i * m + j, :] * V[jp * p + k, :]
                    b[idx] = 1.0 if j == jp else 0.0
                    idx += 1
            
            # Solve A w = b
            try:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if solution is good
                if np.linalg.norm(A @ sol - b) > tol * max(1.0, np.linalg.norm(b)):
                    return None
                W[w_idx, :] = sol
            except np.linalg.LinAlgError:
                return None
    return W


def _find_rank_reduction(U, V, W, n, m, p, deadline):
    """Local search: randomly perturb U and V, re-solve for W, check validity."""
    rank = U.shape[1]
    t0 = time.time()
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    while time.time() - t0 < deadline and rank > 1:
        # Try to drop a column? Or just perturb?
        # Strategy: Randomly zero out entries in U and V, then solve for W.
        U_try = U.copy()
        V_try = V.copy()
        
        # Perturb: zero out some random entries
        for _ in range(10):
            if random.random() < 0.5:
                r = np.random.randint(U_try.shape[0])
                c = np.random.randint(rank)
                U_try[r, c] = 0.0
            else:
                r = np.random.randint(V_try.shape[0])
                c = np.random.randint(rank)
                V_try[r, c] = 0.0
        
        W_try = _solve_for_W(U_try, V_try, n, m, p, rank, tol=1e-6)
        if W_try is not None:
            if _decomposition_valid(U_try, V_try, W_try, n, m, p, tol=1e-6):
                # Check if we can drop a column
                # A column r can be dropped if U[:, r], V[:, r], W[:, r] are all zero?
                # Or if the rank of the decomposition is actually lower?
                # Simple check: if any column is all zeros in U and V, we can drop it.
                drop_col = -1
                for c in range(rank):
                    if np.allclose(U_try[:, c], 0, atol=1e-9) and np.allclose(V_try[:, c], 0, atol=1e-9):
                        drop_col = c
                        break
                
                if drop_col >= 0:
                    U_try = np.delete(U_try, drop_col, axis=1)
                    V_try = np.delete(V_try, drop_col, axis=1)
                    W_try = np.delete(W_try, drop_col, axis=1)
                    rank = rank - 1
                    U, V, W = U_try, V_try, W_try
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = rank
                else:
                    # No drop, but valid solution with same rank. Update best if we want?
                    # We are looking for rank reduction.
                    pass
        # Add a small random noise to escape local minima?
        # Maybe skip for now to keep it simple.
        
    return best_U, best_V, best_W, best_rank


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    deadline = _DEADLINE - 10.0  # safety margin

    # Start from schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    rank = n * m * p

    # Validate baseline
    assert _decomposition_valid(U, V, W, n, m, p, tol=1e-8), "schoolbook failed"

    # Local pruning / rank reduction
    U, V, W, rank = _find_rank_reduction(U, V, W, n, m, p, deadline - 20.0)

    # Final polish: round to nearest half-integer to clean up floating point
    def _round_half(X):
        return (2.0 * X).round() / 2.0
    U2 = _round_half(U)
    V2 = _round_half(V)
    W2 = _round_half(W)
    if _decomposition_valid(U2, V2, W2, n, m, p, tol=1e-8):
        U, V, W = U2, V2, W2

    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    print("rank:", len(result["U"][0]) if result["U"] else 0)
