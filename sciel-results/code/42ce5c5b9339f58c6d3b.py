"""Matrix multiplication 2x3 by 3x3 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    deadline = time.time() + 140  # 20s safety margin
    start = time.time()

    # Try rank 17 first (theoretical lower bound is 17)
    result = _find_rank17(n, m, p, deadline)
    if result is not None:
        return result

    # Fallback: schoolbook rank 18
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition exactly."""
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _is_half_integer(arr, tol=1e-8):
    """Check if all entries are close to half-integers."""
    doubled = 2.0 * arr
    return np.allclose(doubled, np.round(doubled), atol=tol)

def _round_half_integer(arr):
    """Round to nearest half-integer."""
    return np.round(2.0 * arr) / 2.0

def _find_rank17(n, m, p, deadline):
    """
    Search for a rank-17 half-integer decomposition.
    
    Key idea: Fix W (generic), then the system is LINEAR in the products
    U[i*m+j1, r] * V[j2*p+k, r]. For each (i,k), we have a 9x9 linear system
    in the 9 products (j1,j2). We solve it via SVD and check if solutions
    are consistent and half-integer.
    """
    rank = 17
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6

    np.random.seed(12345)

    # Precompute the (i,k) -> (j1,j2) mapping for the linear system
    # For each (i,k): matrix M[i,k] where M[i,k][(j1,j2), r] = U[i*m+j1, r] * V[j2*p+k, r]
    # We need M[i,k] @ x = e where e[j1*m+j2] = 1 if j1==j2 else 0
    # But x is the vector of products... actually we need to solve for U,V jointly.
    
    # Alternative: use the fact that for fixed W, define for each r:
    #   u_r = U[:, r] (6-dim), v_r = V[:, r] (9-dim)
    # The constraint is: for each (i,k), sum_r W[kn+i, r] * u_r[i*m+j1] * v_r[j2*p+k] = delta(j1,j2)
    
    # This is still bilinear. But we can use a different strategy:
    # Randomly initialize U and V with half-integers, solve for W via least squares,
    # then round W and verify. The key improvement: use MANY more random trials
    # and a smarter initialization.

    # Strategy: random half-integer U, V; solve linear system for W exactly;
    # check if W is half-integer; verify.
    
    # For fixed U, V, the system for W is:
    # For each (i,k): sum_r W[kn+i, r] * (U[i*m+j1, r] * V[j2*p+k, r]) = delta(j1,j2) for all j1,j2
    # This is a 9x9 linear system for the 17 unknowns W[kn+i, 0..16]
    # (9 equations, 17 unknowns) - underdetermined! So we can use minimum-norm solution.
    # But we need W to be half-integer, which is a discrete constraint.
    
    # Better strategy: use the structure more directly.
    # The matricization T_{(i,k), (j1,j2)} has rank 5. So we can use a rank-5
    # decomposition of this matricization to guide the construction.
    
    # Let's try a direct algebraic approach:
    # We know the tensor rank is 17. Let's use the known structure.
    # The 2x3x3 mat-mul tensor can be decomposed using the fact that
    # it's related to the 3x3 tensor (rank 23 or so) but with one dimension smaller.
    
    # Actually, let me try the most direct approach: random search with
    # structured initialization and exact linear solve for W.

    for attempt in range(5000):
        if time.time() > deadline:
            break

        # Random half-integer U and V
        scale = 4
        U = np.random.randint(-scale, scale + 1, size=(u_dim, rank)) / 2.0
        V = np.random.randint(-scale, scale + 1, size=(v_dim, rank)) / 2.0

        # Solve for W: for each (i,k), we have a 9x17 system
        W = np.zeros((w_dim, rank))
        valid = True
        for i in range(n):
            for k in range(p):
                # Build 9x9 matrix A where A[j1*m+j2, r] = U[i*m+j1, r] * V[j2*p+k, r]
                A = np.zeros((m*m, rank))
                for j1 in range(m):
                    for j2 in range(m):
                        A[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                
                rhs = np.zeros(m*m)
                for j in range(m):
                    rhs[j*m+j] = 1.0

                # Solve A @ w = rhs for w (17-dim)
                # Use SVD for minimum-norm solution
                try:
                    U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
                except np.linalg.LinAlgError:
                    valid = False
                    break

                # Minimum norm solution: w = Vt^T @ (U_svd^T @ rhs / s)
                rhs_proj = U_svd.T @ rhs
                # Handle small singular values
                w = np.zeros(rank)
                for idx in range(min(len(s), rank)):
                    if s[idx] > 1e-10:
                        w[idx] = rhs_proj[idx] / s[idx]
                    else:
                        # Check consistency
                        if abs(rhs_proj[idx]) > 1e-8:
                            valid = False
                            break
                if not valid:
                    break

                # Check if w is close to half-integer
                if not _is_half_integer(w, tol=1e-6):
                    valid = False
                    break

                W[k*n+i, :] = _round_half_integer(w)
            if not valid:
                break

        if valid:
            # Final verification
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return None
# EVOLVE-BLOCK-END
