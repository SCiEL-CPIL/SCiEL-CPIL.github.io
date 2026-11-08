"""Rank-minimal tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy: Use alternating linear solves to find a low-rank decomposition.
Start from schoolbook (rank 120), then iteratively try to reduce rank by
1 by fixing two factor matrices and solving for the third via least squares,
checking if the residual is exactly zero.
"""
import time
import random

try:
    import numpy as np
except Exception:
    np = None

try:
    import sympy
except Exception:
    sympy = None

DIMENSIONS = [4, 5, 6]
SEED = 42


def _build_schoolbook(n=4, m=5, p=6):
    """Build the schoolbook decomposition: rank = n*m*p = 120."""
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W, R


def _verify_numpy(U, V, W, n, m, p):
    """Verify the decomposition using numpy."""
    if np is None:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            target = np.eye(m, dtype=np.float64)
            if not np.allclose(M, target, atol=1e-9):
                return False
    return True


def _verify_sympy(U, V, W, n, m, p):
    """Verify the decomposition using sympy for exactness."""
    if sympy is None:
        return False
    R = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    s = sympy.Integer(0)
                    for c in range(R):
                        wv = sympy.Integer(W[k * n + i][c])
                        uu = sympy.Integer(U[i * m + j][c])
                        vv = sympy.Integer(V[j2 * p + k][c])
                        if wv == 0 or uu == 0 or vv == 0:
                            continue
                        s += wv * uu * vv
                    target = sympy.Integer(1) if j == j2 else sympy.Integer(0)
                    if s != target:
                        return False
    return True


def _random_init(U, V, W, n, m, p, R, rng):
    """Randomly initialize factor matrices with small integer/half-integer entries."""
    U_new = [[0] * R for _ in range(n * m)]
    V_new = [[0] * R for _ in range(m * p)]
    W_new = [[0] * R for _ in range(p * n)]
    
    for r in range(n * m):
        for c in range(R):
            U_new[r][c] = rng.choice([-2, -1, 0, 0, 0, 1, 2])
    for r in range(m * p):
        for c in range(R):
            V_new[r][c] = rng.choice([-2, -1, 0, 0, 0, 1, 2])
    for r in range(p * n):
        for c in range(R):
            W_new[r][c] = rng.choice([-2, -1, 0, 0, 0, 1, 2])
    
    return U_new, V_new, W_new


def _solve_for_W(U, V, n, m, p, R):
    """Given U and V, solve for W such that the decomposition holds.
    For each (i,k), we need: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = delta(j,j2) for all j,j2.
    This is a linear system in W[k*n+i, :].
    """
    if np is None:
        return None
    
    W = [[0] * R for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            # Build linear system: A * w = b
            # A[j, j2, c] = U[i*m+j, c] * V[j*p+k, c]
            # b[j, j2] = delta(j, j2)
            A = np.zeros((m * m, R), dtype=np.float64)
            b = np.zeros(m * m, dtype=np.float64)
            
            for j in range(m):
                for j2 in range(m):
                    idx = j * m + j2
                    b[idx] = 1.0 if j == j2 else 0.0
                    for c in range(R):
                        A[idx, c] = U[i * m + j][c] * V[j2 * p + k][c]
            
            # Solve A * w = b
            if R == 0:
                continue
            try:
                # Use least squares
                w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if solution is exact
                residual = np.linalg.norm(A @ w - b)
                if residual > 1e-8:
                    return None
                # Round to nearest integer/half-integer
                w_rounded = np.round(w * 2) / 2
                # Verify exactness with rounded values
                A_wr = np.array([[A[i, c] * 2 for c in range(R)] for i in range(m*m)], dtype=np.float64)
                wr2 = w_rounded * 2
                if np.max(np.abs(A_wr @ wr2 - b * 2)) > 1e-6:
                    return None
                for c in range(R):
                    W[k * n + i][c] = w_rounded[c]
            except Exception:
                return None
    
    return W


def _solve_for_U(V, W, n, m, p, R):
    """Given V and W, solve for U."""
    if np is None:
        return None
    
    U = [[0] * R for _ in range(n * m)]
    
    for i in range(n):
        for j in range(m):
            # Build linear system: A * u = b
            # A[k, j2, c] = V[j*p+k, c] * W[k*n+i, c]
            # b[k, j2] = delta(j, j2) for the (k,j2) entry... 
            # Actually: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should contribute to C[k,i]
            # For fixed i,j: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 0 for k != ... 
            # Wait, this is tricky. Let me think differently.
            # The constraint is: for all i,k: sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = delta...
            # Actually the full constraint is: for all i,k,j,j2: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j2*p+k, c] = delta(j,j2)
            # For fixed i,j, unknown is U[i*m+j, :], and the equation is:
            # sum_c U[i*m+j, c] * (sum_{k} ... ) -- no, this doesn't separate cleanly.
            # Let me use a different approach: treat all U entries for fixed i as unknowns.
            pass
    
    # This approach is complex. Let's use a simpler iterative method instead.
    return None


def _try_reduce_rank(U, V, W, n, m, p, deadline, max_iterations=500):
    """Try to reduce rank by removing columns that are redundant using random restarts."""
    if np is None:
        return U, V, W
    
    R = len(U[0])
    best_U, best_V, best_W = U, V, W
    best_R = R
    
    rng = random.Random(SEED)
    
    # First, try simple column removal
    changed = True
    while changed and time.time() < deadline - 30.0:
        changed = False
        R = len(U[0])
        for c in range(R):
            if time.time() > deadline - 30.0:
                break
            keep = [x for x in range(R) if x != c]
            U2 = [[U[r][x] for x in keep] for r in range(n * m)]
            V2 = [[V[r][x] for x in keep] for r in range(m * p)]
            W2 = [[W[r][x] for x in keep] for r in range(p * n)]
            if _verify_numpy(U2, V2, W2, n, m, p):
                U, V, W = U2, V2, W2
                changed = True
                break
    
    # Then, try random perturbations and re-solve
    R = len(U[0])
    for attempt in range(max_iterations):
        if time.time() > deadline - 30.0:
            break
        if R <= 100:  # Only try if we're below 100
            break
        
        # Try to reduce by 1: pick a random column to eliminate
        # Randomly perturb and try to find a solution with R-1 columns
        target_R = R - 1
        if target_R < 1:
            break
        
        # Random initialization
        U_rand, V_rand, W_rand = _random_init(U, V, W, n, m, p, target_R, rng)
        
        # Iterate: solve for W given U,V; solve for U given V,W (if possible)
        converged = False
        for it in range(20):
            if time.time() > deadline - 30.0:
                break
            W_new = _solve_for_W(U_rand, V_rand, n, m, p, target_R)
            if W_new is None:
                break
            U_rand, V_rand, W_rand = U_rand, V_rand, W_new
            converged = True
        
        if converged:
            if _verify_numpy(U_rand, V_rand, W_rand, n, m, p):
                U, V, W = U_rand, V_rand, W_rand
                R = target_R
                best_U, best_V, best_W = U, V, W
                best_R = R
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    
    try:
        # Build schoolbook decomposition (rank 120)
        U, V, W, R = _build_schoolbook(n, m, p)
        
        # Verify
        ok = _verify_numpy(U, V, W, n, m, p)
        
        if not ok:
            return {"U": U, "V": V, "W": W}
        
        # Try to reduce rank
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
        
        return {"U": U, "V": V, "W": W}
    
    except Exception:
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
