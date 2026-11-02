"""Bootstrap SEARCH for exact tensor decomposition of 3x4x7 matrix multiplication."""
import time
import itertools
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as sla

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Return the best exact tensor decomposition found within a time budget.
    The interface is construct() -> {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start = time.time()
    deadline = start + 140.0  # safety margin below 150s kill

    n, m, p = DIMENSIONS
    total_triples = n * m * p  # 84

    def _schoolbook():
        """Return the classical rank-nmp decomposition."""
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
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reconstructs the matrix multiplication."""
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Ua[i*m + j, :] @ (Va[j*p + k, :] * Wa[k*n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _try_rank(R, max_iters=500, seed=42):
        """Try to find a rank-R decomposition using ALS-like iteration."""
        if R < 1:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W with random values
        U = rng.randn(n * m, R)
        V = rng.randn(m * p, R)
        W = rng.randn(p * n, R)
        
        # Normalize columns to prevent blowup
        for r in range(R):
            nU = np.linalg.norm(U[:, r])
            nV = np.linalg.norm(V[:, r])
            nW = np.linalg.norm(W[:, r])
            if nU > 0 and nV > 0 and nW > 0:
                U[:, r] /= nU
                V[:, r] /= nV
                W[:, r] /= nW
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all k
            # This is a linear system in U[i,j,:]
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    b = np.ones(p)
                    # Solve least squares
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m + j, :] = sol
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b = np.ones(n)
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p + k, :] = sol
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b = np.ones(m)
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n + i, :] = sol
                    except:
                        pass
        
        # Check if we got a good solution
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = U[i*m + j, :] @ (V[j*p + k, :] * W[k*n + i, :])
                    err = max(err, abs(s - 1.0))
        
        if err < 1e-8:
            # Try to round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            if _verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), tol=1e-8):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
            
            # Try integer rounding
            U_round = np.round(U)
            V_round = np.round(V)
            W_round = np.round(W)
            
            if _verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), tol=1e-8):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        
        return None

    def _strassen_embedded():
        """
        Construct a decomposition by replacing non-overlapping 2x2x2 schoolbook blocks
        with Strassen's rank-7 decomposition.
        """
        def strassen_block(i_off, j_off, k_off, U, V, W, col):
            """Add Strassen decomposition for block at given offsets. Returns next col."""
            i0, i1 = i_off, i_off + 1
            j0, j1 = j_off, j_off + 1
            k0, k1 = k_off, k_off + 1
            
            # M1 = (A00+A11)(B00+B11)
            U[i0*m + j0][col] = 1
            U[i1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            # M2 = (A10+A11)B00
            U[i1*m + j0][col] = 1
            U[i1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + i1][col] = 1
            W[k1*n + i1][col] = -1
            col += 1
            
            # M3 = A00(B10-B11)
            U[i0*m + j0][col] = 1
            V[j1*p + k0][col] = 1
            V[j1*p + k1][col] = -1
            W[k0*n + i0][col] = 1
            W[k1*n + i0][col] = -1
            col += 1
            
            # M4 = A11(B01-B00)
            U[i1*m + j1][col] = 1
            V[j0*p + k1][col] = 1
            V[j0*p + k0][col] = -1
            W[k0*n + i0][col] = -1
            W[k1*n + i0][col] = 1
            col += 1
            
            # M5 = (A00+A10)B11
            U[i0*m + j0][col] = 1
            U[i1*m + j0][col] = 1
            V[j1*p + k1][col] = 1
            W[k1*n + i0][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            # M6 = (A01-A00)(B00+B10)
            U[i0*m + j1][col] = 1
            U[i0*m + j0][col] = -1
            V[j0*p + k0][col] = 1
            V[j1*p + k0][col] = 1
            W[k1*n + i1][col] = -1
            col += 1
            
            # M7 = (A10-A11)(B01+B11)
            U[i1*m + j0][col] = 1
            U[i1*m + j1][col] = -1
            V[j0*p + k1][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            return col
        
        blocks = [
            (0, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (0, 2, 2),
            (0, 0, 4),
            (0, 2, 4),
        ]
        
        num_blocks = len(blocks)
        R = num_blocks * 7 + (n * m * p - num_blocks * 8)
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        
        for i_off, j_off, k_off in blocks:
            col = strassen_block(i_off, j_off, k_off, U, V, W, col)
        
        covered = set()
        for i_off, j_off, k_off in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i_off + di, j_off + dj, k_off + dk))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        if col != R:
            return _schoolbook()
            
        return {"U": U, "V": V, "W": W}

    # First try Strassen-embedded
    best = _strassen_embedded()
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # Try numerical search for lower ranks
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    for R in range(70, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank(R, max_iters=200, seed=42)
        if result is not None:
            best_result = result
            best_rank = R
            break
    
    if best_result is not None:
        return best_result
    
    return _schoolbook()

# EVOLVE-BLOCK-END
