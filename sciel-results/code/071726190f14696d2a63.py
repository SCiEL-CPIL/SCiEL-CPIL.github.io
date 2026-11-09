import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p, rank):
    """Verify exact decomposition using Fraction arithmetic."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, rank, tol=1e-9):
    """Quick float verification."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _solve_for_UV_given_W(n, m, p, rank, W):
    """
    Given W (p*n x rank), try to find U (n*m x rank) and V (m*p x rank)
    such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    We can vectorize: for each (i,j), the constraint is:
    sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
    
    This is a bilinear constraint. Let's use an alternating approach:
    - Fix V, solve for U (linear)
    - Fix U, solve for V (linear)
    """
    # Initialize V randomly
    rng = np.random.RandomState(42)
    V = rng.randint(-2, 3, size=(m*p, rank)).astype(float)
    
    for _ in range(200):
        # Solve for U given V, W
        U = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                # For each k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is: A @ u = ones(p) where A[c,k] = V[j*p+k,c] * W[k*n+i,c]
                A = np.zeros((rank, p))
                for k in range(p):
                    A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    u, res, rnk, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
                except:
                    return None
        
        # Solve for V given U, W
        V_new = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                # For each i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is: B @ v = ones(n) where B[c,i] = U[i*m+j,c] * W[k*n+i,c]
                B = np.zeros((rank, n))
                for i in range(n):
                    B[:, i] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v, res, rnk, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = v
                except:
                    return None
        
        # Check convergence
        if np.allclose(V, V_new, atol=1e-8):
            V = V_new
            # Verify
            if _verify_float(U.tolist(), V.tolist(), W, n, m, p, rank):
                # Round to half-integers and verify exactly
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(np.array(W) * 2) / 2
                if _verify_float(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, rank):
                    # Final exact check
                    U_list = [[float(x) for x in row] for row in U_r]
                    V_list = [[float(x) for x in row] for row in V_r]
                    W_list = [[float(x) for x in row] for row in W_r]
                    if _verify_exact(U_list, V_list, W_list, n, m, p, rank):
                        return U_list, V_list, W_list
                # Try without rounding
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = [[float(x) for x in row] for row in W]
                if _verify_exact(U_list, V_list, W_list, n, m, p, rank):
                    return U_list, V_list, W_list
            return None
        V = V_new
    
    return None

def _try_rank(n, m, p, rank, time_budget=30.0):
    """Try to find a decomposition of given rank."""
    start = time.time()
    
    # Strategy 1: Fix W with simple entries, solve for U, V
    # W has shape (p*n, rank) = (8, rank)
    # Try W entries in {0, 1, -1}
    
    # Generate candidate W matrices
    candidates = []
    
    # Candidate 1: W with sparse structure
    # Each row of W should have a few non-zero entries
    
    for attempt in range(50):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.RandomState(attempt)
        
        # Generate W with entries in {-1, 0, 1}
        W = rng.randint(-1, 2, size=(p*n, rank))
        
        # Ensure no zero rows
        for r in range(p*n):
            if np.all(W[r, :] == 0):
                W[r, rng.randint(rank)] = 1
        
        result = _solve_for_UV_given_W(n, m, p, rank, W.tolist())
        if result:
            return result
    
    # Strategy 2: ALS with better initialization
    for attempt in range(30):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.RandomState(1000 + attempt)
        U = rng.randn(n*m, rank) * 0.5
        V = rng.randn(m*p, rank) * 0.5
        W = rng.randn(p*n, rank) * 0.5
        
        for it in range(300):
            # Update U
            for i in range(n):
                for j in range(m):
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # For fixed (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                    A = np.zeros((rank, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((rank, n))
                    for i in range(n):
                        B[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (s - 1.0)**2
            err = np.sqrt(err / (n*m*p))
            
            if err < 1e-10:
                # Round and verify
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                if _verify_exact(U_list, V_list, W_list, n, m, p, rank):
                    return U_list, V_list, W_list
                # Try exact values
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_exact(U_list, V_list, W_list, n, m, p, rank):
                    return U_list, V_list, W_list
                break
        
        if time.time() - start > time_budget:
            break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Try rank 19 first (known optimal)
    for rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        result = _try_rank(n, m, p, rank, time_budget=min(remaining - 5, 60.0))
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
