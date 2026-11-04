"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] for t in range(len(U[0])))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _rationalize(M, tol=1e-8, max_den=1000):
    """Try to find rational entries close to float matrix M."""
    rows, cols = M.shape
    out = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            v = M[r, c]
            if abs(v) < tol:
                out[r][c] = 0
            else:
                found = False
                for den in range(1, max_den + 1):
                    num = round(v * den)
                    if abs(v - num / den) < tol:
                        if num % den == 0:
                            out[r][c] = num // den
                        else:
                            out[r][c] = num / den
                        found = True
                        break
                if not found:
                    # Try half-integers
                    num2 = round(v * 2)
                    if abs(v - num2 / 2) < tol:
                        out[r][c] = num2 / 2
                        found = True
                if not found:
                    out[r][c] = round(v)
    return out


def _als_step(U, V, W, n, m, p):
    """One full ALS iteration: update U, then V, then W."""
    rank = U.shape[1]
    
    # Update U: for each (i,j), solve A^T @ u = ones(p)
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                U[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
            except:
                pass
    
    # Update V: for each (j,k), solve B^T @ v = ones(n)
    for j in range(m):
        for k in range(p):
            B = np.zeros((n, rank))
            for i in range(n):
                B[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                V[j * p + k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
            except:
                pass
    
    # Update W: for each (k,i), solve C^T @ w = ones(m)
    for k in range(p):
        for i in range(n):
            C = np.zeros((m, rank))
            for j in range(m):
                C[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                W[k * n + i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
            except:
                pass


def _compute_error(U, V, W, n, m, p):
    """Compute the maximum absolute error."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                max_err = max(max_err, abs(val - 1.0))
    return max_err


def _polish(U, V, W, n, m, p, max_iter=100, tol=1e-12):
    """High-precision polish pass with tighter tolerance."""
    for _ in range(max_iter):
        err = _compute_error(U, V, W, n, m, p)
        if err < tol:
            break
        _als_step(U, V, W, n, m, p)
    return _compute_error(U, V, W, n, m, p)


def _try_rank_als(rank, n, m, p, time_budget=10.0, n_restarts=20):
    """Attempt to find a rank-r decomposition via ALS with multiple restarts."""
    start = time.time()
    
    for restart in range(n_restarts):
        if time.time() - start > time_budget:
            break
        
        # Structured initialization: use schoolbook-like structure but compressed
        rng = np.random.RandomState(42 + rank * 100 + restart * 7919)
        
        # Try different initialization strategies
        if restart % 3 == 0:
            # Random initialization
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
        elif restart % 3 == 1:
            # Sparse-like initialization
            U = rng.randn(n * m, rank) * 0.3
            V = rng.randn(m * p, rank) * 0.3
            W = rng.randn(p * n, rank) * 0.3
            # Add some structure
            for t in range(min(rank, n*m)):
                U[t % (n*m), t] += 1.0
            for t in range(min(rank, m*p)):
                V[t % (m*p), t] += 1.0
            for t in range(min(rank, p*n)):
                W[t % (p*n), t] += 1.0
        else:
            # Diagonal-dominant initialization
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for t in range(rank):
                U[t % (n*m), t] = 1.0
                V[t % (m*p), t] = 1.0
                W[t % (p*n), t] = 1.0
            # Add random perturbation
            U += rng.randn(n * m, rank) * 0.1
            V += rng.randn(m * p, rank) * 0.1
            W += rng.randn(p * n, rank) * 0.1
        
        # Run ALS
        for it in range(200):
            if time.time() - start > time_budget:
                break
            _als_step(U, V, W, n, m, p)
            
            # Check convergence every 10 iterations
            if it % 10 == 9:
                err = _compute_error(U, V, W, n, m, p)
                if err < 1e-7:
                    # Polish
                    final_err = _polish(U, V, W, n, m, p, max_iter=50, tol=1e-12)
                    if final_err < 1e-10:
                        U_r = _rationalize(U)
                        V_r = _rationalize(V)
                        W_r = _rationalize(W)
                        # Verify after rationalization
                        if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                            return U_r, V_r, W_r
                        # If rationalization failed, try with smaller tolerance
                        U_r2 = _rationalize(U, tol=1e-6, max_den=100)
                        V_r2 = _rationalize(V, tol=1e-6, max_den=100)
                        W_r2 = _rationalize(W, tol=1e-6, max_den=100)
                        if _verify(U_r2, V_r2, W_r2, n, m, p, tol=1e-6):
                            return U_r2, V_r2, W_r2
    
    return None


def _structured_search(rank, n, m, p, time_budget=15.0):
    """
    Exploit the small n=2 structure.
    For each k, W has two rows: W[2k,:] and W[2k+1,:].
    The equations are:
      For each j: U[j, :] · (V[j*p+k,:] * W[2k,:]) = 1  (i=0)
      For each j: U[5+j, :] · (V[j*p+k,:] * W[2k+1,:]) = 1  (i=1)
    
    We can try to fix W and solve for U and V, or vice versa.
    """
    start = time.time()
    
    for attempt in range(30):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.RandomState(12345 + attempt * 104729)
        
        # Initialize W with structure
        W = rng.randn(p * n, rank) * 0.5
        
        # For each k, the two rows of W should be "compatible" with the same V rows
        # Try to make W[2k,:] and W[2k+1,:] related
        for k in range(p):
            # Make the second row a scaled version with some perturbation
            scale = rng.uniform(0.5, 2.0)
            W[2*k+1, :] = scale * W[2*k, :] + rng.randn(rank) * 0.1
        
        # Now solve for U and V using alternating approach
        U = rng.randn(n * m, rank) * 0.3
        V = rng.randn(m * p, rank) * 0.3
        
        for it in range(150):
            if time.time() - start > time_budget:
                break
            
            # Update U given V, W
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V given U, W
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W given U, V
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            if it % 15 == 14:
                err = _compute_error(U, V, W, n, m, p)
                if err < 1e-7:
                    final_err = _polish(U, V, W, n, m, p, max_iter=50, tol=1e-12)
                    if final_err < 1e-10:
                        U_r = _rationalize(U)
                        V_r = _rationalize(V)
                        W_r = _rationalize(W)
                        if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                            return U_r, V_r, W_r
                        U_r2 = _rationalize(U, tol=1e-6, max_den=100)
                        V_r2 = _rationalize(V, tol=1e-6, max_den=100)
                        W_r2 = _rationalize(W, tol=1e-6, max_den=100)
                        if _verify(U_r2, V_r2, W_r2, n, m, p, tol=1e-6):
                            return U_r2, V_r2, W_r2
    
    return None


def _schoolbook():
    n, m, p = DIMENSIONS
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


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    max_rank = n * m * p  # 60
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But actually for matmul, rank >= max(n*m, m*p, p*n) is a known lower bound
    # For 2x5x6, let's try from 30 upward
    
    min_possible = max(n * m, m * p, p * n)
    
    # Try structured search first for lower ranks
    for rank in range(min_possible, max_rank + 1):
        if time.time() - start > _DEADLINE - 10:
            break
        
        remaining = _DEADLINE - 10 - (time.time() - start)
        if remaining < 3.0:
            break
        
        time_budget = min(12.0, remaining)
        
        # Try structured search
        result = _structured_search(rank, n, m, p, time_budget=time_budget * 0.5)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() - start > _DEADLINE - 10:
            break
        
        remaining = _DEADLINE - 10 - (time.time() - start)
        if remaining < 3.0:
            break
        
        time_budget = min(12.0, remaining)
        
        # Try standard ALS
        result = _try_rank_als(rank, n, m, p, time_budget=time_budget * 0.5)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
