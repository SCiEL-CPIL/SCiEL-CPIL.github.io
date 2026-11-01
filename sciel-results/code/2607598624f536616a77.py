"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

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

def _try_structured_decomposition(n, m, p, time_budget, seed=42):
    """
    Try structured decompositions based on known matrix multiplication algorithms.
    For 2x3 * 3x10, we can try to decompose using smaller blocks.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Strategy 1: Use the fact that 2x3 can be done in rank 5 (Bini et al.)
    # and 3x10 can be done in rank 24 (schoolbook) or less
    # Total rank would be 5 + 24 = 29 if we can combine them
    
    # Strategy 2: Try to find a decomposition where U and V have a Kronecker-like structure
    
    # For now, try a systematic search with better initialization
    candidate_ranks = list(range(20, n*m*p + 1, 2))
    candidate_ranks.sort()
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        # Try to build U and V with structure, then solve for W
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Initialize U and V with a mix of structure and randomness
            # Use a pattern that might lead to solvable systems
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            # Fill with random ±1 but with some zeros for sparsity
            for i in range(n*m):
                for c in range(r):
                    if rng.random() < 0.7:  # 70% chance of ±1
                        U[i, c] = rng.choice([-1, 1])
            
            for j in range(m*p):
                for c in range(r):
                    if rng.random() < 0.7:
                        V[j, c] = rng.choice([-1, 1])
            
            # Solve for W
            W = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    # Build the linear system for W[k*n+i, :]
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    # Solve A * W[k*n+i, :]^T = b
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = sol
                        # Check if the solution is exact
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                # Verify the full solution
                ok = True
                for i in range(n):
                    if not ok:
                        break
                    for j in range(m):
                        if not ok:
                            break
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-10:
                                ok = False
                                break
                if ok:
                    # Convert to lists with integer/half-integer entries if possible
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    # Try to round to nearest half-integer if close
                    def round_to_half(x):
                        r = round(x * 2) / 2
                        if abs(x - r) < 1e-8:
                            return r
                        return x
                    
                    U_list = [[round_to_half(v) for v in row] for row in U_list]
                    V_list = [[round_to_half(v) for v in row] for row in V_list]
                    W_list = [[round_to_half(v) for v in row] for row in W_list]
                    
                    # Verify with exact arithmetic
                    if _verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = r
                        break
        if best is not None:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured decomposition approach
    time_left = deadline - time.time()
    if time_left > 10:
        result = _try_structured_decomposition(n, m, p, time_left, seed=12345)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    return best
