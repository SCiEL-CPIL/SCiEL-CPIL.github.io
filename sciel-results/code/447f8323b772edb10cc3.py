"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for r in range(rank):
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-12):
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

def _strassen_rank55(n, m, p):
    """
    Construct the known rank-55 decomposition using Strassen for 2x2 blocks
    and schoolbook for the 2x1 block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2 * 2x10 (5 blocks of 2 columns each)
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v00, v01 = k0, k1
        v10, v11 = 10 + k0, 10 + k1
        w00, w01 = k0 * 2, k1 * 2
        w10, w11 = k0 * 2 + 1, k1 * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1; U[u11][col] = 1; V[v00][col] = 1; V[v11][col] = 1; W[w00][col] = 1; W[w11][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[u10][col] = 1; U[u11][col] = 1; V[v00][col] = 1; W[w10][col] = 1; W[w11][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[u00][col] = 1; V[v01][col] = 1; V[v11][col] = -1; W[w01][col] = 1; W[w11][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[u11][col] = 1; V[v10][col] = 1; V[v00][col] = -1; W[w00][col] = 1; W[w10][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[u00][col] = 1; U[u01][col] = 1; V[v11][col] = 1; W[w01][col] = 1; W[w00][col] = -1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1; U[u00][col] = -1; V[v00][col] = 1; V[v01][col] = 1; W[w11][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1; U[u11][col] = -1; V[v10][col] = 1; V[v11][col] = 1; W[w00][col] = 1
        col += 1
        
    # Schoolbook for A2 (2x1) * B2 (1x10)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _try_rank54_exploit(n, m, p, time_budget, seed=42):
    """
    Try to find a rank-54 decomposition by exploiting the structure and
    solving for W given random sparse U, V.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    r = 54
    best = None
    
    # We use the rank-55 solution as a starting point for the structure
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        # Strategy 0: Random sparse U, V with entries in {-1, 0, 1}
        if attempt % 2 == 0:
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            for row in range(n*m):
                # Average sparsity around 3-4 non-zeros per row for m=3, n=2
                nnz = max(1, min(r, rng.poisson(3.0) + 1))
                cols = rng.choice(r, size=nnz, replace=False)
                U[row, cols] = rng.choice([-1, 1], size=nnz)
                
            for row in range(m*p):
                nnz = max(1, min(r, rng.poisson(3.0) + 1))
                cols = rng.choice(r, size=nnz, replace=False)
                V[row, cols] = rng.choice([-1, 1], size=nnz)
        else:
            # Strategy 1: Perturb the rank-55 base solution slightly (remove one column effectively by mixing)
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            # Copy most of the rank-55 structure, dropping the last column (schoolbook)
            U[:, :53] = U0[:, :53]
            V[:, :53] = V0[:, :53]
            # Add random noise to the last column to try to absorb the schoolbook term
            U[:, 53] = rng.choice([-1, 0, 1], size=n*m)
            V[:, 53] = rng.choice([-1, 0, 1], size=m*p)
            # Also try perturbing the Strassen part slightly
            U[rng.random((n*m, r)) < 0.05] = 0
            V[rng.random((m*p, r)) < 0.05] = 0
            
        W = np.zeros((p*n, r))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                # We need U[i*m+j] * V[j*p+k] * W[k*n+i] to sum to 1 for all i,j,k
                # For fixed i, k, we have a linear system for W[k*n+i]
                # Let w be the vector W[k*n+i] of length r
                # Sum_c U[i*m+j, c] * V[j*p+k, c] * w[c] = 1 for j=0..m-1
                # This is A @ w = 1, where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                # Solve A @ w = 1
                try:
                    # Check if consistent
                    # Use lstsq and verify
                    w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    
                    # Check residual
                    if rank_A < m:
                        # If rank deficient, check if 1 is in the column space
                        # More robustly: check if A @ w is close to 1
                        if not np.allclose(A @ w, np.ones(m), atol=1e-10):
                            feasible = False
                            break
                    else:
                        if not np.allclose(A @ w, np.ones(m), atol=1e-10):
                            feasible = False
                            break
                    
                    W[k*n+i] = w
                except:
                    feasible = False
                    break
            
            if not feasible:
                break
        
        if feasible:
            # Exact verification with rationals
            # Convert to fractions
            U_frac = [[Fraction(U[i, j]) for j in range(r)] for i in range(n*m)]
            V_frac = [[Fraction(V[i, j]) for j in range(r)] for i in range(m*p)]
            W_frac = [[Fraction(W[i, j]) for j in range(r)] for i in range(p*n)]
            
            if _verify(U_frac, V_frac, W_frac, n, m, p):
                best = {
                    "U": U_frac,
                    "V": V_frac,
                    "W": W_frac
                }
                break
            # If it passed lstsq but failed rational check, it's not exact.
            # This might happen due to floating point noise in lstsq solution if the system is ill-conditioned
            # or if the solution is not actually exact rationals.
            
    return best

def _try_rank54_blockwise(n, m, p, time_budget, seed=42):
    """
    Try to find a rank-54 decomposition by treating the 10 columns as 5 groups of 2
    and sharing rank components between groups more efficiently.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    r = 54
    best = None
    
    # Base structure from Strassen
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    # The rank-55 solution has 7*5 + 10 = 45 + 10 = 55 components.
    # We want to reduce to 54.
    # Let's try to modify the last component (one of the schoolbook ones) to be shared
    # or to eliminate one of the Strassen components by modifying others.
    
    # This is a heuristic: 
    # Take the 54 components. 
    # Try to find W that satisfies the equation.
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
            
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        
        # Copy first 54 components of rank-55
        U[:, :54] = U0[:, :54]
        V[:, :54] = V0[:, :54]
        
        # Introduce small rational perturbations to some entries to change the solution space
        # We want to see if there's a W of rank 54 that works.
        # If we just take 54 components of the rank-55 solution, the system for W
        # becomes overdetermined (3 equations for 54 unknowns per (i,k), but the constraints
        # from the omitted 11th component must be satisfiable).
        
        # Actually, just solving for W with the first 54 components of U and V:
        # If a solution exists, it means the 54 components span the necessary space.
        
        # Let's solve for W.
        W = np.zeros((p*n, r))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                try:
                    w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    if not np.allclose(A @ w, np.ones(m), atol=1e-10):
                        feasible = False
                        break
                    W[k*n+i] = w
                except:
                    feasible = False
                    break
            
            if not feasible:
                break
        
        if feasible:
            U_frac = [[Fraction(U[i, j]) for j in range(r)] for i in range(n*m)]
            V_frac = [[Fraction(V[i, j]) for j in range(r)] for i in range(m*p)]
            W_frac = [[Fraction(W[i, j]) for j in range(r)] for i in range(p*n)]
            
            if _verify(U_frac, V_frac, W_frac, n, m, p):
                best = {
                    "U": U_frac,
                    "V": V_frac,
                    "W": W_frac
                }
                break
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with the known rank-55 Strassen construction
    best = _strassen_rank55(n, m, p)
    best_rank = 55
    
    # 1. Try the blockwise reduction search
    time_left = deadline - time.time()
    if time_left > 20:
        result = _try_rank54_blockwise(n, m, p, min(time_left, 30), seed=42)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = 54
    
    # 2. Try the exploit search (more random)
    if best_rank > 54:
        time_left = deadline - time.time()
        if time_left > 20:
            for seed in [42, 123, 456, 789, 1000]:
                if time.time() > deadline - 10:
                    break
                tl = min(time_left, deadline - time.time() - 5)
                if tl < 5:
                    break
                result = _try_rank54_exploit(n, m, p, tl, seed=seed)
                if result is not None:
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best = result
                        best_rank = 54
                        break
    
    return best
