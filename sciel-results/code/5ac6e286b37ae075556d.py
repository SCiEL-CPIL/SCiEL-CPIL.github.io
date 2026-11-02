"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _is_half_integer(x, tol=1e-9):
    # Check if x is close to an integer or half-integer
    x2 = 2 * x
    nearest = round(x2)
    if abs(x2 - nearest) < tol:
        return True
    return False

def _verify_half_integers(mat, tol=1e-9):
    for row in mat:
        for val in row:
            if not _is_half_integer(val, tol):
                return False
    return True

def _try_find_decomposition(n, m, p, rank, max_trials=2000):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We fix U, V to random integers and solve for W.
    # System: A w = b
    # A is (n*m*p) x (rank * p*n)
    # b is (n*m*p)
    
    nm = n * m
    mp = m * p
    pn = p * n
    num_eqs = n * m * p
    num_unk = rank * pn
    
    # Precompute indices for efficiency
    # Equation index: e = i*m*p + j*p + k
    # Unknown index: u = r*pn + k*n + i
    
    for trial in range(max_trials):
        # Random U, V with entries in {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(nm)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(mp)]
        
        # Build A and b
        A = np.zeros((num_eqs, num_unk), dtype=float)
        b = np.zeros(num_eqs, dtype=float)
        
        # Fill b (T is 1 for all valid i,j,k)
        b[:] = 1.0
        
        # Fill A
        # A[e, u] = U[i*m+j, r] * V[j*p+k, r]
        # e = i*m*p + j*p + k
        # u = r*pn + k*n + i
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = i * m * p + j * p + k
                    for r in range(rank):
                        u = r * pn + k * n + i
                        A[e, u] = U[i * m + j][r] * V[j * p + k][r]
        
        # Solve A w = b
        # Use least squares if not square/invertible, but we want exact solution if possible.
        # Since it's underdetermined, we can use np.linalg.lstsq to get a particular solution.
        # However, lstsq gives minimum norm solution, which might not be rational.
        # We need a rational solution.
        # If A has integer entries, any basic solution is rational.
        # We can use Gaussian elimination with fractions? Too slow for large systems.
        # Alternative: Use sympy? Not available.
        # Alternative: Check if the solution from lstsq is close to half-integers.
        # If it is, verify exactness using fractions.
        
        try:
            w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Reshape w to W
        W_flat = w.reshape(rank, pn)
        W = [[W_flat[r, k * n + i] for i in range(n)] for k in range(p) for r in range(rank)]
        # Wait, W shape is (p*n, rank).
        # W[k*n+i, r] = W_flat[r, k*n+i]
        W_mat = [[0] * rank for _ in range(pn)]
        for r in range(rank):
            for idx in range(pn):
                W_mat[idx][r] = W_flat[r, idx]
                
        # Check if W_mat entries are half-integers
        if _verify_half_integers(W_mat, tol=1e-7):
            # Verify exactness with Fractions
            U_frac = [[Fraction(int(U[i][r]) if abs(U[i][r] - round(U[i][r])) < 1e-9 else 0) for r in range(rank)] for i in range(nm)]
            V_frac = [[Fraction(int(V[i][r]) if abs(V[i][r] - round(V[i][r])) < 1e-9 else 0) for r in range(rank)] for i in range(mp)]
            W_frac = [[Fraction(int(round(W_mat[i][r]*2))/2) for r in range(rank)] for i in range(pn)]
            
            # Check tensor equation
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for r in range(rank):
                            s += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                return U, V, W_mat, rank
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Try to find a lower rank
    # Start from a reasonable lower bound, e.g., max(nm, mp, pn)
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound to 35
    for rank in range(lower_bound, 36):
        if time.time() > deadline:
            break
            
        # Adjust trials based on rank
        # Lower ranks are harder to find? Or easier?
        # Let's try a few hundred for each
        max_trials = 500 if rank < 30 else 200
        
        result = _try_find_decomposition(n, m, p, rank, max_trials)
        if result is not None:
            U, V, W, r = result
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
# EVOLVE-BLOCK-END
