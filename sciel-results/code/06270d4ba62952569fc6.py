"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import sys
import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START

import numpy as np

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if U, V, W form a valid matrix multiplication decomposition."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k])
                # Actually need triple product
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                s = 0.0
                for c in range(rank):
                    s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, rank, tol=1e-9):
    """Faster verification using numpy broadcasting."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            # For fixed i,k, we need: for all j, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a bilinear check
            pass
    
    # More efficient: build the full tensor and check
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Use einsum or manual
    result = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_np[:, c].reshape(n, m)
        v_c = V_np[:, c].reshape(m, p)
        w_c = W_np[:, c].reshape(p, n)
        # The contribution to T[i,j,k] is u_c[i,j] * v_c[j,k] * w_c[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
    
    if not np.allclose(result, 1.0, atol=tol):
        return False
    return True

def _is_valid_entry(x):
    """Check if x is an integer or half-integer."""
    x2 = x * 2
    return abs(x2 - round(x2)) < 1e-9

def _snap_to_half(x):
    """Snap x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    # Strassen's 7-multiplication algorithm for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # We need to express this in the tensor decomposition format
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # Indices: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C[0,0], C[0,1], C[1,0], C[1,1]
    
    # For each multiplication M_c, we need:
    # U row for each (i,j) in A: coefficient of A[i,j] in M_c
    # V row for each (j,k) in B: coefficient of B[j,k] in M_c
    # W row for each (k,i) in C: coefficient of C[k,i]... wait, W[k*n+i] with n=2
    
    # Let me define the 7 multiplications:
    # M1 = (a+d)(e+h): U[0,0]=1,U[1,1]=1; V[0,0]=1,V[1,1]=1; W: contributes to C[0,0] and C[1,1]
    # Actually this is getting complex. Let me just use a known rank-7 decomposition.
    
    # For now, return None to indicate not implemented
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Verify schoolbook
    if _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p, best_rank):
        pass  # Good
    else:
        # Should never happen
        pass
    
    # Try to find lower-rank decompositions
    # Strategy: use random search with rational/half-integer entries
    
    def _random_half_int(lo=-4, hi=4):
        """Generate a random half-integer in [lo, hi]."""
        val = random.randint(int(lo * 2), int(hi * 2)) / 2.0
        return val
    
    def _try_rank(r, max_attempts=100):
        """Try to find a rank-r decomposition."""
        if r >= best_rank:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            for i in range(n * m):
                for c in range(r):
                    U[i][c] = _random_half_int()
            for j in range(m * p):
                for c in range(r):
                    V[j][c] = _random_half_int()
            for k in range(p * n):
                for c in range(r):
                    W[k][c] = _random_half_int()
            
            if _verify_decomposition(U, V, W, n, m, p, r, tol=1e-9):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from just below schoolbook down to some lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    lower_bound = max(n*m, m*p, p*n)
    
    for r in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank(r, max_attempts=50)
        if result is not None:
            best_rank = r
            best_result = result
            # Try to go even lower
            continue
    
    return best_result

# EVOLVE-BLOCK-END
