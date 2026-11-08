"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from scipy.optimize import linprog
from fractions import Fraction

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
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
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_merge_schoolbook(n, m, p, deadline):
    """
    Try to reduce rank by merging schoolbook terms.
    
    Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged into one rank index
    if we can find coefficients a,b,c,d such that:
      For all (i,j,k): the contribution to T[i,j,k] from the merged term equals
                       the sum of contributions from the two original terms.
    
    Actually, a simpler approach: consider whether we can replace two rank-1 terms
    with one. This requires the tensor entries to be compatible.
    
    More practically: try all pairs of schoolbook indices and check if they can
    be combined. For 2x3x8, we have 48 terms. We'll try to greedily merge.
    
    Even better: use the structure. The tensor is T[i,j,k]=1 for all i,j,k.
    A rank-1 component c contributes U_c[i,j]*V_c[j,k]*W_c[k,i] to T[i,j,k].
    
    For the schoolbook, each term c=(i,j,k) has U=1 only at (i,j), V=1 only at (j,k),
    W=1 only at (k,i).
    
    To merge terms, we need more flexible patterns. Let's try:
    - For each pair of (i,j) with same i, see if we can combine their k-dependence.
    
    Actually, let's use a different strategy: formulate as ILP.
    """
    start = time.time()
    # Lower bound for rank: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # But also need to cover all 48 tensor entries.
    
    # Try a structured search: each rank index c is defined by:
    #   U pattern: which (i,j) pairs are non-zero
    #   V pattern: which (j,k) pairs are non-zero  
    #   W pattern: which (k,i) pairs are non-zero
    # with the constraint that for each (i,j,k):
    #   sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    
    # For small dimensions, we can try to find a decomposition with rank R
    # by searching over structured patterns.
    
    # Key insight: if we fix the U and V patterns, W is determined:
    #   W_c[k,i] = (sum over all (i',j,k') ... ) 
    # No, W is also a free variable. The constraint is:
    #   For all (i,j,k): sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    # This is a system of 48 equations in R*6 + R*24 + R*16 = R*46 unknowns (if R < 46, underdetermined).
    
    # Let's try R from 24 upward.
    # For each R, we need to find U (6xR), V (24xR), W (16xR) such that
    # the tensor equation holds.
    
    # Use a greedy/iterative approach:
    # 1. Start with schoolbook (R=48)
    # 2. Try to eliminate one rank index at a time by redistributing its contribution.
    
    # Alternative: use the fact that this is a bilinear problem.
    # For fixed U and V, the constraint on W is:
    #   For each (i,k): W_c[k,i] must satisfy sum_j U_c[i,j]*V_c[j,k]*W_c[k,i] ... 
    # No, that's not right either.
    
    # Let me think again. The tensor decomposition is:
    #   T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    # where T[i,j,k] = 1 for all i,j,k.
    
    # This is equivalent to: for each c, define a 3-tensor slice T_c[i,j,k] = U_c[i,j]*V_c[j,k]*W_c[k,i]
    # and we need sum_c T_c = T.
    
    # For rank R, we have R*46 variables and 48 constraints.
    # For R >= 24, the system is underdetermined, so solutions should exist.
    # The question is whether they have nice (integer/half-integer) values.
    
    # Let's try a specific structured ansatz:
    # Group rank indices by their "i-index" pattern.
    
    # Actually, let's just try to solve it directly with a simple search.
    # Given the small size, we can try random initializations and Newton's method.
    
    for R in range(24, 49):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(attempt * 1000 + R)
            
            # Initialize with small random values
            U = rng.randn(n*m, R) * 0.1
            V = rng.randn(m*p, R) * 0.1
            W = rng.randn(p*n, R) * 0.1
            
            # Iteratively update to satisfy the constraints
            # The constraint is: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            
            # This is a nonlinear system. Let's use a simple gradient-free approach:
            # For each (i,j,k), if the current sum is s, adjust the variables.
            
            converged = False
            for it in range(1000):
                if time.time() > deadline - 3:
                    break
                
                # Compute residuals
                max_res = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(R):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            res = 1.0 - s
                            max_res = max(max_res, abs(res))
                
                if max_res < 1e-10:
                    converged = True
                    break
                
                # Simple update: for each (i,j,k), distribute the residual
                # across the rank indices that contribute to it
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(R):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            res = 1.0 - s
                            if abs(res) < 1e-15:
                                continue
                            
                            # Find rank indices that have non-zero products for this (i,j,k)
                            for c in range(R):
                                prod = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                if abs(prod) > 1e-10:
                                    # Scale all three factors by (1 + res/prod)^(1/3)
                                    factor = (1 + res / prod) ** (1/3)
                                    if factor > 0:
                                        U[i*m+j, c] *= factor
                                        V[j*p+k, c] *= factor
                                        W[k*n+i, c] *= factor
                                else:
                                    # Try to set one of them
                                    if abs(U[i*m+j, c]) > 1e-10:
                                        V[j*p+k, c] = res / (U[i*m+j, c] * W[k*n+i, c]) if abs(W[k*n+i, c]) > 1e-10 else res / (U[i*m+j, c] * 0.1)
                                    elif abs(V[j*p+k, c]) > 1e-10:
                                        U[i*m+j, c] = res / (V[j*p+k, c] * W[k*n+i, c]) if abs(W[k*n+i, c]) > 1e-10 else res / (V[j*p+k, c] * 0.1)
                                    else:
                                        U[i*m+j, c] = 0.1
                                        V[j*p+k, c] = 0.1
                                        W[k*n+i, c] = res / 0.01
                
                if it % 100 == 0:
                    # Check convergence
                    max_res = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(R):
                                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                max_res = max(max_res, abs(1.0 - s))
                    if max_res < 1e-10:
                        converged = True
                        break
            
            if converged:
                # Check if values are close to integer/half-integer
                # Round to nearest half-integer and verify
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(R):
                                s += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                            if abs(s - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    U_list = [[float(U_rounded[r, c]) for c in range(R)] for r in range(n*m)]
                    V_list = [[float(V_rounded[r, c]) for c in range(R)] for r in range(m*p)]
                    W_list = [[float(W_rounded[r, c]) for c in range(R)] for r in range(p*n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, try to find a better decomposition
    result = _try_merge_schoolbook(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fallback: schoolbook decomposition
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
