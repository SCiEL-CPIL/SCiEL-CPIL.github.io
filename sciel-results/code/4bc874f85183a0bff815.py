"""Matrix multiplication 2,4,7 rank-minimal tensor decomposition search."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    
    Returns:
        dict with keys 'U', 'V', 'W' of shapes (n*m, rank), (m*p, rank), (p*n, rank)
        with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # We need U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c = delta(i,j,k) for valid (i,j,k)
    # Actually, the tensor is T[i*m+j, k*n+i] style... let me reconsider.
    
    # The matrix multiplication tensor: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] which is 1 for all valid i,j,k (it's the identity-like tensor)
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a rank-r decomposition where each term is a rank-1 tensor.
    # The schoolbook gives rank = n*m*p = 56.
    
    # Known bounds: For 2x4x7, the rank is at most... let's think.
    # Lower bound: max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Upper bound via various constructions.
    
    # Let's try to find a decomposition with rank < 56.
    # One approach: use the fact that we can combine some terms.
    
    # Actually, let me think about this more carefully.
    # The tensor is T in V^* ⊗ W^* ⊗ Z^* where V = R^{n*m}, W = R^{m*p}, Z = R^{p*n}
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    
    # A rank-r decomposition is: T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    
    # For the schoolbook: each term has exactly one nonzero in each of U, V, W.
    
    # Can we do better? Let's try a greedy/iterative approach.
    
    # Strategy: Start with schoolbook rank 56, then try to reduce by finding linear dependencies.
    
    # Actually, let me try a different approach: use known results or try small ranks.
    
    # For 2x4x7, let's see if we can achieve rank < 56.
    # The Strassen-like approach might help for some subproblems.
    
    # Let me try to construct a decomposition with rank r for various r.
    # We'll use a simple approach: try to find U, V, W such that the tensor equation holds.
    
    # For exact rational/half-integer solutions, we can set up a system of equations.
    
    # The constraint is: for all (i,j,k) in [0,n) x [0,m) x [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 56 equations with r*(nm + mp + pn) = r*(8+28+14) = 50r unknowns.
    # For r < 56, we have more unknowns than equations, so solutions should exist generically.
    # But we need them to be rational/half-integer.
    
    # Let me try to find a solution with rank as low as possible.
    
    # Start with a reasonable upper bound and try to reduce.
    # Known: rank(2x4x7) <= ?
    
    # Let's try rank 32 first (half of 56), then try lower.
    
    def try_rank(r, max_time=10):
        """Try to find a rank-r decomposition using random initialization and refinement."""
        t_start = time.time()
        t_deadline = t_start + max_time
        
        # We'll use a simple approach: initialize with random values and try to solve.
        # Since the problem is bilinear/trilinear, we can use alternating minimization.
        
        best_solution = None
        best_err = float('inf')
        
        for attempt in range(20):
            if time.time() > t_deadline:
                break
            
            # Initialize
            np.random.seed(attempt * 12345 + 67890)
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Alternating minimization
            for iteration in range(100):
                if time.time() > t_deadline:
                    break
                
                # Fix V, W, solve for U
                # For each (i,j), we need: sum_c U[i*m+j, c] * (V[:, c] * W[:, c] elementwise) = 1
                # Actually, the constraint is:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is a linear system in U[i*m+j, :] for each fixed (i,j)
                
                # Let's reformulate: for fixed (i,j), the equation over all k is:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for k = 0,...,p-1
                # This is: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # Let M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then: U[i*m+j, :] @ M.T = ones(p)
                # So: M @ U[i*m+j, :].T = ones(p)
                # U[i*m+j, :].T = pinv(M) @ ones(p)
                
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        if np.linalg.matrix_rank(M) >= p:
                            U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        else:
                            # Underdetermined, use minimum norm
                            U[i*m+j, :] = np.linalg.pinv(M) @ np.ones(p)
                
                # Similarly for V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        if np.linalg.matrix_rank(M) >= n:
                            V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                        else:
                            V[j*p+k, :] = np.linalg.pinv(M) @ np.ones(n)
                
                # Similarly for W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        if np.linalg.matrix_rank(M) >= m:
                            W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        else:
                            W[k*n+i, :] = np.linalg.pinv(M) @ np.ones(m)
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += abs(s - 1.0)
                
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-10:
                    break
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10 and best_solution is not None:
            return best_solution
        return None
    
    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = 28
    for r in range(28, 57):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_time=min(5, deadline - time.time()))
        if result is not None:
            U, V, W = result
            
            # Convert to Fraction for exactness
            U_frac = [[Fraction(U[i, c]).limit_denominator(1000) for c in range(r)] for i in range(nm)]
            V_frac = [[Fraction(V[i, c]).limit_denominator(1000) for c in range(r)] for i in range(mp)]
            W_frac = [[Fraction(W[i, c]).limit_denominator(1000) for c in range(r)] for i in range(pn)]
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Convert back to list of lists
            U_out = [[float(U_frac[i][c]) for c in range(r)] for i in range(nm)]
            V_out = [[float(V_frac[i][c]) for c in range(r)] for i in range(mp)]
            W_out = [[float(W_frac[i][c]) for c in range(r)] for i in range(pn)]
            
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook decomposition
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
