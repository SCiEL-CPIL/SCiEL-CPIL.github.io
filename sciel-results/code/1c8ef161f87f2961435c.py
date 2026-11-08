"""Tensor decomposition for 3x3 by 3x7 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal exact tensor decomposition.
    
    Returns dict with 'U', 'V', 'W' matrices.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Entries can be integers or half-integers (multiples of 1/2).
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Deadline: 145 seconds
    deadline = time.time() + 145
    start_time = time.time()
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p  # 63
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
    
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def verify_numpy(U, V, W):
        """Fast verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i*m+j]
                    v_val = V_np[j*p+k]
                    w_val = W_np[k*n+i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    # Kronecker product approach:
    # View 3x3 matrix A as rows: a_0, a_1, a_2 (each 1x3)
    # View 3x7 matrix B as columns: b_0, b_1, ..., b_6 (each 3x1)
    # Result C = A*B has entries C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # We can decompose this using the tensor structure.
    # The multiplication tensor T has indices (i, j, k) where:
    #   i in [0,3) is row of A
    #   j in [0,3) is column of A / row of B  
    #   k in [0,7) is column of B
    #
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy 1: Try structured decompositions based on Strassen-like ideas
    # For 3x3 * 3x3, Strassen uses rank 5.
    # For 3x3 * 3x7, we can potentially exploit structure.
    
    # Let's try a randomized search for low-rank decompositions
    # using a local search / simulated annealing approach.
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_random_rank(R, max_iters=10000):
        """Try to find a rank-R decomposition via random search."""
        if R >= best_rank:
            return None
        
        # Random initialization with small values
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        # Local search
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        residual += (s - 1.0) ** 2
            
            if residual < 1e-12:
                # Convert to fractions
                U_frac = [[Fraction(U[i,j]).limit_denominator(1000) for j in range(R)] for i in range(n*m)]
                V_frac = [[Fraction(V[i,j]).limit_denominator(1000) for j in range(R)] for i in range(m*p)]
                W_frac = [[Fraction(W[i,j]).limit_denominator(1000) for j in range(R)] for i in range(p*n)]
                
                # Verify with exact arithmetic
                if verify(U_frac, V_frac, W_frac):
                    return U_frac, V_frac, W_frac
                else:
                    # Try with different denominator
                    U_frac = [[Fraction(U[i,j]).limit_denominator(100) for j in range(R)] for i in range(n*m)]
                    V_frac = [[Fraction(V[i,j]).limit_denominator(100) for j in range(R)] for i in range(m*p)]
                    W_frac = [[Fraction(W[i,j]).limit_denominator(100) for j in range(R)] for i in range(p*n)]
                    if verify(U_frac, V_frac, W_frac):
                        return U_frac, V_frac, W_frac
            
            # Update: pick a random element and perturb it
            mat = random.choice([U, V, W])
            row = random.randint(0, mat.shape[0]-1)
            col = random.randint(0, mat.shape[1]-1)
            mat[row, col] += random.gauss(0, 0.1)
        
        return None
    
    # Try various ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # For 3x3*3x3, rank 5 is known. For 3x3*3x7, try ranks 15-40.
    
    for R in range(15, min(45, best_rank)):
        if time.time() > deadline - 10:
            break
        
        result = try_random_rank(R, max_iters=5000)
        if result is not None:
            U_frac, V_frac, W_frac = result
            best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
            best_rank = R
            break  # Found a valid decomposition, try to improve further if time
    
    # If we found something better, try to improve
    if best_rank < n * m * p:
        for R in range(best_rank - 1, 14, -1):
            if time.time() > deadline - 10:
                break
            result = try_random_rank(R, max_iters=3000)
            if result is not None:
                U_frac, V_frac, W_frac = result
                best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                best_rank = R
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Safety margin
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
