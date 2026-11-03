"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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

def _schoolbook_solution(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the factor matrices exactly decompose the matrix multiplication tensor.
    For each (i,j,k), the tensor entry T[i*m+j, j*p+k, k*n+i] should equal
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    Returns True if all entries match exactly (using Fraction arithmetic).
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                # The tensor entry T[i*m+j, j*p+k, k*n+i] is 1 if (i,j,k) is a valid index, else 0
                # For matrix multiplication, T[a,b,c] = 1 if a corresponds to (i,j), b to (j,k), c to (k,i)
                # Specifically, T[i*m+j, j*p+k, k*n+i] = 1
                if total != 1:
                    return False
    return True

def _try_low_rank_search(n, m, p, time_limit=100):
    """
    Attempt to find a lower-rank exact decomposition using iterative refinement.
    Starts from random initializations and tries to converge to an exact solution.
    Returns a valid decomposition dict if found, else None.
    """
    start_time = time.time()
    best_solution = None
    best_rank = n * m * p
    
    # Try ranks from a reasonable lower bound up
    # Lower bound for matrix multiplication rank is max(n*m, m*p, p*n) for rectangular cases
    # For 4x4 * 4x9: max(16, 36, 36) = 36
    min_possible_rank = max(n * m, m * p, p * n)
    
    for target_rank in range(min_possible_rank, best_rank):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Multiple random restarts for each rank
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Initialize with random small values
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Random initialization
            U[:, :] = np.random.randn(n * m, target_rank) * 0.1
            V[:, :] = np.random.randn(m * p, target_rank) * 0.1
            W[:, :] = np.random.randn(p * n, target_rank) * 0.1
            
            # Iterative refinement: alternate updating each factor
            # The tensor equation is: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # We can fix two factors and solve for the third linearly
            converged = False
            for iteration in range(50):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Update U: for each (i,m+j) and c, solve linear system
                # Fix V and W, update U
                # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed (i,j): T[i*m+j, j*p+k, k*n+i] for all k
                # = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                
                # Build linear system for U
                # For each row u_idx = i*m+j, we have p equations (one for each k)
                # Unknowns are U[u_idx, 0..target_rank-1]
                # Equation for k: sum_c U[u_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                
                # This is a linear system: A_u @ U[u_idx,:] = b_u
                # where A_u[k,c] = V[j*p+k,c] * W[k*n+i,c], b_u[k] = 1
                
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            v_idx = j * p + k
                            w_idx = k * n + i
                            for c in range(target_rank):
                                A[k, c] = V[v_idx, c] * W[w_idx, c]
                        b = np.ones(p)
                        
                        # Solve A @ U[u_idx,:] = b
                        try:
                            U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            u_idx = i * m + j
                            w_idx = k * n + i
                            for c in range(target_rank):
                                A[i, c] = U[u_idx, c] * W[w_idx, c]
                        b = np.ones(n)
                        try:
                            V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            for c in range(target_rank):
                                A[j, c] = U[u_idx, c] * V[v_idx, c]
                        b = np.ones(m)
                        try:
                            W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = 0
                            for c in range(target_rank):
                                val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Convert to exact fractions if possible
                U_frac = [[Fraction(U[i, c]).limit_denominator(1000) for c in range(target_rank)] for i in range(n * m)]
                V_frac = [[Fraction(V[i, c]).limit_denominator(1000) for c in range(target_rank)] for i in range(m * p)]
                W_frac = [[Fraction(W[i, c]).limit_denominator(1000) for c in range(target_rank)] for i in range(p * n)]
                
                # Verify exactness
                if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                    best_solution = {"U": U_frac, "V": V_frac, "W": W_frac}
                    best_rank = target_rank
                    break
                else:
                    # Try with higher precision
                    U_frac = [[Fraction(U[i, c]).limit_denominator(10000) for c in range(target_rank)] for i in range(n * m)]
                    V_frac = [[Fraction(V[i, c]).limit_denominator(10000) for c in range(target_rank)] for i in range(m * p)]
                    W_frac = [[Fraction(W[i, c]).limit_denominator(10000) for c in range(target_rank)] for i in range(p * n)]
                    if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                        best_solution = {"U": U_frac, "V": V_frac, "W": W_frac}
                        best_rank = target_rank
                        break
    
    return best_solution

# EVOLVE-BLOCK-END
