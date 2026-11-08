"""Matrix multiplication 3x5 by 5x5 - rank-minimal tensor decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]
DEADLINE = 140.0  # Safety margin before 150s limit

def _verify(U, V, W, n=3, m=5, p=5):
    """Verify that U, V, W form a valid decomposition of matrix multiplication.
    
    U: (n*m, rank)
    V: (m*p, rank)  
    W: (p*n, rank)
    
    For each output element C[i,k] = sum_j A[i,j]*B[j,k],
    we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is the (i,j,k) term
    Actually, the tensor T_{i,j,k} = 1 for all i,j,k, and the decomposition is:
    T = sum_c U_c ⊗ V_c ⊗ W_c where U_c is a vector of length n*m, etc.
    
    More precisely: for the multiplication tensor, T[i,j,k] = 1 for all i,j,k.
    The decomposition satisfies: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    # Verify: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a trilinear form check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=5, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: use random initialization and iterative refinement
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For small ranks, try random search with verification
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # Also rank >= n*m*p / max(n,m,p) = 75/5 = 15 (weak bound)
    # Strassen-like bounds may apply
    
    # Let's try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, p*n) = 25
    # But actually for matrix multiplication, the rank is at least the dimension of the output space
    # which is n*p = 15, but also at least the "support" considerations
    
    # Try a range of ranks
    lower_bound = max(n * m, m * p, p * n)  # 25
    upper_bound = best_rank  # 75
    
    # Try random search for lower ranks
    np.random.seed(42)
    
    for rank in range(lower_bound, upper_bound + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try a few random initializations for this rank
        max_tries = 50
        for attempt in range(max_tries):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Iterate: fix two, solve for the third
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # For fixed U, V: W[k*n+i, c] = T[i,j,k] / (U[i*m+j,c] * V[j*p+k,c]) if denominator nonzero
            # But this is overconstrained, so we use least squares
            
            converged = False
            for iteration in range(20):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Update W: for each (i,k), W[k*n+i, :] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # This is: for fixed (i,k), W[k*n+i, :] is the solution to
                # sum_j (U[i*m+j,:] * V[j*p+k,:]) * W[k*n+i,:] = 1 (vector equation)
                # Let A_ik = sum_j outer(U[i*m+j,:], V[j*p+k,:])  -- no, that's not right
                # Actually: sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # = sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
                # Let S_ik[c] = sum_j U[i*m+j,c] * V[j*p+k,c]
                # Then: sum_c S_ik[c] * W[k*n+i,c] = 1
                # This is a scalar equation, so we have many solutions.
                # We want to minimize ||W|| or just find any solution.
                # Simplest: set W[k*n+i, c] = S_ik[c] / (sum_c S_ik[c]^2) if sum != 0
                # This gives sum_c S_ik[c] * W[k*n+i,c] = sum_c S_ik[c]^2 / sum_c S_ik[c]^2 = 1
                
                W_new = np.zeros((p * n, rank))
                for i in range(n):
                    for k in range(p):
                        S = np.zeros(rank)
                        for j in range(m):
                            S += U[i * m + j, :] * V[j * p + k, :]
                        denom = np.dot(S, S)
                        if abs(denom) > 1e-12:
                            W_new[k * n + i, :] = S / denom
                        else:
                            W_new[k * n + i, :] = 0
                
                # Similarly update U
                U_new = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        S = np.zeros(rank)
                        for k in range(p):
                            S += V[j * p + k, :] * W[k * n + i, :]
                        denom = np.dot(S, S)
                        if abs(denom) > 1e-12:
                            U_new[i * m + j, :] = S / denom
                        else:
                            U_new[i * m + j, :] = 0
                
                # Similarly update V
                V_new = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        S = np.zeros(rank)
                        for i in range(n):
                            S += U[i * m + j, :] * W[k * n + i, :]
                        denom = np.dot(S, S)
                        if abs(denom) > 1e-12:
                            V_new[j * p + k, :] = S / denom
                        else:
                            V_new[j * p + k, :] = 0
                
                # Check convergence
                if (np.max(np.abs(W_new - W)) < 1e-8 and 
                    np.max(np.abs(U_new - U)) < 1e-8 and 
                    np.max(np.abs(V_new - V)) < 1e-8):
                    converged = True
                    break
                
                W, U, V = W_new, U_new, V_new
            
            # Verify the result
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                # Found a valid decomposition of this rank!
                if rank < best_rank:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break  # Move to next (lower) rank
                else:
                    break  # Already have this rank or better
    
    # Convert to lists of lists with proper types
    # Ensure entries are integers or half-integers if possible
    result = {}
    for key in ['U', 'V', 'W']:
        arr = np.array(best[key])
        # Check if entries are close to integers or half-integers
        rounded = np.round(arr * 2) / 2
        if np.max(np.abs(arr - rounded)) < 1e-9:
            result[key] = rounded.tolist()
        else:
            result[key] = arr.tolist()
    
    return result
