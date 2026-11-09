"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 90

    # Try to find a lower rank decomposition using random initialization + gradient descent
    # The tensor T has entries T[i, j, k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    # We want U, V, W such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Actually, the decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # This is a rank-r decomposition of the matrix multiplication tensor.

    # Let's try a few random initializations and do gradient descent on the constraint
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1 = 0

    # Use a smarter approach: try known lower bounds and random search
    # For 3x5x6, the lower bound is max(3*5, 5*6, 6*3) = 30, but actual min rank is likely higher
    
    # Let's try rank values from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    
    best_result = None
    best_rank = schoolbook_rank

    # Create the schoolbook decomposition as fallback
    def schoolbook():
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

    fallback = schoolbook()

    # Try to find lower rank via random initialization and local search
    # We'll use the fact that for small ranks, we can try to solve the system
    
    # For a given rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that for all (i,j,k): sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    
    # This is a system of n*m*p = 90 equations with 3*r*(nm+mp+pn) unknowns
    # For r=30: 3*30*(15+30+18) = 3*30*63 = 5670 unknowns, 90 equations - very underdetermined
    
    # Let's try a few random initializations for various ranks
    
    np.random.seed(42)
    
    for rank in range(30, min(schoolbook_rank, 60)):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Do a few steps of gradient descent to minimize the error
            lr = 0.01
            for step in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute error and gradients
                err = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            
                            # e = sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r] - 1
                            e = 0.0
                            for r in range(rank):
                                e += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                            e -= 1.0
                            err += e * e
                            
                            for r in range(rank):
                                grad_U[idx_u, r] += 2 * e * V[idx_v, r] * W[idx_w, r]
                                grad_V[idx_v, r] += 2 * e * U[idx_u, r] * W[idx_w, r]
                                grad_W[idx_w, r] += 2 * e * U[idx_u, r] * V[idx_v, r]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if err < 1e-10:
                    break
            
            if err < 1e-10:
                # Found a valid decomposition!
                # Convert to integer/half-integer if possible
                U_int = U.astype(int)
                V_int = V.astype(int)
                W_int = W.astype(int)
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(rank):
                                s += U_int[i*m+j][r] * V_int[j*p+k][r] * W_int[k*n+i][r]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = rank
                    best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    break
        
        if best_result is not None:
            break

    if best_result is not None:
        return best_result
    else:
        return fallback
# EVOLVE-BLOCK-END
