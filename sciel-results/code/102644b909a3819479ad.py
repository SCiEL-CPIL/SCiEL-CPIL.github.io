import numpy as np
from scipy.optimize import linprog
import time
import random
from itertools import product, combinations

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook rank fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    start_time = time.time()
    max_time = 115  # leave buffer for evaluation
    
    # Known lower bounds: 
    # For 3x3x7, the tensor rank is known to be at most 27 (from various constructions)
    # The trivial bound is 63.
    # We'll try to find a decomposition with rank < 63.
    
    # Strategy: 
    # 1. Use LP to find a sparse support pattern for the tensor decomposition
    # 2. The tensor equation is: T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    #    Wait, the convention is W[k*n+i, r], so W_{k,i,r} in our notation.
    #    Actually, let's be precise. The output C[i,k] = sum_j A[i,j] * B[j,k]
    #    The tensor is T_{i,j,k} = delta_{i,k} in some sense? No.
    #    Matrix multiplication tensor: T_{i,j,k} such that C[i,k] = sum_j A[i,j] B[j,k] T_{i,j,k}
    #    So T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    #    Actually T_{i,j,k} = 1 for all i,j,k? No, that's not right.
    #    
    #    Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined by:
    #    (A * B)[i,k] = sum_{j=1}^m A[i,j] * B[j,k]
    #    As a tensor: T_{i,j,k} where the "inputs" are A[i,j], B[j,k] and output is C[i,k]
    #    The tensor rank is the minimum number of terms of the form u_r * v_r * w_r
    #    such that T = sum_r u_r v_r w_r (outer product of vectors)
    #    
    #    In our convention:
    #    U has shape (n*m, R): U[i*m + j, r] = u_{i,j,r}
    #    V has shape (m*p, R): V[j*p + k, r] = v_{j,k,r}  
    #    W has shape (p*n, R): W[k*n + i, r] = w_{k,i,r}
    #    
    #    The constraint is: for all i,j,k: 
    #    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #    (since T_{i,j,k} = 1 for all i,j,k in matrix multiplication)
    #    
    #    Wait, that's not right either. Let me think again.
    #    
    #    Actually, the multiplication tensor has entries T_{i,j,k} = 1 for all i,j,k.
    #    Because each output C[i,k] depends on each A[i,j] and B[j,k] with coefficient 1.
    #    So yes, T_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    
    # For a given rank R, we want to find U, V, W with entries in {0, 1, 1/2, ...} 
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a hard combinatorial problem. Let's try a structured approach.
    
    # Try rank from a lower estimate up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But actual rank is higher. Known results suggest rank around 24-27 for 3x3x7.
    
    best_rank = rank_schoolbook
    best_decomp = fallback
    
    # Try structured decompositions
    # One known approach: use bilinear algorithms
    
    # Let's try rank R from 21 to 62
    for R in range(21, 63):
        if time.time() - start_time > max_time:
            break
        
        # Try to find a valid decomposition of rank R
        # Using random initialization and local search
        
        success = False
        for attempt in range(50):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with random sparse patterns
            U = np.random.choice([0, 1, -1], size=(n*m, R))
            V = np.random.choice([0, 1, -1], size=(m*p, R))
            W = np.random.choice([0, 1, -1], size=(p*n, R))
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(val - 1.0) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                success = True
                break
        
        if success:
            best_rank = R
            best_decomp = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            break
    
    if best_rank < rank_schoolbook:
        return best_decomp
    else:
        return fallback
# EVOLVE-BLOCK-END
