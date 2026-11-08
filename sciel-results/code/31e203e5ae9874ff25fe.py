import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]
DEADLINE = 150

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = nm * mp  # n*m*m*p = 108 for 3,3,4
    
    # Schoolbook decomposition as baseline
    U0 = np.zeros((nm, schoolbook_rank))
    V0 = np.zeros((mp, schoolbook_rank))
    W0 = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1
                V0[j*p + k, col] = 1
                W0[k*n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U0, V0, W0
    
    start_time = time.time()
    
    # Strategy: Use the known rank-23 algorithm for 3x3x3 matrix multiplication
    # and combine with vector-matrix multiplication for the extra column
    # 
    # For 3x3x3, the Strassen-based algorithm has rank 23 (over reals)
    # For the 4th column, we need additional terms
    #
    # The tensor T_{i,j,k} for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # can be decomposed as sum_r U[i,j,r] * V[j,k,r]
    #
    # Let's try a direct rank minimization using random initialization and
    # iterative refinement via the tensor train / alternating least squares approach
    
    # First, let's try to use known low-rank constructions
    # For 3x3x4: we can think of it as 3x3x3 + 3x3x1
    # Rank(3x3x3) = 23 (Strassen-like), Rank(3x3x1) <= 9
    # Total <= 32, but we can do better with shared structure
    
    # Let's try a systematic search for rank reduction
    # Start with schoolbook and try to merge terms
    
    # Alternative: use the fact that for fixed k, we're doing 3x3x3 mult
    # The 3x3x3 multiplication tensor has rank at most 23
    # So total rank <= 23 + 9 = 32 if we separate, but can share
    
    # Let's try a rank-R decomposition with R from 30 down to 23
    # using alternating minimization
    
    for target_rank in range(32, 22, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Initialize with random rank-factor decomposition
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Normalize columns
            for r in range(target_rank):
                norm = np.linalg.norm(U[:, r])
                if norm > 1e-10:
                    U[:, r] /= norm
                    V[:, r] *= norm
            
            # Alternating minimization to fit the multiplication tensor
            converged = False
            for it in range(200):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Fix V and W, solve for U
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # For fixed V, W: U[i,j,:] should minimize
                # sum_k (T[i,j,k] - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r])^2
                # This is a linear system in U[i,j,:]
                
                # Build the linear system
                # Let a_r = V[j,k,r]*W[k,i,r] for each r
                # Then sum_r U[i,j,r]*a_r = T[i,j,k] for all k
                # But T depends on i,j,k... this is not straightforward
                # because W depends on i as well.
                
                # Let's use a different approach: gradient descent
                # T_{ijk} = sum_r U_{ij}^r * V_{jk}^r * W_{ki}^r
                # (using W[k,i,r] = W_{ki}^r)
                
                lr = 0.01
                for r in range(target_rank):
                    # Gradient of U
                    dU = np.zeros((nm, 1))
                    for i in range(n):
                        for j in range(m):
                            g = 0.0
                            for k in range(p):
                                # T_ijk = delta(i,k) * A_ik... no
                                # The tensor is: T[i,j,k] = 1 if this is a valid mult
                                # Actually the multiplication tensor has T[i,j,k] = 1
                                # meaning (A*B)_{ik} = sum_j A_{ij} B_{jk}
                                # So T[i,j,k] = 1 for all i,j,k
                                # And the decomposition: 1 = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                                # Wait, that's not right either.
                                # 
                                # The matrix multiplication tensor T has T[i,j,k] = 1
                                # and we want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                # So for each (i,j,k): sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1
                                
                                g += (1.0 - sum(U[i*m+j, s]*V[j*p+k, s]*W[k*n+i, s] for s in range(target_rank))) * V[j*p+k, r] * W[k*n+i, r]
                            dU[i*m+j, 0] = g
                    
                    U -= lr * dU * 0.01  # Small step
                
                # This is too slow. Let's use a vectorized approach.
                break  # Skip the slow gradient descent
            
            # Let's try a completely different approach:
            # Use the known 3x3x3 rank-23 construction and extend to 4 columns
            
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # 3x3x3 Strassen-like rank 23 construction
            # This is complex, so let's use a simpler approach:
            # Try to find a low-rank decomposition via random projection and SVD
            
            # Build the full tensor
            T = np.ones((n, m, p))  # T[i,j,k] = 1 for all
            
            # Reshape to matrix: mode-1 (i*m) x mode-2 (j*p) ... no
            # The tensor is 3-way: T[i, j, k]
            # Mode-1 unfolding: matrix of size (n*m) x (p) ... no
            # Let's think: T is n x m x p
            # We want: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is a 3-way tensor decomposition with a specific structure
            
            # Let's just try to verify the schoolbook and return it
            # since the search is complex and time-limited
            
    # Return the schoolbook decomposition (safest)
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
