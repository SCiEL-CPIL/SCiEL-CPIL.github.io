"""Matrix multiplication 3x5x7 rank-minimal decomposition search."""
import numpy as np
import time
import itertools
from fractions import Fraction
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with schoolbook rank = n*m*p = 105
    # Try to find lower rank via random search + local optimization
    
    deadline = time.time() + 110  # safety margin
    
    # Build the target tensor as a 3D array
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Flatten indices
    nrm = n * m  # 15
    mpr = m * p  # 35
    pnr = p * n  # 21
    
    # Schoolbook solution
    rank_school = n * m * p
    U_sch = np.zeros((nrm, rank_school))
    V_sch = np.zeros((mpr, rank_school))
    W_sch = np.zeros((pnr, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1.0
                V_sch[j*p+k, col] = 1.0
                W_sch[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U = U_sch.copy()
    best_V = V_sch.copy()
    best_W = W_sch.copy()
    
    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the target tensor."""
        r = U.shape[1]
        # Compute sum_c U[:,c] * V[:,c] * W[:,c] as a 3D tensor
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for c in range(r):
            u_vec = U[:, c]  # shape (n*m,)
            v_vec = V[:, c]  # shape (m*p,)
            w_vec = W[:, c]  # shape (p*n,)
            # u_vec[i*m+j], v_vec[j*p+k], w_vec[k*n+i]
            # Reshape to 2D: u_2d[i,j] = u_vec[i*m+j]
            u_2d = u_vec.reshape(n, m)
            v_2d = v_vec.reshape(m, p)
            w_2d = w_vec.reshape(p, n)
            # T[i,j,k] += u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i,j,k] += u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        return np.allclose(T_recon, T, atol=1e-9, rtol=1e-9)
    
    def check_decomposition_fast(U, V, W):
        """Faster check using vectorized operations."""
        r = U.shape[1]
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for c in range(r):
            u_2d = U[:, c].reshape(n, m)
            v_2d = V[:, c].reshape(m, p)
            w_2d = W[:, c].reshape(p, n)
            # T[i,j,k] += u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
            for i in range(n):
                for j in range(m):
                    T_recon[i,j,:] += u_2d[i,j] * v_2d[j,:] * w_2d[:,i]
        return np.allclose(T_recon, T, atol=1e-9, rtol=1e-9)
    
    # Try ranks from 105 down
    for rank in range(rank_school, 40, -1):
        if time.time() > deadline - 10:
            break
        
        # Try random initialization and iterative refinement
        max_iters = 500
        found = False
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small entries
            U = np.random.randn(nrm, rank) * 0.1
            V = np.random.randn(mpr, rank) * 0.1
            W = np.random.randn(pnr, rank) * 0.1
            
            # Iterative refinement: fix two, optimize the third
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Optimize U given V, W
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed V, W: U[i*m+j,c] should satisfy
                # sum_c U[i*m+j,c] * (sum_k V[j*p+k,c]*W[k*n+i,c]) = T[i,j,k] for all k
                # This is complex. Instead, use gradient-free approach.
                
                # Simple approach: try to reduce rank by merging terms
                # Or use a different strategy: start from schoolbook and try to eliminate terms
                
                break  # Placeholder for now
            
            # Alternative: direct search for low-rank solutions
            # For rank r, we need to find U, V, W such that the tensor equation holds
            # This is a system of n*m*p = 105 equations with r*(n*m + m*p + p*n) = r*71 unknowns
            # For r = 15: 1072 unknowns, 105 equations - underdetermined
            # For r = 10: 710 unknowns, 105 equations - very underdetermined
            # For r = 5: 355 unknowns, 105 equations - underdetermined
            # For r = 3: 213 unknowns, 105 equations - underdetermined
            # For r = 2: 142 unknowns, 105 equations - underdetermined
            # For r = 1: 71 unknowns, 105 equations - overdetermined
            
            # So rank 1 is impossible (71 < 105)
            # Rank 2 might be possible (142 > 105)
            
            # Let's try rank 2 with a structured approach
            if rank <= 2 and not found:
                # Try rank 2
                for a in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
                    for b in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
                        U2 = np.zeros((nrm, 2))
                        V2 = np.zeros((mpr, 2))
                        W2 = np.zeros((pnr, 2))
                        
                        # U[i*m+j, 0] = a, U[i*m+j, 1] = b
                        # V[j*p+k, 0] = 1/a, V[j*p+k, 1] = 1/b (if a,b != 0)
                        # W[k*n+i, 0] = 1, W[k*n+i, 1] = 0
                        # Then T[i,j,k] = a*(1/a)*1 + b*(1/b)*0 = 1 + 0 = 1 ✓ (if b != 0)
                        # Wait, that's not right. Let me reconsider.
                        
                        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # With 2 terms:
                        # T[i,j,k] = U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] + U[i*m+j,1]*V[j*p+k,1]*W[k*n+i,1]
                        
                        # Try: U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1
                        #      U[i*m+j,1] = 0, V[j*p+k,1] = 0, W[k*n+i,1] = 0
                        # Then T = 1*1*1 + 0 = 1. That's rank 1!
                        
                        U1 = np.ones((nrm, 1))
                        V1 = np.ones((mpr, 1))
                        W1 = np.ones((pnr, 1))
                        if check_decomposition_fast(U1, V1, W1):
                            best_rank = 1
                            best_U = U1
                            best_V = V1
                            best_W = W1
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        
        if found:
            break
    
    # Convert to list format
    result_U = best_U.tolist()
    result_V = best_V.tolist()
    result_W = best_W.tolist()
    
    return {"U": result_U, "V": result_V, "W": result_W}
