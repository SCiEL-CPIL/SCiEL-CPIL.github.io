"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[(i,j),(j',k),(k',i')] = 1 if j==j' and i==i' and k==k', else 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    
    # Try to find minimum rank decomposition
    # T[alpha, beta, gamma] = sum_c U[alpha,c] * V[beta,c] * W[gamma,c]
    
    # For a given rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that for all alpha,beta,gamma:
    #   sum_c U[alpha,c]*V[beta,c]*W[gamma,c] = T[alpha,beta,gamma]
    
    # This is a bilinear system in (U,V) for fixed W, etc.
    # Let's try a direct approach: fix W, then the problem becomes linear in U and V.
    
    # Actually, let's try a simpler approach: use the known structure.
    # The schoolbook decomposition has rank 60. Let's see if we can do better.
    
    # Try ranks from a lower bound up to 60
    # Lower bound: max(nm, mp, pn) = 20
    
    best_rank = nm * mp * pn  # fallback to schoolbook
    best_result = None
    
    deadline = time.time() + 100  # 100 second deadline
    
    # Try to find a decomposition of rank r
    def try_rank(r):
        """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Iterate: fix two, solve for the third
            for iteration in range(100):
                if time.time() > deadline:
                    return None
                
                # Fix U and V, solve for W
                # For each gamma: sum_c U[alpha,c]*V[beta,c]*W[gamma,c] = T[alpha,beta,gamma]
                # This is: (U * V^T) contracted... 
                # Let M[alpha,beta] = sum_c U[alpha,c]*V[beta,c]
                # Then sum_c W[gamma,c] * M[alpha,beta] ... no, that's not right.
                
                # Actually: sum_c U[alpha,c]*V[beta,c]*W[gamma,c] 
                # = sum_c (U[alpha,c]*V[beta,c]) * W[gamma,c]
                
                # For fixed U,V, define for each c: M_c[alpha,beta] = U[alpha,c]*V[beta,c]
                # Then sum_c M_c[alpha,beta] * W[gamma,c] = T[alpha,beta,gamma]
                
                # This is linear in W[gamma,:] for each gamma:
                # For each (alpha,beta,gamma): sum_c (U[alpha,c]*V[beta,c]) * W[gamma,c] = T[alpha,beta,gamma]
                
                # Group by gamma: for each gamma, we have a system in W[gamma,:]:
                # sum_c W[gamma,c] * (U[alpha,c]*V[beta,c]) = T[alpha,beta,gamma] for all alpha,beta
                
                # So for each gamma, A_gamma[c, (alpha,beta)] = U[alpha,c]*V[beta,c]
                # And we want A_gamma @ W[gamma,:] = T[:,:,gamma].ravel()
                
                # Build A_gamma for each gamma
                W_new = np.zeros((pn, r))
                for gamma in range(pn):
                    # A_gamma is r x (nm*mp)
                    A_gamma = np.zeros((r, nm * mp))
                    for c in range(r):
                        for alpha in range(nm):
                            for beta in range(mp):
                                A_gamma[c, alpha * mp + beta] = U[alpha, c] * V[beta, c]
                    
                    b_gamma = T[:, :, gamma].ravel()
                    
                    # Solve A_gamma @ W[gamma,:] = b_gamma
                    try:
                        W_new[gamma, :] = np.linalg.lstsq(A_gamma, b_gamma, rcond=None)[0]
                    except:
                        return None
                
                # Fix V and W, solve for U
                U_new = np.zeros((nm, r))
                for alpha in range(nm):
                    A_alpha = np.zeros((r, mp * pn))
                    for c in range(r):
                        for beta in range(mp):
                            for gamma in range(pn):
                                A_alpha[c, beta * pn + gamma] = V[beta, c] * W[gamma, c]
                    
                    b_alpha = T[alpha, :, :].ravel()
                    
                    try:
                        U_new[alpha, :] = np.linalg.lstsq(A_alpha, b_alpha, rcond=None)[0]
                    except:
                        return None
                
                # Fix U and W, solve for V
                V_new = np.zeros((mp, r))
                for beta in range(mp):
                    A_beta = np.zeros((r, nm * pn))
                    for c in range(r):
                        for alpha in range(nm):
                            for gamma in range(pn):
                                A_beta[c, alpha * pn + gamma] = U[alpha, c] * W[gamma, c]
                    
                    b_beta = T[:, beta, :].ravel()
                    
                    try:
                        V_new[beta, :] = np.linalg.lstsq(A_beta, b_beta, rcond=None)[0]
                    except:
                        return None
                
                # Check convergence
                U = U_new
                V = V_new
                W = W_new
            
            # Verify the solution
            max_err = 0.0
            for alpha in range(nm):
                for beta in range(mp):
                    for gamma in range(pn):
                        val = np.sum(U[alpha, :] * V[beta, :] * W[gamma, :])
                        err = abs(val - T[alpha, beta, gamma])
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                return (U, V, W)
        
        return None
    
    # Try ranks from 20 to 60
    for r in range(20, 61):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # If no better found, use schoolbook
    if best_result is None:
        rank = nm * mp * pn
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_result = (U, V, W)
    
    U, V, W = best_result
    
    # Convert to lists of floats (or ints if they are integers)
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                # Round to nearest integer if close
                if abs(val - round(val)) < 1e-6:
                    r.append(int(round(val)))
                else:
                    r.append(float(val))
            result.append(r)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
