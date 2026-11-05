import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
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
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Use tensor decomposition approach
    # The multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the 3D tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try various ranks
    for rank in range(70, min(84, int(time.time() - (deadline - 110)) > 0 and 84 or 70)):
        if time.time() > deadline - 10:
            break
        
        # Try to find rank-r decomposition
        # Use iterative refinement
        np.random.seed(rank)
        
        # Initialize with random factors
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Iterative refinement
        for it in range(50):
            if time.time() > deadline - 5:
                break
            
            # Update U
            WV = np.einsum('kr,lr->kl', W, V)  # (p*n, m*p)
            U_new = np.einsum('ikl,l->ik', T, WV)
            U = U_new
            
            # Update V  
            UW = np.einsum('ik,kr->ir', U, W)  # (n*m, p*n)
            V_new = np.einsum('ikj,k->ij', T, UW)
            V = V_new
            
            # Update W
            UV = np.einsum('ik,il->kl', U, V)  # (n*m, m*p)
            W_new = np.einsum('ikl,k->il', T, UV)
            W = W_new
        
        # Check if decomposition is good
        T_recon = np.einsum('ik,jk,lk->ijl', U, V, W)
        error = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
        
        if error < 1e-6:
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_recon2 = np.einsum('ik,jk,lk->ijl', U_rounded, V_rounded, W_rounded)
            error2 = np.linalg.norm(T - T_recon2) / np.linalg.norm(T)
            
            if error2 < 1e-6:
                # Convert to lists
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                # Verify
                for i in range(n*m):
                    for j in range(m*p):
                        for k in range(p*n):
                            expected = T[i,j,k]
                            actual = sum(U_list[i][r] * V_list[j][r] * W_list[k][r] for r in range(rank))
                            if abs(expected - actual) > 1e-9:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break
    
    return best_result
