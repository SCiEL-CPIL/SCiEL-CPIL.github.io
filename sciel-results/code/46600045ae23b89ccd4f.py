"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p  # 48
    
    # Try to find a lower rank decomposition
    # The tensor T[i*m+j, k*n+i] structure... 
    # Actually, let's build the target tensor and try ALS or other methods
    
    # Build target multiplication tensor
    # T[a, b, c] where a = i*m+j (input A index), b = j*p+k (input B index), c = k*n+i (output index)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    
    # For matrix mult (n,m,p): 
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = 1
    # Flattened: U index = i*m + j, V index = j*p + k, W index = k*n + i
    
    # Target tensor in flattened form
    # T_flat[a, b, c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Try various ranks from low to high
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # Deadline guard
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try to find exact decomposition at lower ranks
    # Use randomized ALS / gradient-based approach
    
    for rank in range(20, rank_school + 1):
        if time.time() > deadline:
            break
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline:
                break
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS iteration
            for it in range(200):
                if time.time() > deadline:
                    break
                # Update U
                # For each a, U[a,:] should satisfy: sum_c W[c,:]^T * (target[a,:,c] reshaped) 
                # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
                # For fixed a: sum_r U[a,r] * V[:,r] outer W[:,r] = target[a,:,:]
                # => U[a,:] = (target[a,:,:] * (V^T W elementwise... ))
                # Actually: target[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r]^T)
                # This is: target[a] = U[a] @ (V^T * W) where V^T*W is rank x (m*p*p*n)
                # Better: vectorize
                # target[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix U: for each r, V[:,r] and W[:,r]...
                
                # Simpler ALS:
                # Fix V, W: U[a,r] = sum_{b,c} target[a,b,c] V[b,r] W[c,r] / sum_{b,c} V[b,r]^2 W[c,r]^2
                VW = np.einsum('br,cr->bc', V, W)  # (m*p, p*n)
                # U[a,r] = sum_{b,c} target[a,b,c] * V[b,r] * W[c,r] / norm
                # = einsum('abc,br,cr->ar', target, V, W) / einsum('br,cr->r', V*V, W*W)
                num_U = np.einsum('abc,br,cr->ar', target, V, W)
                den_U = np.einsum('br,cr->r', V*V, W*W)
                den_U[den_U < 1e-10] = 1e-10
                U_new = num_U / den_U
                
                # Fix U, W: V[b,r] = sum_{a,c} target[a,b,c] U[a,r] W[c,r] / norm
                num_V = np.einsum('abc,ar,cr->br', target, U_new, W)
                den_V = np.einsum('ar,cr->r', U_new*U_new, W*W)
                den_V[den_V < 1e-10] = 1e-10
                V_new = num_V / den_V
                
                # Fix U, V: W[c,r] = sum_{a,b} target[a,b,c] U[a,r] V[b,r] / norm
                num_W = np.einsum('abc,ar,br->cr', target, U_new, V_new)
                den_W = np.einsum('ar,br->r', U_new*U_new, V_new*V_new)
                den_W[den_W < 1e-10] = 1e-10
                W_new = num_W / den_W
                
                # Check convergence
                diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
                U, V, W = U_new, V_new, W_new
                if diff < 1e-12:
                    break
            
            # Check if decomposition is exact
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - target))
            if err < 1e-9:
                # Found exact decomposition at this rank
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
    
    # If no better found, use schoolbook
    if best_U is None:
        rank = rank_school
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
