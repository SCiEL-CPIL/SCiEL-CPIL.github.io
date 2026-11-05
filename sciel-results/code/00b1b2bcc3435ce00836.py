import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # Actually T[i*m+j, j*p+k, k*n+i] = 1
    
    # For the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to find lower rank using a structured approach
    # Key idea: look for linear combinations of schoolbook terms that cancel
    
    # For each pair of terms, check if they can be combined
    # Term (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    
    # Two terms (i1,j1,k1) and (i2,j2,k2) can potentially be combined if
    # they share some indices, allowing a 2-term rank-1 decomposition to replace them
    
    # Let's try a randomized search for rank 27
    for target_rank in range(27, 24, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline - 15:
                break
            if found:
                break
                
            # Start from schoolbook and try to merge terms
            # Pick target_rank random "groups" of schoolbook terms
            indices = list(range(rank_sb))
            random.shuffle(indices)
            
            # Create groups
            groups = []
            for c in range(target_rank):
                start = c * (rank_sb // target_rank)
                end = start + (rank_sb // target_rank)
                if c == target_rank - 1:
                    end = rank_sb
                groups.append(indices[start:end])
            
            # For each group, find a rank-1 tensor that approximates the sum
            U_new = np.zeros((n*m, target_rank))
            V_new = np.zeros((m*p, target_rank))
            W_new = np.zeros((p*n, target_rank))
            
            for c, group in enumerate(groups):
                if not group:
                    continue
                # Sum of rank-1 tensors in the group
                # We need to find u, v, w such that u*v*w ≈ sum of the group's tensors
                # This is a rank-1 approximation problem
                
                # Build the sum tensor
                T_sum = np.zeros((n*m, m*p, p*n))
                for col_idx in group:
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_sum[i*m+j, j*p+k, k*n+i] += 1
                
                # Find best rank-1 approximation using SVD-like approach
                # Reshape and use SVD
                # This is approximate, so let's try a simple approach
                
                # Initialize with the first term's structure
                i0, j0, k0 = group[0] // (m*p), (group[0] // p) % m, group[0] % p
                # Actually this is getting complicated. Let me try a different approach.
                
                # Simple: just use the average
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                for col_idx in group:
                    i = col_idx // (m*p)
                    rem = col_idx % (m*p)
                    j = rem // p
                    k = rem % p
                    u[i*m+j] += 1
                    v[j*p+k] += 1
                    w[k*n+i] += 1
                
                scale = len(group)
                U_new[:, c] = u / scale
                V_new[:, c] = v / scale
                W_new[:, c] = w / scale
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_new[i*m+j], V_new[j*p+k] * W_new[k*n+i])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
