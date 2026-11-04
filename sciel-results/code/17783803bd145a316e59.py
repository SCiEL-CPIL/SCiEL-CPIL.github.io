"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Also rank >= n + m + p - 2 = 10 (trivial)
    # Strassen-like: for 2x2x2, rank 7. For larger, we search.
    
    # We'll try to find a decomposition with rank < schoolbook_rank
    # Strategy: 
    # 1. Start with structured random tensors
    # 2. Use alternating minimization / gradient descent on the factor matrices
    # 3. Verify exactness with rational/integer arithmetic
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Build the target tensor T[i,j,k] for matrix mult: T[i,j,k] = delta_{i,k} (if we think of it as A[i,j]*B[j,k])
    # Actually the matrix multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] where output index is (i,k), first input (i,j), second input (j,k)
    # T[i,j,k] = 1 if the "k" index in B matches... 
    # Standard: M_{n,m,p} tensor: T[a,b,c] where a in [n], b in [m], c in [p]
    # T[a,b,c] = 1 if b is the "shared" index... 
    # For C = A * B: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) where the multiplication contributes
    # Actually T[i,j,k] = delta (it's always 1 for the structure, but the "value" is that 
    # tensor product gives C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? No, that's not right.
    
    # The matrix multiplication tensor: T in R^{n x m x p} where
    # T[i,j,k] = 1 if we're multiplying A[i,j] by B[j,k] to get contribution to C[i,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that would mean every entry is 1, which gives rank n*m*p with the trivial decomposition.
    
    # Actually, the standard matrix multiplication tensor M_{n,m,p} is defined as:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # No, that's the all-ones tensor.
    
    # Let me reconsider. The matrix multiplication tensor for C = A*B where A is n×m and B is m×p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T such that C = T(A,B) has:
    # T[i,j,k] = 1 for all i,j,k (since each A[i,j]*B[j,k] contributes to C[i,k])
    
    # So the tensor is indeed the all-ones tensor of shape (n, m, p)?? 
    # No! That's not right either. Let me think more carefully.
    
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map from R^{n×m} × R^{m×p} -> R^{n×p}
    # As a tensor T in R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}, 
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'... 
    
    # Actually, the standard way: the matrix multiplication tensor M_{n,m,p} is an element of
    # R^{nm} ⊗ R^{mp} ⊗ R^{np} (or with indices arranged differently)
    # where M_{n,m,p}[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i' and j=j' and k=k'
    
    # So in flattened form:
    # First index: a = i*m + j (row i, col j of A), ranges [0, nm)
    # Second index: b = j'*p + k (row j', col k of B), ranges [0, mp)
    # Third index: c = k'*n + i' (row k', col i' of C), ranges [0, np)
    
    # M[a, b, c] = 1 if:
    #   i = a // m, j = a % m
    #   j' = b // p, k = b % p
    #   k' = c // n, i' = c % n
    #   and i == i' AND j == j' AND k == k'
    
    # So M[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0.
    
    # This is exactly the schoolbook decomposition: each term e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # Now, for the factorization:
    # We want U (nm × r), V (mp × r), W (np × r) such that
    # T = sum_{t=0}^{r-1} U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # i.e., T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    
    # Let's build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try various ranks and search strategies
    # Start from lower ranks and go up
    min_possible_rank = max(n*m, m*p, p*n)  # 20
    
    # Try ranks from min_possible_rank upward
    for target_rank in range(min_possible_rank, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Multiple random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with structured random factors
            # Use sparse initialization to help convergence
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Add some structured components: include the schoolbook terms sparsely
            # Randomly select some of the schoolbook basis vectors
            if target_rank <= schoolbook_rank:
                # Use a subset of schoolbook terms as initial structure
                n_struct = min(target_rank, schoolbook_rank)
                indices = random.sample(range(schoolbook_rank), n_struct)
                for idx in indices:
                    i = idx // (m*p)
                    jk = idx % (m*p)
                    j = jk // p
                    k = jk % p
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, idx] = 1.0
                    V[b, idx] = 1.0
                    W[c, idx] = 1.0
                # Randomize the rest
                for t in range(n_struct, target_rank):
                    U[:, t] = np.random.randn(n*m) * 0.05
                    V[:, t] = np.random.randn(m*p) * 0.05
                    W[:, t] = np.random.randn(p*n) * 0.05
            
            # Alternating minimization
            # For each factor, given the other two, solve a linear system
            # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # Fix V and W: for each a, T[a,:,:] = sum_t U[a,t] (V[:,t] ⊗ W[:,t])
            # This is: vec(T[a,:,:]) = sum_t U[a,t] * vec(V[:,t] ⊗ W[:,t])
            # So we have a linear system in U[a,:]
            
            # Build the Gram-like matrix for alternating minimization
            # For U: fix V, W. For each a:
            # T[a, b, c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # Let's define for each t: v_t = V[:,t] (mp-vector), w_t = W[:,t] (np-vector)
            # outer = v_t ⊗ w_t (mp*np vector)
            # T[a, :, :] reshaped to (mp*np,) = sum_t U[a,t] * outer_t
            # So: G_U[t1, t2] = sum_{b,c} V[b,t1]*V[b,t2]*W[c,t1]*W[c,t2]
            #     rhs[a, t] = sum_{b,c} T[a,b,c] * V[b,t] * W[c,t]
            
            max_iters = 50
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                    
                # Update U
                # G_U[t1,t2] = (V[:,t1] * V[:,t2]).dot() * (W[:,t1] * W[:,t2]).dot()
                # Actually: sum_{b,c} V[b,t1] V[b,t2] W[c,t1] W[c,t2]
                #           = (sum_b V[b,t1] V[b,t2]) * (sum_c W[c,t1] W[c,t2])
                VV = V.T @ V  # (r, r)
                WW = W.T @ W  # (r, r)
                G_U = VV * WW  # element-wise product, (r, r)
                
                # rhs[a, t] = sum_{b,c} T[a,b,c] V[b,t] W[c,t]
                # T has shape (nm, mp, np)
                # For each a: sum_{b,c} T[a,b,c] V[b,t] W[c,t]
                # = sum_b V[b,t] * (sum_c T[a,b,c] W[c,t])
                # Let's compute: tmp[a, b, t] = sum_c T[a,b,c] W[c,t]
                # Then rhs[a, t] = sum_b tmp[a,b,t] V[b,t]
                
                # Efficient: tmp = np.einsum('abc,ct->abt', T, W)
                # rhs = np.einsum('abt,bt->at', tmp, V)
                
                try:
                    tmp = np.einsum('abc,ct->abt', T, W, optimize=True)
                    rhs_U = np.einsum('abt,bt->at', tmp, V, optimize=True)
                    
                    # Solve G_U @ U.T = rhs_U.T
                    # U.T has shape (r, nm)
                    try:
                        U_new = np.linalg.solve(G_U + 1e-10 * np.eye(target_rank), rhs_U.T).T
                    except np.linalg.LinAlgError:
                        U_new = np.linalg.lstsq(G_U + 1e-10 * np.eye(target_rank), rhs_U.T, rcond=None)[0].T
                    U = U_new
                    
                except Exception:
                    break
                
                # Update V
                # Similarly: G_V[t1,t2] = (U[:,t1]*U[:,t2]).dot() * (W[:,t1]*W[:,t2]).dot()
                UU = U.T @ U
                G_V = UU * WW
                tmp = np.einsum('abc,ct->abt', T, W, optimize=True)
                rhs_V = np.einsum('abt,at->bt', tmp, U, optimize=True)
                
                try:
                    V_new = np.linalg.solve(G_V + 1e-10 * np.eye(target_rank), rhs_V.T).T
                except np.linalg.LinAlgError:
                    V_new = np.linalg.lstsq(G_V + 1e-10 * np.eye(target_rank), rhs_V.T, rcond=None)[0].T
                V = V_new
                
                # Update W
                # G_W[t1,t2] = (U[:,t1]*U[:,t2]).dot() * (V[:,t1]*V[:,t2]).dot()
                G_W = UU * VV
                tmp = np.einsum('abc,bt->act', T, V, optimize=True)
                rhs_W = np.einsum('act,at->ct', tmp, U, optimize=True)
                
                try:
                    W_new = np.linalg.solve(G_W + 1e-10 * np.eye(target_rank), rhs_W.T).T
                except np.linalg.LinAlgError:
                    W_new = np.linalg.lstsq(G_W + 1e-10 * np.eye(target_rank), rhs_W.T, rcond=None)[0].T
                W = W_new
                
                # Check convergence / error
                if it % 5 == 0 or it == max_iters - 1:
                    # Compute reconstruction error
                    # T_recon[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                    try:
                        T_recon = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
                        err = np.max(np.abs(T_recon - T))
                        if err < 1e-6:
                            # Found a good solution! Try to clean it up
                            # Check if entries are close to integers or half-integers
                            # Round and verify
                            U_rounded = np.round(U * 2) / 2
                            V_rounded = np.round(V * 2) / 2
                            W_rounded = np.round(W * 2) / 2
                            T_rounded = np.einsum('at,bt,ct->abc', U_rounded, V_rounded, W_rounded, optimize=True)
                            err_rounded = np.max(np.abs(T_rounded - T))
                            if err_rounded < 1e-9:
                                # Success!
                                result = {
                                    "U": U_rounded.tolist(),
                                    "V": V_rounded.tolist(),
                                    "W": W_rounded.tolist()
                                }
                                if best_result is None or target_rank < best_rank:
                                    best_result = result
                                    best_rank = target_rank
                                break
                    except Exception:
                        pass
            else:
                continue
            break  # break out of restart loop if we found a solution
        
        if best_result is not None and target_rank == best_rank:
            break
    
    # If no better solution found, return schoolbook
    if best_result is None:
        U = [[0] * schoolbook_rank for _ in range(n*m)]
        V = [[0] * schoolbook_rank for _ in range(m*p)]
        W = [[0] * schoolbook_rank for _ in range(p*n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
