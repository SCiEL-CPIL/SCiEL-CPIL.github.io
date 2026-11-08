"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _find_rank_11_233(time_limit=30.0):
    """
    Attempts to find a rank 11 decomposition for 2x3x3 matrix multiplication.
    Returns (U, V, W) if found and valid, else None.
    U: (6, 11), V: (9, 11), W: (6, 11)
    """
    n, m, p = 2, 3, 3
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1 # Tensor T_{ijk} = 1 if delta? No.
                # The tensor for matrix mult is T_{i,j,k} = 1 if i,j,k correspond to valid multiplication?
                # No, the identity tensor for matrix mult is such that 
                # sum_{j} T_{i,j,k} A_{i,j} B_{j,k} is not quite right.
                # The structure is: C_{ik} = sum_j A_{ij} B_{jk}.
                # This is a 3-tensor T with indices i, j, k.
                # T_{i,j,k} = 1 if the term A_{ij} B_{jk} appears.
                # Since every (i,j,k) triplet is valid, T_{i,j,k} = 1 for all i,j,k?
                # Wait. The schoolbook decomposition has rank nmp.
                # The tensor T is defined by T[i,j,k] = 1?
                # Let's check the contraction in the seed code.
                # U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
                # Sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 if i,j,k is the specific triplet.
                # So the target tensor T has T[i,j,k] = 1 for all i,j,k.
                # Yes, because for every i,j,k, there is a rank-1 component 1*1*1.
                pass
    
    # Actually, the target tensor T is just all ones?
    # Let's verify.
    # If T[i,j,k] = 1 for all i,j,k, then the tensor is the "all-ones" tensor.
    # Is the matrix multiplication tensor the all-ones tensor?
    # No. The matrix multiplication tensor is usually defined such that 
    # <T, A \otimes B> = vec(A B).
    # The components are T_{i,j,k} = 1?
    # Let's test with n=1, m=1, p=1. T[0,0,0]=1. Rank 1. Correct.
    # n=1, m=2, p=1. T[i=0, j=0, k=0]=1, T[0,1,0]=1.
    # C = A B. A is 1x2, B is 2x1. C is 1x1.
    # C = A0 B0 + A1 B1.
    # Tensor T has non-zeros at (0,0,0) and (0,1,0).
    # So T[i,j,k] = 1 if the index combination is valid.
    # For 2x3x3, all i in 0..1, j in 0..2, k in 0..2 are valid.
    # So T is indeed all ones.
    
    T = np.ones((n, m, p))
    T_flat = T.flatten() # Shape (18,)
    
    R = 11
    U_size = n * m # 6
    V_size = m * p # 9
    W_size = p * n # 6
    
    best_err = float('inf')
    best_params = None
    start_time = time.time()
    
    # Random restarts
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize randomly
        U = np.random.randn(U_size, R)
        V = np.random.randn(V_size, R)
        W = np.random.randn(W_size, R)
        
        # Newton/Gauss-Newton steps
        for step in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current decomposition
            # Res = T_flat - sum_r U[:,r] \otimes V[:,r] \otimes W[:,r] flattened
            # We want to minimize || Res ||^2
            
            # Gradient calculation
            # Let f = sum_r u_r v_r^T w_r (tensor)
            # Res = T - f
            # Grad_U[:,r] = Res \cdot_{(1)} (v_r \otimes w_r) ?
            
            # Efficient way:
            # Res_matrix_i = sum_{j,k} Res[i,j,k] * ...
            
            # Let's use a simple numerical gradient or analytic gradient.
            # Analytic gradient for U:
            # d/dU_{a,r} (sum_{i,j,k} (T_{ijk} - sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r})^2)
            # = -2 sum_{j,k} Res[i,j,k] V_{j,k,r} W_{k,i,r}
            
            Res = T.copy()
            # Compute decomposition
            for r in range(R):
                # u_r shape (n,m)
                # v_r shape (m,p)
                # w_r shape (p,n)
                # product is (n,m) \otimes (m,p) \otimes (p,n) -> (n,m,p)
                # This is an outer product.
                # We can do:
                # term = np.einsum('ij,jk,ki->ijk', U[:,r].reshape(n,m), V[:,r].reshape(m,p), W[:,r].reshape(p,n))
                # Res -= term
                pass # Too slow if done in Python loop for every step.
            
            # Better: Vectorized
            # U (6, 11), V (9, 11), W (6, 11)
            # Decomposition tensor D[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            
            # We can compute this by reshaping and einsum
            U_reshaped = U.reshape(n, m, R)
            V_reshaped = V.reshape(m, p, R)
            W_reshaped = W.reshape(p, n, R)
            
            D = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
            Res = T - D
            
            err = np.linalg.norm(Res)
            if err < best_err:
                best_err = err
                best_params = (U.copy(), V.copy(), W.copy())
                
            if err < 1e-10:
                break
            
            # Calculate gradients
            # Grad_U[i,j,r] = -2 sum_{k} Res[i,j,k] V[j,k,r] W[k,i,r]
            # Grad_V[j,k,r] = -2 sum_{i} Res[i,j,k] U[i,j,r] W[k,i,r]
            # Grad_W[k,i,r] = -2 sum_{j} Res[i,j,k] U[i,j,r] V[j,k,r]
            
            # Using einsum
            # Res shape (n,m,p)
            # V_reshaped shape (m,p,R)
            # W_reshaped shape (p,n,R)
            
            # Grad_U_reshaped (n,m,R)
            # Sum over k (axis 2 of Res, axis 1 of V, axis 0 of W)
            # Res[i,j,k] * V[j,k,r] * W[k,i,r]
            # Note: V index is j,k. W index is k,i.
            # Res axis: i, j, k
            # V axis: j, k, r
            # W axis: k, i, r
            
            # This contraction is tricky.
            # Let's compute intermediate sums.
            
            # For U gradient:
            # Temp1[j,k,r] = Res[i,j,k] * W[k,i,r] summed over i? No.
            # Grad_U[i,j,r] = -2 * sum_k Res[i,j,k] * V[j,k,r] * W[k,i,r]
            
            # We can do:
            # X[i,j,k,r] = Res[i,j,k] * W[k,i,r]
            # X = np.einsum('ijk,kir->ijkr', Res, W_reshaped)
            # Then Grad_U[i,j,r] = -2 * sum_k X[i,j,k,r] * V[j,k,r]
            # Grad_U = -2 * np.einsum('ijkr,jkr->ijr', X, V_reshaped)
            
            X_U = np.einsum('ijk,kir->ijkr', Res, W_reshaped, optimize=True)
            Grad_U_reshaped = -2 * np.einsum('ijkr,jkr->ijr', X_U, V_reshaped, optimize=True)
            Grad_U = Grad_U_reshaped.reshape(U_size, R)
            
            # For V gradient:
            # Grad_V[j,k,r] = -2 * sum_i Res[i,j,k] * U[i,j,r] * W[k,i,r]
            # Y[i,j,k,r] = Res[i,j,k] * U[i,j,r]
            # Y = np.einsum('ijk,ijr->ijkr', Res, U_reshaped)
            # Grad_V[j,k,r] = -2 * sum_i Y[i,j,k,r] * W[k,i,r]
            # Grad_V = -2 * np.einsum('ijkr,kir->jkr', Y, W_reshaped)
            
            Y_V = np.einsum('ijk,ijr->ijkr', Res, U_reshaped, optimize=True)
            Grad_V_reshaped = -2 * np.einsum('ijkr,kir->jkr', Y_V, W_reshaped, optimize=True)
            Grad_V = Grad_V_reshaped.reshape(V_size, R)
            
            # For W gradient:
            # Grad_W[k,i,r] = -2 * sum_j Res[i,j,k] * U[i,j,r] * V[j,k,r]
            # Z[i,j,k,r] = Res[i,j,k] * V[j,k,r]
            # Z = np.einsum('ijk,jkr->ijkr', Res, V_reshaped)
            # Grad_W[k,i,r] = -2 * sum_j Z[i,j,k,r] * U[i,j,r]
            # Grad_W = -2 * np.einsum('ijkr,ijr->kir', Z, U_reshaped)
            
            Z_W = np.einsum('ijk,jkr->ijkr', Res, V_reshaped, optimize=True)
            Grad_W_reshaped = -2 * np.einsum('ijkr,ijr->kir', Z_W, U_reshaped, optimize=True)
            Grad_W = Grad_W_reshaped.reshape(W_size, R)
            
            # Simple gradient step with adaptive learning rate
            # Or use scipy optimize if available? No, just simple.
            # Let's try a small step
            lr = 0.01
            U -= lr * Grad_U
            V -= lr * Grad_V
            W -= lr * Grad_W
            
        if best_err < 1e-8:
            # Found a good candidate, try to polish or round
            break
            
    if best_err < 1e-8:
        U, V, W = best_params
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify exactness
        U_r = U_rounded.reshape(n, m, R)
        V_r = V_rounded.reshape(m, p, R)
        W_r = W_rounded.reshape(p, n, R)
        D_r = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        if np.allclose(D_r, T, atol=1e-9):
            return U_rounded, V_rounded, W_rounded
    return None

def _construct_236_from_233(U_233, V_233, W_233):
    """
    Constructs 2x3x6 factors from two 2x3x3 decompositions.
    U_233, V_233, W_233 are (6,11), (9,11), (6,11) for the first half.
    We assume the second half is identical or we find another one.
    For simplicity and speed, we use the same decomposition for both halves 
    but shift the indices in V and W.
    """
    n, m, p = 2, 3, 6
    R_233 = U_233.shape[1]
    R_total = 2 * R_233
    
    U_236 = np.zeros((n*m, R_total))
    V_236 = np.zeros((m*p, R_total))
    W_236 = np.zeros((p*n, R_total))
    
    # First half (cols 0..2 of B)
    # U is same
    U_236[:, 0:R_233] = U_233
    
    # V maps j,k (k in 0..2) to j*6+k
    # V_233 has shape (9, 11) with indices j*3+k
    # We need to place V_233[j*3+k] into V_236[j*6+k]
    for j in range(m):
        for k in range(3):
            idx_233 = j * 3 + k
            idx_236 = j * 6 + k
            V_236[idx_236, 0:R_233] = V_233[idx_233, :]
            
    # W maps k,i (k in 0..2) to k*2+i
    # W_233 has shape (6, 11) with indices k*2+i
    # W_236 needs indices k*2+i.
    # So W_236[:, 0:R_233] = W_233 directly?
    # Yes, because i is 0..1, k is 0..2. Index is k*n+i.
    # In 236, k is 0..5. For first half, k is 0..2.
    # So indices are identical.
    W_236[:, 0:R_233] = W_233
    
    # Second half (cols 3..5 of B)
    # U is same
    U_236[:, R_233:R_total] = U_233
    
    # V maps j,k (k in 3..5) to j*6+k
    # k_local = k - 3 (0..2)
    for j in range(m):
        for k in range(3):
            idx_233 = j * 3 + k
            idx_236 = j * 6 + (k + 3)
            V_236[idx_236, R_233:R_total] = V_233[idx_233, :]
            
    # W maps k,i (k in 3..5) to k*2+i
    # k_local = k - 3
    for k in range(3):
        for i in range(n):
            idx_233 = k * n + i
            idx_236 = (k + 3) * n + i
            W_236[idx_236, R_233:R_total] = W_233[idx_233, :]
            
    return U_236, V_236, W_236

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    deadline = time.time() + 100 # 120s limit, keep 20s margin
    
    # 1. Try to find rank 11 decomposition for 2x3x3
    # This is the hard part.
    U_233, V_233, W_233 = _find_rank_11_233(time_limit=80.0)
    
    if U_233 is not None and time.time() < deadline:
        # 2. Construct 2x3x6 decomposition
        U, V, W = _construct_236_from_233(U_233, V_233, W_233)
        
        # Convert to list of lists with integer/half-integer values
        # The evaluator expects integer/half-integer entries.
        # Our rounding ensures this.
        
        # Final verification before returning
        # Check if U, V, W are valid
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Optional: Quick check
        # But we already verified 233.
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # 3. Fallback to Schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
