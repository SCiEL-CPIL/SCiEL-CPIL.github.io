<<<<<<< SEARCH
def _try_rank(r, n, m, p, max_iters=200, lr=0.01, seed=None):
    """Try to find a rank-r decomposition using gradient descent."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    # We want to minimize the error of the tensor decomposition.
    # The target tensor is: T[(i,j),(j',k),(k',i')] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # Error = sum over all (i,j,j',k,k',i') of (sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] - T[...])^2
    
    # This is expensive to compute fully. Let me use a sampled approach.
    
    for it in range(max_iters):
        # Sample a few conditions
        errors = []
        grads_U = np.zeros_like(U)
        grads_V = np.zeros_like(V)
        grads_W = np.zeros_like(W)
        
        for _ in range(50):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            jp = random.randint(0, m-1)
            k = random.randint(0, p-1)
            kp = random.randint(0, p-1)
            ip = random.randint(0, n-1)
            
            target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
            
            # Compute current value
            val = 0.0
            for r in range(r):
                val += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
            
            err = val - target
            
            # Gradients
            for r in range(r):
                grads_U[i*m+j, r] += err * V[jp*p+k, r] * W[kp*n+ip, r]
                grads_V[jp*p+k, r] += err * U[i*m+j, r] * W[kp*n+ip, r]
                grads_W[kp*n+ip, r] += err * U[i*m+j, r] * V[jp*p+k, r]
        
        # Update
        U -= lr * grads_U
        V -= lr * grads_V
        W -= lr * grads_W
        
        # Round to half-integers periodically
        if it % 10 == 9:
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
    
    # Final rounding
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _is_exact(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None
=======
def _try_rank(r, n, m, p, max_iters=500, lr=0.05, seed=None, momentum=0.9):
    """Try to find a rank-r decomposition using gradient descent with momentum."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5
    
    # Momentum buffers
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    # Precompute index arrays for efficiency
    # The tensor condition: for all i, j, j', k, k', i':
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = 1 if (i==i' and j==j' and k==k') else 0
    
    # We can compute gradients more efficiently using tensor contractions
    # For each r, the contribution to the tensor is U_r ⊗ V_r ⊗ W_r
    # The error tensor E[i,j,j',k,k',i'] = sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] - T[i,j,j',k,k',i']
    
    # Let's use a smarter approach: compute the full error tensor and its gradients
    # But that's (n*m)*(m*p)*(p*n) = 2*4*4*8*8*2 = 4096 elements, which is manageable
    
    for it in range(max_iters):
        # Compute the current tensor T_current[(i,j,j',k,k',i')]
        # = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        
        # Reshape U, V, W for tensor operations
        # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
        
        # Compute tensor contraction
        # T[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        
        # Use einsum or manual loops
        # For efficiency, compute as: T = sum_r U_r[:,None,None] * V_r[None,:,None] * W_r[None,None,:]
        
        T_current = np.zeros((n*m, m*p, p*n))
        for rr in range(r):
            T_current += np.outer(U[:, rr], V[:, rr]) * W[:, rr][np.newaxis, :]
        
        # Target tensor: T_target[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0
        # This means T_target[a, b, c] = 1 if the indices match appropriately
        # a = i*m+j, b = j'*p+k, c = k'*n+i'
        # Condition: i == i', j == j', k == k'
        # i = a // m, j = a % m, j' = b // p, k = b % p, k' = c // n, i' = c % n
        # So: a//m == c%n, a%m == b//p, b%p == c//n
        
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T_target[a, b, c] = 1.0
        
        # Error
        E = T_current - T_target
        loss = np.sum(E**2)
        
        # Compute gradients
        # dL/dU[i*m+j, r] = 2 * sum_{j',k,k',i'} E[i*m+j, j'*p+k, k'*n+i'] * V[j'*p+k, r] * W[k'*n+i', r]
        # This can be computed as: grad_U = 2 * E.reshape(n*m, -1) @ (V * W.T)
        # Actually: grad_U[a, r] = 2 * sum_b,c E[a,b,c] * V[b,r] * W[c,r]
        # = 2 * (E.reshape(n*m, -1) * (V * W.T).reshape(n*m, -1)).sum(axis=1)
        # Wait, let me be more careful.
        
        # grad_U[a, r] = 2 * sum_{b,c} E[a,b,c] * V[b,r] * W[c,r]
        # = 2 * sum_{b,c} E[a,b,c] * (V[b,r] * W[c,r])
        
        # We can compute this as: for each r, grad_U[:, r] = 2 * (E * V[r][:,None,None] * W[r][None,:]).sum(axis=(1,2))
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for rr in range(r):
            # grad_U[:, rr] = 2 * sum_{b,c} E[a,b,c] * V[b,rr] * W[c,rr]
            # = 2 * (E * V[rr][:,None] * W[rr][None,:]).sum(axis=(1,2))
            temp = E * V[rr][:, np.newaxis] * W[rr][np.newaxis, :]
            grad_U[:, rr] = 2.0 * temp.sum(axis=(1, 2))
            
            # grad_V[b, rr] = 2 * sum_{a,c} E[a,b,c] * U[a,rr] * W[c,rr]
            # = 2 * (E * U[rr][:,None] * W[rr][None,:]).sum(axis=(0,2))
            temp = E * U[rr][:, np.newaxis] * W[rr][np.newaxis, :]
            grad_V[:, rr] = 2.0 * temp.sum(axis=(0, 2))
            
            # grad_W[c, rr] = 2 * sum_{a,b} E[a,b,c] * U[a,rr] * V[b,rr]
            # = 2 * (E * U[rr][:,None] * V[rr][None,:]).sum(axis=(0,1))
            temp = E * U[rr][:, np.newaxis] * V[rr][np.newaxis, :]
            grad_W[:, rr] = 2.0 * temp.sum(axis=(0, 1))
        
        # Update with momentum
        vU = momentum * vU - lr * grad_U
        vV = momentum * vV - lr * grad_V
        vW = momentum * vW - lr * grad_W
        
        U += vU
        V += vV
        W += vW
        
        # Check time
        if it % 50 == 49:
            elapsed = time.time() - start_time_global
            if elapsed > TIME_LIMIT - 15:
                break
        
        # Periodic rounding to half-integers
        if it % 20 == 19:
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
    
    # Final rounding
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _is_exact(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None
>>>>>>> REPLACE

