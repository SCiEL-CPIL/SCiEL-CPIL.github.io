<<<<<<< SEARCH
def residuals(x, n=2, m=4, p=4, rank=24):
    """Calculate residuals for the optimization problem."""
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    return T_recon.flatten() - 1.0
=======
def residuals(x, n=2, m=4, p=4, rank=24):
    """Calculate residuals for the optimization problem (vectorized)."""
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    # Efficient computation of the tensor contraction
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We can compute this by iterating over r and using outer products, 
    # or by reshaping and using einsum.
    # For small sizes, a loop over r with vectorized ops is often fast.
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        # u_r is shape (n*m), v_r is (m*p), w_r is (p*n)
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        # Reshape to 3D
        # u_r[i, j] = u_r[i*m + j]
        # v_r[j, k] = v_r[j*p + k]
        # w_r[k, i] = w_r[k*n + i]
        
        u_3d = u_r.reshape(n, m)
        v_3d = v_r.reshape(m, p)
        w_3d = w_r.reshape(p, n)
        
        # The contribution to T[i,j,k] is u_3d[i,j] * v_3d[j,k] * w_3d[k,i]
        # This is an element-wise product of three tensors broadcast to (n,m,p)
        # u_3d[i,j, :] (broadcast over k)
        # v_3d[:, j, k] (broadcast over i) -> need to be careful with axes
        # w_3d[:, :, i] (broadcast over j)
        
        # Let's use einsum for clarity and speed
        # T_r[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is equivalent to:
        # T_r = np.einsum('ir,jr,kr->ijk', u_3d, v_3d, w_3d) -- NO, indices are mixed.
        
        # Let's stick to the loop over i,j,k inside the r-loop? No, that's slow.
        # Let's do:
        # Term[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can compute this by:
        # 1. Create a tensor A_r of shape (n, m, 1) from u_3d
        # 2. Create a tensor B_r of shape (1, m, p) from v_3d
        # 3. Create a tensor C_r of shape (p, 1, n) from w_3d
        # But C_r needs to align k with B and i with A.
        # C_r is indexed by (k, i). We want it at (k, i) in the (n,m,p) space.
        # So we need to permute w_3d to (i, 1, k)? No, w_3d is (p, n) -> (k, i).
        # We want C_r[i, j, k] = w_3d[k, i].
        
        # Let's just use a simple loop over i, j, k for the r-term accumulation, 
        # but vectorize over r? No, x is flat.
        
        # Alternative:
        # T_recon[i,j,k] += u_3d[i,j] * v_3d[j,k] * w_3d[k,i]
        
        # We can do this with broadcasting if we reshape appropriately.
        # u_3d: (n, m) -> (n, m, 1)
        # v_3d: (m, p) -> (1, m, p)
        # w_3d: (p, n) -> we need (n, 1, p) such that w_3d_new[i, 0, k] = w_3d[k, i]
        # w_3d.T is (n, p). So w_3d.T[i, k] = w_3d[k, i].
        # So w_3d.T reshaped to (n, 1, p) works.
        
        term = u_3d[:, :, np.newaxis] * v_3d[np.newaxis, :, :] * w_3d.T[:, np.newaxis, :]
        T_recon += term
    
    return T_recon.flatten() - 1.0
>>>>>>> REPLACE

Now, add the `try_row_sharing_decomposition` function. This function will initialize the variables with a structure that encourages row sharing.

