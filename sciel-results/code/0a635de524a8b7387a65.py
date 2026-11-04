"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s safety margin within 150s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    T_true = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1
                
    # Start with schoolbook (rank = n*m*p = 75)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    def verify_decomp(U, V, W, n, m, p):
        """Verify that the decomposition exactly reconstructs the matrix mult tensor."""
        rank = U.shape[1]
        if rank == 0:
            return False
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        # Build true T
        T_true = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i*m+j, j*p+k, k*n+i] = 1
        return np.allclose(T_recon, T_true, atol=1e-10)

    def check_rank_one_residual(T_res):
        """Check if T_res is a rank-1 tensor. Returns (u, v, w) if yes, else None."""
        # Find a non-zero entry
        nz = np.nonzero(T_res)
        if len(nz[0]) == 0:
            return None # Zero tensor, rank 0
        
        a0, b0, c0 = nz[0][0], nz[1][0], nz[2][0]
        val = T_res[a0, b0, c0]
        if val == 0:
            return None
            
        u = T_res[:, b0, c0] / val
        v = T_res[a0, :, c0] / val
        w = T_res[a0, b0, :] / val
        
        # Verify
        recon = np.einsum('a,b,c->abc', u, v, w)
        if np.allclose(recon, T_res, atol=1e-10):
            return u, v, w
        return None

    # Search for lower rank
    current_U, current_V, current_W = U, V, W
    current_rank = rank
    
    # Try to reduce rank by 1 repeatedly
    # We try to eliminate one column at a time
    # To be efficient, we shuffle the columns and try to remove them
    
    for _ in range(50): # Try up to 50 reductions
        if time.time() > deadline - 10:
            break
            
        reduced = False
        # Shuffle columns to avoid bias
        cols = list(range(current_rank))
        np.random.shuffle(cols)
        
        for c in cols:
            if time.time() > deadline - 5:
                break
                
            # Remove column c
            mask = [i for i in range(current_rank) if i != c]
            
            U_sub = current_U[:, mask]
            V_sub = current_V[:, mask]
            W_sub = current_W[:, mask]
            
            # Calculate residual
            T_sub = np.einsum('at,bt,ct->abc', U_sub, V_sub, W_sub)
            T_res = T_true - T_sub
            
            # Check if residual is rank 1
            res_vecs = check_rank_one_residual(T_res)
            if res_vecs is not None:
                u_new, v_new, w_new = res_vecs
                # Append the new vector
                current_U = np.column_stack([U_sub, u_new])
                current_V = np.column_stack([V_sub, v_new])
                current_W = np.column_stack([W_sub, w_new])
                current_rank -= 1
                reduced = True
                break
        
        if not reduced:
            break
            
    # Verify final result
    if verify_decomp(current_U, current_V, current_W, n, m, p):
        return {
            "U": current_U.tolist(),
            "V": current_V.tolist(),
            "W": current_W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
