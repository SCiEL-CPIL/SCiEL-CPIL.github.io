<<<<<<< SEARCH
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Vectorized check:
    # T[i,l,k] = sum_r U[i,l,r] * V[l,k,r] * W[k,i,r]
    # We want T[i,l,k] == 1 for all i,l,k.
    # Let's construct the tensor T.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Indices: a = i*m+l, b = l*p+k, c = k*n+i
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        
        # u is vector of length n*m
        # v is vector of length m*p
        # w is vector of length p*n
        
        # Reshape u to (n, m)
        u_mat = u.reshape(n, m)
        # Reshape v to (m, p)
        v_mat = v.reshape(m, p)
        # Reshape w to (p, n)
        w_mat = w.reshape(p, n)
        
        # Contribution to T[i,l,k] is u[i,l] * v[l,k] * w[k,i]
        # This is not a simple matrix product.
        # T[i,l,k] = sum_r u_r[i,l] v_r[l,k] w_r[k,i]
        
        # We can loop over l since m is small (5)
        for l in range(m):
            u_l = u_mat[:, l] # length n
            v_l = v_mat[l, :] # length p
            # w_mat is (p, n), we need w[k,i]
            # term[i,k] = u_l[i] * v_l[k] * w_mat[k,i]
            # outer product of u_l and v_l is (n, p)
            prod = np.outer(u_l, v_l) # prod[i,k] = u_l[i] * v_l[k]
            # Multiply element-wise with w_mat.T (which is n, p)
            # w_mat is (p, n), so w_mat.T is (n, p)
            # w_mat.T[i,k] = w_mat[k,i]
            T[:, l, :] += prod * w_mat.T
            
    # Check if T is all ones
    return np.allclose(T, 1.0, atol=1e-9)
=======
import time
import numpy as np

def _verify_decomp_fast(U, V, W, n, m, p, rank):
    """Fast verification using numpy vectorization."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r].reshape(n, m)
        v = V_arr[:, r].reshape(m, p)
        w = W_arr[:, r].reshape(p, n)
        for l in range(m):
            T[:, l, :] += np.outer(u[:, l], v[l, :]) * w.T
    
    return np.allclose(T, 1.0, atol=1e-9)

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    return _verify_decomp_fast(U, V, W, n, m, p, rank)
>>>>>>> REPLACE

