import scipy
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    n, m, p = 3, 3, 6
    
    # Target tensor T for n x m by m x p
    # T[i, j, k] = 1 if i,j,k are the indices of the product term
    # Actually, T is defined on (i,j), (j',k), (i',k')
    # T[(i,j), (j',k), (i',k')] = 1 if i==i', j==j', k==k'
    # We flatten to 1D indices for U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 if (i,j,k) is valid, 0 else
    # Wait, the sum is over r. The result is a tensor of shape (n*m, m*p, p*n).
    # Let's call the result tensor R.
    # R[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # We want R[a, b, c] to be 1 if a,b,c correspond to a valid multiplication term, 0 otherwise.
    # Valid term: a = i*m+j, b = j*p+k, c = k*n+i.
    # Note that j in a and b must match, and k in b and c must match, and i in c and a must match.
    # If they don't match, the term is 0.
    # So we want R[a, b, c] = 1 if a,b,c form a consistent (i,j,k), 0 otherwise.
    
    # Construct target tensor T_target
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def residual(U, V, W):
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        # U: (dim_U, R), V: (dim_V, R), W: (dim_W, R)
        # Result: (dim_U, dim_V, dim_W)
        # Efficient computation:
        # R = np.einsum('ir,jr,kr->ijk', U, V, W)
        R = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((R - T_target)**2)

    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        # Check exactness
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        # Check if R matches T_target exactly
        if np.allclose(R, T_target, atol=1e-9):
            # Convert to lists
            U_list = U_s.tolist()
            V_list = V_s.tolist()
            W_list = W_s.tolist()
            return True, U_list, V_list, W_list
        return False, None, None, None

    # Try ranks from 46 upwards
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Schoolbook initialization
    def get_schoolbook(rank):
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # If rank is less than schoolbook, we need a different init.
    # We can try random init.
    
    for rank in range(46, 55):
        if time.time() > deadline - 20: break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10: break
            
            # Initialize
            if rank == 54:
                U, V, W = get_schoolbook(rank)
            else:
                # Random init scaled
                U = np.random.randn(dim_U, rank) * 0.1
                V = np.random.randn(dim_V, rank) * 0.1
                W = np.random.randn(dim_W, rank) * 0.1
                
            # Optimize
            # Use simple gradient descent or scipy
            # Given small size, let's use scipy.optimize.minimize
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(params):
                U_p = params[:dim_U*rank].reshape(dim_U, rank)
                V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                return residual(U_p, V_p, W_p)
            
            # L-BFGS-B
            try:
                res = scipy.optimize.minimize(obj, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
                U_opt = res.x[:dim_U*rank].reshape(dim_U, rank)
                V_opt = res.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                W_opt = res.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s, "V": V_s, "W": W_s}
            except Exception:
                pass

    # Fallback to schoolbook
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
