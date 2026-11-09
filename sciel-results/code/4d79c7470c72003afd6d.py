import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[alpha, beta, gamma] = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    # alpha: 0..5, beta: 0..14, gamma: 0..9
    nm = n * m
    mp = m * p
    pn = p * n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    
    T_flat = T.flatten()
    
    def compute_residual(params):
        # params: flattened U (nm*r), V (mp*r), W (pn*r)
        r = params.size // (nm + mp + pn)
        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r+mp*r].reshape(mp, r)
        W = params[nm*r+mp*r:].reshape(pn, r)
        
        # Compute sum_s U[:,s] V[:,s]^T W[:,s]^T
        # Result shape (nm, mp, pn)
        # Efficient computation:
        # A = U @ V.T (nm, mp) -- No, this is rank 1 outer product of U_s and V_s?
        # Term s is U_s (vector) V_s (vector) W_s (vector).
        # Entry (a,b,c) is U[a,s] * V[b,s] * W[c,s].
        
        # We can compute tensor = sum_s np.einsum('as,bs,cs->abc', U, V, W)
        tensor = np.zeros((nm, mp, pn))
        for s in range(r):
            # u_s is vector of length nm
            # v_s is vector of length mp
            # w_s is vector of length pn
            # outer product
            tensor += np.einsum('i,j,k->ijk', U[:, s], V[:, s], W[:, s])
            
        return (tensor.flatten() - T_flat)
    
    def try_rank(r, max_restarts=5):
        total_size = r * (nm + mp + pn)
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale factors might matter. Let's try uniform distribution.
            # To help convergence, we can use the schoolbook solution as a hint?
            # Schoolbook has 30 terms. If r < 30, we can't use it directly.
            # Random is safer.
            np.random.seed(42 + attempt)
            init = np.random.uniform(-1, 1, total_size)
            
            res = least_squares(compute_residual, init, method='lm', max_nfev=200, xtol=1e-12, ftol=1e-12)
            
            if res.success or res.cost < 1e-10:
                # Check if we can snap to half-integers
                params_opt = res.x
                U_opt = params_opt[:nm*r].reshape(nm, r)
                V_opt = params_opt[nm*r:nm*r+mp*r].reshape(mp, r)
                W_opt = params_opt[nm*r+mp*r:].reshape(pn, r)
                
                # Round to nearest 0.5
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify exactness
                tensor_snap = np.zeros((nm, mp, pn))
                for s in range(r):
                    tensor_snap += np.einsum('i,j,k->ijk', U_snap[:, s], V_snap[:, s], W_snap[:, s])
                
                if np.allclose(tensor_snap, T, atol=1e-9):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
        
        return None

    # Try rank 29
    result = try_rank(29, max_restarts=10)
    if result:
        return result

    # Fallback to schoolbook rank 30
    rank = rank_schoolbook
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
