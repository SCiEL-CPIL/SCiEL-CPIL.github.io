import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n, m, p = 3, 3, 5
    # Target tensor T[I, J, K]
    # I = i*m + j (0..8)
    # J = j'*p + k (0..14)
    # K = k'*n + i' (0..14)
    # T[I, J, K] = 1 if i=i', j=j', k=k'
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def compute_residual(U, V, W):
        R = np.einsum('ic,jc,kc->ijk', U, V, W)
        R -= T
        return R

    def objective_and_grad(params, r, mask_U, mask_V, mask_W):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = params[:size_U].reshape(n*m, r)
        V = params[size_U:size_U+size_V].reshape(m*p, r)
        W = params[size_U+size_V:].reshape(p*n, r)
        
        # Apply masks to enforce block structure
        # mask_U, mask_V, mask_W are boolean arrays of same shape as U, V, W
        U_masked = U * mask_U
        V_masked = V * mask_V
        W_masked = W * mask_W
        
        R = compute_residual(U_masked, V_masked, W_masked)
        loss = np.sum(R**2)
        
        grad_U = 2.0 * np.einsum('ijk,jc,kc->ic', R, V_masked, W_masked) * mask_U
        grad_V = 2.0 * np.einsum('ijk,ic,kc->jc', R, U_masked, W_masked) * mask_V
        grad_W = 2.0 * np.einsum('ijk,ic,jc->kc', R, U_masked, V_masked) * mask_W
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def check_exactness(U, V, W):
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if np.max(np.abs(U - U_r)) > 1e-4 or np.max(np.abs(V - V_r)) > 1e-4 or np.max(np.abs(W - W_r)) > 1e-4:
            return False, None, None, None
            
        R = compute_residual(U_r, V_r, W_r)
        if np.max(np.abs(R)) < 1e-6:
            return True, U_r, V_r, W_r
        return False, None, None, None

    # Try ranks from 40 to 44
    # We use the block structure masks
    # Block 1: 3x3x3 (cols 0..22)
    # Block 2: 3x3x2 (cols 23..39)
    
    # Masks for rank 40
    # U: (9, 40). All 1s.
    # V: (15, 40). 
    #   Cols 0..22: Non-zero only in rows 0..8 (j'=0,1,2 -> J=0..8? No, J = j*p+k. j=0,1,2 -> J=0..14? No.
    #   B is 3x5. Rows of V correspond to (j,k) pairs.
    #   j=0,1,2. k=0..4.
    #   For 3x3x3 part, we use B_1 (cols 0,1,2 of B). So k=0,1,2.
    #   J indices for k=0,1,2:
    #   j=0: J=0,1,2
    #   j=1: J=3,4,5
    #   j=2: J=6,7,8
    #   So V mask for cols 0..22: Rows 0..8 are 1, Rows 9..14 are 0.
    #   For 3x3x2 part, we use B_2 (cols 3,4 of B). So k=3,4.
    #   J indices for k=3,4:
    #   j=0: J=9,10
    #   j=1: J=12,13
    #   j=2: J=15,16? No, m=3, p=5. J max is 14.
    #   j=0: J=3,4? No. J = j*5 + k.
    #   j=0: J=0,1,2,3,4
    #   j=1: J=5,6,7,8,9
    #   j=2: J=10,11,12,13,14
    #   k=0,1,2 -> J=0,1,2 (j=0), 5,6,7 (j=1), 10,11,12 (j=2).
    #   k=3,4 -> J=3,4 (j=0), 8,9 (j=1), 13,14 (j=2).
    
    # W: (15, 40).
    #   C is 3x5. Rows of W correspond to (k,i) pairs.
    #   K = k*n + i.
    #   For 3x3x3 part, C_1 (cols 0,1,2). k=0,1,2.
    #   K indices for k=0,1,2:
    #   k=0: K=0,1,2
    #   k=1: K=3,4,5
    #   k=2: K=6,7,8
    #   So W mask for cols 0..22: Rows 0..8 are 1, Rows 9..14 are 0.
    #   For 3x3x2 part, C_2 (cols 3,4). k=3,4.
    #   K indices for k=3,4:
    #   k=3: K=9,10,11
    #   k=4: K=12,13,14
    #   So W mask for cols 23..39: Rows 9..14 are 1, Rows 0..8 are 0.

    mask_U_40 = np.ones((n*m, 40))
    mask_V_40 = np.zeros((m*p, 40))
    mask_W_40 = np.zeros((p*n, 40))
    
    # Block 1 (0..22)
    mask_V_40[0:9, 0:23] = 1  # J=0..8
    mask_W_40[0:9, 0:23] = 1  # K=0..8
    
    # Block 2 (23..39)
    mask_V_40[9:15, 23:40] = 1  # J=9..14
    mask_W_40[9:15, 23:40] = 1  # K=9..14

    best_result = None
    
    # Try rank 40
    if time.time() < deadline - 5:
        r = 40
        np.random.seed(42)
        params = np.random.randn(r * (n*m + m*p + p*n)) * 0.1
        try:
            from scipy.optimize import minimize
            res = minimize(
                lambda x: objective_and_grad(x, r, mask_U_40, mask_V_40, mask_W_40)[0],
                params,
                jac=lambda x: objective_and_grad(x, r, mask_U_40, mask_V_40, mask_W_40)[1],
                method='BFGS',
                options={'gtol': 1e-8, 'maxiter': 500}
            )
            if res.fun < 1e-8:
                U = res.x[:n*m*r].reshape(n*m, r)
                V = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                is_exact, U_r, V_r, W_r = check_exactness(U, V, W)
                if is_exact:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        except Exception:
            pass

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, column] = 1
                V[j*p + k, column] = 1
                W[k*n + i, column] = 1
                column += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
