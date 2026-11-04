import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target: 2x3 matrix * 3x4 matrix
    # Tensor shape: (n, m, p) = (2, 3, 4)
    # W index: k*n + i
    
    # Fallback: Schoolbook rank 24
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find rank 20
    # Strategy: Find rank 10 for 2x3x2, then duplicate for 2x3x4
    # 2x3x2: n=2, m=3, p=2
    n2, m2, p2 = 2, 3, 2
    rank_target = 10
    T_232 = np.zeros((n2, m2, p2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T_232[i, j, k] = 1
                
    best_232 = None
    best_err = float('inf')
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s for construction and verification
    
    # Local search for rank 10
    # Initialize U, V, W
    # U: (6, 10), V: (6, 10), W: (4, 10)
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n2 * m2, rank_target)
        V = np.random.randn(m2 * p2, rank_target)
        W = np.random.randn(p2 * n2, rank_target)
        
        # Normalize to help convergence
        for c in range(rank_target):
            U[:, c] /= (np.linalg.norm(U[:, c]) + 1e-9)
            V[:, c] /= (np.linalg.norm(V[:, c]) + 1e-9)
            W[:, c] /= (np.linalg.norm(W[:, c]) + 1e-9)
            
        # Simple gradient descent or just check if random init is close?
        # Random init is unlikely to be close.
        # Use a few steps of gradient descent.
        # Error E = || T - sum U_c V_c W_c ||^2
        # dE/dU = 2 * sum (T - approx) * V_c W_c
        
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        for step in range(200):
            if time.time() > deadline:
                break
                
            # Compute current approximation
            approx = np.tensordot(U_curr, V_curr, axes=1) # (n, m, p) ? No.
            # U is (nm, r), V is (mp, r), W is (pn, r)
            # Term c: U[:, c] (nm,) -> reshape (n, m)
            #          V[:, c] (mp,) -> reshape (m, p)
            #          W[:, c] (pn,) -> reshape (p, n)
            # Product: (n, m) * (m, p) * (p, n) is not a standard contraction for tensor.
            # The tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            approx = np.zeros((n2, m2, p2))
            for c in range(rank_target):
                u_vec = U_curr[:, c].reshape(n2, m2)
                v_vec = V_curr[:, c].reshape(m2, p2)
                w_vec = W_curr[:, c].reshape(p2, n2)
                # The term is sum_{i,j,k} u_vec[i,j] v_vec[j,k] w_vec[k,i] T[i,j,k]
                # This is a trace-like operation.
                # Let's just compute the error directly for gradient.
                pass
                
            # Efficient error computation
            # E = || T - A ||_F^2
            # A[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            
            # We can use einsum
            # A = np.einsum('ic,jc,kc->ijk', 
            #               U_curr.reshape(n2, m2, rank_target), 
            #               V_curr.reshape(m2, p2, rank_target), 
            #               W_curr.reshape(p2, n2, rank_target))
            # Note: indices for W are k*n+i. So W[k, i, c].
            # W_curr is (p*n, r). Reshape to (p, n, r).
            # U_curr is (n*m, r). Reshape to (n, m, r).
            # V_curr is (m*p, r). Reshape to (m, p, r).
            
            U_r = U_curr.reshape(n2, m2, rank_target)
            V_r = V_curr.reshape(m2, p2, rank_target)
            W_r = W_curr.reshape(p2, n2, rank_target)
            
            A = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            err = np.linalg.norm(T_232 - A)
            
            if err < best_err:
                best_err = err
                if err < 1e-6:
                    best_232 = (U_curr.copy(), V_curr.copy(), W_curr.copy())
                    break
            
            # Gradient descent
            # dE/dU[i, m, c] = 2 * sum_{j,k} (T[i,j,k] - A[i,j,k]) V[j,k,c] W[k,i,c]
            # dE/dV[j, p, c] = 2 * sum_{i,k} (T[i,j,k] - A[i,j,k]) U[i,j,c] W[k,i,c]
            # dE/dW[k, n, c] = 2 * sum_{i,j} (T[i,j,k] - A[i,j,k]) U[i,j,c] V[j,k,c]
            
            diff = T_232 - A
            # Gradient for U
            grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
            # Gradient for V
            grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
            # Gradient for W
            grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
            
            lr = 0.01
            U_curr -= lr * grad_U.reshape(n2*m2, rank_target)
            V_curr -= lr * grad_V.reshape(m2*p2, rank_target)
            W_curr -= lr * grad_W.reshape(p2*n2, rank_target)
            
            if best_err < 1e-6:
                break
                
        if best_err < 1e-6:
            break

    if best_232 is not None:
        U1, V1, W1 = best_232
        # Construct rank 20 for 2x3x4
        # U: (6, 20) = [U1, U1]
        # V: (12, 20) = [V1_top, V1_bottom] where V1_top is V1 for first 3x2 block, V1_bottom for second
        # Actually, V for 2x3x4 is (m*p, r) = (3*4, 20) = (12, 20).
        # The first 6 rows correspond to j=0,1,2 and k=0,1 (first 2 cols of B).
        # The next 6 rows correspond to j=0,1,2 and k=2,3 (second 2 cols of B).
        # So V = [V1; V1]
        # W: (8, 20) = [W1_top, W1_bottom]
        # W for 2x3x4 is (p*n, r) = (4*2, 20) = (8, 20).
        # Indices k*n+i.
        # k=0,1 for first block. k=2,3 for second block.
        # W1 is (p2*n, r) = (2*2, 10) = (4, 10).
        # So W = [W1; W1]
        
        U_20 = np.hstack([U1, U1])
        V_20 = np.vstack([V1, V1])
        W_20 = np.vstack([W1, W1])
        
        # Snap to half-integers
        U_20 = np.round(U_20 * 2) / 2
        V_20 = np.round(V_20 * 2) / 2
        W_20 = np.round(W_20 * 2) / 2
        
        # Verify
        U_v = U_20.reshape(n, m, 20)
        V_v = V_20.reshape(m, p, 20)
        W_v = W_20.reshape(p, n, 20)
        
        T_target = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, k] = 1
                    
        A_check = np.einsum('ijc,jkc,kic->ijk', U_v, V_v, W_v)
        
        if np.allclose(A_check, T_target, atol=1e-9):
            # Convert to lists
            U_out = U_20.tolist()
            V_out = V_20.tolist()
            W_out = W_20.tolist()
            return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback
    return {"U": U_school, "V": V_school, "W": W_school}
