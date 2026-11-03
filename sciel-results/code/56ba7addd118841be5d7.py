import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 120
    
    # Build Kronecker-embedded Strassen (rank 65)
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    def add_strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai+1)] = 1
        w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+1)] = 1
        v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ai)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = -1
        w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj+1)] = 1
        u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        
        # Naive for col bk+2
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m)
                u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p)
                v[get_v_idx(bj+dj, bk+2)] = 1
                w = np.zeros(p*n)
                w[get_w_idx(bk+2, ai+di)] = 1
                add_term(u, v, w)

    # C00 = A00 B00 + A01 B10
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    # C01 = A00 B01 + A01 B11
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    # C10 = A10 B00 + A11 B10
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    # C11 = A10 B01 + A11 B11
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    base_rank = U_mat.shape[1]
    
    def verify(U, V, W):
        # Check tensor decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    expected = 1.0 if i == j and j == k else 0.0
                    if abs(val - expected) > 1e-6:
                        return False
        return True
    
    def round_half(M):
        return np.round(M * 2) / 2
    
    def try_als(U_init, V_init, W_init, target_rank, max_iters=200, lr=0.05):
        """ALS with rounding, try to reduce rank"""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute residual
            R = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R[i, j, k] = 1.0 if i == j and j == k else 0.0
                        for r in range(target_rank):
                            R[i, j, k] -= U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            if np.max(np.abs(R)) < 1e-6:
                break
            
            # Gradient descent for U
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        g = 0.0
                        for k in range(p):
                            g += R[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                        U[i*m+j, r] -= lr * g
            
            # Gradient descent for V
            for r in range(target_rank):
                for j in range(m):
                    for k in range(p):
                        g = 0.0
                        for i in range(n):
                            g += R[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                        V[j*p+k, r] -= lr * g
            
            # Gradient descent for W
            for r in range(target_rank):
                for k in range(p):
                    for i in range(n):
                        g = 0.0
                        for j in range(m):
                            g += R[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
                        W[k*n+i, r] -= lr * g
            
            # Periodic rounding
            if it % 5 == 4:
                U = round_half(U)
                V = round_half(V)
                W = round_half(W)
            
            if np.max(np.abs(R)) < 1e-4:
                break
        
        if verify(U, V, W):
            return U, V, W
        return None
    
    # Try to reduce rank
    best_U, best_V, best_W = U_mat, V_mat, W_mat
    best_rank = base_rank
    
    # Try rank reduction from 65 down to 60
    for target_rank in range(base_rank - 1, max(59, base_rank - 8), -1):
        if time.time() > deadline - 15:
            break
        
        # Initialize by truncating the base solution
        U_init = U_mat[:, :target_rank].copy()
        V_init = V_mat[:, :target_rank].copy()
        W_init = W_mat[:, :target_rank].copy()
        
        result = try_als(U_init, V_init, W_init, target_rank, max_iters=100, lr=0.03)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
