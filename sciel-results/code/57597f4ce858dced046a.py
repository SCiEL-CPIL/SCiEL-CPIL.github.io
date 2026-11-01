import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Precompute T flattened for fast residual
    T_flat = T.flatten()
    
    def compute_residual(U, V, W):
        """Compute ||U V^T W^T - T||_F^2 efficiently"""
        R = np.einsum('as,bs,cs->abc', U, V, W)
        return np.sum((R - T) ** 2)
    
    def als_step(x, U, V, W, rank):
        """One ALS iteration: fix two, solve for third"""
        # Solve for U: for each row a, U[a,:] should satisfy
        # sum_s U[a,s] * (V[b,s]*W[c,s]) = T[a,b,c] for all b,c
        # This is: U[a,:] @ (V*W elementwise summed over b,c) ... 
        # Actually: for fixed b,c, sum_s U[a,s] V[b,s] W[c,s] = T[a,b,c]
        # Let M[a, s] = U[a,s], and for each (b,c), we have a linear equation in U[a,:]
        # Better: reshape. For each a, the equation is:
        # sum_s U[a,s] * (V[:,s] * W[:,s] reshaped) ... 
        # Let's do: for each a, U[a,:] * (V.T @ diag stuff) ...
        # Simpler: R = einsum('as,bs,cs->abc', U,V,W)
        # dR/dU[a,s] = sum_{b,c} (R[a,b,c]-T[a,b,c]) * V[b,s] * W[c,s]
        # Set to 0: sum_{b,c} V[b,s]W[c,s] U[a,s] = sum_{b,c} T[a,b,c] V[b,s] W[c,s]
        # Let G[s,s'] = sum_{b,c} V[b,s] W[c,s] V[b,s'] W[c,s']  -- no that's wrong
        # Actually the normal equations for U:
        # For each a: sum_s' U[a,s'] * (sum_{b,c} V[b,s] W[c,s] V[b,s'] W[c,s']) = sum_{b,c} T[a,b,c] V[b,s] W[c,s]
        # Hmm, this is getting complex. Let me just do direct linear solve.
        
        # For U: fix V, W. For each a, we need:
        # sum_s U[a,s] * (V[b,s]*W[c,s]) = T[a,b,c] for all b,c
        # This is a linear system in U[a,:]. The matrix is (m*p)*(p*n) x rank
        # M_a[s, (b,c)] = V[b,s]*W[c,s]
        # We want: M_a @ U[a,:] = T[a,:,:].flatten()
        
        VW = np.einsum('bs,cs->bcs', V, W)  # shape (m*p, p*n, rank)
        VW_flat = VW.reshape(m*p*p*n, rank)  # (m*p*p*n, rank)
        T_a = T.reshape(n*m, m*p*p*n)  # (n*m, m*p*p*n)
        
        # Solve: VW_flat^T @ VW_flat @ U[a,:] = VW_flat^T @ T[a,:]
        # Normal equations
        A_mat = VW_flat.T @ VW_flat  # (rank, rank)
        # Add small regularization for stability
        A_mat += 1e-12 * np.eye(rank)
        
        U_new = np.zeros((n*m, rank))
        for a in range(n*m):
            b_vec = VW_flat.T @ T[a, :]
            U_new[a, :] = np.linalg.solve(A_mat, b_vec)
        
        # Similarly for V
        UW = np.einsum('as,cs->acs', U_new, W)  # shape (n*m, p*n, rank)
        UW_flat = UW.reshape(n*m*p*n, rank)
        T_b = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)  # T[b,a,c]
        
        A_mat_v = UW_flat.T @ UW_flat + 1e-12 * np.eye(rank)
        V_new = np.zeros((m*p, rank))
        for b in range(m*p):
            b_vec = UW_flat.T @ T_b[b, :]
            V_new[b, :] = np.linalg.solve(A_mat_v, b_vec)
        
        # For W
        UV = np.einsum('as,bs->abs', U_new, V_new)  # shape (n*m, m*p, rank)
        UV_flat = UV.reshape(n*m*m*p, rank)
        T_c = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)  # T[c,a,b]
        
        A_mat_w = UV_flat.T @ UV_flat + 1e-12 * np.eye(rank)
        W_new = np.zeros((p*n, rank))
        for c in range(p*n):
            b_vec = UV_flat.T @ T_c[c, :]
            W_new[c, :] = np.linalg.solve(A_mat_w, b_vec)
        
        return U_new, V_new, W_new
    
    # Try rank 10 first (known minimal), then 11, 12...
    for rank in range(10, 19):
        if time.time() > deadline - 15:
            break
        
        best_U, best_V, best_W = None, None, None
        best_cost = np.inf
        
        # Multiple random restarts
        num_restarts = 5 if rank == 10 else 3
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(1234 + rank * 1000 + restart * 7)
            
            # Random initialization with some structure
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Run ALS iterations
            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                U, V, W = als_step(None, U, V, W, rank)
                
                cost = compute_residual(U, V, W)
                if cost < best_cost:
                    best_cost = cost
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if cost < 1e-10:
                    break
            
            if best_cost < 1e-10:
                break
        
        if best_cost < 1e-10 and best_U is not None:
            # Round to half-integers and verify
            U_out = [[round(x*2)/2 for x in row] for row in best_U]
            V_out = [[round(x*2)/2 for x in row] for row in best_V]
            W_out = [[round(x*2)/2 for x in row] for row in best_W]
            
            U_np = np.array(U_out, dtype=float)
            V_np = np.array(V_out, dtype=float)
            W_np = np.array(W_out, dtype=float)
            
            R = np.einsum('as,bs,cs->abc', U_np, V_np, W_np)
            if np.allclose(R, T, atol=1e-9):
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # If rounding failed, try without rounding (but task says integer/half-integer)
            # Try different rounding strategies
            for scale in [1, 2, 4]:
                U_r = [[round(x*scale)/scale for x in row] for row in best_U]
                V_r = [[round(x*scale)/scale for x in row] for row in best_V]
                W_r = [[round(x*scale)/scale for x in row] for row in best_W]
                U_np2 = np.array(U_r, dtype=float)
                V_np2 = np.array(V_r, dtype=float)
                W_np2 = np.array(W_r, dtype=float)
                R2 = np.einsum('as,bs,cs->abc', U_np2, V_np2, W_np2)
                if np.allclose(R2, T, atol=1e-9):
                    return {"U": U_r, "V": V_r, "W": W_r}
    
    # Fallback: schoolbook rank 18
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
