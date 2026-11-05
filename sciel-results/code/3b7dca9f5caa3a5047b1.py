import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Target tensor: D[i,j,j',k] = 1 if j==j', else 0
    Y = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Y[i, j, j, k] = 1.0
    Y_flat = Y.flatten()  # shape (189,)
    
    def solve_for_W(U, V, rank):
        """Given U (n*m, rank) and V (m*p, rank), solve for W (p*n, rank)."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Reshape U and V
        U_r = U.reshape(n, m, rank)  # U[i,j,c]
        V_r = V.reshape(m, p, rank)  # V[j',k,c]
        
        # Build linear system: for each (i,j,j',k), sum_c U[i,j,c]*V[j',k,c]*W[k,i,c] = Y[i,j,j',k]
        # W is indexed by (k,i,c), so W_flat index = (k*n+i)*rank + c
        num_eq = n * m * m * p  # 189
        num_unk = pn * rank  # 21*rank
        
        A = np.zeros((num_eq, num_unk))
        b = np.zeros(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        b[eq_idx] = 1.0 if j == jp else 0.0
                        for c in range(rank):
                            # W[k,i,c] is at flat index (k*n+i)*rank + c
                            w_idx = (k * n + i) * rank + c
                            A[eq_idx, w_idx] = U_r[i, j, c] * V_r[jp, k, c]
                        eq_idx += 1
        
        # Solve least squares
        W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
        W = W_flat.reshape(pn, rank)
        
        # Compute residual
        W_r = W.reshape(p, n, rank)
        D = np.einsum('ijc,jkc,kic->ijjk', U_r, V_r, W_r)
        err = np.max(np.abs(D - Y))
        
        return W, err
    
    def solve_for_U(U, V, W, rank):
        """Given V and W, solve for U."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        V_r = V.reshape(m, p, rank)  # V[j',k,c]
        W_r = W.reshape(p, n, rank)  # W[k,i,c]
        
        num_eq = n * m * m * p  # 189
        num_unk = nm * rank  # 9*rank
        
        A = np.zeros((num_eq, num_unk))
        b = np.zeros(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        b[eq_idx] = 1.0 if j == jp else 0.0
                        for c in range(rank):
                            # U[i,j,c] is at flat index (i*m+j)*rank + c
                            u_idx = (i * m + j) * rank + c
                            A[eq_idx, u_idx] = V_r[jp, k, c] * W_r[k, i, c]
                        eq_idx += 1
        
        U_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
        U = U_flat.reshape(nm, rank)
        
        U_r = U.reshape(n, m, rank)
        D = np.einsum('ijc,jkc,kic->ijjk', U_r, V_r, W_r)
        err = np.max(np.abs(D - Y))
        
        return U, err
    
    def solve_for_V(U, V, W, rank):
        """Given U and W, solve for V."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        U_r = U.reshape(n, m, rank)  # U[i,j,c]
        W_r = W.reshape(p, n, rank)  # W[k,i,c]
        
        num_eq = n * m * m * p  # 189
        num_unk = mp * rank  # 21*rank
        
        A = np.zeros((num_eq, num_unk))
        b = np.zeros(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        b[eq_idx] = 1.0 if j == jp else 0.0
                        for c in range(rank):
                            # V[jp,k,c] is at flat index (jp*p+k)*rank + c
                            v_idx = (jp * p + k) * rank + c
                            A[eq_idx, v_idx] = U_r[i, j, c] * W_r[k, i, c]
                        eq_idx += 1
        
        V_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
        V = V_flat.reshape(mp, rank)
        
        V_r = V.reshape(m, p, rank)
        D = np.einsum('ijc,jkc,kic->ijjk', U_r, V_r, W_r)
        err = np.max(np.abs(D - Y))
        
        return V, err
    
    def verify(U, V, W, rank):
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        D = np.einsum('ijc,jkc,kic->ijjk', U_r, V_r, W_r)
        err = np.max(np.abs(D - Y))
        return err < 1e-6, err
    
    # Schoolbook decomposition as baseline
    best_rank = n * m * p  # 63
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1
    
    # Alternating minimization for various ranks
    search_deadline = start_time + 130
    
    for rank in range(40, 64):
        if time.time() > search_deadline:
            break
        
        found = False
        num_restarts = 5 if rank <= 50 else 3
        
        for restart in range(num_restarts):
            if time.time() > search_deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() > search_deadline:
                    break
                
                U, err_U = solve_for_U(U, V, W, rank)
                if err_U < 1e-8:
                    break
                V, err_V = solve_for_V(U, V, W, rank)
                if err_V < 1e-8:
                    break
                W, err_W = solve_for_W(U, V, rank)
                if err_W < 1e-8:
                    break
            
            # Check if we found a valid decomposition
            valid, err = verify(U, V, W, rank)
            if valid:
                # Try rounding to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                valid_r, err_r = verify(U_rounded, V_rounded, W_rounded, rank)
                if valid_r:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                else:
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
                found = True
                break
        
        if found:
            # Continue to try lower ranks
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
