import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 140
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find lower rank solutions
    for rank in range(20, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        np.random.seed(rank)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Alternating least squares
        for iteration in range(1000):
            if time.time() > deadline - 30:
                break
            
            # Update U
            for i in range(n*m):
                # For each row i of U, solve for U[i,:]
                # We need: sum_c U[i,c] * V[b,c] * W[c_idx,c] = T[i,b,c_idx]
                # This is linear in U[i,:] for fixed V, W
                # For each (b, c_idx), we have: sum_c U[i,c] * (V[b,c] * W[c_idx,c]) = T[i,b,c_idx]
                
                # Build the system
                A_mat = np.zeros((m*p * p*n, rank))
                b_vec = np.zeros(m*p * p*n)
                
                for b in range(m*p):
                    for c_idx in range(p*n):
                        row = b * p*n + c_idx
                        A_mat[row, :] = V[b, :] * W[c_idx, :]
                        # T[i, b, c_idx]
                        # Check if (i, b, c_idx) is a valid triple
                        i_idx = i // m
                        j_idx = i % m
                        b_j = b // p
                        b_k = b % p
                        c_k = c_idx // n
                        c_i = c_idx % n
                        if j_idx == b_j and i_idx == c_i and b_k == c_k:
                            b_vec[row] = 1.0
                        else:
                            b_vec[row] = 0.0
                
                # Solve least squares
                U[i, :], residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update V
            for j in range(m*p):
                A_mat = np.zeros((n*m * p*n, rank))
                b_vec = np.zeros(n*m * p*n)
                
                for a in range(n*m):
                    for c_idx in range(p*n):
                        row = a * p*n + c_idx
                        A_mat[row, :] = U[a, :] * W[c_idx, :]
                        i_idx = a // m
                        j_idx = a % m
                        j_b = j // p
                        b_k = j % p
                        c_k = c_idx // n
                        c_i = c_idx % n
                        if j_idx == j_b and i_idx == c_i and b_k == c_k:
                            b_vec[row] = 1.0
                        else:
                            b_vec[row] = 0.0
                
                V[j, :], residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update W
            for c_idx in range(p*n):
                A_mat = np.zeros((n*m * m*p, rank))
                b_vec = np.zeros(n*m * m*p)
                
                for a in range(n*m):
                    for b in range(m*p):
                        row = a * m*p + b
                        A_mat[row, :] = U[a, :] * V[b, :]
                        i_idx = a // m
                        j_idx = a % m
                        b_j = b // p
                        b_k = b % p
                        c_k = c_idx // n
                        c_i = c_idx % n
                        if j_idx == b_j and i_idx == c_i and b_k == c_k:
                            b_vec[row] = 1.0
                        else:
                            b_vec[row] = 0.0
                
                W[c_idx, :], residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Check if the solution is valid
        T = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            T += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        
        max_error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c_idx = k*n + i
                    max_error = max(max_error, abs(T[a,b,c_idx] - 1.0))
                    for a2 in range(n*m):
                        for b2 in range(m*p):
                            for c2 in range(p*n):
                                if (a2,b2,c2) != (a,b,c_idx):
                                    max_error = max(max_error, abs(T[a2,b2,c2]))
        
        if max_error < 1e-6:
            best_rank = rank
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            break
    
    if best_U is not None:
        # Round to nearest half-integer
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify
        T = np.zeros((n*m, m*p, p*n))
        for c in range(best_rank):
            T += np.einsum('i,j,k->ijk', U_rounded[:,c], V_rounded[:,c], W_rounded[:,c])
        
        max_error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c_idx = k*n + i
                    max_error = max(max_error, abs(T[a,b,c_idx] - 1.0))
                    for a2 in range(n*m):
                        for b2 in range(m*p):
                            for c2 in range(p*n):
                                if (a2,b2,c2) != (a,b,c_idx):
                                    max_error = max(max_error, abs(T[a2,b2,c2]))
        
        if max_error < 1e-9:
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
