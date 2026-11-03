import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target rank values to try
    target_ranks = [45, 44, 43, 42, 41, 40]
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook as fallback
    U_fb = np.zeros((n * m, best_rank))
    V_fb = np.zeros((m * p, best_rank))
    W_fb = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1

    for R in target_ranks:
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt)
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.zeros((p * n, R))
            
            # Alternating minimization
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W
                for i in range(n):
                    for k in range(p):
                        # Equation: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta[j, j']
                        # For fixed i, k, we have a linear system in W[k*n+i, :]
                        # Let A[j, j'] = sum_r U[i*m+j, r] * V[j'*p+k, r]
                        # Then sum_r A[j, j'] * W[k*n+i, r] = delta[j, j']
                        # This is a system: sum_r W_r * (U_i[:, r] * V_k[:, r])^T?
                        # Let's build the matrix M where M[j, j'] = sum_r U[i*m+j, r] * V[j'*p+k, r]
                        # We want sum_r M[j, j'] * W_r = delta[j, j']
                        # This is not a standard matrix equation.
                        # Let's rewrite:
                        # For each j, j', sum_r (U[i*m+j, r] * V[j'*p+k, r]) * W[k*n+i, r] = 1 if j==j' else 0
                        # Let B be a matrix of size (m*m) x R
                        # B[j*m + j', r] = U[i*m+j, r] * V[j'*p+k, r]
                        # We want B @ W[k*n+i, :] = delta_vector
                        # where delta_vector has 1s at indices j*m+j for j in 0..m-1
                        B = np.zeros((m * m, R))
                        for j in range(m):
                            for jp in range(m):
                                B[j * m + jp, :] = U[i * m + j, :] * V[jp * p + k, :]
                        delta_vec = np.zeros(m * m)
                        for j in range(m):
                            delta_vec[j * m + j] = 1.0
                        
                        # Solve B @ w = delta_vec
                        # Least squares
                        w, residuals, rank, sv = np.linalg.lstsq(B, delta_vec, rcond=None)
                        W[k * n + i, :] = w
                        
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        # Equation: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta[j, j']
                        # For fixed i, j, this is linear in U[i*m+j, :]
                        # Let C[k, j'] = sum_r V[j'*p+k, r] * W[k*n+i, r]
                        # Then sum_k sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta[j, j']?
                        # No, the equation is for specific i, k, j, j'.
                        # Wait, the condition is:
                        # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta[j, j']
                        # This must hold for ALL k.
                        # So for fixed i, j, we have p equations (one for each k).
                        # Let's build a system for U[i*m+j, :]
                        # For each k, let D_k[j'] = sum_r V[j'*p+k, r] * W[k*n+i, r]
                        # Then sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta[j, j']
                        # This is sum_r U_r * (V[j'*p+k, r] * W[k*n+i, r]) = delta[j, j']
                        # Let E be a matrix of size (p*m) x R
                        # E[k*m + j', r] = V[j'*p+k, r] * W[k*n+i, r]
                        # We want E @ U[i*m+j, :] = target_vector
                        # target_vector[k*m + j'] = 1 if j==j' else 0
                        E = np.zeros((p * m, R))
                        for k in range(p):
                            for jp in range(m):
                                E[k * m + jp, :] = V[jp * p + k, :] * W[k * n + i, :]
                        target_vec = np.zeros(p * m)
                        for k in range(p):
                            target_vec[k * m + j] = 1.0
                        
                        u, residuals, rank, sv = np.linalg.lstsq(E, target_vec, rcond=None)
                        U[i * m + j, :] = u
                        
                # Solve for V
                for k in range(p):
                    for jp in range(m):
                        # Equation: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = delta[j, jp]
                        # For fixed k, jp, linear in V[jp*p+k, :]
                        # For each i, let F_i[j] = sum_r U[i*m+j, r] * W[k*n+i, r]
                        # Then sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = delta[j, jp]
                        # This is sum_r V_r * (U[i*m+j, r] * W[k*n+i, r]) = delta[j, jp]
                        # Let G be a matrix of size (n*m) x R
                        # G[i*m + j, r] = U[i*m+j, r] * W[k*n+i, r]
                        # We want G @ V[jp*p+k, :] = target_vector
                        # target_vector[i*m + j] = 1 if j==jp else 0
                        G = np.zeros((n * m, R))
                        for i in range(n):
                            for j in range(m):
                                G[i * m + j, :] = U[i * m + j, :] * W[k * n + i, :]
                        target_vec = np.zeros(n * m)
                        for i in range(n):
                            target_vec[i * m + jp] = 1.0
                        
                        v, residuals, rank, sv = np.linalg.lstsq(G, target_vec, rcond=None)
                        V[jp * p + k, :] = v
                        
            # Check error
            error = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            val = np.sum(U[i * m + j, :] * V[jp * p + k, :] * W[k * n + i, :])
                            target = 1.0 if j == jp else 0.0
                            error += (val - target) ** 2
                            
            if error < 1e-8:
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
                
        if found:
            break
            
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
