"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U0, V0, W0
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Try ranks from 74 down to 60
    for target_rank in range(rank_school - 1, max(rank_school - 15, 0), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Random initialization with small values to avoid divergence
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # ALS Optimization
            # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Fix V, W. Solve for U.
            # For each i, j:
            #   sum_k ( sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) ) = 1 is NOT correct.
            #   The equation is for each (i, j, k):
            #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            #   This is a system of p equations (for k=0..p-1) for the unknown vector U[i*m+j, :] of length R.
            #   Let A[i*m+j, k, c] = V[j*p+k, c] * W[k*n+i, c]
            #   Then sum_c U[i*m+j, c] * A[i*m+j, k, c] = 1 for all k.
            #   This is a least squares problem: min || A_slice @ U_vec - 1 ||^2
            #   where A_slice is (p, R) and U_vec is (R,).
            
            success = False
            for iteration in range(20):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        idx_u = i * m + j
                        # Build A_slice: (p, R)
                        # A_slice[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # V_slice = V[j*p : (j+1)*p, :] shape (p, R)
                        # W_slice = W[:, :] but we need W[k*n+i, c] for k=0..p-1
                        # W_indices = [k*n + i for k in range(p)]
                        # W_vec = W[W_indices, :] shape (p, R)
                        w_indices = np.arange(p) * n + i
                        W_slice = W[w_indices, :] # (p, R)
                        V_slice = V[j*p : (j+1)*p, :] # (p, R)
                        
                        A_slice = V_slice * W_slice # (p, R)
                        # Solve A_slice @ U_vec = 1 (vector of ones of length p)
                        # Use least squares
                        try:
                            U_vec, residuals, rank, s = np.linalg.lstsq(A_slice, np.ones(p), rcond=None)
                            U[idx_u, :] = U_vec
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        idx_v = j * p + k
                        # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                        # Unknown: V[idx_v, :]
                        # Equations for i=0..n-1
                        # B_slice[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        # B_slice @ V_vec = 1 (vector of ones of length n)
                        u_indices = np.arange(n) * m + j
                        U_slice = U[u_indices, :] # (n, R)
                        w_indices = np.arange(p) * n + k # Wait, W index is k*n + i. Here k is fixed.
                        # W indices for i=0..n-1: [0*n+k, 1*n+k, ..., (n-1)*n+k]
                        w_indices_v = np.arange(n) * n + k
                        W_slice = W[w_indices_v, :] # (n, R)
                        
                        B_slice = U_slice * W_slice # (n, R)
                        try:
                            V_vec, residuals, rank, s = np.linalg.lstsq(B_slice, np.ones(n), rcond=None)
                            V[idx_v, :] = V_vec
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        idx_w = k * n + i
                        # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                        # Unknown: W[idx_w, :]
                        # Equations for j=0..m-1
                        # C_slice[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        # C_slice @ W_vec = 1 (vector of ones of length m)
                        u_indices = np.arange(m) * m + j # No, i is fixed.
                        # U indices: [i*m+0, i*m+1, ..., i*m+m-1]
                        u_indices_w = i * m + np.arange(m)
                        U_slice = U[u_indices_w, :] # (m, R)
                        
                        v_indices = np.arange(m) * p + k
                        V_slice = V[v_indices, :] # (m, R)
                        
                        C_slice = U_slice * V_slice # (m, R)
                        try:
                            W_vec, residuals, rank, s = np.linalg.lstsq(C_slice, np.ones(m), rcond=None)
                            W[idx_w, :] = W_vec
                        except:
                            pass
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1)**2
                
                if err < 1e-8:
                    success = True
                    break
            
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U0.tolist()
        V_out = V0.tolist()
        W_out = W0.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
