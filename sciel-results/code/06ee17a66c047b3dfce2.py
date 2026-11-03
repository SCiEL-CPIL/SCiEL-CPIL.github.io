import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    R_max = n * m * p  # 100
    
    # Precompute index mappings
    # For U: index (i*m+j) -> row in U
    # For V: index (j*p+k) -> row in V  
    # For W: index (k*n+i) -> row in W
    
    # Precompute all (i,j,k) triples and their corresponding indices
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    
    # Build the trivial solution
    U_triv = np.zeros((n_m, R_max))
    V_triv = np.zeros((m_p, R_max))
    W_triv = np.zeros((p_n, R_max))
    
    for r, (u_idx, v_idx, w_idx) in enumerate(triples):
        U_triv[u_idx, r] = 1.0
        V_triv[v_idx, r] = 1.0
        W_triv[w_idx, r] = 1.0
    
    # Deadline guard: need to finish within 120s, leave 15s margin
    start_time = time.time()
    deadline = start_time + 105.0  # 105s max for search
    
    # Search for lower rank solutions using ALS with multiple restarts
    # We'll try ranks from R_max-1 down to a lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    min_rank = 25
    
    best_result = None
    best_rank = R_max
    
    # Try different ranks, starting from just below trivial
    for R in range(R_max - 1, min_rank - 1, -1):
        # Check deadline
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random restarts for this rank
        found_solution = False
        max_restarts = 50  # Number of random restarts to try
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize with random perturbation of trivial solution
            # Take first R columns and add noise
            U = U_triv[:, :R].copy()
            V = V_triv[:, :R].copy()
            W = W_triv[:, :R].copy()
            
            # Add small random noise to break symmetry
            np.random.seed(1000 + R * 100 + restart)
            noise_scale = 1e-3
            U += np.random.randn(n_m, R) * noise_scale
            V += np.random.randn(m_p, R) * noise_scale
            W += np.random.randn(p_n, R) * noise_scale
            
            # ALS optimization
            best_error = float('inf')
            best_U, best_V, best_W = None, None, None
            
            for it in range(80):
                if time.time() > deadline - 5.0:
                    break
                
                # Update U
                A_u = np.zeros((n_m, R))
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        # V slice: rows j*p to j*p+p
                        V_slice = V[j*p:(j+1)*p, :]
                        # W slice: rows k*n+i for k in 0..p-1
                        W_rows = [k * n + i for k in range(p)]
                        W_slice = W[W_rows, :]
                        A_u[u_idx, :] = np.sum(V_slice * W_slice, axis=0)
                
                norms_sq = np.sum(A_u * A_u, axis=1)
                norms_sq[norms_sq == 0] = 1e-9
                U = A_u / norms_sq[:, np.newaxis]
                
                # Update V
                A_v = np.zeros((m_p, R))
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        # U slice: rows i*m+j for i in 0..n-1
                        U_rows = [i * m + j for i in range(n)]
                        U_slice = U[U_rows, :]
                        # W slice: rows k*n+i for i in 0..n-1
                        W_rows = [k * n + i for i in range(n)]
                        W_slice = W[W_rows, :]
                        A_v[v_idx, :] = np.sum(U_slice * W_slice, axis=0)
                
                norms_sq_v = np.sum(A_v * A_v, axis=1)
                norms_sq_v[norms_sq_v == 0] = 1e-9
                V = A_v / norms_sq_v[:, np.newaxis]
                
                # Update W
                A_w = np.zeros((p_n, R))
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        # U slice: rows i*m+j for j in 0..m-1
                        U_rows = [i * m + j for j in range(m)]
                        U_slice = U[U_rows, :]
                        # V slice: rows j*p+k for j in 0..m-1
                        V_rows = [j * p + k for j in range(m)]
                        V_slice = V[V_rows, :]
                        A_w[w_idx, :] = np.sum(U_slice * V_slice, axis=0)
                
                norms_sq_w = np.sum(A_w * A_w, axis=1)
                norms_sq_w[norms_sq_w == 0] = 1e-9
                W = A_w / norms_sq_w[:, np.newaxis]
                
                # Calculate error
                err = 0.0
                for u_idx, v_idx, w_idx in triples:
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_error < 1e-10:
                    break
            
            if best_error < 1e-8:
                # Round to half-integers
                U_int = np.round(best_U * 2) / 2
                V_int = np.round(best_V * 2) / 2
                W_int = np.round(best_W * 2) / 2
                
                # Verify exactness
                is_exact = True
                for u_idx, v_idx, w_idx in triples:
                    val = np.dot(U_int[u_idx, :], V_int[v_idx, :] * W_int[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        is_exact = False
                        break
                
                if is_exact:
                    best_result = (U_int, V_int, W_int)
                    best_rank = R
                    found_solution = True
                    break
        
        if found_solution:
            break
    
    if best_result is not None:
        U_int, V_int, W_int = best_result
        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fallback to trivial solution
    return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
# EVOLVE-BLOCK-END
