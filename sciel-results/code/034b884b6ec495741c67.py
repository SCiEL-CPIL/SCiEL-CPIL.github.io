"""Matrix multiplication 3,4,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to find low-rank decomposition using ALS (Alternating Least Squares)
    # Try ranks from a lower bound up to 72
    # Lower bound: max(n*m, m*p, p*n) / min(n,m,p) ... actually use Strassen-like bounds
    # For 3x4x6, try ranks starting from reasonable values
    
    # Known: rank >= max(n*m, m*p, p*n) is not a valid lower bound
    # Better lower bound: the tensor rank is at least the "slice rank"
    # For now, try a range of ranks
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Initialize with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try various ranks with ALS
    # The minimum possible rank for this problem - let's try from 40 up
    for rank in range(40, min(72, n * m * p) + 1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(rank * 1000 + trial)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # ALS iterations
            converged = False
            for iteration in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each row a, solve (V^T ⊙ W^T) * u_a = T[a,:,:].flatten()
                # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                # This is: U[a,:] @ (V[:,b] * W[:,c]) = T[a,b,c] for all b,c
                # So: (V^T * W^T) flattened... let me think
                # Matrix M_b_c = V[:,b] * W[:,c] (element-wise product), shape (rank,)
                # We need U[a,:] @ M = T[a,:,:].flatten()
                # So M is a (m*p * p*n, rank) matrix where M[b*p*n+c, r] = V[b,r]*W[c,r]
                
                # Build M for current V, W
                MW = np.zeros((m * p * p * n, rank))
                for b in range(m * p):
                    for c in range(p * n):
                        MW[b * p * n + c, :] = V[b, :] * W[c, :]
                
                # Solve for U
                # U[a,:] @ MW = T[a,:,:].flatten()
                # U[a,:] = (MW^T MW)^{-1} MW^T T[a,:,:].flatten()
                MWtMW = MW.T @ MW
                MWtT = MW.T @ T.reshape(m * p * p * n, n * m).T  # (rank, n*m)
                try:
                    U_new = np.linalg.solve(MWtMW + 1e-8 * np.eye(rank), MWtT).T
                except:
                    U_new = np.linalg.lstsq(MW, T.reshape(n * m, m * p * p * n), rcond=None)[0]
                
                # Update V
                UW = np.zeros((n * m * p * n, rank))
                for a in range(n * m):
                    for c in range(p * n):
                        UW[a * p * n + c, :] = U[a, :] * W[c, :]
                
                UWtUW = UW.T @ UW
                UWtT = UW.T @ T.reshape(p * n, n * m * m * p).T  # Hmm, need to be careful with reshape
                # T[a,b,c], for fixed c: sum_r U[a,r]*V[b,r] = T[a,b,c]
                # V[b,:] @ (U[a,:] * something)... 
                # For fixed c: sum_r V[b,r] * U[a,r] = T[a,b,c] for all a,b
                # So: V[b,:] @ diag(U[a,:]) ... no
                # V[b,:] * U[a,:] element-wise sum = T[a,b,c]
                # Let N_a_b = U[a,:] (shape rank), we need V[b,:] @ N_a_b = T[a,b,c]
                # So N is (n*m * m*p, rank) where N[a*m*p+b, r] = U[a,r]
                # Then V[b,:] @ N[a*m_p+b, :] = T[a,b,c]
                # So N^T @ V[b,:] = ... 
                # Actually: sum_r V[b,r] * U[a,r] = T[a,b,c]
                # This is V[b,:] dot U[a,:] = T[a,b,c]
                # So for each c, we have a system: V @ U^T = T[:,:,c]
                # V = (U^T U)^{-1} U^T T[:,:,c]... but this varies by c
                
                # Better: build matrix P where P[a*m*p+b, r] = U[a,r] for each c
                # Then V[b,:] @ P[a*m*p+b, :] = T[a,b,c]
                # So P^T @ V[b,:]^T = T[:,b,c]
                # V[b,:] = (P P^T)^{-1} P T[:,b,c]... 
                
                # Let me redo: for fixed c, T_c = T[:,:,c] has shape (n*m, m*p)
                # T_c[a,b] = sum_r U[a,r] V[b,r]
                # So T_c = U @ V^T
                # We want to find V given U and T_c
                # V^T = U^+ T_c where U^+ is pseudo-inverse
                # Or: V = (U^T U)^{-1} U^T T_c^T
                
                # But this is per c, and V must be consistent across all c
                # So we stack: for all c, T[:,:,c] = U @ V^T
                # Flatten: for each (a,b,c), T[a,b,c] = sum_r U[a,r] V[b,r]
                # Group by b: for fixed b, sum_r V[b,r] U[a,r] = T[a,b,c] for all a,c
                # So V[b,:] @ (U[a,:] * 1) ... 
                # Let Q be (n*m * p*n, rank) where Q[a*p*n+c, r] = U[a,r]
                # Then V[b,:] @ Q[a*p*n+c, :] = T[a,b,c]
                # So Q^T @ V[b,:]^T = T[:,b,:].flatten()
                # V[b,:] = (Q Q^T)^{-1} Q T[:,b,:].flatten()
                
                Q = np.zeros((n * m * p * n, rank))
                for a in range(n * m):
                    for c in range(p * n):
                        Q[a * p * n + c, :] = U[a, :]
                
                QtQ = Q.T @ Q
                QtT = Q.T @ T.reshape(m * p, n * m * p * n).T  # (rank, m*p)
                try:
                    V_new = np.linalg.solve(QtQ + 1e-8 * np.eye(rank), QtT).T
                except:
                    V_new = np.linalg.lstsq(Q, T.reshape(m * p, n * m * p * n), rcond=None)[0]
                
                # Update W
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed c: sum_r W[c,r] U[a,r] V[b,r] = T[a,b,c]
                # Let R be (n*m * m*p, rank) where R[a*m*p+b, r] = U[a,r] * V[b,r]
                # Then W[c,:] @ R[a*m_p+b, :] = T[a,b,c]
                # R^T @ W[c,:]^T = T[:,:,c].flatten()
                
                R = np.zeros((n * m * m * p, rank))
                for a in range(n * m):
                    for b in range(m * p):
                        R[a * m * p + b, :] = U[a, :] * V[b, :]
                
                RtR = R.T @ R
                RtT = R.T @ T.reshape(p * n, n * m * m * p).T  # (rank, p*n)
                try:
                    W_new = np.linalg.solve(RtR + 1e-8 * np.eye(rank), RtT).T
                except:
                    W_new = np.linalg.lstsq(R, T.reshape(p * n, n * m * m * p), rcond=None)[0]
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if iteration % 50 == 0:
                    # Compute residual
                    residual = 0
                    for a in range(n * m):
                        for b in range(m * p):
                            for c in range(p * n):
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                residual += (val - T[a, b, c]) ** 2
                    if residual < 1e-10:
                        converged = True
                        break
                
                if time.time() > deadline - 5:
                    break
            
            # Check if we got an exact decomposition
            if converged or iteration >= 499:
                # Verify exactness
                max_err = 0
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err = abs(val - T[a, b, c])
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-6:
                    # Try to clean up to integer/half-integer values
                    U_clean = np.round(U * 2) / 2
                    V_clean = np.round(V * 2) / 2
                    W_clean = np.round(W * 2) / 2
                    
                    # Verify cleaned version
                    max_err_clean = 0
                    for a in range(n * m):
                        for b in range(m * p):
                            for c in range(p * n):
                                val = np.dot(U_clean[a, :], V_clean[b, :] * W_clean[c, :])
                                err = abs(val - T[a, b, c])
                                if err > max_err_clean:
                                    max_err_clean = err
                    
                    if max_err_clean < 1e-6:
                        if rank < best_rank:
                            best_rank = rank
                            best_result = (U_clean, V_clean, W_clean)
                            break  # No need for more trials at this rank
        
        if best_result is not None and best_rank < rank:
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U_list = [[float(x) for x in row] for row in U_school]
        V_list = [[float(x) for x in row] for row in V_school]
        W_list = [[float(x) for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
