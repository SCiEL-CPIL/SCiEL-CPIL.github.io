import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140 seconds to be safe
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find a better decomposition
    # We will try ranks from 16 to best_rank - 1
    for r in range(16, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(attempt * 100 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Iterate
            for _ in range(50):
                if time.time() > deadline:
                    break
                    
                # Update W
                # M[i,k,t] = sum_j U[i,j,t] V[j,k,t]
                # We need sum_t M[i,k,t] W[k,i,t] = 1
                # Reshape U to (n, m, r), V to (m, p, r)
                U_reshaped = U.reshape(n, m, r)
                V_reshaped = V.reshape(m, p, r)
                # M[i,k,t] = sum_j U_reshaped[i,j,t] V_reshaped[j,k,t]
                M = np.einsum('ijt,jkt->ikt', U_reshaped, V_reshaped)
                
                # Solve for W
                # For each i, k, sum_t M[i,k,t] W[k,i,t] = 1
                # W is (p, n, r)
                W_new = np.zeros((p, n, r))
                for i in range(n):
                    for k in range(p):
                        # M[i,k,:] is a vector of length r
                        # We need w such that dot(M[i,k,:], w) = 1
                        # Simple solution: w = M[i,k,:] / dot(M[i,k,:], M[i,k,:])
                        m_vec = M[i, k, :]
                        norm_sq = np.dot(m_vec, m_vec)
                        if norm_sq > 1e-10:
                            W_new[k, i, :] = m_vec / norm_sq
                        else:
                            # Fallback: set one component to 1
                            W_new[k, i, 0] = 1
                W = W_new.reshape(p * n, r)
                
                # Update U
                # N[j,i,t] = sum_k V[j,k,t] W[k,i,t]
                V_reshaped = V.reshape(m, p, r)
                W_reshaped = W.reshape(p, n, r)
                N = np.einsum('jkt,kit->jit', V_reshaped, W_reshaped)
                
                U_new = np.zeros((n, m, r))
                for i in range(n):
                    for j in range(m):
                        n_vec = N[j, i, :]
                        norm_sq = np.dot(n_vec, n_vec)
                        if norm_sq > 1e-10:
                            U_new[i, j, :] = n_vec / norm_sq
                        else:
                            U_new[i, j, 0] = 1
                U = U_new.reshape(n * m, r)
                
                # Update V
                # P[i,k,t] = sum_j U[i,j,t] W[k,i,t]? No.
                # Sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1
                # Fix U, W. Solve for V.
                # For each j, k: sum_t (sum_i U[i,j,t] W[k,i,t]) V[j,k,t] = 1
                U_reshaped = U.reshape(n, m, r)
                W_reshaped = W.reshape(p, n, r)
                # Q[j,k,t] = sum_i U[i,j,t] W[k,i,t]
                Q = np.einsum('ijt,kit->jkt', U_reshaped, W_reshaped)
                
                V_new = np.zeros((m, p, r))
                for j in range(m):
                    for k in range(p):
                        q_vec = Q[j, k, :]
                        norm_sq = np.dot(q_vec, q_vec)
                        if norm_sq > 1e-10:
                            V_new[j, k, :] = q_vec / norm_sq
                        else:
                            V_new[j, k, 0] = 1
                V = V_new.reshape(m * p, r)
                
            # Check error
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            # Error = sum_t U[i,j,t] V[j,k,t] W[k,i,t] - 1
            prod = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
            error = np.sum(np.abs(prod - 1))
            
            if error < 1e-5:
                # Try to round to half-integers
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                U_r = U_half.reshape(n, m, r)
                V_r = V_half.reshape(m, p, r)
                W_r = W_half.reshape(p, n, r)
                prod_r = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
                error_r = np.sum(np.abs(prod_r - 1))
                
                if error_r < 1e-9:
                    best_result = {"U": U_half, "V": V_half, "W": W_half}
                    best_rank = r
                    break # Found a better rank, break out of attempt loop
            
            if time.time() > deadline:
                break
            
        if time.time() > deadline:
            break
        
        # If we found a better rank, we can stop early if we hit a known lower bound
        if best_rank <= 35:
            break

    return best_result
# EVOLVE-BLOCK-END
