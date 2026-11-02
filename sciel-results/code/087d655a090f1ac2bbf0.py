"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p  # 30
    
    # Create schoolbook decomposition as fallback
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_result = None
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Precompute index arrays for vectorized computation
    # U indices: i*m+j -> (i, j)
    # V indices: j*p+k -> (j, k)
    # W indices: k*n+i -> (k, i)
    
    # For each (i,j,k) triple, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Vectorized: create arrays of indices
    I, J, K = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    I = I.ravel()
    J = J.ravel()
    K = K.ravel()
    
    u_idx = I * m + J  # shape (n*m*p,)
    v_idx = J * p + K  # shape (n*m*p,)
    w_idx = K * n + I  # shape (n*m*p,)
    
    num_triples = n * m * p  # 30
    
    for rank in range(20, schoolbook_rank):
        if time.time() > deadline:
            break
        
        for restart in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 1000 + restart)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            lr = 0.05
            prev_err = np.inf
            
            for step in range(3000):
                if time.time() > deadline:
                    break
                
                # Compute P[i,j,k] = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                # Vectorized: 
                # For each triple t, P[t] = dot(U[u_idx[t]], V[v_idx[t]] * W[w_idx[t]])
                # = sum_r U[u_idx[t], r] * V[v_idx[t], r] * W[w_idx[t], r]
                
                # Efficient computation:
                # U_sel = U[u_idx, :]  shape (num_triples, rank)
                # V_sel = V[v_idx, :]  shape (num_triples, rank)
                # W_sel = W[w_idx, :]  shape (num_triples, rank)
                # P = sum_r U_sel * V_sel * W_sel along axis=1
                
                U_sel = U[u_idx, :]
                V_sel = V[v_idx, :]
                W_sel = W[w_idx, :]
                
                P = np.sum(U_sel * V_sel * W_sel, axis=1)
                E = P - 1.0
                err = np.sum(E * E)
                
                if err < 1e-12:
                    best_rank = rank
                    best_result = (U.copy(), V.copy(), W.copy())
                    break
                
                if err > prev_err * 1.01 and step > 100:
                    lr *= 0.5
                prev_err = err
                
                # Gradients (vectorized)
                # dU[u_idx[t], r] += E[t] * V_sel[t, r] * W_sel[t, r]
                # We need to accumulate contributions to each row of U, V, W
                
                # For U: for each triple t, contribution to row u_idx[t] is E[t] * V_sel[t] * W_sel[t]
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Use np.add.at for accumulated gradients
                grad_U = E[:, None] * V_sel * W_sel  # shape (num_triples, rank)
                grad_V = E[:, None] * U_sel * W_sel
                grad_W = E[:, None] * U_sel * V_sel
                
                np.add.at(dU, u_idx, grad_U)
                np.add.at(dV, v_idx, grad_V)
                np.add.at(dW, w_idx, grad_W)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if step % 200 == 0:
                    lr *= 0.98
            
            if best_result is not None:
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Try to round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify rounded version
        U_sel = U_r[u_idx, :]
        V_sel = V_r[v_idx, :]
        W_sel = W_r[w_idx, :]
        P_check = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        if np.max(np.abs(P_check - 1.0)) < 1e-6:
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
        else:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    else:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
# EVOLVE-BLOCK-END

