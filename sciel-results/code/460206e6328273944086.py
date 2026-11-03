"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank using alternating least squares
    # The constraint: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try various target ranks
    for target_rank in range(60, rank_sb + 1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            if found:
                break
                
            # Random init with some structure
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize
            for c in range(target_rank):
                norm = np.sqrt(np.sum(U[:,c]**2) * np.sum(V[:,c]**2) * np.sum(W[:,c]**2))
                if norm > 1e-10:
                    U[:,c] /= norm**(1/3)
                    V[:,c] /= norm**(1/3)
                    W[:,c] /= norm**(1/3)
            
            # Alternating minimization
            for it in range(500):
                if time.time() > deadline - 5:
                    break
                    
                # Update W: for each (k,i), we have sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c] summed over j) = 1
                # For fixed k,i: sum_c W[k*n+i,c] * sum_j U[i*m+j,c] * V[j*p+k,c] = 1
                # Let S_c = sum_j U[i*m+j,c] * V[j*p+k,c]
                # Then: sum_c W[k*n+i,c] * S_c = 1
                # This is underdetermined (target_rank > 1), so we need to minimize norm subject to constraint
                
                # Actually, this is a system of equations. For each (i,j,k):
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # 
                # If we fix U and V, then for each (k,i), we have:
                # sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
                # Let A_c = sum_j U[i*m+j,c] * V[j*p+k,c] for fixed (k,i)
                # Then: sum_c W[k*n+i,c] * A_c = 1
                # This is one equation in target_rank unknowns. Under-determined.
                # We want to find W that satisfies all (k,i) equations and minimizes something.
                
                # This is getting complex. Let me use a different approach.
                
                # Use gradient descent on the loss: sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                
                # Compute gradients
                # dL/dU[i*m+j,c] = sum_k 2*(pred[i,j,k]-1) * V[j*p+k,c] * W[k*n+i,c]
                # where pred[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                
                # This is expensive. Let me use a simpler heuristic.
                
                # Just do random perturbations and keep if loss decreases
                loss_before = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            pred = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
                            loss_before += (pred - 1)**2
                
                # Perturb W
                dW = np.random.randn(p*n, target_rank) * 0.01
                W_new = W + dW
                loss_after = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            pred = np.sum(U[i*m+j,:] * V[j*p+k,:] * W_new[k*n+i,:])
                            loss_after += (pred - 1)**2
                
                if loss_after < loss_before:
                    W = W_new
                    loss = loss_after
                else:
                    loss = loss_before
                
                if loss < 1e-6:
                    found = True
                    break
            
            if found and loss < 1e-6:
                # Check if we can round to integers/halves
                # For now, accept if loss is small
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to list and round if close to integer/half-integer
    def round_matrix(M):
        M_r = M.copy()
        for i in range(M_r.shape[0]):
            for j in range(M_r.shape[1]):
                # Try to round to nearest half-integer
                val = M_r[i,j]
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_half) < 1e-3:
                    M_r[i,j] = nearest_half
                else:
                    M_r[i,j] = round(val)
        return M_r
    
    # Use schoolbook if no better found
    if best_rank >= rank_sb:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Verify the decomposition
    U_r = round_matrix(best_U)
    V_r = round_matrix(best_V)
    W_r = round_matrix(best_W)
    
    # Check if rounded version still works
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pred = np.sum(U_r[i*m+j,:] * V_r[j*p+k,:] * W_r[k*n+i,:])
                max_err = max(max_err, abs(pred - 1))
    
    if max_err < 1e-6:
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
