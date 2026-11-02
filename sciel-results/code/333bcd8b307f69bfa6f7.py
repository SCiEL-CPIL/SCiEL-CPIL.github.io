import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Build the rank 32 base solution
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M2
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        # M3
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M4
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        # M5
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        # M6
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M7
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1; col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1; col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1; col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    # Target tensor T[i, j, k] is 1 if i,j,k are consistent indices for multiplication
    # T[i, j, k] = 1 for all i in [0,2), j in [0,9), k in [0,2)
    # Wait, the tensor is T_{alpha, beta, gamma}.
    # alpha = i*m+k, beta = k*p+j, gamma = j*n+i.
    # T[alpha, beta, gamma] = 1 if k_alpha == k_beta and j_beta == j_gamma and i_gamma == i_alpha.
    # In our case, every (i,j,k) triple is valid.
    # So T is 1 for all alpha, beta, gamma such that they form a valid chain.
    # Actually, the decomposition sums over r:
    # T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
    # We need this to be 1 for valid chains, 0 for invalid.
    
    # However, the "valid chain" condition is inherent in the indices.
    # If we just check the residual on the valid indices, we assume the invalid ones are 0.
    # But the decomposition might produce non-zero values for invalid indices.
    # We must ensure T[alpha, beta, gamma] = 0 for invalid indices too.
    
    # Invalid indices:
    # alpha = i*m+k, beta = k'*p+j, gamma = j'*n+i'
    # Invalid if k != k' or j != j' or i != i'.
    
    # We will minimize the sum of squares of the residual over ALL indices.
    
    def compute_residual(U, V, W):
        # U: (4, R), V: (18, R), W: (18, R)
        # Residual for all alpha, beta, gamma
        # This is a 3-tensor of size 4x18x18 = 1296 entries.
        # We can compute the tensor product sum.
        # T_reconstructed = sum_r U[:, r] * V[:, r] * W[:, r] (outer product)
        # T_reconstructed[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
        
        # Efficient computation:
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target T_target:
        # T_target[alpha, beta, gamma] = 1 if valid(alpha, beta, gamma) else 0
        
        # Construct T_target
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                alpha = i*m + k
                for j in range(p):
                    beta = k*p + j
                    gamma = j*n + i
                    T_target[alpha, beta, gamma] = 1.0
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return T_recon - T_target

    def objective(params):
        size_u = n*m * rank
        size_v = m*p * rank
        U = params[:size_u].reshape(n*m, rank)
        V = params[size_u:size_u+size_v].reshape(m*p, rank)
        W = params[size_u+size_v:].reshape(p*n, rank)
        
        res = compute_residual(U, V, W)
        return np.sum(res**2)

    # Try to reduce rank to 31
    # We can try to set one column of U, V, W to zero and optimize the rest.
    # Or optimize all and check if rank < 32.
    
    best_U, best_V, best_W = U, V, W
    best_obj = objective(np.concatenate([U.flatten(), V.flatten(), W.flatten()]))
    
    # Local search for rank 31
    # Try removing each term and re-optimizing
    for r_remove in range(rank):
        if time.time() > deadline - 10:
            break
            
        # Create a copy with column r_remove zeroed
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, r_remove] = 0
        V_test[:, r_remove] = 0
        W_test[:, r_remove] = 0
        
        params = np.concatenate([U_test.flatten(), V_test.flatten(), W_test.flatten()])
        
        try:
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                size_u = n*m * rank
                size_v = m*p * rank
                best_U = best_params[:size_u].reshape(n*m, rank)
                best_V = best_params[size_u:size_u+size_v].reshape(m*p, rank)
                best_W = best_params[size_u+size_v:].reshape(p*n, rank)
        except Exception:
            pass

    # Check if we can round to half-integers
    def can_round(arr):
        return np.all(np.abs(2*arr - np.round(2*arr)) < 1e-6)

    if can_round(best_U) and can_round(best_V) and can_round(best_W):
        best_U = np.round(2*best_U) / 2
        best_V = np.round(2*best_V) / 2
        best_W = np.round(2*best_W) / 2

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
