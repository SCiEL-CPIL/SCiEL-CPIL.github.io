import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    num_a = n * m
    num_b = m * p
    num_c = p * n
    
    # Build the target sparse tensor T
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # a in [0, 9), b in [0, 18), c in [0, 18)
    T = np.zeros((num_a, num_b, num_c), dtype=np.int8)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
                
    schoolbook_rank = num_a * num_b * num_c # No, n*m*p
    schoolbook_rank = n * m * p
    # Actually schoolbook rank is n*m*p = 54.
    
    # Helper to build schoolbook decomposition
    def get_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(num_a)]
        V = [[0] * schoolbook_rank for _ in range(num_b)]
        W = [[0] * schoolbook_rank for _ in range(num_c)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Helper to verify a decomposition
    def verify(U, V, W):
        # U, V, W are lists of lists or numpy arrays
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check if entries are half-integers (multiples of 0.5)
        # Convert to integers by multiplying by 2
        if not np.allclose(U_np * 2, np.round(U_np * 2)): return False
        if not np.allclose(V_np * 2, np.round(V_np * 2)): return False
        if not np.allclose(W_np * 2, np.round(W_np * 2)): return False
        
        # Reconstruct tensor
        # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a batched matrix multiplication or einsum
        # T_recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        T_recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        # Check exact match with T (integer comparison)
        # T_recon should be integers or half-integers?
        # The product of half-integers is quarter-integers.
        # The sum must result in exactly 1 or 0.
        # So T_recon entries must be exactly 0 or 1.
        if not np.allclose(T_recon, np.round(T_recon)):
            return False
            
        if np.array_equal(np.round(T_recon), T):
            return True
        return False

    # Gradient Descent Search
    def search_rank(R, time_limit=20.0):
        start_time = time.time()
        # Try multiple restarts
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize
            # Use small random init
            U = np.random.randn(num_a, R) * 0.1
            V = np.random.randn(num_b, R) * 0.1
            W = np.random.randn(num_c, R) * 0.1
            
            # Learning rate decay
            lr = 0.01
            best_loss = float('inf')
            
            for it in range(500): # Max iterations per restart
                if time.time() - start_time > time_limit:
                    break
                    
                # Compute Loss: sum (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
                # Gradient calculation
                # L = || T_recon - T ||_F^2
                # dL/dU[a,r] = 2 * sum_{b,c} (T_recon[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
                
                # Efficient calculation
                # Residual R_res[a,b,c] = T_recon[a,b,c] - T[a,b,c]
                # dU = 2 * np.einsum('abc,br,cr->ar', R_res, V, W)
                # dV = 2 * np.einsum('abc,ar,cr->br', R_res, U, W)
                # dW = 2 * np.einsum('abc,ar,br->cr', R_res, U, V)
                
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                R_res = T_recon - T
                
                dU = 2.0 * np.einsum('abc,br,cr->ar', R_res, V, W)
                dV = 2.0 * np.einsum('abc,ar,cr->br', R_res, U, W)
                dW = 2.0 * np.einsum('abc,ar,br->cr', R_res, U, V)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Decay LR
                lr *= 0.99
                
                # Check convergence
                loss = np.sum(R_res**2)
                if loss < 1e-6:
                    # Potential solution found, try rounding
                    U_r = np.round(U * 2) / 2.0
                    V_r = np.round(V * 2) / 2.0
                    W_r = np.round(W * 2) / 2.0
                    if verify(U_r, V_r, W_r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                elif loss > best_loss * 1.1:
                    # Diverged or stuck, break this restart
                    break
                best_loss = min(best_loss, loss)
                
                # Periodic rounding check
                if it % 50 == 0:
                    U_r = np.round(U * 2) / 2.0
                    V_r = np.round(V * 2) / 2.0
                    W_r = np.round(W * 2) / 2.0
                    if verify(U_r, V_r, W_r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
        return None

    # Try ranks in increasing order
    # Known upper bound is 46 (2*23).
    # We try 40, 42, 44, 46.
    # If we find one, we return it.
    ranks_to_try = [40, 42, 44, 46]
    
    for R in ranks_to_try:
        result = search_rank(R, time_limit=20.0)
        if result:
            return result
            
    # Fallback to schoolbook
    return get_schoolbook()
