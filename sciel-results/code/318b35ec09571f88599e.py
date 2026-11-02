"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # 1. Build Schoolbook Decomposition (Rank = n*m*p)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = rank_schoolbook

    # 2. Attempt to find lower rank via Optimization
    # We only try a few ranks below schoolbook to save time.
    # The tensor T is sparse.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_exact(U, V, W, n, m, p, T):
        """Check if decomposition is exact with integer/half-integer entries."""
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    # val should be close to x.0 or x.5
                    nearest_half = round(val * 2) / 2
                    if abs(val - nearest_half) > 1e-9:
                        return False
                    # Check if it's effectively integer or half
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        
        # Check reconstruction
        r = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Reconstruct T_rec
        # T_rec[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
        # This is a tensor contraction.
        # We can do this by iterating over k
        T_rec = np.zeros((n*m, m*p, p*n))
        for k in range(r):
            T_rec += np.einsum('a,b,c->abc', U_mat[:, k], V_mat[:, k], W_mat[:, k])
            
        # Check exact match with T
        if not np.allclose(T_rec, T, atol=1e-9, rtol=1e-9):
            return False
            
        return True

    # Optimization Loop
    np.random.seed(42)
    
    # Try ranks from 59 down to 55 (small range to keep it fast)
    for r in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 40), -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for attempt in range(2):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Gradient Descent
            # Loss: sum( (sum_k U_a_k V_b_k W_c_k - T_abc)^2 )
            # We will run a small number of steps
            
            lr = 0.01
            steps = 200
            
            for step in range(steps):
                if time.time() > deadline - 10:
                    break
                    
                # Compute gradients
                # T_rec = sum_k outer(U[:,k], V[:,k], W[:,k])
                # grad_U[a,k] = sum_{b,c} (T_rec[a,b,c] - T[a,b,c]) * V[b,k] * W[c,k]
                
                T_rec = np.zeros((n*m, m*p, p*n))
                for k in range(r):
                    T_rec += np.einsum('a,b,c->abc', U[:, k], V[:, k], W[:, k])
                    
                diff = T_rec - T
                
                # Compute grads
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for k in range(r):
                    # dU[:, k] = einsum('abc,bc->a', diff, V[:,k]*W[:,k])
                    # dV[:, k] = einsum('abc,ac->b', diff, U[:,k]*W[:,k])
                    # dW[:, k] = einsum('abc,ab->c', diff, U[:,k]*V[:,k])
                    
                    dU[:, k] = np.einsum('abc,bc->a', diff, V[:, k] * W[:, k])
                    dV[:, k] = np.einsum('abc,ac->b', diff, U[:, k] * W[:, k])
                    dW[:, k] = np.einsum('abc,ab->c', diff, U[:, k] * V[:, k])
                    
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
            # After optimization, check if it's close enough to be rounded
            # Calculate final error
            T_rec = np.zeros((n*m, m*p, p*n))
            for k in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, k], V[:, k], W[:, k])
            
            error = np.linalg.norm(T_rec - T)
            
            if error < 1e-5:
                # Try rounding to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Convert to lists for verification function
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if verify_exact(U_list, V_list, W_list, n, m, p, T):
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to Schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
