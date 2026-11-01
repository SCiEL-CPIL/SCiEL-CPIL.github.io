"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for multiplying
    a 2x5 matrix by a 5x5 matrix.
    
    The schoolbook rank is n*m*p = 50. We attempt to find a lower rank
    decomposition using numerical optimization if time permits.
    """
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    schoolbook_rank = n * m * p  # 50
    
    # First, create the schoolbook decomposition as baseline
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # The schoolbook decomposition is always valid
    schoolbook_result = make_schoolbook()
    
    # Check if we have time to attempt optimization
    if time.time() > deadline - 30:
        return schoolbook_result
    
    # Create the target tensor in vectorized form
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1

    # Helper to verify a decomposition
    def verify_decomposition(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Reconstruct T
        # T[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
        # Efficiently: T = sum_k U[:,k] outer V[:,k] outer W[:,k]
        # We can check || T - T_reconstructed || < tol
        # But we need exact integer/half-integer check.
        # The prompt says "exactly decompose".
        # So we should check if U, V, W are half-integers and if the product matches T exactly.
        
        # Check half-integers
        def is_half_integer(x):
            # x should be close to k/2
            return np.allclose(x * 2, np.round(x * 2), atol=1e-9)
        
        if not (is_half_integer(U) and is_half_integer(V) and is_half_integer(W)):
            return False
            
        # Exact check using integers (multiply by 2)
        Ui = (U * 2).astype(np.int64)
        Vi = (V * 2).astype(np.int64)
        Wi = (W * 2).astype(np.int64)
        
        # T_reconstructed[a,b,c] = sum_k Ui[a,k] Vi[b,k] Wi[c,k] / 8
        # T is 0 or 1.
        # So we need sum_k Ui[a,k] Vi[b,k] Wi[c,k] == 8 * T[a,b,c]
        
        # This is expensive to compute for all a,b,c if we do it naively.
        # But dimensions are small: 10, 25, 10.
        # r is ~50.
        # 10*25*10*50 = 125,000 ops. Very fast.
        
        T_rec = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for k in range(Ui.shape[1]):
            # Outer product of vectors
            # Ui[:, k] shape (10,)
            # Vi[:, k] shape (25,)
            # Wi[:, k] shape (10,)
            # T_rec += Ui[:,k, None, None] * Vi[:, None, k, None] * Wi[:, None, None, k]
            # This is not right.
            # T[a,b,c] += U[a,k] V[b,k] W[c,k]
            T_rec += np.einsum('a,b,c->abc', Ui[:,k], Vi[:,k], Wi[:,k])
            
        # Check if T_rec == 8 * T
        # T is float 0/1. 8*T is int 0/8.
        if not np.allclose(T_rec, 8 * T, atol=0):
            return False
            
        return True

    best_result = schoolbook_result
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition
    # We will try ranks 49, 48, ...
    # Using random restarts and gradient descent
    
    for target_rank in range(schoolbook_rank - 1, 40, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(10):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + target_rank * 100 + trial)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Gradient Descent
            # Loss L = || T - sum_k U_k V_k W_k ||^2
            # dL/dU = 2 * sum_k (T - T_rec) * (V_k W_k)
            
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                    
                # Compute T_rec
                T_rec = np.zeros_like(T)
                for k in range(target_rank):
                    T_rec += np.einsum('a,b,c->abc', U[:,k], V[:,k], W[:,k])
                
                diff = T - T_rec
                loss = np.sum(diff**2)
                
                if loss < 1e-10:
                    # Try to round and verify
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if verify_decomposition(U_rounded, V_rounded, W_rounded):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    # If rounding fails, continue optimizing or break?
                    # Maybe the solution is not half-integer?
                    # But the problem requires half-integer.
                    # If we found a real solution, a half-integer one might exist.
                    # Let's try to project to half-integers?
                    # For now, just break if loss is very small but rounding fails.
                    break
                
                # Gradients
                # dL/dU[:,k] = 2 * sum_{a,b,c} diff[a,b,c] * V[b,k] * W[c,k]
                # This is: 2 * einsum('abc,bk,ck->ak', diff, V, W)
                grad_U = 2 * np.einsum('abc,bk,ck->ak', diff, V, W)
                grad_V = 2 * np.einsum('abc,ak,ck->bk', diff, U, W)
                grad_W = 2 * np.einsum('abc,ak,bk->ck', diff, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Simple decay
                if step % 100 == 0:
                    lr *= 0.5
                    
            # After optimization, try rounding
            if loss < 1e-6:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if verify_decomposition(U_rounded, V_rounded, W_rounded):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
