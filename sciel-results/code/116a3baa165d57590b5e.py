import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Precompute flattened views of T
    T_flat_a = T.reshape(n*m, -1)  # (n*m, m*p*p*n)
    T_flat_b = T.transpose(1,0,2).reshape(m*p, -1)  # (m*p, n*m*p*n)
    T_flat_c = T.transpose(2,0,1).reshape(p*n, -1)  # (p*n, n*m*m*p)
    
    # Schoolbook as fallback
    def schoolbook():
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n*m*p
    
    def check_half_integer(U, V, W):
        """Check if U, V, W are half-integers and exactly decompose T."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if np.allclose(U_r, U, atol=1e-10) and np.allclose(V_r, V, atol=1e-10) and np.allclose(W_r, W, atol=1e-10):
            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_check, T, atol=1e-10):
                return True, U_r, V_r, W_r
        return False, None, None, None
    
    def als_iteration(U, V, W, R, max_iter=50):
        """Perform ALS iterations with periodic rounding."""
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return U, V, W
            
            # Solve for U
            M = V[:, :, None] * W[:, None, :]  # (R, m*p, p*n)
            M_flat = M.reshape(R, -1)
            try:
                U_new = T_flat_a @ np.linalg.pinv(M_flat.T)
            except:
                return U, V, W
            U = U_new
            
            # Solve for V
            M = U[:, :, None] * W[:, None, :]  # (R, n*m, p*n)
            M_flat = M.reshape(R, -1)
            try:
                V_new = T_flat_b @ np.linalg.pinv(M_flat.T)
            except:
                return U, V, W
            V = V_new
            
            # Solve for W
            M = U[:, :, None] * V[:, None, :]  # (R, n*m, m*p)
            M_flat = M.reshape(R, -1)
            try:
                W_new = T_flat_c @ np.linalg.pinv(M_flat.T)
            except:
                return U, V, W
            W = W_new
            
            # Periodically round to half-integers to guide convergence
            if it % 10 == 9:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
        
        return U, V, W
    
    # Search for lower rank decompositions
    # Try ranks from a lower bound upward
    # Known lower bound for 3x3 x 3x8 is at least 27 (from n*m*p / something), but let's be more aggressive
    for R in range(20, n*m*p):
        if time.time() > deadline - 10:
            break
        
        # Number of trials depends on remaining time
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 1:
            break
        
        # Estimate time per trial
        time_per_trial = 2.0  # Conservative estimate
        num_trials = max(1, int(remaining_time / time_per_trial))
        num_trials = min(num_trials, 20)
        
        found = False
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Initialize with structure-aware randomization
            # Use a mix of random and structured initialization
            if trial % 3 == 0:
                # Pure random
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
            elif trial % 3 == 1:
                # Initialize with some structure from schoolbook
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
                # Add some schoolbook-like structure
                for r in range(min(R, 10)):
                    i = r % n
                    j = r % m
                    k = r % p
                    U[i*m+j, r] += 1.0
                    V[j*p+k, r] += 1.0
                    W[k*n+i, r] += 1.0
            else:
                # Sparse initialization
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for r in range(R):
                    a = np.random.randint(0, n*m)
                    b = np.random.randint(0, m*p)
                    c = np.random.randint(0, p*n)
                    U[a, r] = np.random.choice([-1, 1])
                    V[b, r] = np.random.choice([-1, 1])
                    W[c, r] = np.random.choice([-1, 1])
            
            # Run ALS
            U, V, W = als_iteration(U, V, W, R, max_iter=80)
            
            # Check if we found an exact half-integer solution
            is_exact, U_r, V_r, W_r = check_half_integer(U, V, W)
            if is_exact:
                best_rank = R
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
