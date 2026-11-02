import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
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
    
    # Try structured decompositions based on fast matrix multiplication
    # For 3x3 matrices, we can use a rank-5 decomposition (Strassen-like)
    # For 3x3 x 3x8, we can combine this with the 8 columns
    
    # Strategy: Use the fact that we're multiplying A(3x3) * B(3x8)
    # We can think of B as 8 separate 3x1 vectors, or group them
    
    # Try a decomposition that exploits the structure
    # For each of the 8 columns of B, we multiply A(3x3) * b_j(3x1)
    # A rank-3 decomposition per column would give rank 24, but we can share structure
    
    # Let's try to find a decomposition with rank < 72 using structured search
    
    # Precompute for efficient checking
    T_flat_a = T.reshape(n*m, -1)
    T_flat_b = T.transpose(1,0,2).reshape(m*p, -1)
    T_flat_c = T.transpose(2,0,1).reshape(p*n, -1)
    
    def fast_als(U, V, W, R, max_iter=30):
        """Fast ALS using least squares instead of pinv."""
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Solve for U given V, W
            # T[a, :, :] = sum_r U[a,r] * V[:,r] * W[:,r]
            # For each a: T[a] = sum_r U[a,r] * (V * W)_r
            VW = V * W  # (m*p, R) * (p*n, R) -> need outer product per r
            # M[r, b, c] = V[b,r] * W[c,r]
            M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, R)
            M_flat = M.transpose(2, 0, 1).reshape(R, -1)  # (R, m*p*p*n)
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_flat.T, T_flat_a.T, rcond=None)
                U = U_new.T
            except:
                break
            
            # Solve for V given U, W
            M = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, R)
            M_flat = M.transpose(2, 0, 1).reshape(R, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_flat.T, T_flat_b.T, rcond=None)
                V = V_new.T
            except:
                break
            
            # Solve for W given U, V
            M = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, R)
            M_flat = M.transpose(2, 0, 1).reshape(R, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_flat.T, T_flat_c.T, rcond=None)
                W = W_new.T
            except:
                break
            
            # Round to half-integers periodically
            if it % 5 == 4:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
        
        return U, V, W
    
    # Try structured initializations
    # Key insight: For matrix multiplication, the tensor has a specific algebraic structure
    # We can try initializations that respect this structure
    
    # Try ranks from a reasonable lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But we need at least enough to capture all 72 nonzero entries
    
    # Try a few structured approaches
    for R in [24, 27, 30, 36, 40, 48, 54, 60, 64, 68]:
        if time.time() > deadline - 15:
            break
        
        remaining_time = deadline - 15 - time.time()
        if remaining_time < 2:
            break
        
        # Number of trials based on time
        num_trials = max(1, min(5, int(remaining_time / 3)))
        
        found = False
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            # Structured initialization
            if trial == 0:
                # Initialize with schoolbook-like pattern but compressed
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for r in range(R):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
            elif trial == 1:
                # Random with some structure
                U = np.random.randn(n*m, R) * 0.5
                V = np.random.randn(m*p, R) * 0.5
                W = np.random.randn(p*n, R) * 0.5
                # Add some deterministic structure
                for r in range(min(R, 9)):
                    i = r % n
                    j = (r // n) % m
                    U[i*m+j, r] += 1.0
                    V[j*p + (r % p), r] += 1.0
                    W[(r % p)*n + i, r] += 1.0
            else:
                # Sparse random
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for r in range(R):
                    a = np.random.randint(0, n*m)
                    b = np.random.randint(0, m*p)
                    c = np.random.randint(0, p*n)
                    U[a, r] = np.random.choice([-1.0, 1.0])
                    V[b, r] = np.random.choice([-1.0, 1.0])
                    W[c, r] = np.random.choice([-1.0, 1.0])
            
            U, V, W = fast_als(U, V, W, R, max_iter=20)
            
            is_exact, U_r, V_r, W_r = check_half_integer(U, V, W)
            if is_exact:
                best_rank = R
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        
        if found:
            break
    
    # If we found a better rank, verify and return
    if best_rank < n*m*p:
        is_exact, U_r, V_r, W_r = check_half_integer(best_U, best_V, best_W)
        if is_exact:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
