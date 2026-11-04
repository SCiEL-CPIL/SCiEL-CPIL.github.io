import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
    n, m, p = 4, 4, 5
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Constraint: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # The tensor T has T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,5)
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 80
    
    # First, let's get the schoolbook solution
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    # Now try to find lower rank decompositions
    # Use the Strassen decomposition for 4x4x4 and extend to 4x4x5
    
    def check_decomposition_fast(U, V, W):
        """Faster check using numpy."""
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # Create the tensor and check
        # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Reshape and use einsum
        # U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # We need: for each c, the contribution is u_c ⊗ v_c ⊗ w_c
        # where u_c[i*m+j] = U[i*m+j,c], v_c[j*p+k] = V[j*p+k,c], w_c[k*n+i] = W[k*n+i,c]
        
        # Let's compute the full tensor
        # T = sum_c np.outer(np.outer(u_c, v_c), w_c) reshaped appropriately
        # But this is expensive. Let's use einsum.
        
        # Actually, let's just do it directly but vectorized
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    errors.append(s - 1.0)
        return max(abs(e) for e in errors) < 1e-6
    
    # Build Strassen-based 4x4x4 decomposition
    def build_strassen_4x4x4():
        """
        Build a rank-49 decomposition for 4x4x4 matrix multiplication.
        This uses the known Bini et al. decomposition.
        """
        # Strassen's 2x2x2 decomposition has rank 7
        # We can tile this to get 4x4x4 with rank 7*4 = 28? No, that's not right.
        # Actually, the optimal rank for 4x4x4 is 49 (Bini et al.)
        
        # Let me use a simpler approach: start with schoolbook and try to reduce
        # For now, let's use a structured approach
        
        # Actually, let me just use the schoolbook and try linear combination removal
        return None
    
    # Try to reduce rank by finding linear dependencies
    def try_reduce_rank(U, V, W, target_rank, time_budget):
        """
        Try to reduce the rank by finding linear combinations of terms
        that can be eliminated.
        """
        r = U.shape[1]
        if r <= target_rank:
            return U, V, W, r
        
        # Try to find which terms are linearly dependent
        # This is a hard problem, so let's use a heuristic approach
        
        # For each pair of terms, check if one can be expressed as a linear
        # combination of others
        
        # Actually, let's use a simpler heuristic: try random linear combinations
        # and see if we can maintain the decomposition with fewer terms
        
        np.random.seed(42)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_r = r
        
        # Try to remove terms one at a time
        for col_idx in range(r - 1, target_rank - 1, -1):
            if time.time() > time_budget:
                break
            
            # Try to absorb this term into others
            # This is complex, so let's just skip for now
            pass
        
        return best_U, best_V, best_W, best_r
    
    # Try random search for lower rank
    np.random.seed(42)
    
    for target_rank in [79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            r = target_rank
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Use Adam-like optimization with better initialization
            lr = 0.05
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            t = 0
            
            for step in range(500):
                if time.time() > deadline - 30:
                    break
                
                t += 1
                # Compute gradients
                loss = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = s - 1.0
                            loss += err * err
                            # d/dU[i*m+j,c] = 2*err * V[j*p+k,c]*W[k*n+i,c]
                            grad_U[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                            grad_V[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                            grad_W[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
                
                # Adam update
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U**2
                m_hat_U = m_U / (1 - beta1**t)
                v_hat_U = v_U / (1 - beta2**t)
                U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V**2
                m_hat_V = m_V / (1 - beta1**t)
                v_hat_V = v_V / (1 - beta2**t)
                V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W**2
                m_hat_W = m_W / (1 - beta1**t)
                v_hat_W = v_W / (1 - beta2**t)
                W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
                
                if loss < 1e-10:
                    break
            
            if loss < 1e-8:
                # Check if entries are close to integer/half-integer
                # For now, accept if it's a valid decomposition
                if check_decomposition_fast(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists with rounding to nearest half-integer if close
    def round_half(x):
        return np.round(x * 2) / 2
    
    if best_U is not None:
        U_out = round_half(best_U)
        V_out = round_half(best_V)
        W_out = round_half(best_W)
        
        # Verify the rounded version
        if check_decomposition_fast(U_out, V_out, W_out):
            best_U, best_V, best_W = U_out, V_out, W_out
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
