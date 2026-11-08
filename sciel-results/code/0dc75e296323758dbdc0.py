"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _is_half_int(x):
    return abs(x * 2 - round(x * 2)) < 1e-9

def _round_to_half_int(x):
    return round(x * 2) / 2.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _construct_block_decomposition(n, m, p):
    """
    Construct decomposition using block structure.
    A is n×m, B is m×p. Split B into blocks of width 2.
    For each block B_block (m×2), compute C_block = A × B_block (n×2).
    
    For n=2, m=3: C_block[i,k] = sum_j A[i,j]*B_block[j,k]
    This is a 2×3×2 tensor with schoolbook rank 12.
    
    We try to find a lower-rank decomposition for 2×3×2, then combine.
    """
    U_dim = n * m  # 6
    V_dim = m * p  # 18
    W_dim = p * n  # 12
    
    # For 2×3×2 tensor: C[i,k] = sum_j A[i,j]*B[j,k], i in {0,1}, k in {0,1}, j in {0,1,2}
    # Schoolbook: 12 terms. Let's see if we can do better.
    # 
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    #
    # Note: C[0,0]*C[1,1] - C[0,1]*C[1,0] = det(A)*det(B) ... not directly helpful.
    #
    # Let's try rank 10 for 2×3×2:
    # Use the identity: for 2×2×2, rank 7 (Strassen). 
    # For 2×3×2, we can view it as: 
    #   C = [A0 A1 A2] * B where B is 3×2
    #   = A0*B[0,:] + A1*B[1,:] + A2*B[2,:]
    #   where A0,A1,A2 are 2×1 column vectors of A.
    #
    # Actually, let me just try to optimize directly for the full 2×3×6 problem.
    return None

def _optimize_decomposition(n, m, p, target_rank, deadline, max_iterations=5000):
    """
    Use gradient-based optimization to find a rank decomposition.
    We minimize the error of the tensor decomposition.
    """
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
            
        np.random.seed(attempt * 12345 + 7)
        
        # Initialize with small random values
        U = np.random.randn(U_dim, target_rank) * 0.5
        V = np.random.randn(V_dim, target_rank) * 0.5
        W = np.random.randn(W_dim, target_rank) * 0.5
        
        lr = 0.01
        for it in range(max_iterations):
            if time.time() > deadline - 30:
                break
            
            # Compute error: for each (i,ip,j,jp,k,kp), 
            # error = sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] - delta
            # This is expensive. Use a sampled approach.
            
            # Compute the tensor product for random samples
            errors = []
            # Sample a subset of index combinations
            num_samples = 200
            for _ in range(num_samples):
                i = np.random.randint(n)
                ip = np.random.randint(n)
                j = np.random.randint(m)
                jp = np.random.randint(m)
                k = np.random.randint(p)
                kp = np.random.randint(p)
                
                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                
                # Compute sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c]
                prod = U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :]
                s = np.sum(prod)
                errors.append(s - expected)
            
            err = np.mean(np.array(errors) ** 2)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-8:
                break
            
            # Simple gradient step (approximate)
            # For each c, the contribution to the error gradient is complex.
            # Use a simpler approach: perturb and accept if better.
            if it % 10 == 0:
                # Try a small perturbation
                dU = np.random.randn(U_dim, target_rank) * lr
                dV = np.random.randn(V_dim, target_rank) * lr
                dW = np.random.randn(W_dim, target_rank) * lr
                
                U_new = U + dU
                V_new = V + dV
                W_new = W + dW
                
                # Quick error check
                num_check = 50
                errs = []
                for _ in range(num_check):
                    i = np.random.randint(n)
                    ip = np.random.randint(n)
                    j = np.random.randint(m)
                    jp = np.random.randint(m)
                    k = np.random.randint(p)
                    kp = np.random.randint(p)
                    expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                    prod = U_new[i*m+j, :] * V_new[jp*p+k, :] * W_new[kp*n+ip, :]
                    s = np.sum(prod)
                    errs.append((s - expected) ** 2)
                new_err = np.mean(errs)
                
                if new_err < err:
                    U, V, W = U_new, V_new, W_new
                    err = new_err
                else:
                    lr *= 0.9
    
    if best_err < 1e-6:
        # Round to half-integers and verify
        U_r = np.array([[_round_to_half_int(x) for x in row] for row in best_U])
        V_r = np.array([[_round_to_half_int(x) for x in row] for row in best_V])
        W_r = np.array([[_round_to_half_int(x) for x in row] for row in best_W])
        
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, target_rank):
            return U_r, V_r, W_r
    
    return None

def _try_structured_search(n, m, p, best_rank, deadline):
    """Try structured approaches based on block decomposition."""
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Try optimization for ranks from low to high
    for rank in range(12, min(best_rank, 25)):
        if time.time() > deadline - 40:
            break
        
        result = _optimize_decomposition(n, m, p, rank, deadline, max_iterations=2000)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try to find a lower rank
    found = _try_structured_search(n, m, p, n * m * p, deadline)
    if found is not None:
        return found
    
    return result
# EVOLVE-BLOCK-END
