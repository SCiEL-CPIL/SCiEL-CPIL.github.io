"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i, j, k] should be 1 for all i, j, k
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _solve_tensor_decomposition(n, m, p, rank, time_limit):
    """
    Attempt to find a rank-r decomposition of the n x m x p matrix multiplication tensor.
    Uses an alternating least squares approach to solve for U, V, W.
    The equation is: sum_c U[:, c] outer V[:, c] outer W[:, c] = T
    where T is the tensor with 1s everywhere.
    
    We can view this as: For a fixed c, we add a rank-1 tensor.
    Actually, it's easier to solve for one matrix at a time given the others.
    Let's flatten the equation.
    T_{ijk} = sum_c U_{ij, c} V_{jk, c} W_{ki, c}
    
    This is a 3-way tensor decomposition.
    We can linearize it.
    Let's try to solve for W given U and V.
    For each (i, k), we have:
    1 = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    This is linear in W.
    Let A_{(i,k), c} = sum_j U[i*m+j, c] V[j*p+k, c]
    Then 1 = sum_c A_{(i,k), c} W[k*n+i, c]
    
    This is a system of n*p equations for n*p*rank unknowns? No.
    W has shape (p*n, rank).
    The index of W is (k*n+i, c).
    So for each pair (i, k), we have one equation:
    sum_c A_{(i,k), c} W[k*n+i, c] = 1
    
    This is a system of n*p linear equations with n*p*rank unknowns.
    It is underdetermined. We can find a minimum norm solution or just a solution.
    
    However, U and V are also unknowns.
    We can use ALS:
    1. Initialize U, V, W randomly.
    2. Solve for W given U, V.
    3. Solve for V given U, W.
    4. Solve for U given V, W.
    5. Repeat until convergence or time limit.
    """
    
    start_time = time.time()
    
    # Initialize
    # A good initialization might be structured, but random is a safe start for search.
    # To help convergence, we can initialize with a known valid decomposition (schoolbook) 
    # and then try to compress it? No, that's hard.
    # Let's try random initialization with small values.
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Precompute indices for efficiency
    # Equation for W:
    # For each i in range(n), k in range(p):
    #   idx_w = k * n + i
    #   sum_c ( sum_j U[i*m+j, c] * V[j*p+k, c] ) * W[idx_w, c] = 1
    
    # Let's build the matrix A_w of shape (n*p, rank)
    # A_w[idx_w, c] = sum_j U[i*m+j, c] V[j*p+k, c]
    
    def update_W(U, V):
        A_w = np.zeros((n * p, rank))
        for i in range(n):
            for k in range(p):
                idx_w = k * n + i
                for c in range(rank):
                    s = 0.0
                    for j in range(m):
                        s += U[i * m + j, c] * V[j * p + k, c]
                    A_w[idx_w, c] = s
        # Solve A_w @ W_col = 1 for each column? No, W is (p*n, rank).
        # The system is: A_w @ W.T = 1_vector ?
        # Let's check dimensions.
        # A_w is (n*p, rank). W is (p*n, rank).
        # We want: sum_c A_w[idx, c] * W[idx, c] = 1.
        # This is element-wise dot product of rows of A_w and W.
        # This is not a standard linear system like M x = b.
        # It is: A_w * W = 1 (element-wise)? No, sum over c.
        # This is equivalent to: (A_w * W).sum(axis=1) = 1? No.
        # Let x_c = W[:, c]. Then for each row idx, sum_c A_w[idx, c] x_c = 1.
        # This is a system of linear equations for the columns of W?
        # Actually, since the equations are decoupled by index (idx), 
        # for each idx, we have a linear equation in the vector W[idx, :].
        # sum_c A_w[idx, c] W[idx, c] = 1.
        # This is underdetermined. We can pick a solution.
        # To minimize norm, W[idx, :] = A_w[idx, :] / (A_w[idx, :] @ A_w[idx, :])
        # But we want to find *any* solution that allows subsequent steps to work.
        # Using minimum norm solution is a standard heuristic.
        
        W_new = np.zeros_like(W)
        for idx in range(n * p):
            a = A_w[idx, :]
            norm_sq = np.dot(a, a)
            if norm_sq > 1e-10:
                W_new[idx, :] = a / norm_sq
            else:
                # If A_w row is zero, we cannot satisfy the equation with this W row.
                # This implies the current U, V are "bad".
                # We might need to restart or adjust U, V.
                # For now, set to 0, which will fail verification.
                W_new[idx, :] = 0
        return W_new

    def update_V(U, W):
        # Equation for V:
        # For each j in range(m), k in range(p):
        #   idx_v = j * p + k
        #   sum_i sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        #   Let B_{(j,k), c} = sum_i U[i*m+j, c] * W[k*n+i, c]
        #   Then sum_c B_{(j,k), c} V[j*p+k, c] = 1
        A_v = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                idx_v = j * p + k
                for c in range(rank):
                    s = 0.0
                    for i in range(n):
                        s += U[i * m + j, c] * W[k * n + i, c]
                    A_v[idx_v, c] = s
        
        V_new = np.zeros_like(V)
        for idx in range(m * p):
            a = A_v[idx, :]
            norm_sq = np.dot(a, a)
            if norm_sq > 1e-10:
                V_new[idx, :] = a / norm_sq
            else:
                V_new[idx, :] = 0
        return V_new

    def update_U(V, W):
        # Equation for U:
        # For each i in range(n), j in range(m):
        #   idx_u = i * m + j
        #   sum_k sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        #   Let C_{(i,j), c} = sum_k V[j*p+k, c] * W[k*n+i, c]
        #   Then sum_c C_{(i,j), c} U[i*m+j, c] = 1
        A_u = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                for c in range(rank):
                    s = 0.0
                    for k in range(p):
                        s += V[j * p + k, c] * W[k * n + i, c]
                    A_u[idx_u, c] = s
        
        U_new = np.zeros_like(U)
        for idx in range(n * m):
            a = A_u[idx, :]
            norm_sq = np.dot(a, a)
            if norm_sq > 1e-10:
                U_new[idx, :] = a / norm_sq
            else:
                U_new[idx, :] = 0
        return U_new

    # ALS Loop
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # We need to be careful: the "minimum norm" solution for each step might drive the system to zero or instability.
    # A better approach for tensor decomposition is often to use a rank-1 update step or a more robust solver.
    # However, given the time constraints, let's try this simple ALS first.
    # To improve stability, we can add a regularization term or use a different update rule.
    # Actually, the problem is that we are solving for *exact* 1s. 
    # The system is linear. If a solution exists, ALS should converge to one if initialized well.
    # Random initialization might fail.
    
    # Alternative Strategy: 
    # Use the fact that we can start with a known valid decomposition (Schoolbook, rank n*m*p)
    # and then try to "compress" it by merging terms? No, that's not how ALS works.
    
    # Let's try a different initialization: 
    # Start with rank = n*m*p (Schoolbook) and see if we can reduce rank?
    # No, the task is to find *a* decomposition of minimal rank.
    # The prompt suggests using the structure 8=2*2*2.
    
    # Let's try a specific structured initialization based on Kronecker products.
    # But for now, let's stick to random init and multiple restarts if it fails.
    
    max_iters = 200
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        W = update_W(U, V)
        V = update_V(U, W)
        U = update_U(V, W)
        
        # Check convergence
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    error += (s - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if error < 1e-8:
            return best_U, best_V, best_W

    # If error is still high, this rank/initialization failed
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # We want to find the minimal rank.
    # Known lower bound is max(n*m, m*p, p*n) ? No, that's for square matrices roughly.
    # For 3x4x8, schoolbook is 96.
    # Strassen-like methods might reduce this.
    # Let's try ranks from a lower bound up to schoolbook.
    
    # A reasonable lower bound is max(n*m, m*p, p*n) is not correct.
    # The rank of the matrix multiplication tensor M_{n,m,p} is at least max(n*m, m*p, p*n) is false.
    # It is at least max(n*m, m*p, p*n) / something?
    # Actually, the rank is at least max(n*m, m*p, p*n) is a common misconception.
    # For square n, rank is at least n^2.
    # For rectangular, rank is at least max(n*m, m*p, p*n) is not a proven lower bound in general, 
    # but it is a good heuristic start.
    # Let's start from a bit lower than schoolbook.
    
    # Try ranks: 80, 85, 90, 95, 96
    # Since we have time, we can try a few random restarts for each rank.
    
    candidate_ranks = [80, 85, 90, 95, 96]
    
    # Sort by rank (lower is better)
    candidate_ranks.sort()
    
    for rank in candidate_ranks:
        if time.time() - start_time > deadline - 5:
            break
            
        # Try multiple random initializations for this rank
        found = False
        for attempt in range(5):
            if time.time() - start_time > deadline - 5:
                break
                
            U, V, W = _solve_tensor_decomposition(n, m, p, rank, 10) # 10s per attempt
            
            if U is not None:
                # Verify
                if _verify(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If we didn't find a solution for this rank, try next higher rank
        # Note: It's possible that a solution exists but ALS failed to find it.
        # We might need more sophisticated methods or more attempts.
        
    # Fallback to Schoolbook
    rank = schoolbook_rank
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Should not reach here
    raise Exception("Failed to construct valid decomposition")

# EVOLVE-BLOCK-END
