import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as guaranteed fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook.copy()
    best_V = V_schoolbook.copy()
    best_W = W_schoolbook.copy()
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def alternating_minimize(U, V, W, max_iter=100, tol=1e-10):
        """Perform alternating least-squares minimization."""
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Solve for W given U, V
            # For each (i,k): sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is linear in W[k*n+i,c]
            
            # Build system for W
            # Variables: W flattened (p*n*rank)
            # Equations: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # But W[k*n+i,c] appears in equations for all j
            
            # Let's use a different approach: for fixed (i,k), the equation is:
            # sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # Let A[i,k,c] = sum_j U[i*m+j,c]*V[j*p+k,c]
            # Then sum_c A[i,k,c]*W[k*n+i,c] = 1
            
            A_W = np.zeros((n * p, n * p * 1))  # This won't work directly
            
            # Better: treat each (i,k) pair independently for W
            # For fixed (i,k): sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
            # This gives n*p equations, each involving rank variables W[k*n+i,:]
            
            # Actually, W[k*n+i,c] is shared across different (i,k) with same (k,i)
            # Let me restructure: for each (k,i), W[k*n+i,:] is a vector of length rank
            # The constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            
            # Group by (k,i): for fixed (k,i), the constraints are over j:
            # For each j: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is m equations in rank unknowns W[k*n+i,:]
            
            # So we have n*p groups, each with m equations
            # Total: n*p*m = 72 equations, n*p*rank = 24*rank unknowns
            
            # Build the full system
            num_eqs = n * p * m  # 72
            num_vars_W = p * n * len(U[0])  # 24 * rank
            A_W = np.zeros((num_eqs, num_vars_W))
            b_W = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(len(U[0])):
                            A_W[eq_idx, (k * n + i) * len(U[0]) + c] = U[i * m + j, c] * V[j * p + k, c]
                        eq_idx += 1
            
            W_sol, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = W_sol.reshape(p * n, len(U[0]))
            
            # Solve for U given V, W
            num_vars_U = n * m * len(U[0])
            A_U = np.zeros((num_eqs, num_vars_U))
            b_U = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(len(U[0])):
                            A_U[eq_idx, (i * m + j) * len(U[0]) + c] = V[j * p + k, c] * W[k * n + i, c]
                        eq_idx += 1
            
            U_sol, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = U_sol.reshape(n * m, len(U[0]))
            
            # Solve for V given U, W
            num_vars_V = m * p * len(U[0])
            A_V = np.zeros((num_eqs, num_vars_V))
            b_V = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(len(U[0])):
                            A_V[eq_idx, (j * p + k) * len(U[0]) + c] = U[i * m + j, c] * W[k * n + i, c]
                        eq_idx += 1
            
            V_sol, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = V_sol.reshape(m * p, len(U[0]))
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        error += (val - 1) ** 2
            error = np.sqrt(error)
            
            if error < tol:
                break
        
        return U, V, W, error
    
    # Try target ranks from high to low
    # Focus on ranks that might be achievable
    target_ranks = [64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Try multiple initializations
        for init_type in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(init_type * 1000 + target_rank + 7)
            
            # Different initialization strategies
            if init_type == 0:
                # Random initialization
                U = np.random.randn(n * m, target_rank) * 0.1
                V = np.random.randn(m * p, target_rank) * 0.1
                W = np.random.randn(p * n, target_rank) * 0.1
            elif init_type == 1:
                # Block-structured: initialize based on schoolbook pattern
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                # Spread schoolbook pattern across target_rank
                for c in range(min(target_rank, n * m * p)):
                    i = (c // (m * p)) % n
                    j = (c // p) % m
                    k = c % p
                    U[i * m + j, c % target_rank] = 1.0 / np.sqrt(target_rank)
                    V[j * p + k, c % target_rank] = 1.0 / np.sqrt(target_rank)
                    W[k * n + i, c % target_rank] = 1.0 / np.sqrt(target_rank)
            elif init_type == 2:
                # Kronecker-like structure
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for c in range(target_rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    if c < n * m * p:
                        U[i * m + j, c] = 1.0
                        V[j * p + k, c] = 1.0
                        W[k * n + i, c] = 1.0
            elif init_type == 3:
                # Sparse random
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for c in range(target_rank):
                    for _ in range(3):
                        U[np.random.randint(n * m), c] = np.random.choice([-1, 1])
                        V[np.random.randint(m * p), c] = np.random.choice([-1, 1])
                        W[np.random.randint(p * n), c] = np.random.choice([-1, 1])
            else:
                # Mix of structured and random
                U = np.random.randn(n * m, target_rank) * 0.05
                V = np.random.randn(m * p, target_rank) * 0.05
                W = np.random.randn(p * n, target_rank) * 0.05
                # Add some structure
                for c in range(min(target_rank, 12)):
                    U[c % (n * m), c] += 1.0
                    V[c % (m * p), c] += 1.0
                    W[c % (p * n), c] += 1.0
            
            # Run alternating minimization
            U_new, V_new, W_new, error = alternating_minimize(U, V, W, max_iter=80)
            
            if error < 1e-8:
                # Try rounding to integers or half-integers
                U_r = np.round(U_new * 2) / 2
                V_r = np.round(V_new * 2) / 2
                W_r = np.round(W_new * 2) / 2
                
                if check_decomposition(U_r, V_r, W_r):
                    best_rank = target_rank
                    best_U = U_r.copy()
                    best_V = V_r.copy()
                    best_W = W_r.copy()
                    found = True
                    break
                else:
                    # Use the unrounded solution if it's valid
                    if check_decomposition(U_new, V_new, W_new):
                        best_rank = target_rank
                        best_U = U_new.copy()
                        best_V = V_new.copy()
                        best_W = W_new.copy()
                        found = True
                        break
        
        if found:
            break
    
    # Final verification
    if not check_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U = U_schoolbook
        best_V = V_schoolbook
        best_W = W_schoolbook
        best_rank = rank_schoolbook
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
