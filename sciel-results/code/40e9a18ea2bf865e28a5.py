"""Matrix multiplication 2x3x5 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def verify_decomposition(U, V, W, n=2, m=3, p=5, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+l,c] = delta_{i,l}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(n):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+l, c] for c in range(rank))
                    expected = 1.0 if i == l else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    best_rank = n * m * p  # schoolbook = 30
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook fallback
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # For each target rank, try to find a decomposition
    # Strategy: optimize U,V, solve for W via least squares, round, verify
    for target_rank in range(15, min(best_rank, 30)):
        if time.time() > deadline - 5:
            break
        
        found = False
        # More restarts for lower ranks
        n_restarts = 5 if target_rank <= 18 else 3
        
        for restart in range(n_restarts):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(42 + target_rank * 100 + restart * 7)
            
            # Initialize U and V randomly
            U_mat = np.random.randn(n*m, target_rank) * 0.5
            V_mat = np.random.randn(m*p, target_rank) * 0.5
            
            # Optimize U, V to minimize error after solving for W
            # Error for fixed (i,k): sum_j [sum_c U[i*m+j,c] * V[j*p+k,c]] * W[k*n+i,c] should equal delta_{i,?}
            # Wait, the constraint is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+l,c] = delta_{i,l}
            # For fixed (j,k,l): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+l,c] = delta_{i,l}
            # This is linear in W for fixed U,V.
            # For each (j,k,l), we have a constraint on W[k*n+l,:]
            # Let's reformulate: for fixed (i,l), we need:
            # sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+l,c] = delta_{i,l} for all k
            # Define A_{i,j,k,c} = U[i*m+j,c] * V[j*p+k,c]
            # Then sum_c A_{i,j,k,c} * W[k*n+l,c] = delta_{i,l} / m ... no
            # Actually: for each (i,k,l): sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+l,c] = delta_{i,l}
            # Let B_{i,k,j,c} = U[i*m+j,c] * V[j*p+k,c]
            # Then: sum_j sum_c B_{i,k,j,c} * W[k*n+l,c] = delta_{i,l}
            # This is: (sum_j B_{i,k,j,:}) dot W[k*n+l,:] = delta_{i,l}
            # Define S_{i,k,c} = sum_j U[i*m+j,c] * V[j*p+k,c]
            # Then: S_{i,k} dot W[k*n+l] = delta_{i,l}
            
            # So for each k, we have a system:
            # S[:, k, :] @ W[k*n + l, :] = delta_{i,l} for i=0,1
            # S is (n, p, rank) = (2, 5, rank)
            # W[k*n + l, :] is a vector of length rank
            # For fixed k: S[:, k, :] is (2, rank), and we need:
            #   S[0,k,:] dot W[k*n+0,:] = 1
            #   S[1,k,:] dot W[k*n+1,:] = 1
            #   S[0,k,:] dot W[k*n+1,:] = 0
            #   S[1,k,:] dot W[k*n+0,:] = 0
            # This is a 4 x rank system for each k!
            
            def compute_S(U_mat, V_mat):
                """Compute S[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]"""
                S = np.zeros((n, p, target_rank))
                for i in range(n):
                    for k in range(p):
                        for c in range(target_rank):
                            S[i, k, c] = sum(U_mat[i*m+j, c] * V_mat[j*p+k, c] for j in range(m))
                return S
            
            def solve_for_W(S):
                """Solve for W given S. For each k, solve 4 x rank system."""
                W_mat = np.zeros((p*n, target_rank))
                for k in range(p):
                    # S[:, k, :] is (2, rank)
                    # Need: S[0,k,:] dot W[k*n+0,:] = 1
                    #       S[1,k,:] dot W[k*n+1,:] = 1  
                    #       S[0,k,:] dot W[k*n+1,:] = 0
                    #       S[1,k,:] dot W[k*n+0,:] = 0
                    # Build 4 x rank matrix A and 4-vector b
                    A = np.zeros((4, target_rank))
                    b = np.zeros(4)
                    A[0, :] = S[0, k, :]
                    A[1, :] = S[1, k, :]
                    A[2, :] = S[0, k, :]
                    A[3, :] = S[1, k, :]
                    b[0] = 1.0  # S[0,k] dot W[k*n+0] = 1
                    b[1] = 1.0  # S[1,k] dot W[k*n+1] = 1
                    b[2] = 0.0  # S[0,k] dot W[k*n+1] = 0
                    b[3] = 0.0  # S[1,k] dot W[k*n+0] = 0
                    
                    # Solve A @ [W[k*n+0,:], W[k*n+1,:]]^T = b
                    # Actually this is two separate equations:
                    # S[0,k,:] dot W[k*n+0,:] = 1
                    # S[1,k,:] dot W[k*n+0,:] = 0
                    # S[0,k,:] dot W[k*n+1,:] = 0
                    # S[1,k,:] dot W[k*n+1,:] = 1
                    
                    # For W[k*n+0,:]: solve [S[0,k,:]; S[1,k,:]] @ W[k*n+0,:] = [1; 0]
                    # For W[k*n+1,:]: solve [S[0,k,:]; S[1,k,:]] @ W[k*n+1,:] = [0; 1]
                    
                    S_ik = S[:, k, :]  # (2, rank)
                    try:
                        if target_rank >= 2:
                            W0, _, _, _ = np.linalg.lstsq(S_ik, np.array([1.0, 0.0]), rcond=None)
                            W1, _, _, _ = np.linalg.lstsq(S_ik, np.array([0.0, 1.0]), rcond=None)
                        else:
                            # rank < 2, may not have solution
                            W0 = np.zeros(target_rank)
                            W1 = np.zeros(target_rank)
                        W_mat[k*n+0, :] = W0
                        W_mat[k*n+1, :] = W1
                    except:
                        W_mat[k*n+0, :] = 0
                        W_mat[k*n+1, :] = 0
                return W_mat
            
            def compute_error(U_mat, V_mat):
                """Compute total error after solving for W."""
                S = compute_S(U_mat, V_mat)
                W_mat = solve_for_W(S)
                
                # Verify the full tensor equation
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for l in range(n):
                                val = sum(U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+l, c] for c in range(target_rank))
                                expected = 1.0 if i == l else 0.0
                                error += (val - expected) ** 2
                return error, W_mat
            
            # Gradient descent on U, V
            lr = 0.05
            U_cur = U_mat.copy()
            V_cur = V_mat.copy()
            
            max_iter = 100
            prev_error = float('inf')
            for it in range(max_iter):
                if time.time() > deadline - 2:
                    break
                
                error, W_cur = compute_error(U_cur, V_cur)
                
                if error < 1e-12:
                    break
                
                # Check for convergence
                if abs(prev_error - error) < 1e-15:
                    break
                prev_error = error
                
                # Compute gradients numerically (finite differences) for robustness
                # Or use analytic gradients
                # Let's use analytic: error = sum_{i,j,k,l} (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+l,c] - delta_{i,l})^2
                # This is complex. Let's use a simpler approach: just do a few steps of gradient descent
                # with finite differences for the key variables
                
                # Actually, let's just do a simple random restart approach with better initialization
                # and more iterations
                
                # Perturb U and V slightly
                if it % 10 == 9:
                    U_cur += np.random.randn(n*m, target_rank) * 0.01
                    V_cur += np.random.randn(m*p, target_rank) * 0.01
                
                # Simple gradient step using finite differences on a few elements
                # This is slow but robust
                grad_U = np.zeros_like(U_cur)
                grad_V = np.zeros_like(V_cur)
                eps = 1e-5
                
                # Only compute gradients for a subset of elements to save time
                if it % 5 == 0:
                    # Sample some elements
                    indices_U = np.random.choice(n*m * target_rank, min(20, n*m*target_rank), replace=False)
                    for idx in indices_U:
                        row = idx // target_rank
                        col = idx % target_rank
                        U_plus = U_cur.copy()
                        U_plus[row, col] += eps
                        err_plus, _ = compute_error(U_plus, V_cur)
                        grad_U[row, col] = (err_plus - error) / eps
                    
                    indices_V = np.random.choice(m*p * target_rank, min(20, m*p*target_rank), replace=False)
                    for idx in indices_V:
                        row = idx // target_rank
                        col = idx % target_rank
                        V_plus = V_cur.copy()
                        V_plus[row, col] += eps
                        err_plus, _ = compute_error(U_cur, V_plus)
                        grad_V[row, col] = (err_plus - error) / eps
                
                U_cur -= lr * grad_U
                V_cur -= lr * grad_V
                
                # Decay learning rate
                if it % 20 == 19:
                    lr *= 0.7
            
            # Final check
            error, W_final = compute_error(U_cur, V_cur)
            
            if error < 1e-8:
                # Round to half-integers
                U_rounded = np.round(U_cur * 2) / 2
                V_rounded = np.round(V_cur * 2) / 2
                W_rounded = np.round(W_final * 2) / 2
                
                # Verify after rounding
                if verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                    found = True
                    break
            
            # Also try without rounding in case the exact solution works
            if error < 1e-10:
                if verify_decomposition(U_cur.tolist(), V_cur.tolist(), W_final.tolist()):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_final.copy()
                    found = True
                    break
        
        if found:
            break
    
    # Return the best solution found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
