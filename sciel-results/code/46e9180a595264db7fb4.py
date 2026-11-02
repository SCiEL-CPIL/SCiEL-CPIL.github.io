import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 48
    
    # The target tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is the matrix multiplication tensor for C = AB where C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook decomposition (rank = n*m*p = 48)
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    def to_list(arr):
        return arr.tolist()
    
    # Check if a decomposition is exact
    def check_exact(U, V, W, tol=1e-9):
        # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using structured search
    # Strategy: Use the fact that we can combine terms
    # For matrix multiplication, known results suggest rank < n*m*p for small cases
    # For 3x4x4, let's try rank 47, 46, etc.
    
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit
    
    # Try random perturbations of the schoolbook decomposition
    # Remove one term and see if we can adjust others to compensate
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try rank 47: remove one column and solve for adjustments
    # This is a linear system problem
    # If we remove column r0, we need to adjust other columns to maintain the decomposition
    
    # Alternative: Use known algebraic identities for matrix multiplication
    # For 2x2x2, rank is 7 (vs 8 schoolbook)
    # For 3x3x3, rank is 23 (vs 27 schoolbook)
    # For 3x4x4, the rank is likely around 40-44
    
    # Let's try a random search with structured initialization
    # Start from schoolbook and randomly merge/combine terms
    
    for trial_rank in range(rank_school - 1, 35, -1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-trial_rank decomposition
        # Use random initialization and gradient descent on the constraint
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        
        U = np.zeros((n*m, trial_rank))
        V = np.zeros((m*p, trial_rank))
        W = np.zeros((p*n, trial_rank))
        
        # Initialize with random values
        np.random.seed(42)
        U[:, :] = np.random.randn(n*m, trial_rank) * 0.5
        V[:, :] = np.random.randn(m*p, trial_rank) * 0.5
        W[:, :] = np.random.randn(p*n, trial_rank) * 0.5
        
        # Use alternating least squares to solve the constraint
        # For fixed V, W: solve for U (linear system)
        # For fixed U, W: solve for V (linear system)
        # For fixed U, V: solve for W (linear system)
        
        max_iters = 200
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Solve for U: For each (i*m+j), we have constraints
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j,k
            # This is: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
            # For fixed i, j, k: sum_r U[i*m+j, r] * c_r = 1 where c_r = V[j*p+k,r]*W[k*n+i,r]
            # But U[i*m+j, r] is shared across different k for the same i,j
            
            # Actually, for fixed i, the constraint is:
            # For each j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a linear system in U[i*m+j, :] for each j
            # But the same U[i*m+j, :] appears in constraints for different k
            
            # Let's set up the linear system properly
            # Unknowns: U[i*m+j, r] for r in range(trial_rank)
            # Equations: for each k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # So for fixed i, j: trial_rank unknowns, p equations
            
            # If p < trial_rank, we have an underdetermined system
            # We can find a minimum-norm solution
            
            improved = False
            for i in range(n):
                for j in range(m):
                    # Build the linear system for U[i*m+j, :]
                    # A @ u = b where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, trial_rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    # Solve A @ u = b in least squares sense
                    try:
                        u_new, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        if np.linalg.norm(u_new - U[i*m+j, :]) > 1e-10:
                            U[i*m+j, :] = u_new
                            improved = True
                    except:
                        pass
            
            # Solve for V similarly
            for j in range(m):
                for k in range(p):
                    # Unknowns: V[j*p+k, r]
                    # Equations: for each i: sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1
                    A = np.zeros((n, trial_rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        v_new, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        if np.linalg.norm(v_new - V[j*p+k, :]) > 1e-10:
                            V[j*p+k, :] = v_new
                            improved = True
                    except:
                        pass
            
            # Solve for W similarly
            for k in range(p):
                for i in range(n):
                    # Unknowns: W[k*n+i, r]
                    # Equations: for each j: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                    A = np.zeros((m, trial_rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        w_new, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        if np.linalg.norm(w_new - W[k*n+i, :]) > 1e-10:
                            W[k*n+i, :] = w_new
                            improved = True
                    except:
                        pass
            
            if not improved:
                break
            
            # Check if we have an exact solution
            if check_exact(U, V, W, tol=1e-6):
                best_rank = trial_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if best_rank < rank_school:
            break
    
    # Return the best decomposition found
    if best_rank == rank_school:
        U_list = to_list(U_school)
        V_list = to_list(V_school)
        W_list = to_list(W_school)
    else:
        U_list = to_list(best_U)
        V_list = to_list(best_V)
        W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
