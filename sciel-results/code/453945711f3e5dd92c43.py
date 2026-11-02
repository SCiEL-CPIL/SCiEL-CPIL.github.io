import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds
    
    # Schoolbook solution
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Pre-compute the constraint structure
    # For each output (i, k), we need sum over j of A[i,j] * B[j,k] = C[i,k]
    # The tensor T[i,j,k] = 1 if this is a valid multiplication term
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Build the linear system matrices for each variable
    # For W: for each (i,k) pair and each j, we have equation involving W[k*n+i, r]
    # Actually, let's think more carefully.
    
    # The constraint is: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
    # where delta is 1 if (i,j,k) is a valid term in matrix multiplication
    
    # Wait, for matrix multiplication C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) triples
    # Actually T[i,j,k] = 1 always (it's the structure tensor of matrix multiplication)
    
    # The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For fixed U and V, this is linear in W:
    # For each (i,k): sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_j T[i,j,k] = m (if all T=1)
    # Wait no, T[i,j,k] = 1 for all i,j,k in matrix multiplication
    
    # Actually for matrix multiplication, the structure tensor is:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So for each (i,k), the constraint is: sum_j T[i,j,k] * (product) = 1? No...
    
    # Let me re-read the task. The output factor follows W[k*n+i, r]
    # So the tensor is T[i,j,k] where the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And this should equal the structure tensor of matrix multiplication
    
    # For matrix multiplication of (n x m) * (m x p) = (n x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # So the structure tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a bilinear problem. For fixed U and V, it's linear in W.
    # For fixed U and W, it's linear in V.
    # For fixed V and W, it's linear in U.
    
    # Let's pre-compute the coefficient matrices
    
    # Number of equations: n * m * p
    # Number of unknowns for W: p * n * target_rank
    
    # For W: equation (i,j,k) involves W[k*n+i, r] with coefficient U[i*m+j, r] * V[j*p+k, r]
    # So A_w[(i,j,k), (k,n,i)*target_rank + r] = U[i*m+j, r] * V[j*p+k, r]
    
    # Similarly for U and V.
    
    # Let's pre-compute the index mappings
    
    # For each equation index eq = i * (m*p) + j * p + k
    # For W: row eq, col = (k*n+i) * target_rank + r, coeff = U[i*m+j, r] * V[j*p+k, r]
    # For U: row eq, col = (i*m+j) * target_rank + r, coeff = V[j*p+k, r] * W[k*n+i, r]
    # For V: row eq, col = (j*p+k) * target_rank + r, coeff = U[i*m+j, r] * W[k*n+i, r]
    
    # The RHS b is always 1 for all equations (since T[i,j,k] = 1)
    
    # Pre-compute the sparse structure of these matrices
    # We'll use dense matrices since the sizes are manageable
    
    def build_matrices(target_rank):
        """Build the coefficient matrix templates for the linear systems."""
        n_eq = n * m * p
        n_unk_w = p * n * target_rank
        n_unk_u = n * m * target_rank
        n_unk_v = m * p * target_rank
        
        # For W: A_w[eq, col] where eq = i*m*p + j*p + k, col = (k*n+i)*target_rank + r
        # Coefficient depends on U and V, so we can't fully pre-compute
        # But we can pre-compute the index structure
        
        # Actually, the coefficients depend on the current U, V, W values,
        # so we need to build the matrices each iteration. But we can vectorize it.
        
        return None
    
    # Let's use a different approach: vectorized construction of the linear systems
    
    def solve_for_W(U, V, target_rank):
        """Solve the linear system for W given U and V."""
        n_eq = n * m * p
        n_unk = p * n * target_rank
        
        A = np.zeros((n_eq, n_unk))
        b = np.ones(n_eq)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for r in range(target_rank):
                        col = (k * n + i) * target_rank + r
                        A[eq, col] = U[i*m+j, r] * V[j*p+k, r]
        
        w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return w_sol.reshape(p*n, target_rank)
    
    def solve_for_U(V, W, target_rank):
        """Solve the linear system for U given V and W."""
        n_eq = n * m * p
        n_unk = n * m * target_rank
        
        A = np.zeros((n_eq, n_unk))
        b = np.ones(n_eq)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for r in range(target_rank):
                        col = (i*m+j) * target_rank + r
                        A[eq, col] = V[j*p+k, r] * W[k*n+i, r]
        
        u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return u_sol.reshape(n*m, target_rank)
    
    def solve_for_V(U, W, target_rank):
        """Solve the linear system for V given U and W."""
        n_eq = n * m * p
        n_unk = m * p * target_rank
        
        A = np.zeros((n_eq, n_unk))
        b = np.ones(n_eq)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for r in range(target_rank):
                        col = (j*p+k) * target_rank + r
                        A[eq, col] = U[i*m+j, r] * W[k*n+i, r]
        
        v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return v_sol.reshape(m*p, target_rank)
    
    def check_error(U, V, W, target_rank):
        """Check the decomposition error."""
        n_eq = n * m * p
        n_unk = p * n * target_rank
        
        A = np.zeros((n_eq, n_unk))
        b = np.ones(n_eq)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for r in range(target_rank):
                        col = (k * n + i) * target_rank + r
                        A[eq, col] = U[i*m+j, r] * V[j*p+k, r]
        
        error = np.linalg.norm(A @ W.flatten() - b)
        return error
    
    # Try to find lower rank solutions using alternating minimization
    for target_rank in range(53, best_rank):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(seed + target_rank * 100)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Alternating minimization
            for iteration in range(50):
                if time.time() > deadline - 30:
                    break
                
                # Solve for W (linear in W)
                W = solve_for_W(U, V, target_rank)
                
                # Solve for U (linear in U)
                U = solve_for_U(V, W, target_rank)
                
                # Solve for V (linear in V)
                V = solve_for_V(U, W, target_rank)
                
                # Check error
                error = check_error(U, V, W, target_rank)
                
                if error < 1e-10:
                    # Try rounding
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    error2 = check_error(U_r, V_r, W_r, target_rank)
                    
                    if error2 < 1e-10:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
