import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x3 matrix multiplied by 3x6 matrix.
    We treat this as a 3x3x6 tensor multiplication.
    
    Strategy:
    The direct sum of two 3x3x3 multiplications has rank 2*23 = 46 using Bini et al.
    However, we can potentially do better by exploiting the structure.
    
    For now, let's try to implement a rank-46 decomposition using two copies of Bini et al.
    But since I don't have the exact Bini coefficients, I'll first ensure the schoolbook works,
    then try to find a better decomposition.
    
    Actually, let me think about this differently. The task is to multiply A (3x3) by B (3x6).
    This is equivalent to multiplying A by B1 (3x3) and A by B2 (3x3), giving C1 (3x3) and C2 (3x3).
    So we need two 3x3x3 matrix multiplications.
    
    The minimal rank for 3x3x3 is 23 (Bini et al.). So the direct sum has rank 46.
    But maybe we can do better by sharing some terms?
    
    Let me try a different approach: use the fact that the 6 columns are independent.
    We can write B = [B1, B2] where B1, B2 are 3x3.
    Then C = A*B = [A*B1, A*B2].
    
    If we use the same decomposition for both, we get rank 46.
    But can we do better?
    
    Let me search for a better decomposition. I'll use a randomized approach to find a low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    # n=3, m=3, p=6
    
    # Schoolbook rank is n*m*p = 54
    # Two 3x3x3 multiplications: rank 46
    
    # Let's try to find a decomposition with rank < 46
    # We'll use a randomized approach
    
    deadline = time.time() + 120  # 120 second deadline
    
    # First, let's get the schoolbook decomposition as a baseline
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a better decomposition using random initialization and optimization
    # We'll try ranks from 45 down to 40
    
    for target_rank in range(45, 39, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_trials = 50 if target_rank > 42 else 100
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize to avoid numerical issues
            for c in range(target_rank):
                norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
                if norm > 1e-10:
                    U[:, c] /= np.sqrt(np.sum(U[:, c]**2))
                    V[:, c] /= np.sqrt(np.sum(V[:, c]**2))
                    W[:, c] /= np.sqrt(np.sum(W[:, c]**2))
            
            # Check if this decomposition is valid
            # The decomposition is valid if for all i,j,k:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k} (for the matrix multiplication)
            
            # Actually, let me think about this more carefully.
            # The matrix multiplication tensor T[i,j,k] = delta_{i,k} for 3x3x3.
            # For 3x3x6, we have T[i,j,k] where i in [0,3), j in [0,3), k in [0,6).
            # The multiplication is C[i,k] = sum_j A[i,j] * B[j,k].
            # So the tensor is T[i,j,k] = delta_{i,k} for k < 3, and T[i,j,k] = delta_{i,k-3} for k >= 3?
            
            # Wait, no. Let me re-read the task.
            # "Matrix multiplication 3,3,6"
            # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 6 matrix."
            
            # So A is 3x3, B is 3x6, C = A*B is 3x6.
            # C[i,k] = sum_j A[i,j] * B[j,k]
            
            # The tensor T[i,j,k] should be such that C[i,k] = sum_j sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
            
            # Wait, the convention is W[k*n+i, r]. So W is indexed by (k, i) where k is the column of C and i is the row of C.
            
            # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
            
            # So W has shape (p*n, rank) = (6*3, rank) = (18, rank).
            # W[k*n + i, r] where k in [0,6), i in [0,3).
            
            # And U has shape (n*m, rank) = (9, rank), U[i*m + j, r] where i in [0,3), j in [0,3).
            # V has shape (m*p, rank) = (3*6, rank) = (18, rank), V[j*p + k, r] where j in [0,3), k in [0,6).
            
            # The decomposition is: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # For this to equal the matrix multiplication, we need:
            # sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = sum_j A[i,j] * B[j,k]
            
            # This should hold for all A and B, which means the tensor defined by the U,V,W must match the multiplication tensor.
            
            # The multiplication tensor T[i,j,k] is 1 if this term contributes to C[i,k] from A[i,j]*B[j,k].
            # Actually, T[i,j,k] = 1 for all valid (i,j,k) because each term A[i,j]*B[j,k] contributes to C[i,k].
            
            # Wait, no. The tensor decomposition is:
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
            
            # And T[i,j,k] should be 1 for all i,j,k because each (i,j,k) triple corresponds to one term in the matrix multiplication.
            
            # Actually, I think I'm confusing myself. Let me think about this differently.
            
            # The matrix multiplication can be viewed as a tensor T in R^{n x m x p} where T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to the (i,k) entry of C.
            
            # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k because every pair (A[i,j], B[j,k]) contributes to C[i,k].
            
            # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
            
            # This is a system of equations. Let me set it up.
            
            # Number of equations: n*m*p = 3*3*6 = 54
            # Number of unknowns: 3 * target_rank (for U, V, W)
            
            # For target_rank = 45, we have 135 unknowns and 54 equations.
            # This is underdetermined, so there should be solutions.
            
            # Let me try to solve this using least squares or a direct approach.
            
            # Actually, let me try a different approach. I'll use the fact that this is a bilinear problem.
            # For fixed U and V, the equations are linear in W.
            
            # Let me try an alternating minimization approach.
            
            # Initialize U and V randomly
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            
            # For each c, we have U[:, c] and V[:, c]
            # The equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This is linear in W. Let me set up the linear system.
            # For each (i,k), we have one equation for W[k*n+i, c] for all c.
            
            # Actually, for each (i,k), the equation is:
            # sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This can be written as:
            # sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            
            # Let M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            # Then: sum_c M[i,k,c] * W[k*n+i, c] = 1
            
            # This is a linear system in W.
            
            # Let me build this system.
            # Number of equations: n*p = 3*6 = 18
            # Number of unknowns: p*n*target_rank = 18*target_rank
            
            # Wait, that's not right. W has shape (p*n, target_rank) = (18, target_rank).
            # So there are 18*target_rank unknowns in W.
            
            # But the equations are: for each (i,k), sum_c M[i,k,c] * W[k*n+i, c] = 1
            # This is 18 equations with 18*target_rank unknowns.
            
            # This is underdetermined. We can use least squares.
            
            # Actually, I realize I'm making this too complicated. Let me just try the schoolbook for now
            # and return it. The task says the score is the rank, and the schoolbook has rank 54.
            
            # But the inspiration says we can do better with rank 46.
            
            # Let me try to implement the Bini et al. decomposition for 3x3x3 and use it twice.
            
            # The Bini et al. decomposition has rank 23.
            # I need to find the coefficients.
            
            # Let me search for them online... but I can't do that in this environment.
            
            # Let me try to derive them or find them in a different way.
            
            # Actually, let me just return the schoolbook for now to ensure validity, and try to improve in a future iteration.
            
            pass
    
    # Try to find a better decomposition using optimization
    # We'll use alternating least squares
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try ranks from 45 down to 40
    for target_rank in range(45, 39, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        max_trials = 30 if target_rank > 42 else 50
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Alternating least squares
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update W: for each (i,k), solve sum_c M[i,k,c] * W[k*n+i, c] = 1
                # where M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                
                # Build the linear system for W
                # For each (i,k), we have: sum_c M[i,k,c] * W[k*n+i, c] = 1
                # This is 18 equations with 18*target_rank unknowns
                
                # Actually, let me reformulate. For each c, W[:, c] is a vector of length p*n.
                # The equation is: for each (i,k), sum_c M[i,k,c] * W[k*n+i, c] = 1
                
                # This can be written as a linear system: A * W_flat = b
                # where A is 18 x (18*target_rank), W_flat is (18*target_rank) x 1, b is 18 x 1
                
                # Let me build A
                A_W = np.zeros((n*p, p*n*target_rank))
                b_W = np.ones(n*p)
                
                for i in range(n):
                    for k in range(p):
                        row_idx = i * p + k  # equation index
                        for c in range(target_rank):
                            col_idx = (k*n + i) * target_rank + c  # W[k*n+i, c]
                            M_val = 0.0
                            for j in range(m):
                                M_val += U[i*m + j, c] * V[j*p + k, c]
                            A_W[row_idx, col_idx] = M_val
                
                # Solve least squares
                W_flat, residuals, rank_W, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
                W = W_flat.reshape(p*n, target_rank)
                
                # Update U: for each (i,j), solve sum_c U[i*m+j, c] * N[i,j,c] = 1
                # where N[i,j,c] = sum_k V[j*p+k, c] * W[k*n+i, c]
                
                A_U = np.zeros((n*m, n*m*target_rank))
                b_U = np.ones(n*m)
                
                for i in range(n):
                    for j in range(m):
                        row_idx = i * m + j
                        for c in range(target_rank):
                            col_idx = (i*m + j) * target_rank + c
                            N_val = 0.0
                            for k in range(p):
                                N_val += V[j*p + k, c] * W[k*n + i, c]
                            A_U[row_idx, col_idx] = N_val
                
                U_flat, residuals, rank_U, sv = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = U_flat.reshape(n*m, target_rank)
                
                # Update V: for each (j,k), solve sum_c V[j*p+k, c] * P[j,k,c] = 1
                # where P[j,k,c] = sum_i U[i*m+j, c] * W[k*n+i, c]
                
                A_V = np.zeros((m*p, m*p*target_rank))
                b_V = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        row_idx = j * p + k
                        for c in range(target_rank):
                            col_idx = (j*p + k) * target_rank + c
                            P_val = 0.0
                            for i in range(n):
                                P_val += U[i*m + j, c] * W[k*n + i, c]
                            A_V[row_idx, col_idx] = P_val
                
                V_flat, residuals, rank_V, sv = np.linalg.lstsq(A_V, b_V, rcond=None)
                V = V_flat.reshape(m*p, target_rank)
            
            # Check if the decomposition is valid
            if check_decomposition(U, V, W, tol=1e-4):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # Return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
