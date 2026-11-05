import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # For 3x3 * 3x7, known lower bounds suggest rank < 63 might be possible
    # We'll use a randomized approach to search for structured low-rank decompositions
    
    # Strategy: Try various rank values and use iterative refinement
    # The tensor T[i,j,k] = delta_{i,k} (for matrix mult, T[i,j,k] where i=row of A, j=col of A/row of B, k=col of B)
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The multiplication tensor has entries T[i,j,k] = delta_{i,k}... no wait.
    
    # Let me be precise: We want C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The bilinear form for each output C[i,k] uses inputs A[i,j] and B[j,k] for j=0..m-1
    # The tensor T has indices (i, j, k) where:
    #   i indexes rows of A (and C): 0..n-1
    #   j indexes cols of A / rows of B: 0..m-1  
    #   k indexes cols of B (and C): 0..p-1
    # T[i,j,k] = 1 if this term contributes to C[i,k], which is when... 
    # Actually the tensor for matrix mult is: T_{i,j,k} = 1 for all valid (i,j,k)
    # No wait - the standard formulation: the multiplication tensor has a 1 at position (i,j,k) 
    # meaning A[i,j] * B[j,k] contributes to C[i,k]
    # But that's not quite right either. Let me think again.
    
    # The matrix multiplication tensor T in V_{n*m} ⊗ V_{m*p} ⊗ V_{n*p}
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_{(i,k)} ... no
    # 
    # Standard: T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} 
    #            e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    # where e_{(i,j)} is basis vector for A[i,j], e_{(j,k)} for B[j,k], e_{(i,k)} for C[i,k]
    
    # So the tensor has 1 at positions where the indices match: A[i,j], B[j,k], C[i,k]
    
    # For rank decomposition: T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{n*p}
    
    # Let's try a structured approach: exploit the fact that m=3 is small
    
    # Try rank values from lower bound up to schoolbook
    # Lower bound for 3x3*3x7: at least max(3*3, 3*7, 3*7) = 21, but likely higher
    
    # Use a simple iterative method: start with random initialization and refine
    def try_rank(R, max_iter=50, tol=1e-10):
        if time.time() > deadline:
            return None
        
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(n*p, R)
        
        # Target tensor flattened
        # T[i,j,k] = 1 if (i,j,k) corresponds to valid multiplication term
        # In the tensor, position is indexed by (i*m+j, j*p+k, i*p+k)
        # So T has 1s at: for each (i,j,k): U index = i*m+j, V index = j*p+k, W index = i*p+k
        
        # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[i*p+k, r] = 1 for all (i,j,k)
        # and 0 otherwise (but there are no "otherwise" positions in the dense tensor)
        
        # Actually the tensor is dense in the sense that every combination (i,j,k) with 
        # i in [0,n), j in [0,m), k in [0,p) gives a 1 at the corresponding position.
        # But the tensor is sparse in the full (n*m) x (m*p) x (n*p) space.
        
        # Let's verify: the non-zero entries are at positions (i*m+j, j*p+k, i*p+k)
        # for i in range(n), j in range(m), k in range(p)
        
        # We can set up an optimization problem to minimize:
        # sum over all (a,b,c) of (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
        # where T[a,b,c] = 1 if there exists (i,j,k) with a=i*m+j, b=j*p+k, c=i*p+k
        # else 0
        
        # Build the sparse tensor
        T_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = i * p + k
                    T_indices.append((a, b, c))
        
        T_vals = np.ones(len(T_indices))
        
        # Use gradient descent on the factor matrices
        lr = 0.01
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Compute gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for idx, (a, b, c) in enumerate(T_indices):
                # Current value at (a,b,c)
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                err = val - 1.0  # T[a,b,c] = 1
                
                # d/dU[a,r] = err * V[b,r] * W[c,r]
                grad_U[a, :] += err * V[b, :] * W[c, :]
                grad_V[b, :] += err * U[a, :] * W[c, :]
                grad_W[c, :] += err * U[a, :] * V[b, :]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if it % 10 == 0:
                # Compute error
                total_err = 0.0
                for a, b, c in T_indices:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    total_err += (val - 1.0) ** 2
                if total_err < tol:
                    # Found a solution!
                    return U, V, W
        
        return None
    
    # Try various ranks
    for R in range(21, rank_school + 1):
        if time.time() > deadline:
            break
        
        result = try_rank(R, max_iter=100)
        if result is not None:
            U, V, W = result
            # Convert to list format
            U_list = [[U[i][j] for j in range(R)] for i in range(n*m)]
            V_list = [[V[i][j] for j in range(R)] for i in range(m*p)]
            W_list = [[W[i][j] for j in range(R)] for i in range(n*p)]
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = i * p + k
                        val = sum(U_list[a][r] * V_list[b][r] * W_list[c][r] for r in range(R))
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid and R < best_rank:
                best_rank = R
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found a valid lower rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
