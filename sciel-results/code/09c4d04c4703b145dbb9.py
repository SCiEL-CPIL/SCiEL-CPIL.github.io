import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s
    
    # Target ranks to try
    target_ranks = [48, 40, 32]
    
    # Fallback to schoolbook
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Construct the target tensor T
    # T has shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and k==k' and i==i'
    # Actually, the tensor is sparse.
    # We can represent it as a list of non-zero entries.
    # Non-zero entries are at (i*m+j, j*p+k, k*n+i) where i,j,k are valid.
    # Wait, the indices in the problem are:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The condition for a term to be 1 is that the indices match the multiplication.
    # The tensor T_{a,b,c} is 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    # Note that j is determined by a and b, k by b and c, i by c and a.
    # So T is 1 at (i*m+j, j*p+k, k*n+i) for all i,j,k.
    
    # We will use a simple gradient descent on the flattened factors.
    # However, tensor decomposition is non-convex.
    # We will try a few random initializations for lower ranks.
    
    def random_decomposition(rank, seed):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        return U, V, W
    
    def compute_residual(U, V, W):
        # Residual = T - sum(u_r v_r w_r)
        # We compute the sum and compare to T.
        # T is sparse.
        # Instead of building T, we can just compute the sum and check if it matches the pattern.
        # But we need the error for gradient.
        # Let's build T as a dense array if size is small enough.
        # Size: 12 * 32 * 24 = 9216. Very small.
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Compute sum
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        return T - S, T
    
    # Precompute T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    for rank in target_ranks:
        if time.time() > deadline:
            break
            
        for seed in range(50): # 50 restarts per rank
            if time.time() > deadline:
                break
                
            U, V, W = random_decomposition(rank, seed)
            
            # Alternating Least Squares for tensor decomposition
            # For each factor, solve a least-squares problem
            # Residual: T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # When optimizing U: for each (j,k) position in V,W, we have a linear system in U
            # Actually, we can vectorize: for each r, the contribution is outer(U[:,r], V[:,r]) * W[:,r]
            # Better: reshape T to (n*m, m*p*p*n) or similar
            
            # ALS approach:
            # Fix V and W, optimize U:
            # T = sum_r U[:,r] * (V[:,r] ⊗ W[:,r])
            # This is a linear system: U * (V ⊗ W)^T = T_reshaped
            # Where V ⊗ W has shape (rank, m*p * p*n)
            
            # Let's do a simpler approach: gradient descent with careful step size
            # Or use the fact that we can compute gradients efficiently
            
            # Compute gradient for U: dL/dU[:,r] = sum_{j,k} (T[i,j,k] - S[i,j,k]) * V[j,r] * W[k,r]
            # where S = sum_r U[:,r] * V[:,r] * W[:,r]
            
            # Let's implement ALS properly:
            # For U: T_reshaped (n*m, m*p*p*n) = U (n*m, rank) * (V ⊗ W) (rank, m*p*p*n)
            # V ⊗ W: for each r, concatenate V[:,r] and W[:,r] element-wise? No.
            # Actually, the tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # So for fixed V,W: T[:,j,k] = U @ (V[j,:] * W[k,:])^T for each (j,k)
            # This is: T_reshaped (n*m, m*p*p*n) = U @ M where M[r, (j,k)] = V[j,r]*W[k,r]
            
            # Let's just do simple gradient descent with a good step size
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                    
                # Compute S = sum_r U[:,r] * V[:,r] * W[:,r]
                S = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = T - S
                
                # Gradient for U: dL/dU[i,r] = sum_{j,k} residual[i,j,k] * V[j,r] * W[k,r]
                grad_U = np.einsum('ijk,jr,kr->ir', residual, V, W)
                grad_V = np.einsum('ijk,ir,kr->jr', residual, U, W)
                grad_W = np.einsum('ijk,ir,jr->kr', residual, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if step % 20 == 0:
                    if np.linalg.norm(residual) < 1e-8:
                        break
            
            # Check if we found a valid decomposition
            S_final = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(S_final, T, atol=1e-6):
                # Convert to list of lists with integer/half-integer entries
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded solution
                S_rounded = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(S_rounded, T, atol=1e-10):
                    # Found a valid solution with this rank
                    result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    return result
                else:
                    # Try without rounding
                    result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    return result
    
    return best_result
# EVOLVE-BLOCK-END
