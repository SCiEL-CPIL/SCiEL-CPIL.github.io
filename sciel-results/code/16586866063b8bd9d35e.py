"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_rank = rank_sb
    best_U, best_V, best_W = U, V, W

    # Strategy: Try to find a lower rank decomposition using optimization
    # The matrix multiplication tensor T is defined by T[i,j,k] = e_{i,m+j} ⊗ e_{j,p+k} ⊗ e_{k,n+i}
    # Actually, the convention is:
    # Input A is n x m, Input B is m x p.
    # Output C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k).
    # A factorization of rank r is:
    # T[i,j,k] = sum_{l=1}^r U[i,j,l] V[j,k,l] W[i,k,l]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    # Note: The task specifies W[k*n+i, r]. This is the output index ordering.
    # Let's verify the indices.
    # U: index (i*m + j)
    # V: index (j*p + k)
    # W: index (k*n + i)
    # So the equation is:
    # For all i, k:
    #   sum_j A[i,j] B[j,k] = sum_l W[k*n+i, l] * (sum_j U[i*m+j, l] * V[j*p+k, l])?
    # No, the standard decomposition is:
    # C = sum_l W_l * (A U_l) * V_l ? No.
    # The bilinear complexity is the number of multiplications.
    # A multiplication corresponds to a term in the sum.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # If we precompute A' = A U and B' = B V, then C = sum_l W_l (A' U_l) (B' V_l) ?
    # Let's look at the tensor definition.
    # T_{i,j,k} = delta_{i,k}? No.
    # The tensor of matrix multiplication M_{n,m,p} is in A^* ⊗ B^* ⊗ C^*.
    # T(A,B,C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # We want to write T as sum_l U_l ⊗ V_l ⊗ W_l
    # where U_l is a vector in A^* (size nm), V_l in B^* (size mp), W_l in C^* (size np).
    # This means:
    # sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k} = sum_l (sum_{i,j} A_{i,j} U_{l,i,j}) (sum_{j,k} B_{j,k} V_{l,j,k}) (sum_{i,k} C_{i,k} W_{l,i,k})
    # This equality must hold for all A, B, C.
    # This implies that the tensor T (viewed as a 3-way tensor) can be decomposed as a sum of rank-1 tensors.
    # T_{i,j,k} = sum_l U_{l, i*m+j} V_{l, j*p+k} W_{l, k*n+i}
    # So we need to find U, V, W of rank r such that:
    # for all i, j, k:
    #   delta_{i,k}? No.
    #   The entry T_{i,j,k} is 1 if the indices are consistent with multiplication?
    #   Actually, T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} C_{i,k} in the product.
    #   Wait, the product is C = AB. So C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #   The tensor T is defined such that T(A,B) = C.
    #   So T_{i,j,k} is not 0 or 1 in a simple sense.
    #   Let's re-read the standard definition.
    #   The tensor M_{n,m,p} has entries M_{i,j,k} such that:
    #   M_{i,j,k} = 1 if j is in [1..m] (always true) and... wait.
    #   The entries of the tensor are defined by:
    #   T_{i,j,k} = 1 if the indices i, j, k form a valid triple for the multiplication?
    #   No. The tensor is in (R^{nm})^* ⊗ (R^{mp})^* ⊗ (R^{np})^*.
    #   The canonical basis of (R^{nm})^* is e_{i,j}^*.
    #   T = sum_{i,j,k} e_{i,j}^* ⊗ e_{j,k}^* ⊗ e_{i,k}^* ?
    #   Let's check.
    #   sum_l U_l V_l W_l = sum_{i,j,k} e_{i,j}^* ⊗ e_{j,k}^* ⊗ e_{i,k}^*
    #   Then <T, A ⊗ B ⊗ C> = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    #   This matches the definition of the multiplication tensor.
    #   So the tensor T has entries:
    #   T_{i*m+j, j*p+k, k*n+i} = 1
    #   All other entries are 0.
    #   So we need to decompose this specific sparse tensor into a sum of rank-1 tensors.
    #   Rank of M_{3,3,3} is known to be 21 (Strassen-like).
    #   Rank of M_{3,3,4} is unknown but likely 21 + something or close to 21*4/3 = 28?
    #   Or maybe it's 23?
    #   We will try to optimize for rank 21, 22, 23, 24.

    def create_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T

    T_target = create_tensor(n, m, p)
    
    def loss_func(params, r):
        # Reshape params into U, V, W
        U = params[:n*m*r].reshape((n*m, r))
        V = params[n*m*r:n*m*r + m*p*r].reshape((m*p, r))
        W = params[n*m*r + m*p*r:].reshape((p*n, r))
        
        # Compute the decomposed tensor
        # We need to check if sum_l U[:,l] V[:,l] W[:,l] approximates T_target
        # This is a 3-way tensor. We can flatten it.
        # T_decomposed_{i,j,k} = sum_l U[i,l] V[j,l] W[k,l]
        # Here i is index in [0, nm), j in [0, mp), k in [0, pn).
        # But the indices are coupled in the target T.
        # Let's flatten T_target and T_decomposed and compare.
        # However, T_decomposed is dense if r > 0.
        # We can compute the Frobenius norm of the difference.
        
        # Efficient computation:
        # T_decomposed = np.einsum('il,jl,kl->ijk', U, V, W)
        # This might be slow.
        # Alternative:
        # For each l, we add outer(U[:,l], V[:,l], W[:,l]) to the accumulator.
        # This is also slow in Python loops.
        
        # Let's use einsum.
        T_dec = np.einsum('il,jl,kl->ijk', U, V, W)
        
        # Calculate loss
        diff = T_dec - T_target
        loss = np.sum(diff**2)
        
        # Add a regularization to encourage sparsity or structure?
        # Or just L2 loss.
        return loss

    # Try ranks from 21 up to 30
    for rank in range(21, 31):
        if time.time() > deadline - 30.0:
            break
            
        # Initialize parameters
        # We can try several initializations
        found_solution = False
        
        for init_idx in range(5):
            if time.time() > deadline - 10.0:
                break
                
            # Random initialization
            np.random.seed(init_idx * 100 + rank)
            U_init = np.random.randn(n*m, rank) * 0.1
            V_init = np.random.randn(m*p, rank) * 0.1
            W_init = np.random.randn(p*n, rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            res = minimize(loss_func, params_init, args=(rank,), method='BFGS', 
                           options={'gtol': 1e-6, 'maxiter': 1000})
            
            if res.success and res.fun < 1e-6:
                # Verify exactness
                U = res.x[:n*m*rank].reshape((n*m, rank))
                V = res.x[n*m*rank:n*m*rank + m*p*rank].reshape((m*p, rank))
                W = res.x[n*m*rank + m*p*rank:].reshape((p*n, rank))
                
                T_dec = np.einsum('il,jl,kl->ijk', U, V, W)
                if np.allclose(T_dec, T_target, atol=1e-9):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    found_solution = True
                    break # Break init loop
        
        if found_solution:
            break # Break rank loop, found minimal rank in this range

    # Convert best result to lists
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
