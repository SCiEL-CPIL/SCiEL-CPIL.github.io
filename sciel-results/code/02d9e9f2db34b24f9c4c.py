"""Matrix multiplication T_{2,3,5} tensor decomposition using optimization-based rank reduction."""
import numpy as np
import random

def construct():
    n, m, p = 2, 3, 5
    dim_a = n * m  # 6
    dim_b = m * p  # 15
    dim_c = p * n  # 10
    
    # Construct the target tensor T as a flat vector
    T = np.zeros(dim_a * dim_b * dim_c)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = (a * dim_b + b) * dim_c + c
                T[idx] = 1.0
    
    # Unfoldings for initialization
    # T1: shape (dim_a, dim_b * dim_c)
    # T2: shape (dim_b, dim_a * dim_c)
    # T3: shape (dim_c, dim_a * dim_b)
    T1 = T.reshape(dim_a, dim_b * dim_c)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    
    # Try ranks from 29 down to 20
    # Schoolbook is 30. We hope to find something lower.
    for rank in range(29, 19, -1):
        # Try multiple random initializations
        for _ in range(20):
            # Initialize factors
            # Use SVD of unfoldings for better init
            try:
                U_init = np.linalg.svd(T1, full_matrices=False)[0][:, :rank]
                V_init = np.linalg.svd(T2, full_matrices=False)[0][:, :rank]
                W_init = np.linalg.svd(T3, full_matrices=False)[0][:, :rank]
            except:
                U_init = np.random.randn(dim_a, rank)
                V_init = np.random.randn(dim_b, rank)
                W_init = np.random.randn(dim_c, rank)
            
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            # ALS iterations
            for it in range(50):
                # Update W
                # For each a, b, we have sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                # This is hard to do directly in ALS for 3-way without unfolding.
                # Standard ALS for 3-way tensor:
                # Fix U, V. Solve for W.
                # The system is: sum_r U[a,r] V[b,r] W[:,r] = T[:, a, b] ? No.
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Let M_W be a matrix where M_W[c, r] = W[c, r].
                # This is not a standard linear solve for W unless we unfold differently.
                # Unfold T along c: T_c is (dim_c, dim_a * dim_b)
                # T_c[c, a*dim_b+b] = sum_r W[c,r] * (U[a,r] V[b,r])
                # Let X_r = U[:,r] * V[:,r] (element-wise product, size dim_a*dim_b)
                # T_c = W * X, where X is (dim_a*dim_b, rank)
                # So W = T_c * X^+ (pseudo-inverse)
                
                X = np.zeros((dim_a * dim_b, rank))
                for r in range(rank):
                    X[:, r] = (U[:, r] * V[:, r]).flatten()
                
                # Solve W * X = T_c
                # W = T_c @ pinv(X)
                W_new = T3 @ np.linalg.pinv(X)
                
                # Update V
                # Fix U, W.
                # T_b[b, a*dim_c+c] = sum_r V[b,r] * (U[a,r] W[c,r])
                # Y_r = U[:,r] * W[:,r] (size dim_a*dim_c)
                # T_b = V * Y
                Y = np.zeros((dim_a * dim_c, rank))
                for r in range(rank):
                    Y[:, r] = (U[:, r] * W[:, r]).flatten()
                V_new = T2 @ np.linalg.pinv(Y)
                
                # Update U
                # Fix V, W.
                # T_a[a, b*dim_c+c] = sum_r U[a,r] * (V[b,r] W[c,r])
                # Z_r = V[:,r] * W[:,r] (size dim_b*dim_c)
                # T_a = U * Z
                Z = np.zeros((dim_b * dim_c, rank))
                for r in range(rank):
                    Z[:, r] = (V[:, r] * W[:, r]).flatten()
                U_new = T1 @ np.linalg.pinv(Z)
                
                # Check convergence
                if np.allclose(U, U_new) and np.allclose(V, V_new) and np.allclose(W, W_new):
                    break
                U, V, W = U_new, V_new, W_new
            
            # Round to nearest integer or half-integer
            # The task allows integer/half-integer.
            # We can scale by 2 to make them integers, round, then scale back.
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            
            # Verify exactness
            T_recon = np.zeros(dim_a * dim_b * dim_c)
            for r in range(rank):
                term = np.outer(np.outer(U_r[:, r], V_r[:, r]), W_r[:, r]).flatten()
                T_recon += term
            
            if np.allclose(T_recon, T, atol=1e-9):
                # Found a valid decomposition
                # Convert to lists
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no reduced rank found, return schoolbook
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
