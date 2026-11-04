import numpy as np
import random

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    # True multiplication tensor T[i, j, k] = 1 if (i, j, k) is a valid index for A[i,k]*B[k,j]?
    # Convention: 
    # A is n x m (4x4)
    # B is m x p (4x7)
    # C is n x p (4x7)
    # C[i, p_idx] = sum_j A[i, j] * B[j, p_idx]
    # 
    # The tensor T has dimensions (n*m, m*p, p*n)?
    # The interface says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # 
    # The decomposition is T = U V^T W^T ? No.
    # Standard tensor decomposition: T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
    # Here indices are flattened.
    # Index 1: n*m (A's indices) -> i, j? No, A is n x m.
    # Let's map:
    # A index: (i, j) where i in 0..n-1, j in 0..m-1. Flat: i*m + j.
    # B index: (j, k) where j in 0..m-1, k in 0..p-1. Flat: j*p + k.
    # C index: (i, k) where i in 0..n-1, k in 0..p-1. Flat: i*p + k?
    # Wait, the interface says W is (p*n, rank).
    # And the task says "output factor follows the official trace convention W[k*n+i,r]".
    # So W index is (k, i) where k is col of C (0..p-1), i is row of C (0..n-1).
    # Flat index: k*n + i.
    
    # The multiplication tensor T[A_idx, B_idx, C_idx] is 1 if the indices are consistent.
    # A_idx = i*m + j
    # B_idx = j*p + k
    # C_idx = k*n + i
    # T is 1 if (A_idx, B_idx, C_idx) form a valid term in the sum.
    # Specifically, T[a, b, c] = 1 if:
    # i = a // m, j = a % m
    # j' = b // p, k = b % p
    # k' = c // n, i' = c % n
    # And i == i' and j == j' and k == k'.
    
    # Let's build the true tensor T of shape (n*m, m*p, p*n)
    # n*m = 16
    # m*p = 28
    # p*n = 28
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
                
    # We want to find U (16 x R), V (28 x R), W (28 x R)
    # such that T = sum_r U[:, r] V[:, r]^T W[:, r]^T (Outer product sum)
    # This is equivalent to minimizing || T - sum_r U_r V_r W_r ||
    
    # Heuristic: Try ranks from 20 to 112
    # We'll use a simple gradient descent or ALS.
    # ALS is faster and more stable for this.
    
    def try_decompose(rank, iterations=100, tol=1e-6):
        # Initialize U, V, W randomly
        # Scale them to help convergence
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # ALS: Update U, then V, then W
        for _ in range(iterations):
            # Update U: 
            # For each r, U[:, r] = argmin || T - sum_s U_s V_s W_s ||
            # This is linear in U[:, r].
            # T = U_r (V_r W_r^T) + sum_{s!=r} U_s (V_s W_s^T)
            # Let M_r = V_r W_r^T (shape m*p x p*n)
            # We want to match T[:, :, :]
            # Residual R = T - sum_{s!=r} U_s V_s W_s^T
            # U_r = argmin || R - U_r V_r W_r^T ||_F
            # This is a linear system: (V_r W_r^T)^T (V_r W_r^T) U_r^T = (V_r W_r^T)^T R^T ?
            # Actually, vectorizing:
            # Let A_r = V_r W_r^T (28 x 28)
            # We want U_r (16 x 1) such that U_r A_r approximates R (16 x 28 x 28 flattened? No, T is 3D)
            # T is (16, 28, 28).
            # Term r is U_r (16, 1) * V_r (28, 1)^T * W_r (28, 1)^T
            # This is an outer product of vectors.
            # It's a rank-1 update in the 3rd mode? No, it's a rank-1 term in the tensor.
            # T_{abc} = sum_r U_{ar} V_{br} W_{cr}
            # Fix r. Update U_{ar}.
            # T_{abc} = U_{ar} V_{br} W_{cr} + sum_{s!=r} ...
            # For fixed b, c: T_{abc} - sum_{s!=r} ... = U_{ar} V_{br} W_{cr}
            # This is not a standard linear least squares in U if we view it as 3D.
            # However, we can flatten T into a matrix.
            # Mode-1 unfolding: T_(1) is (n*m) x (m*p * p*n).
            # T_(1) = U (V_(1) * W_(1))^T ?
            # V_(1) is (m*p) x rank. W_(1) is (p*n) x rank.
            # The Kronecker product V_(1) ⊗ W_(1) is (m*p * p*n) x rank.
            # So T_(1) = U (V_(1) ⊗ W_(1))^T.
            # To update U:
            # U = T_(1) (V_(1) ⊗ W_(1))^+
            
            # Update V:
            # T_(2) = V (W_(1) ⊗ U_(1))^T ?
            # T_(2) is (m*p) x (p*n * n*m).
            # V = T_(2) (W_(1) ⊗ U_(1))^+
            
            # Update W:
            # T_(3) = W (U_(1) ⊗ V_(1))^T ?
            # T_(3) is (p*n) x (n*m * m*p).
            # W = T_(3) (U_(1) ⊗ V_(1))^+
            
            # Precompute unfoldings
            T1 = T.reshape(n*m, m*p * p*n)
            T2 = T.transpose(1, 0, 2).reshape(m*p, p*n * n*m)
            T3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
            
            # Update U
            # V1 is (m*p, rank), W1 is (p*n, rank)
            # We need V1 ⊗ W1?
            # The term in T1 is sum_r U[:, r] (V[:, r] ⊗ W[:, r])^T ?
            # Let's check indices.
            # T1[a, b*c] = sum_r U[a, r] V[b, r] W[c, r]
            # Yes.
            # So M_U = V ⊗ W? No, it's a matrix where column r is V[:, r] ⊗ W[:, r].
            # V is (28, R), W is (28, R).
            # V ⊗ W (Kronecker) is (28*28, R) ?
            # np.kron(V, W) shape is (28*28, R) if V is (28, R) and W is (28, R)?
            # No, np.kron(A, B) where A is (I, J) and B is (K, L) is (I*K, J*L).
            # We want column r to be V[:, r] ⊗ W[:, r].
            # V[:, r] is (28, 1). W[:, r] is (28, 1).
            # The matrix M_U should have columns V[:, r] ⊗ W[:, r].
            # This is equivalent to np.kron(V, W) if we treat V and W as matrices?
            # Let's construct it explicitly to be safe or use broadcasting.
            # M_U[b, c, r] = V[b, r] * W[c, r]
            # M_U is (28, 28, R). Reshape to (28*28, R).
            M_U = (V[:, :, None] * W[:, None, :]).reshape(m*p * p*n, rank)
            U = T1 @ np.linalg.pinv(M_U)
            
            # Update V
            # T2[b, c*a] = sum_r V[b, r] W[c, r] U[a, r]
            # M_V[c, a, r] = W[c, r] * U[a, r]
            M_V = (W[:, :, None] * U[:, None, :]).reshape(p*n * n*m, rank)
            V = T2 @ np.linalg.pinv(M_V)
            
            # Update W
            # T3[c, a*b] = sum_r W[c, r] U[a, r] V[b, r]
            # M_W[a, b, r] = U[a, r] * V[b, r]
            M_W = (U[:, :, None] * V[:, None, :]).reshape(n*m * m*p, rank)
            W = T3 @ np.linalg.pinv(M_W)
            
            # Check convergence
            # Reconstruct T_approx
            # T_approx = sum_r U_r V_r W_r^T
            # T_approx_flat = (U[:, :, None] * V[:, None, :] * W[None, :, :])?
            # No, T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - T_approx)
            if err < tol:
                return U, V, W, err
                
        return None, None, None, err

    best_rank = 112
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from 20 to 112
    # We'll step by 1 or 2 to save time
    for rank in range(20, 113):
        # Try a few random initializations
        for attempt in range(3):
            U, V, W, err = try_decompose(rank)
            if U is not None:
                # Check if we can round to integers/half-integers
                # And if the rounded version is exact
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_approx_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(T, T_approx_rounded, atol=1e-6):
                    best_rank = rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break
        if best_rank < 112:
            break
            
    if best_U is None:
        # Fallback to schoolbook
        return _schoolbook()
        
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook():
    n, m, p = 4, 4, 7
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                terms.append((u, v, w))
                
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
            
    return {"U": U, "V": V, "W": W}
