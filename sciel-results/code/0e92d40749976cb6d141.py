import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Decompose 2x3 * 3x9 matrix multiplication.
    Strategy: Use tensor product of rank-14 decomposition of 2x3x3 
    and rank-3 decomposition of 1x1x3 to get rank 42.
    Falls back to schoolbook rank 54 if optimization fails.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Target Rank
    target_rank = 42
    fallback_rank = 54
    
    start_time = time.time()
    deadline = start_time + 120  # Leave 30s safety margin
    
    # 1. Construct rank-14 decomposition for 2x3x3
    # Tensor T_sub for 2x3x3
    # U: 6xR, V: 9xR, W: 6xR
    n_sub, m_sub, p_sub = 2, 3, 3
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                alpha = i * m_sub + j
                beta = j * p_sub + k
                gamma = k * n_sub + i
                T_sub[alpha, beta, gamma] = 1.0
                
    U_sub = None
    V_sub = None
    W_sub = None
    found_rank_14 = False
    
    # Try to find rank 14 decomposition
    # Using simple gradient descent
    # We use 3 restarts
    for attempt in range(3):
        if time.time() > deadline - 20:
            break
            
        R = 14
        # Initialize
        scale = 1.0
        U = np.random.randn(n_sub * m_sub, R) * scale
        V = np.random.randn(m_sub * p_sub, R) * scale
        W = np.random.randn(p_sub * n_sub, R) * scale
        
        # Gradient Descent
        lr = 0.01
        for step in range(500):
            if time.time() > deadline - 20:
                break
                
            # Compute current tensor approximation
            # T_approx = sum_r U[:,r] outer V[:,r] outer W[:,r]
            # Efficiently:
            # Residual = T_sub - sum...
            # We can compute gradient directly
            
            # Grad U: sum_{beta,gamma} (T - T_approx)_{alpha,beta,gamma} V_{beta,r} W_{gamma,r}
            # This is equivalent to:
            # For each r:
            #   G_U[:, r] = np.tensordot((T_sub - T_approx) * (V[:,r,None,None] * W[:,None,:,r]), (1,1,1), axes=([1,2],[0,0]))? No.
            
            # Let's do it step by step to be safe and clear, though slower.
            # Given small size, it's okay.
            
            T_approx = np.zeros_like(T_sub)
            for r in range(R):
                T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
            Residual = T_sub - T_approx
            
            # Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(R):
                # dU[:, r] = sum_{b,g} Residual[a,b,g] V[b,r] W[g,r]
                # = np.tensordot(Residual, V[:,r,None] * W[:,None,:,r], axes=([1,2],[0,0]))
                dU[:, r] = np.tensordot(Residual, V[:, r, None] * W[:, None, :, r], axes=([1, 2], [0, 0]))
                
                # dV[:, r] = sum_{a,g} Residual[a,b,g] U[a,r] W[g,r]
                dV[:, r] = np.tensordot(Residual, U[:, r, None] * W[:, None, :, r], axes=([0, 2], [0, 0]))
                
                # dW[:, r] = sum_{a,b} Residual[a,b,g] U[a,r] V[b,r]
                dW[:, r] = np.tensordot(Residual, U[:, r, None] * V[:, None, :, r], axes=([0, 1], [0, 0]))
                
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if step % 10 == 0:
                err = np.linalg.norm(Residual)
                if err < 1e-6:
                    break
        
        # Check if solution is close to half-integers
        # And verify exactness
        U_check = np.round(U * 2) / 2
        V_check = np.round(V * 2) / 2
        W_check = np.round(W * 2) / 2
        
        T_check = np.zeros_like(T_sub)
        for r in range(R):
            T_check += np.einsum('a,b,c->abc', U_check[:, r], V_check[:, r], W_check[:, r])
            
        if np.allclose(T_check, T_sub, atol=1e-9):
            U_sub = U_check
            V_sub = V_check
            W_sub = W_check
            found_rank_14 = True
            break
        else:
            # Try to snap and see if we can fix with local search? 
            # For now, just try next restart if not close enough
            pass
            
    if not found_rank_14:
        # Fallback to Rank 18 for 2x3x3
        R_sub = 18
        U_sub = np.zeros((n_sub * m_sub, R_sub))
        V_sub = np.zeros((m_sub * p_sub, R_sub))
        W_sub = np.zeros((p_sub * n_sub, R_sub))
        for r in range(R_sub):
            i = r // (m_sub * p_sub)
            rem = r % (m_sub * p_sub)
            j = rem // p_sub
            k = rem % p_sub
            U_sub[i * m_sub + j, r] = 1.0
            V_sub[j * p_sub + k, r] = 1.0
            W_sub[k * n_sub + i, r] = 1.0
        target_rank = 54 # 18 * 3

    # 2. Construct rank-42 decomposition for 2x3x9
    # Using tensor product with M_113 (Rank 3)
    # U_113 is 1x3 ones, V_113 is 3x3 I, W_113 is 3x3 I
    
    R_final = target_rank
    U_final = np.zeros((n * m, R_final))
    V_final = np.zeros((m * p, R_final))
    W_final = np.zeros((p * n, R_final))
    
    R_sub_actual = U_sub.shape[1]
    # R_final should be R_sub_actual * 3
    
    for r in range(R_sub_actual):
        for s in range(3):
            idx = 3 * r + s
            
            # U part: U_sub[alpha, r] * 1
            U_final[:, idx] = U_sub[:, r]
            
            # V part: V_sub[beta_sub, r] * delta(k_sub, s)
            # beta_sub is index in 3x3, beta is index in 3x9
            # beta = 9*j + k, k = 3*k1 + s
            # So for a given beta, j = beta // 9, k = beta % 9
            # k1 = k // 3, s_local = k % 3
            # We need V_final[beta, idx] = V_sub[3*j + k1, r] if s == s_local else 0
            
            for beta in range(m * p):
                j = beta // p
                k = beta % p
                k1 = k // 3
                s_local = k % 3
                if s_local == s:
                    beta_sub = 3 * j + k1
                    V_final[beta, idx] = V_sub[beta_sub, r]
                else:
                    V_final[beta, idx] = 0.0
                    
            # W part: W_sub[gamma_sub, r] * delta(k_sub, s)
            # gamma = 2*k + i, k = 3*k1 + s
            # gamma_sub = 2*k1 + i
            # i = gamma % 2, k = gamma // 2
            # k1 = k // 3, s_local = k % 3
            
            for gamma in range(p * n):
                k = gamma // n
                i = gamma % n
                k1 = k // 3
                s_local = k % 3
                if s_local == s:
                    gamma_sub = 2 * k1 + i
                    W_final[gamma, idx] = W_sub[gamma_sub, r]
                else:
                    W_final[gamma, idx] = 0.0

    # Verify final result
    T_final = np.zeros((n * m, m * p, p * n))
    for r in range(R_final):
        T_final += np.einsum('a,b,c->abc', U_final[:, r], V_final[:, r], W_final[:, r])
        
    # Target T
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1.0
                
    if not np.allclose(T_final, T_target, atol=1e-9):
        # Fallback to schoolbook if verification fails
        U_final = np.zeros((n * m, n * m * p))
        V_final = np.zeros((m * p, n * m * p))
        W_final = np.zeros((p * n, n * m * p))
        for r in range(n * m * p):
            i = r // (m * p)
            rem = r % (m * p)
            j = rem // p
            k = rem % p
            U_final[i * m + j, r] = 1.0
            V_final[j * p + k, r] = 1.0
            W_final[k * n + i, r] = 1.0
            R_final = n * m * p

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
