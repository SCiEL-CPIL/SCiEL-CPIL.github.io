import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Rank 28 decomposition (from inspirations)
    def build_rank_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        
        # Part 1: A2 B2 (j=2, k=0..4)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
        # Part 2: A1 B13 (j=0,1, k=4)
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
                
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
        # M1
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M2
        U[1 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1
        W[1 * n + 1][r] = -1
        r += 1
        
        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 1][r] = 1
        V[1 * p + 1][r] = -1
        W[1 * n + 0][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 0][r] = 1
        V[0 * p + 0][r] = -1
        W[0 * n + 0][r] = 1
        W[0 * n + 1][r] = 1
        r += 1
        
        # M5
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = -1
        W[1 * n + 0][r] = 1
        r += 1
        
        # M6
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + 0][r] = 1
        V[0 * p + 1][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M7
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + 0][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        r += 1
        
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
        # M1
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M2
        U[1 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        W[2 * n + 1][r] = 1
        W[3 * n + 1][r] = -1
        r += 1
        
        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 3][r] = 1
        V[1 * p + 3][r] = -1
        W[3 * n + 0][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 2][r] = 1
        V[0 * p + 2][r] = -1
        W[2 * n + 0][r] = 1
        W[2 * n + 1][r] = 1
        r += 1
        
        # M5
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = -1
        W[3 * n + 0][r] = 1
        r += 1
        
        # M6
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + 2][r] = 1
        V[0 * p + 3][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M7
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + 2][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        r += 1
        
        return U, V, W

    U_best, V_best, W_best = build_rank_28()
    
    # Verification function
    def verify(U, V, W, tol=1e-6):
        rank = len(U[0])
        # Convert to numpy for speed
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.sum(U_arr[idx_u, :] * V_arr[idx_v, :] * W_arr[idx_w, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Try to find rank 27
    def try_rank_27():
        if time.time() > deadline - 20:
            return None
            
        # Initialize with random values
        # We can try to initialize by taking rank 28 and zeroing one column, then optimizing
        # Or just random init
        rng = np.random.RandomState(42)
        
        for restart in range(50):
            if time.time() > deadline - 10:
                break
                
            U = rng.randn(n*m, 27) * 0.5
            V = rng.randn(m*p, 27) * 0.5
            W = rng.randn(p*n, 27) * 0.5
            
            # Gradient Descent
            lr = 0.01
            for it in range(2000):
                if time.time() > deadline - 10:
                    return None
                
                # Compute S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                S = np.zeros((n, m, p))
                for r in range(27):
                    # U is (nm, 1), V is (mp, 1), W is (pn, 1)
                    # We need to broadcast
                    # U_flat = U[:, r].reshape(n, m)
                    # V_flat = V[:, r].reshape(m, p)
                    # W_flat = W[:, r].reshape(p, n)
                    # S += np.einsum('ij,jk,ki->ijk', U[:, r].reshape(n,m), V[:, r].reshape(m,p), W[:, r].reshape(p,n))
                    
                    u_r = U[:, r].reshape(n, m)
                    v_r = V[:, r].reshape(m, p)
                    w_r = W[:, r].reshape(p, n)
                    
                    # S[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
                    # Using einsum
                    S += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
                
                # Loss = sum (S - 1)^2
                # Gradient w.r.t U[i,j,r]
                # dL/dU[i,j,r] = 2 * (S[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
                # Note: k is summed over in the gradient for U? No, S is indexed by i,j,k.
                # The term for U[i,j,r] appears in S[i,j,k] for all k.
                # So grad_U[i,j,r] = 2 * sum_k (S[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(27):
                    u_r = U[:, r].reshape(n, m)
                    v_r = V[:, r].reshape(m, p)
                    w_r = W[:, r].reshape(p, n)
                    
                    # grad_U[i,j] = 2 * sum_k (S[i,j,k] - 1) * v_r[j,k] * w_r[k,i]
                    # (S - 1) is (n,m,p). v_r is (m,p). w_r is (p,n).
                    # sum over k: (n,m) = einsum('ijk,jk,ki->ij', S-1, v_r, w_r)
                    grad_U[:, r] = (2 * np.einsum('ijk,jk,ki->ij', S - 1.0, v_r, w_r)).flatten()
                    
                    # grad_V[j,k] = 2 * sum_i (S[i,j,k] - 1) * u_r[i,j] * w_r[k,i]
                    # sum over i: (m,p) = einsum('ijk,ij,ki->jk', S-1, u_r, w_r)
                    grad_V[:, r] = (2 * np.einsum('ijk,ij,ki->jk', S - 1.0, u_r, w_r)).flatten()
                    
                    # grad_W[k,i] = 2 * sum_j (S[i,j,k] - 1) * u_r[i,j] * v_r[j,k]
                    # sum over j: (p,n) = einsum('ijk,ij,jk->ki', S-1, u_r, v_r)
                    grad_W[:, r] = (2 * np.einsum('ijk,ij,jk->ki', S - 1.0, u_r, v_r)).flatten()
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(S - 1.0)) < 1e-6:
                    return U.tolist(), V.tolist(), W.tolist()
                    
                # Adaptive learning rate
                if it % 100 == 0:
                    lr *= 0.9
                    
        return None

    result_27 = try_rank_27()
    if result_27:
        U_best, V_best, W_best = result_27
        if verify(U_best, V_best, W_best):
            pass # Success
        else:
            # Fallback
            U_best, V_best, W_best = build_rank_28()
            
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
