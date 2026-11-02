import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    # 1. Solve for rank 11 decomposition of M_{2,3,2}
    # Tensor T_{2,3,2} has entries T[i, j, k] = 1 for all i,j,k
    # We want U (6x11), V (6x11), W (4x11)
    # such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] == T[i,j,k]
    
    # Initialize
    best_rank = 48
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook for 2,3,8
    def get_schoolbook():
        n, m, p = 2, 3, 8
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    result = get_schoolbook()
    
    # Try to find better decomposition for 2,3,2
    # Target tensor for 2,3,2
    n2, m2, p2 = 2, 3, 2
    T_target = np.ones((n2, m2, p2))
    
    rank_target = 11
    
    def solve_232(rank):
        if time.time() > deadline - 10:
            return None
            
        np.random.seed(42)
        U = np.random.randn(n2 * m2, rank)
        V = np.random.randn(m2 * p2, rank)
        W = np.random.randn(p2 * n2, rank)
        
        # ALS
        for it in range(1000):
            if time.time() > deadline - 10:
                return None
                
            # Update U
            for r in range(rank):
                # V[:, r] and W[:, r] are fixed
                # U[:, r] should be such that U[:, r] V[:, r] W[:, r] approximates T
                # This is linear in U[:, r]
                # T = sum U[:, r] outer(V[:, r], W[:, r])
                # We want to minimize || T - sum U[:, r] M_r ||^2
                # M_r = V[:, r] outer W[:, r] reshaped to (n*m, p*n) ??
                # Actually, the tensor is T[i, j, k].
                # U index (i, j). V index (j, k). W index (k, i).
                # Term r: U[i, j] V[j, k] W[k, i].
                # For fixed r, we want to find u_r such that
                # sum_r u_r[i,j] v_r[j,k] w_r[k,i] = T[i,j,k].
                # This is a linear system for u_r if we fix v_r, w_r.
                # Flatten T to vector t (size 12).
                # Flatten M_r (size 12).
                # u_r is size 6.
                # t = sum u_r dot M_r.
                # We can solve least squares for all r simultaneously?
                # No, ALS updates one factor at a time.
                
                # Let's use a simpler gradient descent step for the whole U
                pass
            
            # Simplified: Use gradient descent on the loss
            loss = np.sum((T_target - np.einsum('ir,jr,kr->ijk', U, V, W))**2)
            if loss < 1e-8:
                return U, V, W
                
            # Compute gradients
            dL_dU = np.einsum('ijk,jr,kr->ir', T_target - np.einsum('ir,jr,kr->ijk', U, V, W), V, W)
            dL_dV = np.einsum('ijk,ir,kr->jr', T_target - np.einsum('ir,jr,kr->ijk', U, V, W), U, W)
            dL_dW = np.einsum('ijk,ir,jr->kr', T_target - np.einsum('ir,jr,kr->ijk', U, V, W), U, V)
            
            lr = 0.1
            U -= lr * dL_dU
            V -= lr * dL_dV
            W -= lr * dL_dW
            
        return None

    # Try a few random restarts
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        sol = solve_232(rank_target)
        if sol:
            U232, V232, W232 = sol
            # Round to nearest integer/half-integer
            U232 = np.round(U232 * 2) / 2
            V232 = np.round(V232 * 2) / 2
            W232 = np.round(W232 * 2) / 2
            
            # Verify
            T_recon = np.einsum('ir,jr,kr->ijk', U232, V232, W232)
            if np.allclose(T_recon, T_target, atol=1e-6):
                # Construct 2,3,8
                # Kronecker with 1,1,4
                # U_238: (2*3) x (11*4) = 6 x 44
                # U_232 is 6 x 11. Tile 4 times.
                U238 = np.tile(U232, (1, 4))
                
                # V_238: (3*8) x 44 = 24 x 44
                # V_232 is 6 x 11.
                # V_114 is I_4.
                # V_238[(j, j'), (r, r')] = V_232[j, r] * delta(j', r')
                V238 = np.zeros((24, 44))
                for j_prime in range(4):
                    for r_prime in range(4):
                        if j_prime == r_prime:
                            V238[j_prime*3 : (j_prime+1)*3, r_prime*11 : (r_prime+1)*11] = V232
                
                # W_238: (2*8) x 44 = 16 x 44
                # W_232 is 4 x 11.
                # W_238[(k, k'), (r, r')] = W_232[k, r] * delta(k', r')
                W238 = np.zeros((16, 44))
                for k_prime in range(4):
                    for r_prime in range(4):
                        if k_prime == r_prime:
                            W238[k_prime*2 : (k_prime+1)*2, r_prime*11 : (r_prime+1)*11] = W232
                
                # Verify full tensor
                T_full = np.einsum('ir,jr,kr->ijk', U238, V238, W238)
                # Check against M_{2,3,8}
                # T_full[i, j, k] should be 1 if i<2, j<3, k<8
                T_check = np.ones((2, 3, 8))
                if np.allclose(T_full, T_check, atol=1e-6):
                    # Convert to lists
                    return {
                        "U": U238.tolist(),
                        "V": V238.tolist(),
                        "W": W238.tolist()
                    }
                    
    return result
