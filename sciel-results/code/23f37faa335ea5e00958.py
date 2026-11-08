import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Known rank-189 solution via Kronecker product of Strassen 2x2 and 3x3 identity
    def get_189_solution():
        # Strassen 2x2 patterns
        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        rank = 189
        U = np.zeros((36, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        
        col_idx = 0
        for I in range(3):
            for J in range(3):
                for K in range(3):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * 2 + pi
                            c = J * 2 + pj
                            U[r * 6 + c, col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * 2 + pi
                            c = K * 2 + pj
                            V[r * 6 + c, col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * 2 + pj
                            i = I * 2 + pi
                            W[k * 6 + i, col_idx] += val
                        col_idx += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(sol, num_trials=3):
        U = np.array(sol['U'], dtype=float)
        V = np.array(sol['V'], dtype=float)
        W = np.array(sol['W'], dtype=float)
        rank = U.shape[1]
        
        for _ in range(num_trials):
            A = np.random.rand(6, 6)
            B = np.random.rand(6, 6)
            C_true = A @ B
            C_calc = np.zeros((6, 6))
            
            # Efficient verification using tensor contraction
            # C[i,k] = sum_{j,r} U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
            for i in range(6):
                for k in range(6):
                    s = 0.0
                    for j in range(6):
                        s += np.dot(U[i*6+j, :], V[j*6+k, :]) * W[k*6+i, :]
                    # Wait, that's not right. Let me fix the contraction.
                    # C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # Let's do it properly
                    s = 0.0
                    for j in range(6):
                        for r in range(rank):
                            s += U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    C_calc[i, k] = s
            
            if not np.allclose(C_calc, C_true, atol=1e-9):
                return False
        return True

    def verify_fast(sol, num_trials=2):
        U = np.array(sol['U'], dtype=float)
        V = np.array(sol['V'], dtype=float)
        W = np.array(sol['W'], dtype=float)
        rank = U.shape[1]
        
        # Reshape for efficient computation
        # U: (6,6,rank), V: (6,6,rank), W: (6,6,rank)
        U_t = U.reshape(6, 6, rank)
        V_t = V.reshape(6, 6, rank)
        W_t = W.reshape(6, 6, rank)
        
        for _ in range(num_trials):
            A = np.random.rand(6, 6)
            B = np.random.rand(6, 6)
            C_true = A @ B
            C_calc = np.zeros((6, 6))
            
            # C[i,k] = sum_{j,r} U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
            for i in range(6):
                for k in range(6):
                    # Sum over j and r
                    # Temp[j,r] = U_t[i,j,r] * V_t[j,k,r]
                    temp = U_t[i, :, :] * V_t[:, k, :]  # shape (6, rank)
                    C_calc[i, k] = np.sum(temp * W_t[k, i, :])
            
            if not np.allclose(C_calc, C_true, atol=1e-9):
                return False
        return True

    # Try to find a lower rank solution using optimization
    def attempt_lower_rank(target_rank, time_budget=10.0):
        """
        Attempt to find a valid decomposition of rank target_rank.
        Uses iterative refinement starting from a random initialization.
        """
        if target_rank >= 189:
            return None
        
        start_time = time.time()
        
        # Initialize with small random values
        np.random.seed(42)
        U = np.random.randn(36, target_rank) * 0.1
        V = np.random.randn(36, target_rank) * 0.1
        W = np.random.randn(36, target_rank) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.01
        decay = 0.99
        
        for iteration in range(5000):
            if time.time() > start_time + time_budget:
                break
            
            # Compute gradient with respect to U, V, W
            # Loss = sum_{i,j,k} (C_true[i,k] - C_calc[i,k])^2
            # C_calc[i,k] = sum_{j,r} U[i*6+j,r] * V[j*6+k,r] * W[k*6+i,r]
            
            A = np.random.rand(6, 6)
            B = np.random.rand(6, 6)
            C_true = A @ B
            C_calc = np.zeros((6, 6))
            
            # Compute C_calc and gradients
            grad_U = np.zeros((36, target_rank))
            grad_V = np.zeros((36, target_rank))
            grad_W = np.zeros((36, target_rank))
            
            loss = 0.0
            
            for i in range(6):
                for k in range(6):
                    diff = 0.0
                    for j in range(6):
                        for r in range(target_rank):
                            val = U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                            diff += val
                            # Gradient contributions
                            grad_U[i*6+j, r] += 2 * (diff - C_true[i,k]) * V[j*6+k, r] * W[k*6+i, r]
                            grad_V[j*6+k, r] += 2 * (diff - C_true[i,k]) * U[i*6+j, r] * W[k*6+i, r]
                            grad_W[k*6+i, r] += 2 * (diff - C_true[i,k]) * U[i*6+j, r] * V[j*6+k, r]
                    C_calc[i, k] = diff
                    loss += (diff - C_true[i,k]) ** 2
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            lr *= decay
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-6:
                break
        
        if best_loss < 1e-6:
            sol = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            if verify_fast(sol):
                return sol
        return None

    # First, get the known valid solution
    sol_189 = get_189_solution()
    
    # Try to find lower rank solutions
    # Known theoretical bounds: rank(6x6) is between 144 and 189
    # Let's try a few target ranks
    target_ranks = [144, 150, 160, 170, 180]
    
    best_sol = sol_189
    best_rank = 189
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        sol = attempt_lower_rank(target_rank, time_budget=10.0)
        if sol is not None:
            best_sol = sol
            best_rank = target_rank
            break
    
    return best_sol
# EVOLVE-BLOCK-END
