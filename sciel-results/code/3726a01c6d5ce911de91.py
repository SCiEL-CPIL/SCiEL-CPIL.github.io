"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    start_time = time.time()
    deadline = 150.0
    safety_margin = 15.0
    
    # Schoolbook decomposition
    def schoolbook():
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def check_decomposition(U_r, V_r, W_r, tol=1e-9):
        """Check if U, V, W exactly decompose the multiplication tensor."""
        r = U_r.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_continuous_optimization(target_rank, max_iterations=200, lr=0.05):
        """
        Use gradient descent to find a real-valued decomposition of target_rank.
        Then try to snap to integers/half-integers.
        """
        if time.time() - start_time > deadline - safety_margin:
            return None
        
        # Initialize with small random values
        U_curr = np.random.randn(n * m, target_rank) * 0.1
        V_curr = np.random.randn(m * p, target_rank) * 0.1
        W_curr = np.random.randn(p * n, target_rank) * 0.1
        
        # Precompute indices? No, just loop.
        
        for iteration in range(max_iterations):
            if time.time() - start_time > deadline - safety_margin:
                break
            
            # Calculate errors and gradients
            grad_U = np.zeros_like(U_curr)
            grad_V = np.zeros_like(V_curr)
            grad_W = np.zeros_like(W_curr)
            
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_vec = U_curr[i * m + j, :]
                        v_vec = V_curr[j * p + k, :]
                        w_vec = W_curr[k * n + i, :]
                        
                        val = np.dot(u_vec, v_vec * w_vec)
                        err = val - 1.0
                        loss += err * err
                        
                        # Gradient contributions
                        grad_U[i * m + j, :] += 2 * err * v_vec * w_vec
                        grad_V[j * p + k, :] += 2 * err * u_vec * w_vec
                        grad_W[k * n + i, :] += 2 * err * u_vec * v_vec
            
            # Update
            U_curr -= lr * grad_U
            V_curr -= lr * grad_V
            W_curr -= lr * grad_W
            
            # Check convergence
            if loss < 1e-6:
                break

        # Check if we found a real solution
        if loss >= 1e-6:
            return None

        # Try to snap to integers or half-integers
        # Candidates: nearest 0.5, nearest integer
        snap_candidates = [
            (np.round(U_curr * 2) / 2, np.round(V_curr * 2) / 2, np.round(W_curr * 2) / 2),
            (np.round(U_curr), np.round(V_curr), np.round(W_curr))
        ]
        
        for c_U, c_V, c_W in snap_candidates:
            if check_decomposition(c_U, c_V, c_W, tol=1e-5):
                return c_U.copy(), c_V.copy(), c_W.copy()
            
        return None

    # Try to find a lower rank
    for target_rank in range(71, 59, -1):
        if time.time() - start_time > deadline - safety_margin:
            break
        
        result = try_continuous_optimization(target_rank, max_iterations=200, lr=0.05)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
