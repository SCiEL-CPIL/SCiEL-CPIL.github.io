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
    
    def try_reduce_rank(target_rank, max_iterations=100):
        """Try to find a decomposition of the given rank."""
        if time.time() - start_time > deadline - safety_margin:
            return None
        
        # Initialize with schoolbook and randomly remove columns, then optimize
        U_curr = np.zeros((n * m, target_rank))
        V_curr = np.zeros((m * p, target_rank))
        W_curr = np.zeros((p * n, target_rank))
        
        # Initialize with random ±1 values
        U_curr[:, :] = np.random.choice([-1, 1], size=(n * m, target_rank))
        V_curr[:, :] = np.random.choice([-1, 1], size=(m * p, target_rank))
        W_curr[:, :] = np.random.choice([-1, 1], size=(p * n, target_rank))
        
        # Try to fix by adjusting
        for iteration in range(max_iterations):
            if time.time() - start_time > deadline - safety_margin:
                break
            
            # Compute error
            errors = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_curr[i * m + j, :], V_curr[j * p + k, :] * W_curr[k * n + i, :])
                        errors[(i, j, k)] = val - 1.0
            
            max_err = max(abs(v) for v in errors.values())
            if max_err < 1e-9:
                return U_curr.copy(), V_curr.copy(), W_curr.copy()
            
            # Find the worst (i,j,k) and try to fix it
            worst_ijk = max(errors.keys(), key=lambda ikj: abs(errors[ikj]))
            i, j, k = worst_ijk
            err = errors[worst_ijk]
            
            # Try flipping a random element in U, V, or W
            choice = np.random.randint(3)
            if choice == 0:
                idx = np.random.randint(n * m)
                col = np.random.randint(target_rank)
                U_curr[idx, col] *= -1
            elif choice == 1:
                idx = np.random.randint(m * p)
                col = np.random.randint(target_rank)
                V_curr[idx, col] *= -1
            else:
                idx = np.random.randint(p * n)
                col = np.random.randint(target_rank)
                W_curr[idx, col] *= -1
        
        return None
    
    # Try to find a lower rank
    for target_rank in range(60, 72):
        if time.time() - start_time > deadline - safety_margin:
            break
        
        result = try_reduce_rank(target_rank, max_iterations=50)
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
