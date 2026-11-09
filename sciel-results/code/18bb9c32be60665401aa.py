"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 150s kill

    # Schoolbook decomposition
    rank_school = n * m * p  # 84
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = rank_school

    def verify(U_list, V_list, W_list, tol=1e-6):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U_list[0])):
                        s += U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Try to find a lower rank decomposition using structured terms
    # Each term r is defined by vectors a_r (length n), b_r (length m), c_r (length p)
    # such that U[i*m+j, r] = a_r[i] * b_r[j], V[j*p+k, r] = b_r[j] * c_r[k], W[k*n+i, r] = c_r[k] * a_r[i]
    # Then the product is a_r[i] * b_r[j] * b_r[j] * c_r[k] * c_r[k] * a_r[i] = (a_r[i] * c_r[k])^2 * b_r[j]^2
    # Wait, that's not right. Let me re-derive.
    
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # If U[i*m+j, r] = u_r[i,j], V[j*p+k, r] = v_r[j,k], W[k*n+i, r] = w_r[k,i]
    # then sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    
    # For rank-1 terms where u_r[i,j] = a_r[i] * b_r[j], v_r[j,k] = b_r[j] * c_r[k], w_r[k,i] = c_r[k] * a_r[i]:
    # product = a_r[i]*b_r[j]*b_r[j]*c_r[k]*c_r[k]*a_r[i] = (a_r[i]*c_r[k])^2 * b_r[j]^2
    # This doesn't simplify nicely to 1.
    
    # Let's try a different approach: use random search with gradient-like updates
    # or use the fact that we can try specific small ranks.
    
    def try_rank(rank_target, max_attempts=20):
        """Try to find a decomposition of the given rank."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U_np = np.random.uniform(-1, 1, (n * m, rank_target))
            V_np = np.random.uniform(-1, 1, (m * p, rank_target))
            W_np = np.random.uniform(-1, 1, (p * n, rank_target))
            
            # Iterate: for each (i,j,k), adjust the terms to make the sum = 1
            converged = False
            for it in range(500):
                if time.time() > deadline - 5:
                    break
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ui = U_np[i * m + j]
                            vk = V_np[j * p + k]
                            wi = W_np[k * n + i]
                            s = np.dot(ui, np.dot(vk, wi))
                            err = s - 1.0
                            max_err = max(max_err, abs(err))
                            
                            # Simple gradient step: adjust U, V, W to reduce error
                            # dU = err * V * W / (3 * ||V*W||^2) roughly
                            # For simplicity, use a fixed learning rate
                            lr = 0.01
                            U_np[i * m + j] -= lr * err * vk * wi
                            V_np[j * p + k] -= lr * err * ui * wi
                            W_np[k * n + i] -= lr * err * ui * vk
                
                if max_err < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Convert to lists
                U_list = U_np.tolist()
                V_list = V_np.tolist()
                W_list = W_np.tolist()
                if verify(U_list, V_list, W_list, tol=1e-6):
                    return U_list, V_list, W_list
        
        return None
    
    # Try ranks from 83 down to some lower bound
    # Theoretical lower bound for 3x4x7 matrix multiplication is not well-known
    # but it's at least max(3*4, 4*7, 7*3) = max(12, 28, 21) = 28
    # Actually, the rank is at least the dimension of the output, which is 3*7=21
    # Let's try a few ranks below 84
    
    for rank_target in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank_target, max_attempts=5)
        if result is not None:
            U_list, V_list, W_list = result
            best_U = U_list
            best_V = V_list
            best_W = W_list
            best_rank = rank_target
            break  # Found a lower rank, stop
    
    # If no lower rank found, return schoolbook
    if best_rank == rank_school:
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
