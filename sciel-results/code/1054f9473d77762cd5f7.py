import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 2x5 by 5x5 matrix multiplication.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Indices for the tensor
    # T[i, j, k] corresponds to U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need the sum over c to be 1 for all valid i, j, k
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is exact (within tolerance)"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        # Check shapes
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        if not (U.shape[1] == V.shape[1] == W.shape[1]):
            return False
        
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol

    def try_rank(r, n, m, p, time_budget=5.0, num_restarts=20):
        """
        Try to find a decomposition of rank r using random restarts and gradient descent.
        """
        end_time = start_time + time_budget
        
        for restart in range(num_restarts):
            if time.time() > end_time:
                break
            
            # Initialize with random values
            # Using small values to avoid large gradients initially
            U_cur = np.random.randn(n*m, r) * 0.5
            V_cur = np.random.randn(m*p, r) * 0.5
            W_cur = np.random.randn(p*n, r) * 0.5
            
            # Gradient descent
            lr = 0.01
            for iteration in range(100):
                if time.time() > end_time:
                    break
                
                # Compute gradients
                U_grad = np.zeros_like(U_cur)
                V_grad = np.zeros_like(V_cur)
                W_grad = np.zeros_like(W_cur)
                
                # Pre-compute products to speed up
                # For each (i,j,k), we need sum_c U[a,c] V[b,c] W[c_idx,c]
                # Gradient w.r.t U[a,c]: sum_{j,k} (V[b,c] W[c_idx,c]) * err
                # This is tricky to vectorize fully without loops over i,j,k due to index dependencies.
                # Given small sizes (10, 25, 10), loops are acceptable.
                
                for i in range(n):
                    for j in range(m):
                        a_idx = i * m + j
                        for k in range(p):
                            b_idx = j * p + k
                            c_idx = k * n + i
                            
                            # Product vector for this (i,j,k)
                            prod = V_cur[b_idx, :] * W_cur[c_idx, :]
                            val = np.dot(U_cur[a_idx, :], prod)
                            err = val - 1.0
                            
                            U_grad[a_idx, :] += err * prod
                            V_grad[b_idx, :] += err * U_cur[a_idx, :] * W_cur[c_idx, :]
                            W_grad[c_idx, :] += err * U_cur[a_idx, :] * V_cur[b_idx, :]
                
                # Update
                U_cur -= lr * U_grad
                V_cur -= lr * V_grad
                W_cur -= lr * W_grad
                
                # Decay learning rate
                lr *= 0.99
                
                # Check convergence
                # Calculate max error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a_idx = i * m + j
                            b_idx = j * p + k
                            c_idx = k * n + i
                            val = np.dot(U_cur[a_idx, :], V_cur[b_idx, :] * W_cur[c_idx, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-5:
                    break
            
            # Try to round to nearest integer or half-integer
            # The problem statement allows integer/half-integer entries.
            # Rounding to nearest 0.5
            U_rounded = np.round(U_cur * 2) / 2
            V_rounded = np.round(V_cur * 2) / 2
            W_rounded = np.round(W_cur * 2) / 2
            
            # Convert to lists for verification
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from a lower bound up to schoolbook_rank - 1
    # Lower bound from matricization rank is 25.
    # We'll try a range of ranks.
    best_result = None
    
    # We can try ranks in increasing order to find the minimal one.
    # However, finding a solution for low ranks is harder.
    # Let's try ranks from 30 to 49.
    for r in range(30, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        # Allocate more time for lower ranks if they are harder?
        # For now, fixed budget per rank.
        result = try_rank(r, n, m, p, time_budget=2.0, num_restarts=10)
        if result is not None:
            best_result = result
            break  # Found the lowest rank in this range, we're done
    
    # If no lower-rank decomposition found, use schoolbook
    if best_result is None:
        rank_final = schoolbook_rank
        U = [[0] * rank_final for _ in range(n * m)]
        V = [[0] * rank_final for _ in range(m * p)]
        W = [[0] * rank_final for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
