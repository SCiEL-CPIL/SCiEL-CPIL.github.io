import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition for the 2x2x7 subproblem
    # If successful, use it. Otherwise, fall back to the known 25-rank decomposition.
    
    def try_optimize_227(rank_target):
        # Tensor T[i, j, k] = 1 if j == k else 0 for i in 0..1, j in 0..1, k in 0..6
        # We want U (2x2, R), V (2x7, R), W (1x7, R) such that T = sum U[:, r] V[:, r] W[:, r]
        # Note: W is 1x7, so W[r, k].
        # Actually, the tensor is T[i, j, k].
        # U has shape (2, R). V has shape (2, R). W has shape (1, R) -> just a vector of length R?
        # No, the output is C[i, k].
        # C[i, k] = sum_j A[i, j] B[j, k].
        # Decomposition: C[i, k] = sum_r U[i, r] V[j, r] W[k, r]?
        # Wait, the standard decomposition is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
        # Here i is row of A, j is col of A / row of B, k is col of B.
        # So U is (2, R), V is (2, R), W is (7, R).
        
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(2, rank_target)
        V = np.random.randn(2, rank_target)
        W = np.random.randn(7, rank_target)
        
        # Gradient descent
        # Loss = || T - sum U[:, r] V[:, r] W[:, r] ||^2
        # T is 2x2x7
        
        T = np.zeros((2, 2, 7))
        for i in range(2):
            for j in range(2):
                for k in range(7):
                    if j == k:
                        T[i, j, k] = 1.0
        
        for step in range(1000):
            # Compute current tensor
            C = np.einsum('ir, jr, kr -> ijk', U, V, W)
            error = T - C
            loss = np.sum(error**2)
            
            if loss < 1e-10:
                return U, V, W
            
            # Gradients
            # dL/dU[i, r] = -2 * sum_{j,k} error[i, j, k] V[j, r] W[k, r]
            # dL/dV[j, r] = -2 * sum_{i,k} error[i, j, k] U[i, r] W[k, r]
            # dL/dW[k, r] = -2 * sum_{i,j} error[i, j, k] U[i, r] V[j, r]
            
            dU = -2 * np.einsum('ijk, jr, kr -> ir', error, V, W)
            dV = -2 * np.einsum('ijk, ir, kr -> jr', error, U, W)
            dW = -2 * np.einsum('ijk, ir, jr -> kr', error, U, V)
            
            # Update
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
        return None

    # Attempt optimization
    opt_result = try_optimize_227(24)
    
    if opt_result:
        U_opt, V_opt, W_opt = opt_result
        # Check if we can rationalize
        # For now, just return the integer solution if optimization fails or if we decide to stick to known
        pass

    # Fallback to known rank 39 solution
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    def add_strassen_block(k_start):
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        add_term([0, 4], [b00, b11], [c00, c11])
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1
        
        add_term([3, 0], [b00, b01], [c11])
        U[0][col-1] = -1
        
        add_term([1, 4], [b10, b11], [c00])
        U[4][col-1] = -1

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
