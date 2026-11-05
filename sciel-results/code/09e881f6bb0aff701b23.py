"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # The tensor constraint: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # U: (n*m) x rank = 9 x rank
    # V: (m*p) x rank = 18 x rank
    # W: (p*n) x rank = 18 x rank
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly satisfies the tensor equation."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.multiply(U[i*m+j], V[j*p+k]), W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_round_and_verify(U, V, W, tol=1e-9):
        """Try rounding to integers or half-integers and verify."""
        # Try integer rounding
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int, tol):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        # Try half-integer rounding
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        if verify(U_half, V_half, W_half, tol):
            return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
        return None
    
    # Strategy: Use known fast matrix multiplication algorithms
    # For 3x3 x 3x3, the best known rank is 23 (Borchers et al. 2005)
    # For 3x3 x 3x6, we can view it as two 3x3x3 multiplications
    # But we can potentially share some structure
    
    # Let's try to construct a solution using the Borchers 3x3x3 algorithm
    # and extend it to handle the 3x6 case
    
    # First, let's try a simple approach: use two independent 3x3x3 decompositions
    # This gives rank = 2 * 23 = 46
    
    # But we can do better by noticing that the two halves share the same A matrix
    # and parts of the B matrix structure
    
    # Let me try a gradient-based optimization approach with multiple restarts
    
    def optimize_rank(rank, n_restarts=50, max_iters=20000, lr=0.05):
        """Optimize to find a valid decomposition of given rank."""
        best_result = None
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(restart * 1000 + rank)
            
            # Initialize with random values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Simple gradient descent on the residual
            target = np.ones((n, m, p))
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual and gradient
                residual = np.zeros((n, m, p))
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # r = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                            u_row = U[i*m+j]
                            v_row = V[j*p+k]
                            w_row = W[k*n+i]
                            
                            prod_uvw = np.multiply(np.multiply(u_row, v_row), w_row)
                            r = np.sum(prod_uvw)
                            residual[i, j, k] = r - 1.0
                            
                            # Gradients
                            # d/dr dU[i*m+j, r'] = V[j*p+k, r'] * W[k*n+i, r']
                            grad_U[i*m+j] += (r - 1.0) * np.multiply(v_row, w_row)
                            grad_V[j*p+k] += (r - 1.0) * np.multiply(u_row, w_row)
                            grad_W[k*n+i] += (r - 1.0) * np.multiply(u_row, v_row)
                
                err = np.sum(residual ** 2)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-12:
                        result = try_round_and_verify(U, V, W, 1e-9)
                        if result is not None:
                            return result
                        # Check if already valid
                        if verify(U, V, W, 1e-9):
                            return U.tolist(), V.tolist(), W.tolist()
                
                # Gradient step
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                if it % 1000 == 999:
                    lr *= 0.95
        
        return None
    
    # Try ranks from a reasonable lower bound
    # Lower bound: at least max(n*p, m*p, n*m) = max(18, 18, 9) = 18
    # But for 3x3x3x3 we know rank >= 23, so for 3x3x3x6 likely >= 23
    
    for rank in range(23, 50):
        if time.time() > deadline - 30:
            break
        
        result = optimize_rank(rank, n_restarts=30, max_iters=15000, lr=0.03)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition (rank = n*m*p = 54)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

