import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_u = n * m  # 20
    dim_v = m * p  # 25
    dim_w = p * n  # 20
    
    # Target Tensor T of shape (dim_u, dim_v, dim_w)
    # T[i, j, k] = 1 if the term U[i] V[j] W[k] corresponds to a valid multiplication term
    # U index: i (0..19) -> (row, col) of A. i = r_a * m + c_a
    # V index: j (0..24) -> (row, col) of B. j = r_b * p + c_b
    # W index: k (0..19) -> (row, col) of C. k = r_c * n + c_c
    # Term A[r_a, c_a] * B[r_b, c_b] contributes to C[r_a, c_b]
    # So we need r_b == c_a (inner dimension match)
    # And W index must correspond to C[r_a, c_b], i.e., k = c_b * n + r_a
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for r_a in range(n):
        for c_a in range(m):
            u_idx = r_a * m + c_a
            for r_b in range(m):
                if r_b != c_a:
                    continue
                for c_b in range(p):
                    v_idx = r_b * p + c_b
                    w_idx = c_b * n + r_a
                    T[u_idx, v_idx, w_idx] = 1.0

    # Try to find a decomposition with rank < 100
    # Start with a rank slightly lower than trivial
    # Trivial is 100.
    # We try ranks from 90 down to 50?
    # Or just one attempt with GD on rank 90?
    
    # Given time constraints, we try a few ranks.
    # If we find a valid one, return it.
    
    ranks_to_try = [99, 98, 97, 96, 95]
    
    for R in ranks_to_try:
        np.random.seed(42)
        U = np.random.randn(dim_u, R) * 0.1
        V = np.random.randn(dim_v, R) * 0.1
        W = np.random.randn(dim_w, R) * 0.1
        
        # Gradient Descent
        lr = 1e-3
        for it in range(200):
            # Forward pass
            # M[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # Using einsum: 'ir,jr,kr->ijk'
            M = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
            
            # Error
            diff = M - T
            err = np.sum(diff * diff)
            
            if err < 1e-8:
                # Check exactness with rounding?
                # The problem requires integer/half-integer entries.
                # GD might not land on half-integers.
                # We might need to round and check.
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                M_int = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int, optimize=True)
                if np.allclose(M_int, T, atol=1e-9):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                
                # If rounding failed, continue or break?
                # If GD found a real solution, rounding might break it.
                # We can try to snap to half-integers more carefully or just return the float solution?
                # The task says "integer/half-integer entries".
                # So we MUST return half-integers.
                # If GD finds a solution, it's likely near integers if the problem is "nice".
                # But matrix multiplication tensors are not necessarily "nice" in that sense for low ranks.
                # However, the trivial solution is integer.
                # If we start near trivial, we might stay near integers.
                
                # Let's try to initialize with the trivial solution and perturb?
                pass

            # Gradients
            # dL/dU[i, r] = 2 * sum_jk diff[i, j, k] V[j, r] W[k, r]
            # dL/dV[j, r] = 2 * sum_ik diff[i, j, k] U[i, r] W[k, r]
            # dL/dW[k, r] = 2 * sum_ij diff[i, j, k] U[i, r] V[j, r]
            
            grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W, optimize=True)
            grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W, optimize=True)
            grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V, optimize=True)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Optional: Project to half-integers every few steps?
            # No, that would stop the gradient flow.
            
            if it % 20 == 0:
                # Check if we are close to half-integers
                U_close = np.round(U * 2) / 2
                if np.allclose(U, U_close, atol=1e-3):
                    pass # Maybe snap?
                    
        # After GD, try to snap
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        M_int = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int, optimize=True)
        if np.allclose(M_int, T, atol=1e-9):
            return {
                "U": U_int.tolist(),
                "V": V_int.tolist(),
                "W": W_int.tolist()
            }

    # Fallback to trivial rank 100
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
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
