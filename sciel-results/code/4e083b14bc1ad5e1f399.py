import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Reconstruct tensor
    T_rec = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    # Check against original tensor
    T_orig = _build_tensor()
    if not np.allclose(T_rec, T_orig, atol=1e-9):
        return False
    
    # Check integer/half-integer constraint
    for matrix in [U, V, W]:
        for val in matrix:
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Try to find a better rank
    # Known best is 23. We try 23, 24, 25, 26.
    # If we find one, we return it.
    
    T = _build_tensor()
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    best_result = _construct_schoolbook()
    
    # We will search for ranks 23 to 26
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
            
        # Number of restarts depends on time left
        # Let's do a few restarts for each rank
        max_restarts = 50
        restarts_done = 0
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(dim_u, target_rank)
            V = np.random.randn(dim_v, target_rank)
            W = np.random.randn(dim_w, target_rank)
            
            # ALS Iterations
            converged = False
            for it in range(50):
                # Update U
                # T = sum_r u_r v_r^T w_r
                # We want to find U such that T ~= sum_r U[:,r] v_r^T w_r
                # Reshape T to (dim_u, dim_v * dim_w)
                T_mat = T.reshape(dim_u, dim_v * dim_w)
                
                # Build the design matrix for U
                # Column r of design matrix is v_r kron w_r
                # v_r is (dim_v, 1), w_r is (dim_w, 1)
                # v_r kron w_r is (dim_v * dim_w, 1)
                
                # Efficiently:
                # T_mat @ X = U
                # X is (dim_v * dim_w, rank)
                # X[:, r] = v_r * w_r (outer product flattened)
                
                X = np.zeros((dim_v * dim_w, target_rank))
                for r in range(target_rank):
                    X[:, r] = np.kron(V[:, r], W[:, r])
                
                # Solve least squares: T_mat @ X = U
                # U = T_mat @ X (X^T X)^{-1} X^T ? No.
                # Minimize || T_mat - U X^T ||_F
                # This is equivalent to finding U such that U X^T approx T_mat.
                # If X has orthonormal columns, U = T_mat X.
                # Let's use QR decomposition for stability.
                Q, R = np.linalg.qr(X)
                # U = T_mat @ Q
                U_new = T_mat @ Q
                
                # Update V
                # Minimize || T - sum u v w ||
                # Reshape T to (dim_u * dim_w, dim_v)
                T_mat_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                # X_v[:, r] = u_r * w_r
                X_v = np.zeros((dim_u * dim_w, target_rank))
                for r in range(target_rank):
                    X_v[:, r] = np.kron(U_new[:, r], W[:, r])
                Q_v, R_v = np.linalg.qr(X_v)
                V_new = T_mat_v @ Q_v
                
                # Update W
                # Reshape T to (dim_u * dim_v, dim_w)
                T_mat_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                # X_w[:, r] = u_r * v_r
                X_w = np.zeros((dim_u * dim_v, target_rank))
                for r in range(target_rank):
                    X_w[:, r] = np.kron(U_new[:, r], V_new[:, r])
                Q_w, R_w = np.linalg.qr(X_w)
                W_new = T_mat_w @ Q_w
                
                # Check convergence
                diff = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                if diff < 1e-8:
                    converged = True
                    break
            
            # If converged, try to snap to integers/half-integers
            # Check if current solution is close to integers/half-integers
            # Scale by 2 to check integers
            U_int = np.round(U * 2) / 2.0
            V_int = np.round(V * 2) / 2.0
            W_int = np.round(W * 2) / 2.0
            
            # Verify exactness
            if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
            restarts_done += 1
            
    return best_result

# EVOLVE-BLOCK-START
# The function construct is defined above.
# We need to ensure the entry point is correct.
# The prompt asks to keep the function names and signatures.
# The current program has `construct()` inside the EVOLVE BLOCK.
# I will place the helper functions and `construct` inside the block or module level?
# The seed program has `construct` inside the block.
# I will define helpers outside or inside?
# "Keep the function names and signatures of the seed program."
# Seed has `construct`.
# I will define helpers at module level for clarity, and `construct` in the block.
# EVOLVE-BLOCK-END
