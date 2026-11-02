"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    Start with the rank-35 Strassen-based decomposition (5 blocks x 7 Strassen products),
    then attempt rank reduction using improved ALS with better initialization and convergence.
    The 5 blocks are independent in the tensor space, so rank 35 may be optimal,
    but we verify by attempting lower ranks.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Build the rank-35 decomposition first
    U35, V35, W35 = _build_strassen_decomposition(n, m, p)
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(35):
        u = np.array(U35)[:, r]
        v = np.array(V35)[:, r]
        w = np.array(W35)[:, r]
        T += np.einsum('i,j,k->ijk', u, v, w)
    
    best_result = {"U": U35, "V": V35, "W": W35}
    best_rank = 35
    
    # Try ranks from 30 up to 34
    for target_rank in range(30, 35):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_decomposition_als(T, n, m, p, target_rank, 
                                             time_budget=min(20, deadline - time.time() - 5))
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target_rank
                break
    
    return best_result


def _build_strassen_decomposition(n, m, p):
    """Build the rank-35 Strassen-based decomposition."""
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1 (e+h)
        [1, 0, 0, 0],  # P2 (e)
        [0, 1, 0, -1], # P3 (f-h)
        [-1, 0, 1, 0], # P4 (g-e)
        [0, 0, 0, 1],  # P5 (h)
        [1, 1, 0, 0],  # P6 (e+f)
        [0, 0, 1, 1]   # P7 (g+h)
    ]
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    w_contrib[0][0] = 1
    w_contrib[0][3] = 1
    
    w_contrib[1][2] = 1
    w_contrib[1][3] = -1
    
    w_contrib[2][1] = 1
    w_contrib[2][3] = 1
    
    w_contrib[3][0] = 1
    w_contrib[3][2] = 1
    
    w_contrib[4][0] = -1
    w_contrib[4][1] = 1
    
    w_contrib[5][3] = 1
    
    w_contrib[6][0] = 1
    
    col = 0
    for b in range(5):
        for r in range(7):
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            V[2*b][col] = strassen_B[r][0]
            V[2*b+1][col] = strassen_B[r][1]
            V[10+2*b][col] = strassen_B[r][2]
            V[10+2*b+1][col] = strassen_B[r][3]
            
            W[4*b][col] = w_contrib[r][0]
            W[4*b+2][col] = w_contrib[r][1]
            W[4*b+1][col] = w_contrib[r][2]
            W[4*b+3][col] = w_contrib[r][3]
            
            col += 1
            
    return U, V, W


def _try_rank_decomposition_als(T, n, m, p, target_rank, time_budget=20):
    """
    Try to find a rank-target_rank decomposition of tensor T using ALS.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    start_time = time.time()
    
    # Try multiple random initializations with different strategies
    for trial in range(12):
        if time.time() - start_time > time_budget:
            break
        
        # Different initialization strategies
        if trial < 4:
            # Random Gaussian
            np.random.seed(42 + trial)
            U = np.random.randn(dim_u, target_rank)
            V = np.random.randn(dim_v, target_rank)
            W = np.random.randn(dim_w, target_rank)
        elif trial < 8:
            # Sparse random
            np.random.seed(100 + trial)
            U = np.zeros((dim_u, target_rank))
            V = np.zeros((dim_v, target_rank))
            W = np.zeros((dim_w, target_rank))
            for r in range(target_rank):
                U[np.random.randint(dim_u), r] = np.random.choice([-1, 1])
                V[np.random.randint(dim_v), r] = np.random.choice([-1, 1])
                W[np.random.randint(dim_w), r] = np.random.choice([-1, 1])
        else:
            # Structured initialization based on Strassen
            U_init, V_init, W_init = _build_strassen_decomposition(n, m, p)
            U = np.array(U_init)[:, :target_rank]
            V = np.array(V_init)[:, :target_rank]
            W = np.array(W_init)[:, :target_rank]
        
        # Normalize
        for r in range(target_rank):
            norm_u = np.linalg.norm(U[:, r])
            if norm_u > 1e-10:
                U[:, r] /= norm_u
            norm_v = np.linalg.norm(V[:, r])
            if norm_v > 1e-10:
                V[:, r] /= norm_v
            norm_w = np.linalg.norm(W[:, r])
            if norm_w > 1e-10:
                W[:, r] /= norm_w
        
        # ALS iterations
        for iteration in range(300):
            if time.time() - start_time > time_budget:
                break
            
            # Update U
            K = np.einsum('jr,kr->jkr', V, W).reshape(dim_v * dim_w, target_rank)
            KTK = K.T @ K + 1e-10 * np.eye(target_rank)
            
            for i in range(dim_u):
                Ti = T[i, :, :].reshape(-1)
                rhs = K.T @ Ti
                U[i, :] = np.linalg.solve(KTK, rhs)
            
            # Update V
            L = np.einsum('ir,kr->ikr', U, W).reshape(dim_u * dim_w, target_rank)
            LTL = L.T @ L + 1e-10 * np.eye(target_rank)
            
            for j in range(dim_v):
                Tj = T[:, j, :].reshape(-1)
                rhs = L.T @ Tj
                V[j, :] = np.linalg.solve(LTL, rhs)
            
            # Update W
            M = np.einsum('ir,jr->ijr', U, V).reshape(dim_u * dim_v, target_rank)
            MTM = M.T @ M + 1e-10 * np.eye(target_rank)
            
            for k in range(dim_w):
                Tk = T[:, :, k].reshape(-1)
                rhs = M.T @ Tk
                W[k, :] = np.linalg.solve(MTM, rhs)
            
            # Check convergence
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            loss = np.max(np.abs(T - T_est))
            if loss < 1e-10:
                break
        
        # Final check
        T_final = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.max(np.abs(T - T_final)) < 1e-8:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    T_expected = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_expected[u, v, w] = 1
    
    return np.max(np.abs(T - T_expected)) < 1e-8
# EVOLVE-BLOCK-END
