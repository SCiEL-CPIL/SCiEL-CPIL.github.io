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
    then attempt rank reduction by exploiting the fact that all blocks share the same A matrix.
    The 35 rank-1 terms may not be linearly independent; we use numerical rank computation
    to find the minimal rank and extract a valid decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Build the rank-35 decomposition first
    U35, V35, W35 = _build_strassen_decomposition(n, m, p)
    
    # Try to reduce rank by finding linear dependencies
    # The tensor is T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # We can view this as: for each r, we have a rank-1 tensor u_r ⊗ v_r ⊗ w_r
    # The question is: what is the minimal number of such terms needed?
    
    # Approach: Build the "tensor matricization" and find its rank
    # T has shape (4, 20, 20). Reshape to (4, 400) for one matricization.
    # But tensor rank is different from matrix rank.
    
    # Better approach: Use the fact that we have a valid rank-35 decomposition.
    # Check if we can do better by looking at the structure.
    
    # Key insight: Since A is shared, the U vectors repeat across blocks.
    # For each Strassen product type r (0..6), we have 5 blocks using the same U vector.
    # This creates structure that might allow rank reduction.
    
    # Let's try a different approach: directly search for a lower-rank decomposition
    # by solving for the minimal rank.
    
    # The tensor T[i,j,k] = sum_{r=1}^{R} U[i,r] V[j,r] W[k,r]
    # For fixed i,j: T[i,j,:] = sum_r (U[i,r] V[j,r]) W[:,r]
    # This means the matrix T[i,j,:] (shape 20) lies in the span of the W vectors.
    
    # Actually, let's use a more direct approach: 
    # Build the full tensor and use numerical methods to find minimal rank.
    
    # Construct the tensor T of shape (n*m, m*p, p*n) = (4, 20, 20)
    T = np.zeros((n*m, m*p, p*n))
    for r in range(35):
        u = np.array(U35)[:, r]
        v = np.array(V35)[:, r]
        w = np.array(W35)[:, r]
        T += np.einsum('i,j,k->ijk', u, v, w)
    
    # Now try to find minimal rank decomposition
    # Use iterative rank reduction: try rank R from lower bound up
    
    # Lower bound: max of matrix ranks of any matricization
    # T as (4, 400): rank <= 4
    # T as (20, 80): rank <= 20  
    # T as (20, 80): rank <= 20
    # So lower bound is at most 20, but tensor rank is typically higher
    
    # For matrix multiplication 2x2 by 2xp, the rank is known to be:
    # For p=1: rank 7 (Strassen for 2x2x2)
    # For p=2: rank 14? or less?
    # For general p: the rank should be related to p
    
    # Let's try to find a decomposition with rank < 35
    # Use a greedy approach or solve via SVD on appropriate matricizations
    
    # Alternative: Since we have time budget, try random initialization + optimization
    # for various target ranks
    
    best_result = {"U": U35, "V": V35, "W": W35}
    best_rank = 35
    
    # Try ranks from 28 up to 34 (skip 35 since we already have it)
    for target_rank in range(28, 35):
        if time.time() > deadline - 30:
            break
        result = _try_rank_decomposition(T, n, m, p, target_rank, time_budget=15)
        if result is not None:
            U_new, V_new, W_new = result
            # Verify the decomposition
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
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


def _try_rank_decomposition(T, n, m, p, target_rank, time_budget=15):
    """
    Try to find a rank-target_rank decomposition of tensor T.
    Uses random initialization + gradient descent on the decomposition parameters.
    """
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    
    # Try multiple random initializations
    for trial in range(5):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = np.random.randn(dim_u, target_rank) * 0.1
        V = np.random.randn(dim_v, target_rank) * 0.1
        W = np.random.randn(dim_w, target_rank) * 0.1
        
        # Gradient descent
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current tensor estimate
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            loss = np.sum((T - T_est)**2)
            
            # Compute gradients
            # dL/dU[i,r] = 2 * sum_{j,k} (T[i,j,k] - T_est[i,j,k]) * V[j,r] * W[k,r]
            # dL/dV[j,r] = 2 * sum_{i,k} (T[i,j,k] - T_est[i,j,k]) * U[i,r] * W[k,r]
            # dL/dW[k,r] = 2 * sum_{i,j} (T[i,j,k] - T_est[i,j,k]) * U[i,r] * V[j,r]
            
            diff = T - T_est
            
            grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if loss < 1e-10:
                break
        
        # Final check
        T_final = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.max(np.abs(T - T_final)) < 1e-8:
            # Convert to list format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return U_list, V_list, W_list
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = len(U[0])
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Compute tensor from decomposition
    T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # Compute expected tensor: T[i,j,k] = sum_l A[i//m, l] * B[l, k//n] * (k%n == i%m ? 1 : 0)
    # Wait, the tensor for matrix multiplication C = AB where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # But our tensor is T[i,j,k] where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    # The relation is: T[i,j,k] = 1 if i = i_A * m + j_A, j = j_B * p + k_B, 
    #                 k = k_C * n + i_C, and j_A == j_B and i_A == i_C
    
    # Let me re-derive:
    # A has indices (i, j) where i in 0..n-1, j in 0..m-1. U index = i*m + j
    # B has indices (j, k) where j in 0..m-1, k in 0..p-1. V index = j*p + k
    # C has indices (i, k) where i in 0..n-1, k in 0..p-1. W index = k*n + i (trace convention)
    
    # The multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # As a tensor: T[u, v, w] = 1 if u = i*m+j, v = j'*p+k', w = k''*n+i'', 
    #                and j == j', i == i'', k == k''
    
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
