import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor
    # T[a, b, c] where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n)
    # A is n x m, indexed as A[i_a, i_b] with flat index i_a * m + i_b
    # B is m x p, indexed as B[j_a, j_b] with flat index j_a * p + j_b
    # C output follows trace convention: W[k*n + i, r] where k is column of C, i is row of C
    # C[i_c, k_c] = sum_j A[i_c, j] * B[j, k_c]
    # W index: for C element at position (i_c, k_c), the W index is k_c * n + i_c
    
    T = np.zeros((n * m, m * p, p * n))
    for i_c in range(n):
        for j in range(m):
            for k_c in range(p):
                a_idx = i_c * m + j
                b_idx = j * p + k_c
                c_idx = k_c * n + i_c  # trace convention
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Strategy: Use linear algebra to find minimal rank decomposition
    # The tensor rank is the minimum r such that T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
    
    # Key insight: For a fixed choice of U (n*m x r), the problem becomes:
    # For each c, T[:, :, c] must be expressible as sum_r U[:, r] * (v_r[b] * w_r[c])
    # This means: for each c, T[:, :, c] = U * X_c where X_c is r x (m*p) with X_c[r, b] = v_r[b] * w_r[c]
    
    # Alternative approach: Use the fact that we can fix one factor and solve for others
    # Let's use an iterative approach with multiple random starts
    
    def try_decompose(rank, seed=42):
        """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Iterate using ALS
        for it in range(300):
            # Fix V and W, solve for U
            # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # For each (b, c): T[:, b, c] = U * (V[b, :] * W[c, :])
            # So: T_mat = U @ M where M[b, c, r] = V[b, r] * W[c, r]
            M_u = (V.reshape(m * p, 1, rank) * W.reshape(1, p * n, rank)).reshape(m * p * p * n, rank)
            T_mat_u = T.reshape(n * m, m * p * p * n)
            U_new, _, _, _ = np.linalg.lstsq(M_u, T_mat_u.T, rcond=None)
            U_new = U_new.T
            
            # Fix U and W, solve for V
            M_v = (U.reshape(n * m, 1, rank) * W.reshape(1, p * n, rank)).reshape(n * m * p * n, rank)
            T_mat_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            V_new, _, _, _ = np.linalg.lstsq(M_v, T_mat_v.T, rcond=None)
            V_new = V_new.T
            
            # Fix U and V, solve for W
            M_w = (U.reshape(n * m, 1, rank) * V.reshape(1, m * p, rank)).reshape(n * m * m * p, rank)
            T_mat_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            W_new, _, _, _ = np.linalg.lstsq(M_w, T_mat_w.T, rcond=None)
            W_new = W_new.T
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T) / (np.linalg.norm(T) + 1e-15)
            if err < 1e-10:
                return U, V, W
        
        return None
    
    # Try ranks from low to high with multiple random seeds
    # Lower bound: tensor rank >= max of matricization ranks
    # Matricization of T along mode 0: shape (n*m, m*p*p*n)
    # Its rank gives a lower bound on tensor rank
    T_mat0 = T.reshape(n * m, m * p * p * n)
    rank_lb = np.linalg.matrix_rank(T_mat0, tol=1e-10)
    
    # Also check other matricizations
    T_mat1 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    rank_lb1 = np.linalg.matrix_rank(T_mat1, tol=1e-10)
    
    T_mat2 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    rank_lb2 = np.linalg.matrix_rank(T_mat2, tol=1e-10)
    
    rank_lower = max(rank_lb, rank_lb1, rank_lb2)
    
    # Try ranks from lower bound up to 22
    for rank in range(int(rank_lower), 23):
        for seed in range(10):
            result = try_decompose(rank, seed=seed)
            if result is not None:
                U, V, W = result
                # Try to round to half-integers
                U_list = [[int(round(x * 2)) / 2 for x in row] for row in U]
                V_list = [[int(round(x * 2)) / 2 for x in row] for row in V]
                W_list = [[int(round(x * 2)) / 2 for x in row] for row in W]
                
                # Verify with rounded values
                T_recon = np.einsum('ar,br,cr->abc', 
                                    np.array(U_list), np.array(V_list), np.array(W_list))
                if np.allclose(T_recon, T, atol=1e-10):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If rounding doesn't work, try using the exact floating point values
                # but check if they are close to half-integers
                U_close = np.allclose(U, U_list, atol=1e-8)
                V_close = np.allclose(V, V_list, atol=1e-8)
                W_close = np.allclose(W, W_list, atol=1e-8)
                
                if U_close and V_close and W_close:
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: use the known rank-22 decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    return {"U": U, "V": V, "W": W}
