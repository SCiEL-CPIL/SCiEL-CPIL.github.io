"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def verify_decomposition_fast(U_arr, V_arr, W_arr, n, m, p, tol=1e-9):
        """Fast vectorized verification."""
        # We can iterate over i, j, k and check the sum
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    if abs(np.dot(u_row * v_row, w_row) - 1.0) > tol:
                        return False
        return True

    def build_4x4x6_strassen():
        """
        Rank-77 decomposition for 4x4 * 4x6 -> 4x6.
        Split into 4x4*4x4 (rank 49 via recursive Strassen) and 4x4*4x2 (rank 28 via 4 independent 2x2*2x2 Strassen).
        """
        rank = 77
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen steps for 2x2x2
        # Each step is a tuple of three dicts: alpha, beta, gamma
        # Keys are (i, j) block indices
        strassen_steps_222 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]

        # Part 1: 4x4x4 part (cols 0-48)
        col_start_1 = 0
        
        for s in range(7):
            alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
            for s2 in range(7):
                alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
                
                col = col_start_1 + s * 7 + s2
                
                # Process U (A)
                for (bi, bj), val_a4 in alpha_4.items():
                    for (si, sj), val_a2 in alpha_2.items():
                        r = (bi * 2 + si) * m + (bj * 2 + sj)
                        U[r][col] += val_a4 * val_a2
                        
                # Process V (B1)
                for (bi, bj), val_b4 in beta_4.items():
                    for (si, sj), val_b2 in beta_2.items():
                        r = (bi * 2 + si) * p + (bj * 2 + sj)
                        V[r][col] += val_b4 * val_b2
                        
                # Process W (C1)
                for (bi, bj), val_c4 in gamma_4.items():
                    for (si, sj), val_c2 in gamma_2.items():
                        r = (bj * 2 + sj) * n + (bi * 2 + si)
                        W[r][col] += val_c4 * val_c2

        # Part 2: 4x4x2 part (cols 49-76)
        col_start_2 = 49
        
        mults = [
            ((0,0), (0,0)), # A00, B0 -> C0
            ((0,1), (1,0)), # A01, B1 -> C0
            ((1,0), (0,0)), # A10, B0 -> C1
            ((1,1), (1,0))  # A11, B1 -> C1
        ]
        
        for m_idx, (a_block, b_block) in enumerate(mults):
            c_block = (a_block[0], 0)
            
            for s2 in range(7):
                alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
                col = col_start_2 + m_idx * 7 + s2
                
                # Process U (A)
                for (si, sj), val_a2 in alpha_2.items():
                    r = (a_block[0] * 2 + si) * m + (a_block[1] * 2 + sj)
                    U[r][col] += val_a2
                    
                # Process V (B2)
                for (si, sj), val_b2 in beta_2.items():
                    r = (b_block[0] * 2 + si) * p + (4 + sj)
                    V[r][col] += val_b2
                    
                # Process W (C2)
                for (si, sj), val_c2 in gamma_2.items():
                    r = (4 + sj) * n + (c_block[0] * 2 + si)
                    W[r][col] += val_c2

        return U, V, W

    def reduce_rank(U, V, W, n, m, p, time_budget):
        """Greedy rank reduction using fast verification."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        current_rank = U_arr.shape[1]
        
        if time.time() > time_budget:
            return U_arr, V_arr, W_arr
        
        # Greedy removal
        while time.time() < time_budget:
            removed_any = False
            current_rank = U_arr.shape[1]
            for col in range(current_rank):
                if time.time() > time_budget:
                    break
                
                # Mask out column col
                mask = np.ones(current_rank, dtype=bool)
                mask[col] = False
                
                U_test = U_arr[:, mask]
                V_test = V_arr[:, mask]
                W_test = W_arr[:, mask]
                
                if verify_decomposition_fast(U_test, V_test, W_test, n, m, p):
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    removed_any = True
                    break # Restart the loop with new rank
            
            if not removed_any:
                break
        
        return U_arr, V_arr, W_arr

    # Build the initial rank-77 decomposition
    U, V, W = build_4x4x6_strassen()
    
    # Verify the initial decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if not verify_decomposition_fast(U_arr, V_arr, W_arr, n, m, p):
        # Fallback to schoolbook
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
    
    # Try to reduce the rank
    time_budget = deadline - 10
    U_reduced, V_reduced, W_reduced = reduce_rank(U_arr, V_arr, W_arr, n, m, p, time_budget)
    
    # Final verification
    if verify_decomposition_fast(U_reduced, V_reduced, W_reduced, n, m, p):
        return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
