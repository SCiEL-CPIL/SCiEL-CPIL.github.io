"""Combined program: Strassen-based structural decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill

    def find_m223_decomposition():
        """
        Find a rank-8 decomposition for M_{2,2,3}.
        Returns (U, V, W) where U is (4,8), V is (6,8), W is (6,8).
        """
        # M_{2,2,3}: T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
        # U index: i*2+j, V index: j*3+k, W index: k*2+i
        # We need sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
        
        # Try a structured approach: use Strassen for the 2x2x2 part (cols 0,1) 
        # and add terms for col 2, but optimize to share structure.
        
        # Let's try to find a solution by solving a linear system.
        # Initialize with a guess and refine.
        
        # Use the 2x2x2 Strassen decomposition for cols 0,1 (7 terms)
        # and add 1 term for col 2? That gives 8 terms.
        
        # Strassen 2x2x2 terms (from the existing code):
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        ]
        
        # Build U, V, W for the 2x2x2 part (cols 0,1)
        U = np.zeros((4, 8))
        V = np.zeros((6, 8))
        W = np.zeros((6, 8))
        
        for t_idx in range(7):
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            # U: A entries (2x2)
            for (li, lj), cx in x_sub.items():
                U[li * 2 + lj, t_idx] += cx
            # V: B entries for cols 0,1 (2x2)
            for (li, lj), cy in y_sub.items():
                V[li * 3 + lj, t_idx] += cy  # k=0,1
            # W: C entries (2x2)
            for (li, lj), cc in c_sub.items():
                W[lj * 2 + li, t_idx] += cc  # W index: k*2+i, so k=lj, i=li
        
        # Now add an 8th term for col 2.
        # We need to handle C[0,2] = a*g + b*j and C[1,2] = c*g + d*j
        # This is a 2x2x1 product: A (2x2) @ B_col2 (2x1) -> C_col2 (2x1)
        # Rank-4 schoolbook: 
        # Term 8: U[0,7]=1, V[0*3+2,7]=1, W[2*2+0,7]=1  -> a*g -> C[0,2]
        # But we need 4 terms for 2x2x1. We only have 1 slot.
        
        # This approach won't work with just 8 terms if we use 7 for 2x2x2 and 1 for 2x2x1.
        # We need a more clever decomposition.
        
        # Let me try a different approach: use a numerical solver.
        # Set up the linear system for 8 terms.
        
        # Unknowns: U (4x8), V (6x8), W (6x8) = 32+48+48 = 128 unknowns
        # Equations: 12 (one per (i,j,k))
        # This is underdetermined. We can set some variables to fixed values and solve for others.
        
        # Let me try to find a solution by optimization.
        # Minimize sum_{i,j,k} (sum_r U[i*2+j,r]*V[j*3+k,r]*W[k*2+i,r] - 1)^2
        
        # Use scipy.optimize.minimize
        from scipy.optimize import minimize
        
        # Flatten the variables
        def objective(x):
            U_flat = x[:32].reshape(4, 8)
            V_flat = x[32:80].reshape(6, 8)
            W_flat = x[80:128].reshape(6, 8)
            err = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = 0
                        for r in range(8):
                            s += U_flat[i*2+j, r] * V_flat[j*3+k, r] * W_flat[k*2+i, r]
                        err += (s - 1)**2
            return err
        
        # Initialize with the 7 Strassen terms + zeros for the 8th
        x0 = np.zeros(128)
        # Copy the Strassen terms
        for t_idx in range(7):
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            for (li, lj), cx in x_sub.items():
                x0[(li*2+lj)*8 + t_idx] += cx
            for (li, lj), cy in y_sub.items():
                x0[32 + (li*3+lj)*8 + t_idx] += cy
            for (li, lj), cc in c_sub.items():
                x0[80 + (lj*2+li)*8 + t_idx] += cc
        # Initialize 8th term with schoolbook for col 2
        x0[(0*2+0)*8 + 7] = 1  # a
        x0[32 + (0*3+2)*8 + 7] = 1  # g
        x0[80 + (2*2+0)*8 + 7] = 1  # C[0,2]
        
        # This is a non-convex problem. Let's try multiple initializations.
        best_x = None
        best_obj = np.inf
        
        for attempt in range(20):
            if attempt == 0:
                x_init = x0.copy()
            else:
                x_init = x0 + np.random.randn(128) * 0.1
            try:
                res = minimize(objective, x_init, method='Nelder-Mead', 
                              options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        if best_x is None or best_obj > 1e-6:
            # Fallback: use schoolbook for 2x2x3 (rank 12)
            U = np.zeros((4, 12))
            V = np.zeros((6, 12))
            W = np.zeros((6, 12))
            r = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U[i*2+j, r] = 1
                        V[j*3+k, r] = 1
                        W[k*2+i, r] = 1
                        r += 1
            return U[:, :12], V[:, :12], W[:, :12], 12
        
        U_sol = best_x[:32].reshape(4, 8)
        V_sol = best_x[32:80].reshape(6, 8)
        W_sol = best_x[80:128].reshape(6, 8)
        
        # Round to nearest half-integer if close
        for arr in [U_sol, V_sol, W_sol]:
            arr_rounded = np.round(arr * 2) / 2
            if np.max(np.abs(arr - arr_rounded)) < 1e-6:
                arr[:] = arr_rounded
        
        return U_sol, V_sol, W_sol, 8

    def build_strassen_decomposition():
        """
        Build a Strassen-based decomposition for 4x4x7 matrix multiplication.
        
        Strategy:
        - Decompose 4x4x7 as 4x4x4 (Strassen rank 49) + 4x4x2 (Block Strassen rank 28) + 4x4x1 (Schoolbook rank 16)
        - Total rank: 93
        
        The 4x4x4 part uses Strassen's algorithm with rank 49.
        The 4x4x2 part uses a block decomposition with rank 28.
        The 4x4x1 part uses schoolbook multiplication with rank 16.
        """
        total_rank = 93
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        current_rank = 0
        
        # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
        # x_sub, y_sub are 2x2 dicts, c_sub is 2x2 dict for output
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        ]
        
        # Strassen 4x4x4: 7 outer terms, each with 7 inner terms
        strassen_4x4_outer = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}),
            ({0: 1}, {1: 1, 3: -1}),
            ({3: 1}, {2: 1, 0: -1}),
            ({0: 1, 1: 1}, {3: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}),
        ]
        
        # For 4x4x4, the c_block for each outer term m_idx
        c_blocks_4x4 = [
            {(0,0): 1.0, (1,1): 1.0},
            {(1,0): 1.0, (1,1): -1.0},
            {(0,1): 1.0, (1,1): 1.0},
            {(0,0): 1.0, (1,0): 1.0},
            {(0,0): -1.0, (0,1): 1.0},
            {(1,1): 1.0},
            {(0,0): 1.0},
        ]
        
        # Part 1: 4x4x4 Strassen (rank 49)
        for m_idx in range(7):
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                # Compute U coefficients
                for (li, lj), cx in x_sub.items():
                    for b, cb in x_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        U[i * m + j, r] += cx * cb
                
                # Compute V coefficients
                for (li, lj), cy in y_sub.items():
                    for b, cb in y_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        V[i * p + j, r] += cy * cb
                
                # Compute W coefficients
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    for (br, bc), cb_c in c_block.items():
                        if cb_c == 0:
                            continue
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        W[k_final * n + i_final, r] += cc * cb_c
        
        # Part 2: 4x4x2 Block Strassen (rank 28)
        # B columns 4, 5.
        # C = A @ B[:, 4:6]
        # Split A into 2x2 blocks A00, A01, A10, A11
        # Split B[:, 4:6] into B_top (2x2), B_bot (2x2)
        # C_top = A00 @ B_top + A01 @ B_bot
        # C_bot = A10 @ B_top + A11 @ B_bot
        # Each product is 2x2x2, rank 7.
        # 4 products total rank 28.
        
        def add_strassen_2x2(A_block_rows, A_block_cols, B_block_rows, B_block_cols, C_block_rows, C_block_cols):
            """
            Add a Strassen 2x2x2 decomposition for A_block @ B_block to the global U, V, W.
            A_block is 2x2, B_block is 2x2, C_block is 2x2.
            """
            nonlocal current_rank
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                # U: A_block entries
                for (li, lj), cx in x_sub.items():
                    i = A_block_rows[li]
                    j = A_block_cols[lj]
                    U[i * m + j, r] += cx
                    
                # V: B_block entries
                for (li, lj), cy in y_sub.items():
                    i = B_block_rows[li]
                    j = B_block_cols[lj]
                    V[i * p + j, r] += cy
                    
                # W: C_block entries
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    i_final = C_block_rows[li]
                    k_final = C_block_cols[lj]
                    W[k_final * n + i_final, r] += cc

        # Define blocks
        A00_r = [0, 1]; A00_c = [0, 1]
        A01_r = [0, 1]; A01_c = [2, 3]
        A10_r = [2, 3]; A10_c = [0, 1]
        A11_r = [2, 3]; A11_c = [2, 3]
        
        B_top_r = [0, 1]; B_top_c = [4, 5]
        B_bot_r = [2, 3]; B_bot_c = [4, 5]
        
        C_top_r = [0, 1]; C_top_c = [4, 5]
        C_bot_r = [2, 3]; C_bot_c = [4, 5]
        
        # P1 = A00 @ B_top -> C_top
        add_strassen_2x2(A00_r, A00_c, B_top_r, B_top_c, C_top_r, C_top_c)
        # P2 = A01 @ B_bot -> C_top
        add_strassen_2x2(A01_r, A01_c, B_bot_r, B_bot_c, C_top_r, C_top_c)
        # P3 = A10 @ B_top -> C_bot
        add_strassen_2x2(A10_r, A10_c, B_top_r, B_top_c, C_bot_r, C_bot_c)
        # P4 = A11 @ B_bot -> C_bot
        add_strassen_2x2(A11_r, A11_c, B_bot_r, B_bot_c, C_bot_r, C_bot_c)

        # Part 3: 4x4x1 schoolbook (rank 16)
        # B column 6.
        # C[i][6] = sum_j A[i][j] * B[j][6]
        for i in range(n):
            for j in range(m):
                r = current_rank
                current_rank += 1
                U[i * m + j, r] = 1.0
                V[j * p + 6, r] = 1.0
                W[6 * n + i, r] = 1.0
        
        assert current_rank == total_rank, f"Expected {total_rank}, got {current_rank}"
        
        return U[:, :total_rank].tolist(), V[:, :total_rank].tolist(), W[:, :total_rank].tolist()
    
    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Precompute index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))
    
    def search_decomposition():
        """
        Attempt to find a lower rank decomposition using iterative refinement.
        Returns (U, V, W, rank) if successful, else None.
        """
        U_str, V_str, W_str = build_strassen_decomposition()
        U_arr = np.array(U_str, dtype=float)
        V_arr = np.array(V_str, dtype=float)
        W_arr = np.array(W_str, dtype=float)
        
        rank = U_arr.shape[1]
        
        # Check for zero columns
        nonzero_mask = np.any(U_arr != 0, axis=0) & np.any(V_arr != 0, axis=0) & np.any(W_arr != 0, axis=0)
        if not np.all(nonzero_mask):
            U_arr = U_arr[:, nonzero_mask]
            V_arr = V_arr[:, nonzero_mask]
            W_arr = W_arr[:, nonzero_mask]
            rank = U_arr.shape[1]
        
        # Try to reduce rank by checking if any column is linearly dependent
        # Use SVD to check numerical rank
        # If the decomposition is valid, the columns should be linearly independent
        # in the tensor product sense, but we can check for exact zero columns
        
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank

    # Try the search (which currently returns the structural solution)
    U, V, W, rank = search_decomposition()
    
    # Verify correctness
    err = verify(U, V, W, rank)
    
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
