import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 115

    def build_schoolbook():
        r = n * m * p  # 48
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def build_strassen_42():
        rank = 42
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        def strassen_2x2(U, V, W, rank_offset, a_row, a_col, b_row, b_col, c_row, c_col):
            # M1 = (A11+A22)(B11+B22)
            r = rank_offset
            U[a_row + a_col, r] += 1
            U[a_row+1 + a_col+1, r] += 1
            V[b_row + b_col, r] += 1
            V[b_row+1 + b_col+1, r] += 1
            W[c_row + c_col, r] += 1
            W[c_row+1 + c_col+1, r] += 1
            
            # M2 = (A21+A22)B11
            r = rank_offset + 1
            U[a_row+1 + a_col, r] += 1
            U[a_row+1 + a_col+1, r] += 1
            V[b_row + b_col, r] += 1
            W[c_row+1 + c_col, r] += 1
            
            # M3 = A11(B21-B11)
            r = rank_offset + 2
            U[a_row + a_col, r] += 1
            V[b_row + b_col+1, r] += 1
            V[b_row+1 + b_col+1, r] += -1
            W[c_row + c_col+1, r] += 1
            
            # M4 = A22(B12-B22)
            r = rank_offset + 3
            U[a_row+1 + a_col+1, r] += 1
            V[b_row+1 + b_col, r] += 1
            V[b_row + b_col, r] += -1
            W[c_row+1 + c_col, r] += 1
            
            # M5 = (A11+A12)B22
            r = rank_offset + 4
            U[a_row + a_col, r] += 1
            U[a_row + a_col+1, r] += 1
            V[b_row+1 + b_col+1, r] += 1
            W[c_row + c_col+1, r] += 1
            
            # M6 = (A21-A11)(B11+B12)
            r = rank_offset + 5
            U[a_row+1 + a_col, r] += 1
            U[a_row + a_col, r] += -1
            V[b_row + b_col, r] += 1
            V[b_row + b_col+1, r] += 1
            W[c_row+1 + c_col+1, r] += 1
            
            # M7 = (A12-A22)(B21+B22)
            r = rank_offset + 6
            U[a_row + a_col+1, r] += 1
            U[a_row+1 + a_col+1, r] += -1
            V[b_row+1 + b_col, r] += 1
            V[b_row+1 + b_col+1, r] += 1
            W[c_row + c_col, r] += 1
            
            return rank_offset + 7
        
        rank_offset = 0
        # We have A (2x4) and B (4x6).
        # Split A into two 2x2 blocks: A[:, 0:2] and A[:, 2:4]
        # Split B into two 2x6 blocks: B[0:2, :] and B[2:4, :]
        # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
        
        # The product of a 2x2 and a 2x6 matrix can be done by 3 Strassen calls?
        # No, 2x2 * 2x6 is not directly Strassen.
        # Strassen works for square matrices.
        # 2x2 * 2x2 is rank 7.
        # 2x2 * 2x6 = [2x2*2x2, 2x2*2x2, 2x2*2x2] -> 3 * 7 = 21 rank.
        # So total rank = 21 + 21 = 42.
        
        # Let's construct the 2x2 * 2x6 part.
        # Let A_block be 2x2 (rows 0,1; cols a0, a1)
        # Let B_block be 2x6 (rows b0, b1; cols k0..k5)
        # We want C_block = A_block @ B_block (2x6)
        
        # We can view B_block as three 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6]
        # Then C_block = [A@B0, A@B1, A@B2]
        
        def strassen_2x2_by_2x6(U, V, W, rank_offset, a_row, a_col, b_row, c_row):
            # a_row, a_col: top-left of A block (2x2)
            # b_row: top row of B block (2x6)
            # c_row: top row of C block (2x6)
            
            # We will call strassen_2x2 three times.
            # But wait, Strassen combines A and B.
            # If we do 3 separate Strassen calls, we use 3*7=21 multiplications.
            # Can we do better?
            # The rank of 2x2 * 2x6 is actually known to be 13? No, 2x2*2x2 is 7.
            # 2x2 * 2x4 is 13? (Winograd?)
            # Let's stick to 21 for now, which gives 42 total.
            
            # Block 1: B cols 0,1
            rank_offset = strassen_2x2(U, V, W, rank_offset, a_row, a_col, b_row, 0, c_row, 0)
            # Block 2: B cols 2,3
            rank_offset = strassen_2x2(U, V, W, rank_offset, a_row, a_col, b_row, 2, c_row, 2)
            # Block 3: B cols 4,5
            rank_offset = strassen_2x2(U, V, W, rank_offset, a_row, a_col, b_row, 4, c_row, 4)
            
            return rank_offset

        # A block 1: rows 0,1, cols 0,1
        # B block 1: rows 0,1
        rank_offset = strassen_2x2_by_2x6(U, V, W, rank_offset, 0, 0, 0, 0)
        
        # A block 2: rows 0,1, cols 2,3
        # B block 2: rows 2,3
        rank_offset = strassen_2x2_by_2x6(U, V, W, rank_offset, 0, 2, 2, 0)
        
        return U, V, W

    def tensor_error(U, V, W):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def check_valid(U, V, W, tol=1e-6):
        return tensor_error(U, V, W) < tol

    U42, V42, W42 = build_strassen_42()
    
    if not check_valid(U42, V42, W42):
        U, V, W = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 42
    best_U, best_V, best_W = U42, V42, W42
    
    # Strategy: Alternating Least Squares (ALS) to reduce rank.
    # We fix two factors and solve for the third via linear least squares.
    # This is much faster and more robust than Nelder-Mead for this problem.
    
    def solve_for_factor(fixed1, fixed2, target, n_rows, n_cols, rank):
        """
        Solve X such that sum_r X[:, r] * (fixed1 * fixed2)[:, r] = target
        Here target is a matrix of size n_rows x n_cols? 
        Actually, the tensor equation is:
        sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        
        If we fix V and W, we need to find U.
        For each (i,j), we have:
        sum_r U[i*m+j, r] * (V[j*p+..., r] * W[..., i, r]) = 1
        
        This is not quite a standard matrix multiplication.
        Let's expand the index for U: u_idx = i*m + j.
        The "right hand side" for U[u_idx] is 1 for all (i,j) such that u_idx = i*m+j.
        Wait, the equation is for ALL i,j,k.
        For a fixed i,j, the equation is:
        sum_r U[i*m+j, r] * ( V[j*p+k, r] * W[k*n+i, r] ) = 1  for all k.
        
        This is a system of p equations for each (i,j).
        Let A_uj be a matrix of size p x rank, where A_uj[k, r] = V[j*p+k, r] * W[k*n+i, r].
        Then A_uj @ U[i*m+j, :]^T = 1 (vector of ones).
        
        So for each (i,j), we solve a linear system.
        """
        U_sol = np.zeros((n_rows, rank))
        for i in range(n):
            for j in range(m):
                # Construct A_uj (p x rank)
                A_uj = np.zeros((p, rank))
                for k in range(p):
                    A_uj[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A_uj @ x = ones
                # x is U[i*m+j, :]
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(A_uj, np.ones(p), rcond=None)
                    U_sol[i*m+j, :] = x
                except:
                    U_sol[i*m+j, :] = 0
        return U_sol

    def solve_for_V(V, U, W, rank):
        """
        Fix U and W, solve for V.
        Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        For fixed j,k:
        sum_r V[j*p+k, r] * ( U[i*m+j, r] * W[k*n+i, r] ) = 1 for all i.
        
        Let A_vk be n x rank, where A_vk[i, r] = U[i*m+j, r] * W[k*n+i, r].
        Then A_vk @ V[j*p+k, :]^T = 1 (vector of ones of length n).
        """
        V_sol = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                A_vk = np.zeros((n, rank))
                for i in range(n):
                    A_vk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(A_vk, np.ones(n), rcond=None)
                    V_sol[j*p+k, :] = x
                except:
                    V_sol[j*p+k, :] = 0
        return V_sol

    def solve_for_W(W, U, V, rank):
        """
        Fix U and V, solve for W.
        Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        For fixed k,i:
        sum_r W[k*n+i, r] * ( U[i*m+j, r] * V[j*p+k, r] ) = 1 for all j.
        
        Let A_wk be m x rank, where A_wk[j, r] = U[i*m+j, r] * V[j*p+k, r].
        Then A_wk @ W[k*n+i, :]^T = 1 (vector of ones of length m).
        """
        W_sol = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                A_wk = np.zeros((m, rank))
                for j in range(m):
                    A_wk[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(A_wk, np.ones(m), rcond=None)
                    W_sol[k*n+i, :] = x
                except:
                    W_sol[k*n+i, :] = 0
        return W_sol

    # Try to reduce rank using ALS
    # Start from rank 42, try 41, 40...
    # To speed up, we can try a few ranks.
    
    for target_rank in range(41, 30, -1):
        if time.time() > deadline - 10:
            break
        
        # Initialize from best_U, best_V, best_W truncated
        # To avoid getting stuck in a bad local minimum, we can randomize slightly or use the previous best.
        U_cur = best_U[:, :target_rank].copy()
        V_cur = best_V[:, :target_rank].copy()
        W_cur = best_W[:, :target_rank].copy()
        
        # Run ALS for a few iterations
        for it in range(10):
            if time.time() > deadline - 5:
                break
            U_new = solve_for_factor(U_cur, V_cur, W_cur, n*m, target_rank, target_rank)
            V_new = solve_for_V(V_cur, U_new, W_cur, target_rank)
            W_new = solve_for_W(W_cur, U_new, V_new, target_rank)
            
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            
            err = tensor_error(U_cur, V_cur, W_cur)
            if err < 1e-6:
                # Found a valid decomposition!
                best_rank = target_rank
                best_U, best_V, best_W = U_cur, V_cur, W_cur
                # Try to round to half-integers
                U_r = np.round(U_cur * 2) / 2
                V_r = np.round(V_cur * 2) / 2
                W_r = np.round(W_cur * 2) / 2
                if check_valid(U_r, V_r, W_r, tol=1e-4):
                    best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if best_rank == target_rank:
            break # Found lower rank, stop
    
    # Final check
    if not check_valid(best_U, best_V, best_W, tol=1e-5):
        U, V, W = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
