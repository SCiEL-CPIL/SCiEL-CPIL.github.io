import numpy as np
import scipy.optimize as so
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Pre-compute the 48 tensor positions
    tensor_positions = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                tensor_positions.append((a, b, w))
    
    tensor_positions = np.array(tensor_positions)
    pos_a = tensor_positions[:, 0]
    pos_b = tensor_positions[:, 1]
    pos_w = tensor_positions[:, 2]
    
    start_time = time.time()
    deadline = 105.0  # Safety margin
    
    # 1. Construct the known Rank 42 solution
    U42 = [[0] * 42 for _ in range(n * m)]
    V42 = [[0] * 42 for _ in range(m * p)]
    W42 = [[0] * 42 for _ in range(p * n)]
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        U42[a_idx[0]][start_col + 0] += 1
        U42[a_idx[3]][start_col + 0] += 1
        U42[a_idx[2]][start_col + 1] += 1
        U42[a_idx[3]][start_col + 1] += 1
        U42[a_idx[0]][start_col + 2] += 1
        U42[a_idx[3]][start_col + 3] += 1
        U42[a_idx[0]][start_col + 4] += 1
        U42[a_idx[1]][start_col + 4] += 1
        U42[a_idx[2]][start_col + 5] += 1
        U42[a_idx[0]][start_col + 5] -= 1
        U42[a_idx[1]][start_col + 6] += 1
        U42[a_idx[3]][start_col + 6] -= 1
        
        V42[b_idx[0]][start_col + 0] += 1
        V42[b_idx[3]][start_col + 0] += 1
        V42[b_idx[0]][start_col + 1] += 1
        V42[b_idx[1]][start_col + 2] += 1
        V42[b_idx[3]][start_col + 2] -= 1
        V42[b_idx[2]][start_col + 3] += 1
        V42[b_idx[0]][start_col + 3] -= 1
        V42[b_idx[3]][start_col + 4] += 1
        V42[b_idx[0]][start_col + 5] += 1
        V42[b_idx[1]][start_col + 5] += 1
        V42[b_idx[2]][start_col + 6] += 1
        V42[b_idx[3]][start_col + 6] += 1
        
        W42[w_idx[0]][start_col + 0] += 1
        W42[w_idx[0]][start_col + 3] += 1
        W42[w_idx[0]][start_col + 4] -= 1
        W42[w_idx[0]][start_col + 6] += 1
        
        W42[w_idx[1]][start_col + 2] += 1
        W42[w_idx[1]][start_col + 4] += 1
        
        W42[w_idx[2]][start_col + 1] += 1
        W42[w_idx[2]][start_col + 3] += 1
        
        W42[w_idx[3]][start_col + 0] += 1
        W42[w_idx[3]][start_col + 1] -= 1
        W42[w_idx[3]][start_col + 2] += 1
        W42[w_idx[3]][start_col + 5] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    add_strassen_block(a_L, b_3_B, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)

    U42_np = np.array(U42, dtype=float)
    V42_np = np.array(V42, dtype=float)
    W42_np = np.array(W42, dtype=float)

    # 2. Try to find a lower rank solution (41, 40, etc.)
    # Initialize with the rank 42 solution truncated/padded?
    # Better: Use the rank 42 solution to guide initialization.
    # We can try to fit rank r < 42.
    
    best_solution = None
    best_rank = 42
    
    # Try rank 41
    for r in [41, 40]:
        if time.time() > deadline - 10:
            break
            
        # Initialize with random, but maybe biased?
        # Or try to project the rank 42 solution?
        # Simple random restarts with a few attempts
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(r * 100 + attempt)
            # Use small random noise added to a scaled version of the rank 42 solution?
            # Rank 42 matrices are 8x42, 24x42, 12x42.
            # For rank 41, we need 8x41, 24x41, 12x41.
            # We can take the first 41 columns of the rank 42 solution as a start.
            U0 = U42_np[:, :r] + 0.1 * np.random.randn(dim_A, r)
            V0 = V42_np[:, :r] + 0.1 * np.random.randn(dim_B, r)
            W0 = W42_np[:, :r] + 0.1 * np.random.randn(dim_C, r)
            
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def res_func(t, r=r):
                U = t[:dim_A * r].reshape(dim_A, r)
                V = t[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                W = t[dim_A * r + dim_B * r :].reshape(dim_C, r)
                vals = np.einsum('ar,br,wr->r', U[pos_a], V[pos_b], W[pos_w])
                return vals - 1.0

            try:
                res = so.least_squares(res_func, theta0, method='lm', max_nfev=2000)
                if res.success and res.cost < 1e-10:
                    U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                    V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                    W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                    
                    # Round to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    # Verify
                    vals_int = np.einsum('ar,br,wr->r', U_int[pos_a], V_int[pos_b], W_int[pos_w])
                    if np.allclose(vals_int, 1.0, atol=1e-9):
                        best_solution = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        best_rank = r
                        break # Found a valid lower rank, stop searching for this r
            except Exception:
                pass

    if best_solution is not None:
        return best_solution
    else:
        return {"U": U42, "V": V42, "W": W42}
