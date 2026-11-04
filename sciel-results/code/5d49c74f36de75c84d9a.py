import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
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
                
    start_time = time.time()
    deadline = start_time + 110

    def compute_error_numpy(U, V, W, r, n, m, p):
        err = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U[a], V[b] * W[c])
                    eq_idx = (i * m + j) * p + k
                    err[eq_idx] = s - 1.0
        return err

    def check_integer_half(U, V, W, r, tol=1e-9):
        if not np.all(np.abs(2*U - np.round(2*U)) < tol): return False
        if not np.all(np.abs(2*V - np.round(2*V)) < tol): return False
        if not np.all(np.abs(2*W - np.round(2*W)) < tol): return False
        return True

    def snap_to_grid(U, V, W):
        U_snap = np.round(2 * U) / 2.0
        V_snap = np.round(2 * V) / 2.0
        W_snap = np.round(2 * W) / 2.0
        return U_snap, V_snap, W_snap

    def verify_solution(U, V, W, r, n, m, p):
        err = compute_error_numpy(U, V, W, r, n, m, p)
        if np.max(np.abs(err)) < 1e-8:
            if check_integer_half(U, V, W, r):
                return True
        return False

    def search_rank(n, m, p, r, time_limit):
        best_err = float('inf')
        best_sol = None
        
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            curr_err = np.sum(compute_error_numpy(U, V, W, r, n, m, p) ** 2)
            
            if curr_err < best_err:
                best_err = curr_err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if curr_err < 1e-10:
                U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                    return U_snap, V_snap, W_snap

            no_improve = 0
            max_iters = 2000
            step_size = 0.5
            
            for _ in range(max_iters):
                if time.time() > time_limit:
                    break
                
                var_type = np.random.randint(3)
                if var_type == 0:
                    a = np.random.randint(n * m)
                    t = np.random.randint(r)
                    old_val = U[a, t]
                    U[a, t] += np.random.choice([-1, 1]) * step_size
                elif var_type == 1:
                    b = np.random.randint(m * p)
                    t = np.random.randint(r)
                    old_val = V[b, t]
                    V[b, t] += np.random.choice([-1, 1]) * step_size
                else:
                    c = np.random.randint(p * n)
                    t = np.random.randint(r)
                    old_val = W[c, t]
                    W[c, t] += np.random.choice([-1, 1]) * step_size
                
                new_err = np.sum(compute_error_numpy(U, V, W, r, n, m, p) ** 2)
                
                if new_err < curr_err:
                    curr_err = new_err
                    no_improve = 0
                    if np.random.rand() < 0.1:
                        step_size *= 1.05
                    if curr_err < 1e-10:
                        U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                        if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                            return U_snap, V_snap, W_snap
                else:
                    if var_type == 0:
                        U[a, t] = old_val
                    elif var_type == 1:
                        V[b, t] = old_val
                    else:
                        W[c, t] = old_val
                    no_improve += 1
                    if no_improve % 20 == 0:
                        step_size *= 0.9
                    if no_improve > 100:
                        break
                        
            if curr_err < 0.5:
                U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                snap_err = np.sum(compute_error_numpy(U_snap, V_snap, W_snap, r, n, m, p) ** 2)
                if snap_err < 1e-8:
                    if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                        return U_snap, V_snap, W_snap

        return None

    # Strategy: Decompose into blocks
    # Block 1: 3x3x3
    # Block 2: 3x3x3
    # Block 3: 3x3x2
    
    # Try to find rank 23 for 3x3x3
    # Try to find rank 17 for 3x3x2
    
    time_per_block = 20
    t_start = time.time()
    
    sol_333_1 = None
    sol_333_2 = None
    sol_332 = None
    
    # Search for 3x3x3 rank 23
    for r in [23, 24, 25, 26, 27]:
        if time.time() - t_start > 40: break
        sol_333_1 = search_rank(3, 3, 3, r, t_start + 20)
        if sol_333_1:
            break
            
    # Search for 3x3x3 rank 23 (second one, can reuse if found)
    if sol_333_1:
        sol_333_2 = sol_333_1
    else:
        for r in [23, 24, 25, 26, 27]:
            if time.time() - t_start > 60: break
            sol_333_2 = search_rank(3, 3, 3, r, t_start + 40)
            if sol_333_2:
                break
                
    # Search for 3x3x2 rank 17
    for r in [17, 18, 19]:
        if time.time() - t_start > 80: break
        sol_332 = search_rank(3, 3, 2, r, t_start + 60)
        if sol_332:
            break
            
    # If any block failed, use schoolbook for that block
    # Schoolbook 3x3x3: rank 27
    # Schoolbook 3x3x2: rank 18
    
    def schoolbook_333():
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = i*3 + j*3 + k # Wait, index mapping
                    # U index: i*3+j
                    # V index: j*3+k
                    # W index: k*3+i
                    # We need a consistent column index
                    # Let's just use a counter
        # Simpler:
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return U, V, W

    def schoolbook_332():
        U = np.zeros((9, 18))
        V = np.zeros((6, 18))
        W = np.zeros((6, 18))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return U, V, W

    if not sol_333_1:
        sol_333_1 = schoolbook_333()
    if not sol_333_2:
        sol_333_2 = schoolbook_333()
    if not sol_332:
        sol_332 = schoolbook_332()

    # Combine blocks
    # Block 1: columns 0..r1-1
    # Block 2: columns r1..r1+r2-1
    # Block 3: columns r1+r2..r1+r2+r3-1
    
    r1 = sol_333_1[0].shape[1]
    r2 = sol_333_2[0].shape[1]
    r3 = sol_332[0].shape[1]
    
    total_rank = r1 + r2 + r3
    
    U_final = np.zeros((9, total_rank))
    V_final = np.zeros((24, total_rank))
    W_final = np.zeros((24, total_rank))
    
    # Block 1: 3x3x3 (columns 0..7 of B)
    # B indices: 0..8. We map B[0..2] to block 1, B[3..5] to block 2, B[6..7] to block 3?
    # Wait, B is 3x8.
    # Block 1 uses B columns 0,1,2.
    # Block 2 uses B columns 3,4,5.
    # Block 3 uses B columns 6,7.
    
    # U is shared? No, U is for A. A is the same for all blocks.
    # But in the tensor decomposition, U, V, W are factor matrices.
    # The tensor is T[i, j, k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t].
    # If we split k into k1, k2, k3, the terms are additive.
    # So we can just concatenate the factor matrices.
    
    # U for block 1: U1 (9 x r1)
    # U for block 2: U2 (9 x r2)
    # U for block 3: U3 (9 x r3)
    # Final U: [U1, U2, U3] (9 x total_rank)
    
    # V for block 1: V1 (9 x r1) -> corresponds to B[0..2]
    # V for block 2: V2 (9 x r2) -> corresponds to B[3..5]
    # V for block 3: V3 (6 x r3) -> corresponds to B[6..7]
    # Final V: (24 x total_rank)
    # Rows 0..8 of V_final correspond to B[0..2] (Block 1)
    # Rows 9..17 of V_final correspond to B[3..5] (Block 2)
    # Rows 18..23 of V_final correspond to B[6..7] (Block 3)
    
    # W for block 1: W1 (9 x r1) -> corresponds to C[0..2] (k=0..2)
    # W for block 2: W2 (9 x r2) -> corresponds to C[3..5] (k=3..5)
    # W for block 3: W3 (6 x r3) -> corresponds to C[6..7] (k=6..7)
    # Final W: (24 x total_rank)
    # Rows 0..8 of W_final correspond to C[0..2] (k=0..2)
    # Rows 9..17 of W_final correspond to C[3..5] (k=3..5)
    # Rows 18..23 of W_final correspond to C[6..7] (k=6..7)
    
    U1, V1, W1 = sol_333_1
    U2, V2, W2 = sol_333_2
    U3, V3, W3 = sol_332
    
    U_final[:, 0:r1] = U1
    U_final[:, r1:r1+r2] = U2
    U_final[:, r1+r2:r1+r2+r3] = U3
    
    V_final[0:9, 0:r1] = V1
    V_final[9:18, r1:r1+r2] = V2
    V_final[18:24, r1+r2:r1+r2+r3] = V3
    
    W_final[0:9, 0:r1] = W1
    W_final[9:18, r1:r1+r2] = W2
    W_final[18:24, r1+r2:r1+r2+r3] = W3
    
    # Verify the combined solution
    if verify_solution(U_final, V_final, W_final, total_rank, n, m, p):
        U_list = [[int(x) if x == int(x) else x for x in row] for row in U_final]
        V_list = [[int(x) if x == int(x) else x for x in row] for row in V_final]
        W_list = [[int(x) if x == int(x) else x for x in row] for row in W_final]
        return {"U": U_list, "V": V_list, "W": W_list}
        
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
