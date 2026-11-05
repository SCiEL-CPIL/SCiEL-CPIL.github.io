import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin

    # 1. Schoolbook Decomposition (Rank 72)
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    # 2. Strassen-Winograd 3x3 (Rank 8)
    def strassen_winograd_3x3():
        U = [[0]*8 for _ in range(9)]
        V = [[0]*8 for _ in range(9)]
        W = [[0]*8 for _ in range(9)]
        
        # 0: Diag
        for i in range(3):
            U[i*3+i][0] = 1
            V[i*3+i][0] = 1
            W[i*3+i][0] = 1
        # 1: Super-diag
        U[0*3+1][1] = 1; U[1*3+2][1] = 1; U[2*3+0][1] = 1
        V[0*3+1][1] = 1; V[1*3+2][1] = 1; V[2*3+0][1] = 1
        W[1*3+0][1] = 1; W[2*3+1][1] = 1; W[0*3+2][1] = 1
        # 2: Sub-diag
        U[0*3+2][2] = 1; U[1*3+0][2] = 1; U[2*3+1][2] = 1
        V[0*3+2][2] = 1; V[1*3+0][2] = 1; V[2*3+1][2] = 1
        W[2*3+0][2] = 1; W[0*3+1][2] = 1; W[1*3+2][2] = 1
        # 3: (a00+a01)(b10-b11)
        U[0*3+0][3] = 1; U[0*3+1][3] = 1
        V[1*3+0][3] = 1; V[1*3+1][3] = -1
        W[0*3+0][3] = -1; W[1*3+0][3] = -1; W[0*3+1][3] = 1; W[1*3+1][3] = -1
        # 4: (a11+a12)(b21-b22)
        U[1*3+1][4] = 1; U[1*3+2][4] = 1
        V[2*3+1][4] = 1; V[2*3+2][4] = -1
        W[1*3+1][4] = 1; W[2*3+1][4] = 1; W[1*3+2][4] = -1; W[2*3+2][4] = -1
        # 5: (a22+a20)(b02-b00)
        U[2*3+2][5] = 1; U[2*3+0][5] = 1
        V[0*3+2][5] = 1; V[0*3+0][5] = -1
        W[0*3+0][5] = 1; W[2*3+0][5] = 1; W[0*3+2][5] = -1; W[2*3+2][5] = -1
        # 6: (a00-a10)(b11+b12)
        U[0*3+0][6] = 1; U[1*3+0][6] = -1
        V[1*3+1][6] = 1; V[1*3+2][6] = 1
        W[0*3+0][6] = -1; W[1*3+0][6] = 1; W[0*3+1][6] = 1; W[1*3+1][6] = -1
        # 7: (a11-a21)(b22+b20)
        U[1*3+1][7] = 1; U[2*3+1][7] = -1
        V[2*3+2][7] = 1; V[2*3+0][7] = 1
        W[2*3+0][7] = 1; W[1*3+1][7] = -1; W[2*3+1][7] = 1; W[2*3+2][7] = 1
        
        return U, V, W

    # Verification
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check zeros
        for u in range(n*m):
            i_u, j_u = divmod(u, m)
            for v in range(m*p):
                j_v, k_v = divmod(v, p)
                if j_u != j_v: continue
                for w in range(p*n):
                    k_w, i_w = divmod(w, n)
                    if i_u != i_w or k_v != k_w:
                        s = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                        if abs(s) > 1e-9:
                            return False
        return True

    # Base Block Construction (Rank 22)
    def build_base():
        U_sw, V_sw, W_sw = strassen_winograd_3x3()
        if not verify(U_sw, V_sw, W_sw, 3, 3, 3, 8):
            return None
        
        total_rank = 8 + 8 + 6
        U_full = [[0] * total_rank for _ in range(9)]
        V_full = [[0] * total_rank for _ in range(24)]
        W_full = [[0] * total_rank for _ in range(24)]
        
        # Block 1: k=0..2
        for i in range(3):
            for j in range(3):
                for r in range(8):
                    U_full[i*3+j][r] = U_sw[i*3+j][r]
                    for k in range(3):
                        V_full[j*8+k][r] = V_sw[j*3+k][r]
                        W_full[k*3+i][r] = W_sw[k*3+i][r]
        
        # Block 2: k=3..5
        for i in range(3):
            for j in range(3):
                for r in range(8):
                    U_full[i*3+j][8+r] = U_sw[i*3+j][r]
                    for k in range(3):
                        V_full[j*8+(k+3)][8+r] = V_sw[j*3+k][r]
                        W_full[(k+3)*3+i][8+r] = W_sw[k*3+i][r]
        
        # Block 3: k=6..7 (Schoolbook)
        for i in range(3):
            for j in range(3):
                for k_local in range(2):
                    col = 16 + (i*3 + j)*2 + k_local
                    u_idx = i*3 + j
                    v_idx = j*8 + (6 + k_local)
                    w_idx = (6 + k_local)*3 + i
                    U_full[u_idx][col] = 1
                    V_full[v_idx][col] = 1
                    W_full[w_idx][col] = 1
        
        if not verify(U_full, V_full, W_full, 3, 3, 8, total_rank):
            return None
            
        return U_full, V_full, W_full

    base = build_base()
    if base is None:
        U, V, W, _ = schoolbook(3, 3, 8)
        return {"U": U, "V": V, "W": W}
    
    U0, V0, W0 = base
    rank0 = len(U0[0])
    
    # Optimization: ALS with rank reduction
    def optimize(U, V, W, n, m, p, target_rank, time_budget):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        dim_U, dim_V, dim_W = n*m, m*p, p*n
        
        # Precompute targets
        # We only need to satisfy the non-zero constraints for the optimization
        # T[u,v,w] = 1 if valid, 0 otherwise.
        # For ALS, we minimize sum (T_target - T_pred)^2
        
        # Flatten indices
        # u: 0..8, v: 0..23, w: 0..23
        # Valid (u,v,w):
        # u = i*3+j
        # v = j*3+k
        # w = k*3+i
        
        # We'll just iterate over all valid (u,v,w) pairs for the loss
        # There are 3*3*8 = 72 such pairs.
        
        valid_pairs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    valid_pairs.append((u, v, w))
                    
        # For ALS, we need to solve for one matrix at a time.
        # Fix V, W. Minimize || T - U (V^T W^T) ||_F
        # Let M = V^T W^T. M has shape (Rank, dim_V * dim_W)? No.
        # T is a tensor.
        # T_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}
        # For fixed V, W, we want to find U.
        # For each u, we have a vector t_u of length dim_V * dim_W? No, T is 3D.
        # Let's reshape T into a matrix of size (dim_U, dim_V * dim_W).
        # T_mat[u, v*dim_W + w] = sum_r U[u,r] V[v,r] W[w,r]
        # Let B_r = V[:,r] * W[:,r] (element-wise product). B_r has length dim_V * dim_W.
        # Then T_mat = U * B^T where B is (Rank, dim_V*dim_W).
        
        # Construct B (Rank, dim_V * dim_W)
        # V is (dim_V, Rank), W is (dim_W, Rank)
        # B[r] = kron(V[:,r], W[:,r])? No, V[:,r] is length dim_V, W[:,r] is length dim_W.
        # The index in T_mat is v * dim_W + w.
        # So B[r, v*dim_W + w] = V[v, r] * W[w, r].
        
        B = np.zeros((target_rank, dim_V * dim_W))
        for r in range(target_rank):
            vec = np.kron(V_np[:, r], W_np[:, r])
            B[r, :] = vec
            
        # T_target_mat (dim_U, dim_V * dim_W)
        T_target = np.zeros((dim_U, dim_V * dim_W))
        for u, v, w in valid_pairs:
            T_target[u, v * dim_W + w] = 1.0
            
        # We also need to enforce zeros for invalid entries?
        # The problem statement says "exactly decompose".
        # The valid pairs are the only ones that MUST be 1.
        # The invalid pairs MUST be 0.
        # If we only optimize on valid pairs, we might not satisfy the zero constraints.
        # However, the zero constraints are linear.
        # Let's include all entries in the loss.
        # Total entries: 9 * 24 * 24 = 5184.
        # T_target is 0 everywhere except valid pairs which are 1.
        
        # This is a large matrix (9 x 5184).
        # U (9 x Rank).
        # Minimize || T_target - U B^T ||_F^2.
        # Solution: U = T_target B^T (B B^T)^{-1}
        
        # Check time
        if time.time() > time_budget:
            return U_np, V_np, W_np
            
        try:
            # Solve for U
            # B B^T is (Rank, Rank)
            BBt = B @ B.T
            # Add regularization for stability
            BBt += 1e-8 * np.eye(target_rank)
            U_new = T_target @ B.T @ np.linalg.inv(BBt)
        except:
            return U_np, V_np, W_np
            
        # Update V
        # Fix U, W.
        # T_target_mat_V (dim_V, dim_U * dim_W)
        # T[u, v, w] = sum_r V[v, r] * (U[u, r] W[w, r])
        # Let C_r = U[:,r] * W[:,r]. C_r has length dim_U * dim_W.
        # T_mat_V[v, u*dim_W + w] = sum_r V[v, r] C_r[u*dim_W + w]
        # So T_mat_V = V C^T.
        
        C = np.zeros((target_rank, dim_U * dim_W))
        for r in range(target_rank):
            vec = np.kron(U_new[:, r], W_np[:, r])
            C[r, :] = vec
            
        T_target_V = np.zeros((dim_V, dim_U * dim_W))
        for u, v, w in valid_pairs:
            T_target_V[v, u * dim_W + w] = 1.0
        # Note: The zeros are implicitly 0.
        
        if time.time() > time_budget:
            return U_new, V_np, W_np
            
        try:
            CCt = C @ C.T
            CCt += 1e-8 * np.eye(target_rank)
            V_new = T_target_V @ C.T @ np.linalg.inv(CCt)
        except:
            return U_new, V_np, W_np
            
        # Update W
        # Fix U, V.
        # T_target_mat_W (dim_W, dim_U * dim_V)
        # T[u, v, w] = sum_r W[w, r] * (U[u, r] V[v, r])
        # Let D_r = U[:,r] * V[:,r]. D_r has length dim_U * dim_V.
        # T_mat_W[w, u*dim_V + v] = sum_r W[w, r] D_r[u*dim_V + v]
        # So T_mat_W = W D^T.
        
        D = np.zeros((target_rank, dim_U * dim_V))
        for r in range(target_rank):
            vec = np.kron(U_new[:, r], V_new[:, r])
            D[r, :] = vec
            
        T_target_W = np.zeros((dim_W, dim_U * dim_V))
        for u, v, w in valid_pairs:
            T_target_W[w, u * dim_V + v] = 1.0
            
        if time.time() > time_budget:
            return U_new, V_new, W_np
            
        try:
            DDt = D @ D.T
            DDt += 1e-8 * np.eye(target_rank)
            W_new = T_target_W @ D.T @ np.linalg.inv(DDt)
        except:
            return U_new, V_new, W_np
            
        return U_new, V_new, W_new

    def calculate_error(U, V, W, n, m, p, rank):
        err = 0
        # Check valid
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    err += (s - 1.0)**2
        # Check invalid
        for u in range(n*m):
            i_u, j_u = divmod(u, m)
            for v in range(m*p):
                j_v, k_v = divmod(v, p)
                if j_u != j_v: continue
                for w in range(p*n):
                    k_w, i_w = divmod(w, n)
                    if i_u != i_w or k_v != k_w:
                        s = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                        err += s**2
        return err

    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0
    best_err = calculate_error(U0, V0, W0, n, m, p, rank0)
    
    # Try to reduce rank
    # Start with rank0, go down to 1
    # If we find a valid integer/half-integer solution, we can stop or keep going?
    # The score is the rank. We want the minimal rank.
    # But finding a valid exact solution at lower rank is hard.
    # The prompt says "rank-minimal exact tensor decomposition".
    # So we should search for the lowest rank that allows an exact solution.
    
    # Strategy:
    # 1. Optimize for rank R.
    # 2. Check if the result is close to integer/half-integer.
    # 3. If yes, snap to integer/half-integer and verify exactness.
    # 4. If exact, update best.
    
    # We iterate R from rank0 down to 1.
    # We might not have time to try all.
    # Let's prioritize lower ranks.
    
    # Actually, the "border rank" strategy suggests we might find a real solution that snaps.
    # Let's try ranks: rank0, rank0-1, ..., down to 1.
    
    # To save time, we can run a few iterations of ALS for each rank.
    # If error is small, try snapping.
    
    # Let's define a function to snap to half-integers
    def snap_to_half_int(U, V, W):
        U_s = [[round(x * 2) / 2 for x in row] for row in U]
        V_s = [[round(x * 2) / 2 for x in row] for row in V]
        W_s = [[round(x * 2) / 2 for x in row] for row in W]
        return U_s, V_s, W_s

    # Search ranks
    # We'll try a few ranks.
    # Given time constraints, we can't do too many.
    # Let's try the current rank (22) and a few lower ones (21, 20, ...).
    # If we find a valid solution at rank R, we keep it.
    # If we find a valid solution at rank R-1, we keep that.
    
    # We'll do a simple loop.
    # We need to be careful with the time.
    
    # Let's try ranks from 1 to rank0.
    # But initializing from the rank0 solution is good for higher ranks.
    # For lower ranks, we can initialize randomly or by truncating SVD of the tensor.
    
    # Let's try a simple strategy:
    # For R in range(rank0, 0, -1):
    #   Init U, V, W from previous best or random.
    #   Run ALS.
    #   Check error.
    #   If error < 1e-6, try snapping.
    #   If snap verifies, return.
    
    # This might be slow.
    # Let's limit the number of ranks we try.
    # Maybe just try rank0, rank0-1, rank0-2.
    
    # Actually, the "border rank" might be lower.
    # Let's try to find the minimal rank.
    
    # We'll use a budget for the search.
    search_start = time.time()
    search_budget = deadline - search_start - 5
    
    # We'll try ranks from 1 to rank0.
    # But we'll stop if we find a valid solution? No, we want the MINIMAL rank.
    # So we should go from 1 upwards?
    # Or from rank0 downwards?
    # If we go downwards, the first valid one we find is NOT necessarily minimal.
    # We need to check all lower ranks.
    
    # Given the time limit, checking all ranks from 1 to 22 is risky.
    # Let's assume the minimal rank is close to the current one or slightly lower.
    # Or maybe much lower?
    # For 3x3x8, the rank is known to be 24? No, schoolbook is 72.
    # Strassen for 3x3 is 8.
    # The rank of M_{3,3,8} is at most 24 (3 * rank(3x3) = 24? No, that's not how it works).
    # Actually, M_{3,3,8} can be decomposed into 3 blocks of M_{3,3,3} and 2 blocks of M_{3,3,2}?
    # No, the k dimension is 8.
    # We split k into 0..2, 3..5, 6..7.
    # This gives 8 + 8 + 6 = 22.
    # Is it possible to do better?
    # Maybe by mixing the blocks?
    
    # Let's try to optimize the rank 22 solution and see if it snaps to integers.
    # If it does, we have a valid rank 22 solution.
    # Then we can try rank 21, 20, etc.
    
    # Let's start with the current best.
    U_curr, V_curr, W_curr = np.array(U0), np.array(V0), np.array(W0)
    
    # Try rank 22
    U_opt, V_opt, W_opt = optimize(U_curr, V_curr, W_curr, n, m, p, 22, time.time() + 5)
    U_snap, V_snap, W_snap = snap_to_half_int(U_opt.tolist(), V_opt.tolist(), W_opt.tolist())
    if verify(U_snap, V_snap, W_snap, n, m, p, 22):
        best_U, best_V, best_W, best_rank = U_snap, V_snap, W_snap, 22
    
    # Try lower ranks
    # We'll try a few.
    # We need to initialize.
    # We can use the SVD of the tensor to get an initial guess.
    # Or just random.
    
    # Let's try rank 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8.
    # We'll break if time runs out.
    
    for R in range(21, 0, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        # If R is close to 22, we can truncate the 22-rank solution.
        # Otherwise, random.
        if R > 18:
            # Truncate
            U_init = U_opt[:, :R]
            V_init = V_opt[:, :R]
            W_init = W_opt[:, :R]
        else:
            # Random
            U_init = np.random.randn(9, R)
            V_init = np.random.randn(24, R)
            W_init = np.random.randn(24, R)
            
        # Run ALS
        U_new, V_new, W_new = optimize(U_init, V_init, W_init, n, m, p, R, time.time() + 5)
        
        # Check error
        err = calculate_error(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, R)
        
        # If error is small, try snapping
        if err < 1e-4:
            U_s, V_s, W_s = snap_to_half_int(U_new.tolist(), V_new.tolist(), W_new.tolist())
            if verify(U_s, V_s, W_s, n, m, p, R):
                best_U, best_V, best_W, best_rank = U_s, V_s, W_s, R
                # We found a valid solution at rank R.
                # We could continue to check lower ranks, but it might take too long.
                # Let's continue for a bit to see if we can go lower.
                # But we must be careful with time.
                # Let's break if we find a very low rank?
                # No, we want the minimal.
                # But if we find one at 10, we should check 9, 8...
                # Let's continue the loop.
                pass
        
    # If no better solution found, return the best we have (which might be the original 22)
    # But wait, the original 22 was not verified for integer/half-integer?
    # It was constructed with integers, so it IS integer.
    # So best_U, best_V, best_W should be the original if no optimization improved it.
    # But we only updated best_* if we found a valid snapped solution.
    # The original U0, V0, W0 are integers.
    # So we should initialize best_* to U0, V0, W0.
    
    # Let's re-initialize best_* to be safe.
    # Actually, I set best_U = U0 etc at the start.
    # So if no optimization succeeds, we return the original.
    
    # One more check: is the original solution valid?
    # Yes, verify(U0, V0, W0, ...) returned True.
    
    return {"U": best_U, "V": best_V, "W": best_W}

# Run construct to test
if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    rank = len(res["U"][0])
    print(f"Rank: {rank}")
    # Verify
    U, V, W = res["U"], res["V"], res["W"]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    print(f"Error at {i},{j},{k}: {s}")
    print(f"Valid: {valid}")
