import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave margin

    # Target: 2x3x6
    # Strategy: Split B (3x6) into B1 (3x3) and B2 (3x3).
    # C = A * B1 + A * B2.
    # This is two independent 2x3x3 multiplications.
    # Rank of 2x3x3 is 9.
    # Total rank = 9 + 9 = 18.
    
    # We need a rank-9 decomposition for 2x3x3.
    # U_block (6x9), V_block (9x9), W_block (6x9)
    # such that sum_r U_block[:,r] V_block[:,r] W_block[:,r] = T_block
    
    # T_block indices:
    # A: (i, j) -> a = i*3 + j (0..5)
    # B: (j, k) -> b = j*3 + k (0..8)
    # C: (i, k) -> c = i*3 + k (0..5)
    
    T_block = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = i * 3 + k
                T_block[a, b, c] = 1.0

    # Try to find rank-9 decomposition
    rank_target = 9
    best_U = None
    best_V = None
    best_W = None
    found = False
    
    # Use a few random restarts with ALS
    # To ensure speed, we limit iterations and restarts
    max_restarts = 10
    max_iters = 20
    
    for r in range(max_restarts):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        U = np.random.randn(6, rank_target)
        V = np.random.randn(9, rank_target)
        W = np.random.randn(6, rank_target)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # ALS Step for U
            # Fix V, W. Solve for U.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each a, we have a linear system?
            # Let's vectorize.
            # T_flat (324,)
            # Unknowns: U (54,)
            # A_u (324, 54)
            # A_u[t_idx, u_idx] = V[b, r] * W[c, r]
            
            A_u = np.zeros((324, 54))
            t_vec = T_block.flatten()
            
            # Precompute indices for speed?
            # a in 0..5, b in 0..8, c in 0..5, r in 0..8
            # t_idx = c*54 + b*6 + a
            # u_idx = r*6 + a
            
            for c in range(6):
                for b in range(9):
                    for a in range(6):
                        t_idx = c * 54 + b * 6 + a
                        for r in range(rank_target):
                            u_idx = r * 6 + a
                            A_u[t_idx, u_idx] = V[b, r] * W[c, r]
                            
            try:
                u_vec, _, _, _ = np.linalg.lstsq(A_u, t_vec, rcond=None)
                U = u_vec.reshape(rank_target, 6).T
            except:
                break
                
            # ALS Step for V
            A_v = np.zeros((324, 81)) # 9*9=81
            for c in range(6):
                for b in range(9):
                    for a in range(6):
                        t_idx = c * 54 + b * 6 + a
                        for r in range(rank_target):
                            v_idx = r * 9 + b
                            A_v[t_idx, v_idx] = U[a, r] * W[c, r]
                            
            try:
                v_vec, _, _, _ = np.linalg.lstsq(A_v, t_vec, rcond=None)
                V = v_vec.reshape(rank_target, 9).T
            except:
                break
                
            # ALS Step for W
            A_w = np.zeros((324, 54)) # 6*9=54
            for c in range(6):
                for b in range(9):
                    for a in range(6):
                        t_idx = c * 54 + b * 6 + a
                        for r in range(rank_target):
                            w_idx = r * 6 + c
                            A_w[t_idx, w_idx] = U[a, r] * V[b, r]
                            
            try:
                w_vec, _, _, _ = np.linalg.lstsq(A_w, t_vec, rcond=None)
                W = w_vec.reshape(rank_target, 6).T
            except:
                break
                
            # Check error
            T_rec = np.zeros((6, 9, 6))
            for r in range(rank_target):
                T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T_rec - T_block)
            
            if err < 1e-5:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded
                T_rec_r = np.zeros((6, 9, 6))
                for r in range(rank_target):
                    T_rec_r += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
                err_r = np.linalg.norm(T_rec_r - T_block)
                
                if err_r < 1e-5:
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
            if err < 1e-5 and not found:
                # Keep best real solution just in case, but we need integer/half
                pass
                
        if found:
            break

    if found:
        # Construct full rank-18 solution
        U_full = np.zeros((6, 18))
        V_full = np.zeros((18, 18))
        W_full = np.zeros((12, 18))
        
        U_full[:, 0:9] = best_U
        U_full[:, 9:18] = best_U
        
        V_full[0:9, 0:9] = best_V
        V_full[9:18, 9:18] = best_V
        
        W_full[0:6, 0:9] = best_W
        W_full[6:12, 9:18] = best_W
        
        # Final verification
        # T_full (6, 18, 12)
        # a in 0..5, b in 0..17, c in 0..11
        # T_full[a, b, c] = sum_r U_full[a, r] V_full[b, r] W_full[c, r]
        
        # Target T_target
        T_target = np.zeros((6, 18, 12))
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    a = i * 3 + j
                    b = j * 6 + k
                    c = i * 6 + k
                    T_target[a, b, c] = 1.0
                    
        T_full = np.zeros((6, 18, 12))
        for r in range(18):
            T_full += np.einsum('a,b,c->abc', U_full[:, r], V_full[:, r], W_full[:, r])
            
        err_full = np.linalg.norm(T_full - T_target)
        
        if err_full < 1e-5:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: Schoolbook Rank 36
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
