import numpy as np
import time
import random

def _verify_333(U, V, W):
    """Verify 3x3x3 decomposition. U,V,W are 9xr."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Target tensor entries for 3x3x3
    # a = i*3+j, b = j*3+k, c = k*3+i
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i*3 + j
                b = j*3 + k
                c = k*3 + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-5:
                    return False
    # Off-diagonal
    for a in range(9):
        i_a = a // 3
        j_a = a % 3
        for b in range(9):
            j_b = b // 3
            k_b = b % 3
            for c in range(9):
                k_c = c // 3
                i_c = c % 3
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-5:
                        return False
    return True

def _find_333_decomposition(rank, max_iter=200, deadline=None):
    """Try to find a rank decomposition for 3x3x3 matrix multiplication."""
    np.random.seed(42) # Fixed seed for reproducibility in attempts, but we will try multiple seeds
    
    T = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i*3 + j
                b = j*3 + k
                c = k*3 + i
                T[a, b, c] = 1.0

    for seed in range(50):
        if deadline and time.time() > deadline - 5:
            break
            
        np.random.seed(seed)
        U = np.random.randn(9, rank)
        V = np.random.randn(9, rank)
        W = np.random.randn(9, rank)
        
        prev_error = float('inf')
        
        for it in range(max_iter):
            if deadline and time.time() > deadline - 5:
                break
                
            # Update U
            for a in range(9):
                y = T[a, :, :].flatten()
                # K shape: 81 x rank
                # K[ (b*9+c), r ] = V[b, r] * W[c, r]
                K = np.zeros((81, rank))
                for b in range(9):
                    for c in range(9):
                        K[b*9+c, :] = V[b, :] * W[c, :]
                # Solve K U[a,:]^T = y
                sol, _, _, _ = np.linalg.lstsq(K, y, rcond=None)
                U[a, :] = sol

            # Update V
            for b in range(9):
                y = T[:, b, :].flatten()
                # L shape: 81 x rank
                # L[ (a*9+c), r ] = U[a, r] * W[c, r]
                L = np.zeros((81, rank))
                for a in range(9):
                    for c in range(9):
                        L[a*9+c, :] = U[a, :] * W[c, :]
                sol, _, _, _ = np.linalg.lstsq(L, y, rcond=None)
                V[b, :] = sol

            # Update W
            for c in range(9):
                y = T[:, :, c].flatten()
                # M shape: 81 x rank
                # M[ (a*9+b), r ] = U[a, r] * V[b, r]
                M = np.zeros((81, rank))
                for a in range(9):
                    for b in range(9):
                        M[a*9+b, :] = U[a, :] * V[b, :]
                sol, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
                W[c, :] = sol
            
            # Calculate error
            error = 0.0
            for a in range(9):
                for b in range(9):
                    for c in range(9):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        error += (val - T[a, b, c])**2
                        
            if abs(prev_error - error) < 1e-10:
                break
            prev_error = error
            
            if _verify_333(U, V, W):
                return U, V, W
                
    return None

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    n, m, p = 3, 3, 6
    
    # Baseline: Schoolbook
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find a better decomposition for 3x3x3
    # If we find rank r, we can get rank 2r for 3x3x6
    for r_333 in [19, 20, 21, 22]:
        if time.time() > deadline - 10:
            break
            
        result = _find_333_decomposition(r_333, max_iter=100, deadline=deadline)
        
        if result is not None:
            U_333, V_333, W_333 = result
            rank_336 = 2 * r_333
            
            U_336 = [[0] * rank_336 for _ in range(9)]
            V_336 = [[0] * rank_336 for _ in range(18)]
            W_336 = [[0] * rank_336 for _ in range(18)]
            
            # Embed first 3x3x3 (for B1)
            for a in range(9):
                for r in range(r_333):
                    U_336[a][r] = U_333[a][r]
                    
            for j in range(3):
                for k in range(3):
                    b_333 = j * 3 + k
                    b_336 = j * 6 + k
                    for r in range(r_333):
                        V_336[b_336][r] = V_333[b_333][r]
                        
            for k in range(3):
                for i in range(3):
                    c_333 = k * 3 + i
                    c_336 = k * 3 + i
                    for r in range(r_333):
                        W_336[c_336][r] = W_333[c_333][r]
                        
            # Embed second 3x3x3 (for B2)
            for a in range(9):
                for r in range(r_333):
                    U_336[a][r_333 + r] = U_333[a][r]
                    
            for j in range(3):
                for k in range(3):
                    b_333 = j * 3 + k
                    b_336 = j * 6 + (k + 3)
                    for r in range(r_333):
                        V_336[b_336][r_333 + r] = V_333[b_333][r]
                        
            for k in range(3):
                for i in range(3):
                    c_333 = k * 3 + i
                    c_336 = (k + 3) * 3 + i
                    for r in range(r_333):
                        W_336[c_336][r_333 + r] = W_333[c_333][r]
                        
            # Verify the 3x3x6 decomposition
            # Simple check: verify on diagonal and a few off-diagonals
            valid = True
            U_np = np.array(U_336)
            V_np = np.array(V_336)
            W_np = np.array(W_336)
            
            # Check diagonal
            for i in range(3):
                for j in range(3):
                    for k in range(6):
                        a = i * 3 + j
                        b = j * 6 + k
                        c = k * 3 + i
                        val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                        if abs(val - 1.0) > 1e-5:
                            valid = False
                            break
                if not valid: break
            if not valid: break
            
            # Check off-diagonal (sample)
            for _ in range(100):
                a = random.randint(0, 8)
                b = random.randint(0, 17)
                c = random.randint(0, 17)
                i_a = a // 3
                j_a = a % 3
                j_b = b // 6
                k_b = b % 6
                k_c = c // 3
                i_c = c % 3
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                    if abs(val) > 1e-5:
                        valid = False
                        break
            if not valid: break
            
            if valid:
                best_result = {"U": U_336, "V": V_336, "W": W_336}
                best_rank = rank_336
                break # Found a good one, stop searching for worse ranks? 
                # Actually, lower rank is better. We started with 19. If 19 failed, we try 20.
                # So we break when we find the first valid one (lowest rank).
                
    return best_result

# EVOLVE-BLOCK-END
