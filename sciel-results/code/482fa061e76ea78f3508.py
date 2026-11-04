"""Matrix multiplication 2x3x9: find rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 27, 18
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Strategy: exploit block structure. p=9=3*3, so B has 3 blocks of 3x3.
    # The multiplication C = A*B can be written as:
    # C[:, 3*l+3*(l+1)] = A * B_block_l for l=0,1,2
    # So we need to decompose three 2x3x3 multiplications.
    # If we find a rank-r decomposition for 2x3x3, total rank = 3*r.
    
    # Build the 2x3x3 tensor
    T_block = np.zeros((6, 9, 6))  # (n*m, m*3, 3*n)
    for i in range(n):
        for j in range(m):
            for s in range(3):
                a = i * m + j
                b = j * 3 + s
                c = s * n + i
                T_block[a, b, c] = 1.0
    
    def try_block_decomposition(r_block, max_trials=200):
        """Try to find rank-r_block decomposition of 2x3x3 tensor."""
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            seed = 70000 + r_block * 1000 + trial
            np.random.seed(seed)
            
            V = np.random.randn(9, r_block)
            W = np.random.randn(6, r_block)
            
            # Build M matrix: M[a, r] = sum over b,c of T_block[a,b,c] * V[b,r] * W[c,r]
            # Actually, we need: for each a, sum_r U[a,r] V[b,r] W[c,r] = T_block[a,b,c]
            # So for fixed a: sum_r U[a,r] * (V[:,r] * W[:,r]) = T_block[a,:,:].flatten()
            
            M = np.zeros((54, r_block))  # 54 = 9*6
            for r_idx in range(r_block):
                M[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx]).flatten()
            
            if np.linalg.matrix_rank(M) < r_block:
                continue
            
            U = np.zeros((6, r_block))
            valid = True
            for a in range(6):
                target = T_block[a, :, :].flatten()
                try:
                    sol, residuals, rank_c, sv = np.linalg.lstsq(M, target, rcond=None)
                    U[a, :] = sol
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_approx, T_block, atol=1e-9):
                return U, V, W
        
        return None
    
    # Try to find low-rank block decomposition
    found_block = False
    for r_block in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline:
            break
        
        result = try_block_decomposition(r_block, max_trials=150)
        if result is not None:
            U_block, V_block, W_block = result
            rank_full = 3 * r_block
            
            if rank_full < best_rank:
                # Expand to full 2x3x9
                U_full = np.zeros((nm, rank_full))
                V_full = np.zeros((mp, rank_full))
                W_full = np.zeros((pn, rank_full))
                
                for l in range(3):
                    for r_idx in range(r_block):
                        R = l * r_block + r_idx
                        U_full[:, R] = U_block[:, r_idx]
                        for j in range(3):
                            for s in range(3):
                                k = 3*l + s
                                V_full[j*9 + k, R] = V_block[j*3 + s, r_idx]
                        for s in range(3):
                            for i in range(2):
                                k = 3*l + s
                                W_full[k*2 + i, R] = W_block[s*2 + i, r_idx]
                
                if verify_solution(U_full, V_full, W_full, T):
                    best_U, best_V, best_W = U_full, V_full, W_full
                    best_rank = rank_full
                    found_block = True
                    break
            break
    
    # If block approach didn't help, try direct search on full tensor
    if not found_block:
        for r in [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]:
            if time.time() > deadline:
                break
            
            found = False
            for trial in range(30):
                if time.time() > deadline:
                    break
                
                seed = 100000 + r * 1000 + trial
                np.random.seed(seed)
                
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
                
                M = np.zeros((nm * mp, r))
                for r_idx in range(r):
                    M[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx]).flatten()
                
                if np.linalg.matrix_rank(M) < r:
                    continue
                
                U = np.zeros((nm, r))
                valid = True
                for a in range(nm):
                    target = T[a, :, :].flatten()
                    try:
                        sol, residuals, rank_c, sv = np.linalg.lstsq(M, target, rcond=None)
                        U[a, :] = sol
                    except:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                if verify_solution(U, V, W, T):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found = True
                    break
            
            if found:
                break
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

