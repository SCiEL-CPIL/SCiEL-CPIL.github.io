import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    schoolbook_rank = n * m * p
    
    # Schoolbook fallback
    def make_schoolbook():
        U = np.zeros((dim_u, schoolbook_rank))
        V = np.zeros((dim_v, schoolbook_rank))
        W = np.zeros((dim_w, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U0, V0, W0 = make_schoolbook()
    
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Check all entries
        # Efficient check using einsum
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, 1.0, atol=tol)

    def calc_error(U, V, W, r):
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.sum((T - 1.0)**2)

    def local_search(U, V, W, r, time_limit):
        # Local search that allows moves that reduce error, using step size 0.5
        # This respects the half-integer constraint if we start from half-integers
        U_r = U.reshape(n, m, r).copy()
        V_r = V.reshape(m, p, r).copy()
        W_r = W.reshape(p, n, r).copy()
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        E = np.sum((T - 1.0)**2)
        
        if E < 1e-9:
            return U_r.reshape(-1), V_r.reshape(-1), W_r.reshape(-1), E
        
        improved = True
        step_size = 0.5
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Update U
            for i in range(n):
                for j in range(m):
                    for c in range(r):
                        old_val = U_r[i, j, c]
                        best_delta = 0
                        best_E = E
                        best_T_slice = None
                        for delta in [-step_size, step_size]:
                            # T[i,j,k] += delta * V[j,k,c] * W[k,i,c]
                            updates = delta * (V_r[j, :, c] * W_r[:, i, c])
                            T_slice = T[i, j, :]
                            T_new_slice = T_slice + updates
                            E_slice = np.sum((T_slice - 1.0)**2)
                            E_new_slice = np.sum((T_new_slice - 1.0)**2)
                            E_new = E - E_slice + E_new_slice
                            if E_new < best_E:
                                best_E = E_new
                                best_delta = delta
                                best_T_slice = T_new_slice
                        if best_delta != 0:
                            U_r[i, j, c] += best_delta
                            T[i, j, :] = best_T_slice
                            E = best_E
                            improved = True
                            if E < 1e-9:
                                return U_r.reshape(-1), V_r.reshape(-1), W_r.reshape(-1), E
            
            # Update V
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        old_val = V_r[j, k, c]
                        best_delta = 0
                        best_E = E
                        best_T_slice = None
                        for delta in [-step_size, step_size]:
                            # T[i,j,k] += delta * U[i,j,c] * W[k,i,c]
                            updates = delta * (U_r[:, j, c] * W_r[k, :, c])
                            T_slice = T[:, j, k]
                            T_new_slice = T_slice + updates
                            E_slice = np.sum((T_slice - 1.0)**2)
                            E_new_slice = np.sum((T_new_slice - 1.0)**2)
                            E_new = E - E_slice + E_new_slice
                            if E_new < best_E:
                                best_E = E_new
                                best_delta = delta
                                best_T_slice = T_new_slice
                        if best_delta != 0:
                            V_r[j, k, c] += best_delta
                            T[:, j, k] = best_T_slice
                            E = best_E
                            improved = True
                            if E < 1e-9:
                                return U_r.reshape(-1), V_r.reshape(-1), W_r.reshape(-1), E

            # Update W
            for k in range(p):
                for i in range(n):
                    for c in range(r):
                        old_val = W_r[k, i, c]
                        best_delta = 0
                        best_E = E
                        best_T_slice = None
                        for delta in [-step_size, step_size]:
                            # T[i,j,k] += delta * U[i,j,c] * V[j,k,c]
                            updates = delta * (U_r[i, :, c] * V_r[:, k, c])
                            T_slice = T[i, :, k]
                            T_new_slice = T_slice + updates
                            E_slice = np.sum((T_slice - 1.0)**2)
                            E_new_slice = np.sum((T_new_slice - 1.0)**2)
                            E_new = E - E_slice + E_new_slice
                            if E_new < best_E:
                                best_E = E_new
                                best_delta = delta
                                best_T_slice = T_new_slice
                        if best_delta != 0:
                            W_r[k, i, c] += best_delta
                            T[i, :, k] = best_T_slice
                            E = best_E
                            improved = True
                            if E < 1e-9:
                                return U_r.reshape(-1), V_r.reshape(-1), W_r.reshape(-1), E

        return U_r.reshape(-1), V_r.reshape(-1), W_r.reshape(-1), E

    def to_half_int(x):
        return np.round(x * 2) / 2

    best = None
    best_rank = None
    deadline = time.time() + 135
    
    # Strategy:
    # 1. Start from schoolbook rank 60.
    # 2. Try to reduce rank by 1, 2, ... down to a lower bound.
    # 3. For each target rank r, try to find a decomposition.
    #    We can try to initialize from a known higher-rank solution by "merging" columns or just random init + alternating minimization.
    #    Given the difficulty, alternating minimization from random init is the most robust generic approach.
    
    # Lower bound: max(n*m, m*p, p*n) = 20.
    # Let's try ranks from 59 down to 40 first, as lower ranks are harder.
    # Actually, let's try 50, 55, 60 to see if we can beat 60.
    # The prompt says "rank-minimal".
    
    # We will iterate r from 59 down to 40.
    # If we find a solution at r, we record it and continue to r-1.
    # If we fail at r, we might still find a solution at r-1 (unlikely but possible with different seeds/landscape).
    # However, typically if you can't find rank r, you can't find r-1.
    # So we can stop if we fail for a few consecutive ranks? No, let's just use time.
    
    # To maximize chance of beating 60, we should spend most time on the lowest ranks we can plausibly reach.
    # But if we can't reach 50, we return 60.
    
    # Let's prioritize:
    # 1. Check if we can do 55.
    # 2. If yes, check 50.
    # 3. If yes, check 45.
    # 4. If no 55, try 59, 58...
    
    # A better strategy:
    # Try r in [50, 55, 60, 59, 58, 57, 56, 54, 53, 52, 51, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    # But we have limited time.
    
    # Let's try a range of ranks.
    # We'll use alternating minimization with random restarts.
    
    def alternating_minimization(U, V, W, r, max_iters=100, tol=1e-10):
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
            # Fix V, W.
            # For each (i,j), we want sum_c U[i,j,c] * (V[j,:,c] * W[:,i,c]) = 1_k for all k.
            # Let M[i,j] be a matrix of shape (p, r) where M[i,j,k,c] = V[j,k,c] * W[k,i,c].
            # Then M[i,j].T @ U[i,j,:] = ones(p).
            
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            
            # Compute M: (n, m, p, r)
            M = np.einsum('jkc,kic->ijkc', V_r, W_r, optimize=True)
            M_T = np.transpose(M, (0, 1, 3, 2)) # (n, m, r, p)
            
            # Solve for U[i,j,:]
            # M_T[i,j] is (r, p). U[i,j] is (r,).
            # We solve M_T[i,j] @ U[i,j] = ones(p).
            # This is an underdetermined system if r > p. We want min norm solution or just any solution.
            # np.linalg.lstsq gives min norm.
            
            M_flat = M_T.reshape(n*m, r, p)
            U_flat = U_r.reshape(n*m, r)
            b = np.ones((n*m, p))
            
            # Batch least squares?
            # np.linalg.lstsq doesn't do batch.
            # We can use np.linalg.pinv or solve via SVD.
            # Or just loop. n*m = 12. Loop is fine.
            
            for idx in range(n*m):
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M_flat[idx], b[idx], rcond=None)
                    U_flat[idx] = sol
                except:
                    pass
            U = U_flat.reshape(n*m, r)
            
            # Update V
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
            # Fix U, W.
            # For each (j,k), sum_c V[j,k,c] * (U[:,j,c] * W[k,:,c]) = 1_i for all i.
            # M_V[j,k,i,c] = U[i,j,c] * W[k,i,c]. Shape (m, p, n, r).
            # M_V_T[j,k] is (r, n). V[j,k] is (r,).
            # M_V_T[j,k] @ V[j,k] = ones(n).
            
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            
            M_V = np.einsum('ijc,kic->jkic', U_r, W_r, optimize=True)
            M_V_T = np.transpose(M_V, (0, 1, 3, 2)) # (m, p, r, n)
            
            M_V_flat = M_V_T.reshape(m*p, r, n)
            V_flat = V_r.reshape(m*p, r)
            b_V = np.ones((m*p, n))
            
            for idx in range(m*p):
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M_V_flat[idx], b_V[idx], rcond=None)
                    V_flat[idx] = sol
                except:
                    pass
            V = V_flat.reshape(m*p, r)
            
            # Update W
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
            # Fix U, V.
            # For each (k,i), sum_c W[k,i,c] * (U[i,:,c] * V[:,k,c]) = 1_j for all j.
            # M_W[k,i,j,c] = U[i,j,c] * V[j,k,c]. Shape (p, n, m, r).
            # M_W_T[k,i] is (r, m). W[k,i] is (r,).
            # M_W_T[k,i] @ W[k,i] = ones(m).
            
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            
            M_W = np.einsum('ijc,jkc->kimc', U_r, V_r, optimize=True)
            M_W_T = np.transpose(M_W, (0, 1, 3, 2)) # (p, n, r, m)
            
            M_W_flat = M_W_T.reshape(p*n, r, m)
            W_flat = W_r.reshape(p*n, r)
            b_W = np.ones((p*n, m))
            
            for idx in range(p*n):
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M_W_flat[idx], b_W[idx], rcond=None)
                    W_flat[idx] = sol
                except:
                    pass
            W = W_flat.reshape(p*n, r)
            
            # Check error
            if it % 5 == 0:
                err = calc_error(U, V, W, r)
                if err < 1e-6:
                    # Round to half-integers and verify
                    U_h = to_half_int(U)
                    V_h = to_half_int(V)
                    W_h = to_half_int(W)
                    if verify(U_h, V_h, W_h):
                        return U_h, V_h, W_h, True
                    # If rounding broke it, try local search
                    U_l, V_l, W_l, err_l = local_search(U_h, V_h, W_h, r, time.time() + 2)
                    if err_l < 1e-9 and verify(U_l, V_l, W_l):
                        return U_l, V_l, W_l, True
                        
        return U, V, W, False

    # Search loop
    # We want to find the minimal rank.
    # We'll try ranks in decreasing order of likelihood of success?
    # Or just decreasing rank?
    # Let's try 50, 55, 60, 59...
    
    # Actually, if we can find 50, we should try 49, 48...
    # But if we can't find 50, we should try 55, 59...
    
    # Let's define a list of ranks to try.
    # We'll try 50 first. If found, try 49, 48...
    # If 50 fails, try 55. If found, try 54, 53...
    # If 55 fails, try 60 (baseline).
    
    # But we have time limit.
    # Let's just iterate r from 50 to 60?
    # No, we want minimal.
    
    # Let's try r = 50.
    # If successful, record best_rank = 50.
    # Then try r = 49.
    # ...
    # If we run out of time, return best found.
    
    # To be efficient, we can try multiple restarts for each r.
    
    ranks_to_try = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    
    # If we fail all of these, we should try higher ranks like 55, 59, etc.
    # But 60 is guaranteed.
    
    # Let's structure it:
    # 1. Try to beat 60.
    # 2. Candidates: 50-59.
    
    # We'll try r in [50, 55, 59, 58, 57, 56, 54, 53, 52, 51, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    # But this is a lot.
    
    # Let's just try 50, 55, 59.
    # If 50 works, great.
    # If 55 works, great.
    # If 59 works, great.
    
    # Actually, let's just iterate r from 59 down to 40.
    # For each r, do a few restarts.
    # If we find a solution, update best.
    # Continue to lower r.
    # If we find a solution at r, we can stop trying r+1, r+2... (already done).
    # We stop when time runs out.
    
    # To save time, if we fail at r, we might skip r-1? No, landscape is different.
    
    # Let's try 50 first with many restarts.
    
    # Re-order:
    # Try 50. If fail, try 55. If fail, try 59.
    # This covers the "easy" improvements.
    
    # Let's try a sequence of ranks.
    # We will allocate time per rank.
    
    # 50: 30s
    # 55: 30s
    # 59: 30s
    # 58: 10s
    # 57: 10s
    # 56: 10s
    # 54: 5s
    # 53: 5s
    # 52: 5s
    # 51: 5s
    # 49: 5s
    
    # Total 150s.
    
    # Let's just loop r from 50 to 59 and 40 to 49?
    # No, let's be smart.
    
    # We'll try r in [50, 55, 59, 58, 57, 56, 54, 53, 52, 51, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    # But we need to manage time.
    
    # Let's just try 50, 55, 59.
    # If we find one, we return it.
    # Wait, we want MINIMAL.
    # So if we find 55, we should try 50 again?
    # No, if we find 55, we have a valid solution of rank 55.
    # We should then try to find 50.
    
    # So:
    # 1. Try 50. If found, return 50 (or try 49? Unlikely).
    # 2. If 50 fails, try 55. If found, return 55.
    # 3. If 55 fails, try 59. If found, return 59.
    # 4. If 59 fails, return 60.
    
    # This is a greedy approach.
    # Let's implement this.
    
    target_ranks = [50, 55, 59, 58, 57, 56, 54, 53, 52, 51, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    
    # But we want to try 50 first.
    # If 50 fails, we try 55.
    
    # Let's just iterate through a list of candidate ranks in increasing order of "difficulty" (i.e. lower rank first).
    # But if 50 is impossible, we waste time.
    
    # Let's try 55 first? No, 50 is better if possible.
    
    # Let's try 50 with 20 restarts.
    # If fail, try 55 with 20 restarts.
    # If fail, try 59 with 20 restarts.
    # If fail, return 60.
    
    # This seems reasonable.
    
    for r in [50, 55, 59, 58, 57, 56, 54, 53, 52, 51, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 10:
            break
            
        if best_rank is not None and r >= best_rank:
            continue # We are looking for lower rank
            
        found = False
        for restart in range(20):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(restart * 1000 + r)
            
            U = np.random.randn(dim_u, r) * 0.5
            V = np.random.randn(dim_v, r) * 0.5
            W = np.random.randn(dim_w, r) * 0.5
            
            U, V, W, success = alternating_minimization(U, V, W, r, max_iters=100)
            
            if success:
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break # Found solution for this rank, try lower rank
        
        # If we found a solution, we continue to try lower ranks.
        # If we didn't, we continue to next rank.
        
    if best is None:
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
        
    return best

