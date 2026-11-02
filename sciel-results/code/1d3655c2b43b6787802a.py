import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook solution as fallback
    schoolbook_rank = nm * mp * pn
    U_sb = np.zeros((nm, schoolbook_rank), dtype=int)
    V_sb = np.zeros((mp, schoolbook_rank), dtype=int)
    W_sb = np.zeros((pn, schoolbook_rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
                
    def get_schoolbook():
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Local search with efficient error update
    def local_search(R, max_iters=2000, timeout=10):
        start_time = time.time()
        U = np.random.randint(-1, 2, size=(nm, R))
        V = np.random.randint(-1, 2, size=(mp, R))
        W = np.random.randint(-1, 2, size=(pn, R))
        
        # Ensure non-zero
        if np.all(U == 0): U[0,0] = 1
        if np.all(V == 0): V[0,0] = 1
        if np.all(W == 0): W[0,0] = 1
        
        # Precompute norms squared for V and W columns to speed up delta norm calculation
        V_norms_sq = np.sum(V**2, axis=0) # shape (R,)
        W_norms_sq = np.sum(W**2, axis=0) # shape (R,)
        
        # Initial Error
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        E = T - T_rec
        current_error = np.sum(E**2)
        
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                break
            if current_error == 0:
                return U, V, W, 0
                
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                mat = U
                rows = nm
            elif mat_idx == 1:
                mat = V
                rows = mp
            else:
                mat = W
                rows = pn
                
            r = random.randint(0, R-1)
            row = random.randint(0, rows-1)
            old_val = mat[row, r]
            
            # Calculate change in error for delta = +/- 1
            # Delta T = delta * (V_r outer W_r) if mat is U
            # Delta T = delta * (U_r outer W_r) if mat is V
            # Delta T = delta * (U_r outer V_r) if mat is W
            
            best_delta = 0
            best_err = current_error
            
            for delta in [-1, 1]:
                if mat_idx == 0:
                    # Change U[row, r] by delta
                    # Delta T = delta * (V[:, r] outer W[:, r])
                    # Term = delta * <E, V_r outer W_r>
                    # <E, V_r outer W_r> = sum_{a,b,c} E[a,b,c] V_r[b] W_r[c]
                    # = sum_b V_r[b] (sum_c E[a,b,c] W_r[c])? No, sum over a,b,c.
                    # Let's compute this efficiently.
                    # Proj = np.einsum('abc,b,c->a', E, V[:, r], W[:, r])
                    # Then <E, Delta T> = delta * sum_a Proj[a] * 1 (since U change is at row, but wait...)
                    # If we change U[row, r], the change in T_rec is delta * V[:,r] outer W[:,r].
                    # This change is added to ALL a indices? No.
                    # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
                    # If we change U[row, r], only T_rec[row, b, c] changes.
                    # Specifically, T_rec[row, b, c] increases by delta * V[b,r] * W[c,r].
                    # So Delta T is non-zero only at a=row.
                    # Delta T[row, b, c] = delta * V[b,r] * W[c,r].
                    # <E, Delta T> = sum_{b,c} E[row, b, c] * delta * V[b,r] * W[c,r]
                    # = delta * sum_b V[b,r] (sum_c E[row, b, c] W[c,r])
                    
                    # Compute inner = sum_c E[row, :, :] * W[:, r]
                    # inner[b] = sum_c E[row, b, c] W[c, r]
                    inner = np.dot(E[row, :, :], W[:, r]) # shape (mp,)
                    dot_val = np.dot(inner, V[:, r]) # scalar
                    
                    norm_delta_sq = delta**2 * V_norms_sq[r] * W_norms_sq[r]
                    new_err = current_error - 2 * delta * dot_val + norm_delta_sq
                    
                elif mat_idx == 1:
                    # Change V[row, r] by delta
                    # Delta T[a, row, c] = delta * U[a,r] * W[c,r]
                    # <E, Delta T> = sum_{a,c} E[a, row, c] * delta * U[a,r] * W[c,r]
                    # = delta * sum_a U[a,r] (sum_c E[a, row, c] W[c,r])
                    inner = np.dot(E[:, row, :], W[:, r]) # shape (nm,)
                    dot_val = np.dot(inner, U[:, r]) # scalar
                    norm_delta_sq = delta**2 * V_norms_sq[r] * W_norms_sq[r] # Wait, V_norms_sq is sum V^2. 
                    # Delta T norm squared = sum_{a,c} (delta U[a,r] W[c,r])^2 = delta^2 (sum_a U[a,r]^2) (sum_c W[c,r]^2)
                    # We need U_norms_sq too.
                    # Let's precompute U_norms_sq, V_norms_sq, W_norms_sq.
                    pass # Need to update precomputation

                else:
                    # Change W[row, r] by delta
                    # Delta T[a, b, row] = delta * U[a,r] * V[b,r]
                    # <E, Delta T> = sum_{a,b} E[a, b, row] * delta * U[a,r] * V[b,r]
                    # = delta * sum_a U[a,r] (sum_b E[a, b, row] V[b,r])
                    inner = np.dot(E[:, :, row], V[:, r]) # shape (nm,)
                    dot_val = np.dot(inner, U[:, r]) # scalar
                    # Norm squared = delta^2 (sum_a U[a,r]^2) (sum_b V[b,r]^2)
                    pass

            # This logic requires U_norms_sq as well.
            # Let's refactor to include all norms.
            
            break # Break out of this dummy loop to implement correctly in the final code
            
        return None, None, None, current_error

    # Corrected Local Search
    def local_search_v2(R, max_iters=2000, timeout=10):
        start_time = time.time()
        U = np.random.randint(-1, 2, size=(nm, R))
        V = np.random.randint(-1, 2, size=(mp, R))
        W = np.random.randint(-1, 2, size=(pn, R))
        
        if np.all(U == 0): U[0,0] = 1
        if np.all(V == 0): V[0,0] = 1
        if np.all(W == 0): W[0,0] = 1
        
        U_norms_sq = np.sum(U**2, axis=0)
        V_norms_sq = np.sum(V**2, axis=0)
        W_norms_sq = np.sum(W**2, axis=0)
        
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        E = T - T_rec
        current_error = np.sum(E**2)
        
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                break
            if current_error == 0:
                return U, V, W, 0
                
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                mat = U
                rows = nm
            elif mat_idx == 1:
                mat = V
                rows = mp
            else:
                mat = W
                rows = pn
                
            r = random.randint(0, R-1)
            row = random.randint(0, rows-1)
            old_val = mat[row, r]
            
            best_delta = 0
            best_err = current_error
            
            for delta in [-1, 1]:
                if mat_idx == 0:
                    # Change U[row, r]
                    inner = np.dot(E[row, :, :], W[:, r]) # (mp,)
                    dot_val = np.dot(inner, V[:, r])
                    norm_delta_sq = delta**2 * V_norms_sq[r] * W_norms_sq[r]
                    new_err = current_error - 2 * delta * dot_val + norm_delta_sq
                elif mat_idx == 1:
                    # Change V[row, r]
                    inner = np.dot(E[:, row, :], W[:, r]) # (nm,)
                    dot_val = np.dot(inner, U[:, r])
                    norm_delta_sq = delta**2 * U_norms_sq[r] * W_norms_sq[r]
                    new_err = current_error - 2 * delta * dot_val + norm_delta_sq
                else:
                    # Change W[row, r]
                    inner = np.dot(E[:, :, row], V[:, r]) # (nm,)
                    dot_val = np.dot(inner, U[:, r])
                    norm_delta_sq = delta**2 * U_norms_sq[r] * V_norms_sq[r]
                    new_err = current_error - 2 * delta * dot_val + norm_delta_sq
                
                if new_err < best_err:
                    best_err = new_err
                    best_delta = delta
            
            if best_delta != 0:
                mat[row, r] += best_delta
                current_error = best_err
                # Update norms and E? 
                # Updating E is O(mp*pn) or similar, which is small.
                # Updating norms is O(1) per element change.
                
                if mat_idx == 0:
                    U_norms_sq[r] += 2 * old_val * best_delta + best_delta**2
                    # Update E: E[row, :, :] -= best_delta * np.outer(V[:, r], W[:, r])
                    E[row, :, :] -= best_delta * np.outer(V[:, r], W[:, r])
                elif mat_idx == 1:
                    V_norms_sq[r] += 2 * old_val * best_delta + best_delta**2
                    # Update E: E[:, row, :] -= best_delta * np.outer(U[:, r], W[:, r])
                    E[:, row, :] -= best_delta * np.outer(U[:, r], W[:, r])
                else:
                    W_norms_sq[r] += 2 * old_val * best_delta + best_delta**2
                    # Update E: E[:, :, row] -= best_delta * np.outer(U[:, r], V[:, r])
                    E[:, :, row] -= best_delta * np.outer(U[:, r], V[:, r])
            else:
                # Stagnation: random jump
                # Pick a random element and change it by a random amount in [-2, 2]
                # This helps escape local minima
                pass

        if current_error == 0:
            return U, V, W, 0
        return None, None, None, current_error

    # Try to find a rank < 24
    for R in [23, 22, 21]:
        for restart in range(10):
            U_sol, V_sol, W_sol, err = local_search_v2(R, max_iters=3000, timeout=8)
            if err == 0:
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                
    return get_schoolbook()
