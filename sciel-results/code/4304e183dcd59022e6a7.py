import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    n, m, p = DIMENSIONS
    
    # Build target tensor: T[i,j,j',k,k',i'] = 1 iff i==i', j==j', k==k'
    # Flatten: a = i*3+j (A index), b = j'*3+k (B index), c = k'*3+i (W index = k*n+i)
    T = np.zeros((9, 9, 9), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i  # W[k*n+i]
                T[a, b, c] = 1

    def check(U, V, W):
        """Check exact decomposition."""
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(recon, T)
    
    def get_rank26():
        """Standard Strassen-based rank-26 decomposition."""
        rank = 26
        U = np.zeros((9, rank), dtype=np.int64)
        V = np.zeros((9, rank), dtype=np.int64)
        W = np.zeros((9, rank), dtype=np.int64)
        
        col = 0
        # Strassen for top-left 2x2 block
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1; U[4][col] = 1
        V[0][col] = 1; V[4][col] = 1
        W[0][col] = 1; W[4][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1; W[4][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[0][col] = 1
        V[1][col] = 1; V[4][col] = -1
        W[3][col] = 1; W[4][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[4][col] = 1
        V[3][col] = 1; V[0][col] = -1
        W[0][col] = 1; W[1][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[0][col] = 1; U[1][col] = 1
        V[4][col] = 1
        W[0][col] = -1; W[3][col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[3][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[4][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[1][col] = 1; U[4][col] = -1
        V[3][col] = 1; V[4][col] = 1
        W[0][col] = 1
        col += 1
        
        # Cross terms for the 3rd row/col
        # a b^T (row 2 of A times col 2 of B)
        U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
        U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
        U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
        U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
        
        # A2 b (row 2 of A times col 0,1 of B)
        U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
        U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
        U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
        U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
        
        # a b3 (row 0,1 of A times col 2 of B)
        U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
        U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
        
        # a^T B2 (row 0,1 of A times col 0,1 of B for output row 2)
        U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
        U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
        U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
        U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
        
        # a3 b^T (row 2 of A times col 2 of B for output col 2)
        U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
        U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
        
        # a^T b (row 0,1 of A times col 0,1 of B for output [2,2])
        U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
        U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
        
        # a3 b3
        U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
        
        return U, V, W

    # Verify rank-26
    U26, V26, W26 = get_rank26()
    assert check(U26, V26, W26), "Rank-26 decomposition failed verification"
    
    best_rank = 26
    best_U, best_V, best_W = U26.copy(), V26.copy(), W26.copy()
    
    # Fast error computation using precomputed column contributions
    # recon[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # If we change U[row,col] by delta, the change in recon is:
    #   delta * V[:,col] * W[:,col] (outer product scaled)
    # We can maintain recon incrementally.
    
    def compute_recon(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def fast_local_search(U_init, V_init, W_init, max_iters, time_limit):
        """Local search with incremental error updates."""
        U_c = U_init.astype(np.float64).copy()
        V_c = V_init.astype(np.float64).copy()
        W_c = W_init.astype(np.float64).copy()
        
        recon = compute_recon(U_c, V_c, W_c)
        current_err = int(np.sum((recon - T) ** 2))
        if current_err == 0:
            return U_c.astype(np.int64), V_c.astype(np.int64), W_c.astype(np.int64), 0
        
        best_err = current_err
        best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
        
        rng = np.random.RandomState(42)
        temperature = 5.0
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 3000
        r = U_c.shape[1]
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random element to perturb
            mat_choice = rng.randint(3)
            row = rng.randint(9)
            col = rng.randint(r)
            delta = rng.choice([-1.0, 1.0])
            
            if mat_choice == 0:
                old_val = U_c[row, col]
                U_c[row, col] += delta
                # Update recon: recon[i,j,k] += delta * V_c[j,col] * W_c[k,col] for all i,j,k
                # Only affects entries where i == row
                recon[row, :, :] += delta * np.outer(V_c[:, col], W_c[:, col])
            elif mat_choice == 1:
                old_val = V_c[row, col]
                V_c[row, col] += delta
                recon[:, row, :] += delta * np.outer(U_c[:, col], W_c[:, col])
            else:
                old_val = W_c[row, col]
                W_c[row, col] += delta
                recon[:, :, row] += delta * np.outer(U_c[:, col], V_c[:, col])
            
            new_err = int(np.sum((recon - T) ** 2))
            
            if new_err < current_err or (temperature > 1e-6 and rng.random() < np.exp(-(new_err - current_err) / temperature)):
                current_err = new_err
                no_improve = 0
                if new_err < best_err:
                    best_err = new_err
                    best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
                    if best_err == 0:
                        break
            else:
                # Revert
                if mat_choice == 0:
                    U_c[row, col] = old_val
                    recon[row, :, :] -= delta * np.outer(V_c[:, col], W_c[:, col])
                elif mat_choice == 1:
                    V_c[row, col] = old_val
                    recon[:, row, :] -= delta * np.outer(U_c[:, col], W_c[:, col])
                else:
                    W_c[row, col] = old_val
                    recon[:, :, row] -= delta * np.outer(U_c[:, col], V_c[:, col])
            
            temperature *= cooling
            no_improve += 1
            
            if no_improve > max_no_improve and temperature < 0.5:
                break
        
        # Round to nearest integer and verify
        U_int = np.round(U_c).astype(np.int64)
        V_int = np.round(V_c).astype(np.int64)
        W_int = np.round(W_c).astype(np.int64)
        
        # Check if the rounded version is valid
        recon_check = compute_recon(U_int, V_int, W_int)
        if np.array_equal(recon_check, T):
            return U_int, V_int, W_int, 0
        
        # If not exactly zero, use best found
        return best_Ux.astype(np.int64), best_Vx.astype(np.int64), best_Wx.astype(np.int64), best_err

    # Try to find rank 23 by removing 3 columns
    all_combs = list(combinations(range(26), 3))
    
    rng_comb = np.random.RandomState(123)
    # Try all 2600 combinations with fast local search
    # With fast error, we can do more iterations per combination
    
    # First pass: try all combinations with moderate search
    for idx in range(len(all_combs)):
        if time.time() > deadline - 10:
            break
        
        cols_to_remove = all_combs[idx]
        cols_to_keep = [c for c in range(26) if c not in cols_to_remove]
        
        U_init = np.zeros((9, 23), dtype=np.int64)
        V_init = np.zeros((9, 23), dtype=np.int64)
        W_init = np.zeros((9, 23), dtype=np.int64)
        
        for new_idx, old_idx in enumerate(cols_to_keep):
            U_init[:, new_idx] = U26[:, old_idx]
            V_init[:, new_idx] = V26[:, old_idx]
            W_init[:, new_idx] = W26[:, old_idx]
        
        # Quick check: what's the error of just removing these columns?
        recon_quick = compute_recon(U_init.astype(np.float64), V_init.astype(np.float64), W_init.astype(np.float64))
        quick_err = int(np.sum((recon_quick - T) ** 2))
        
        if quick_err == 0:
            # Already perfect without those columns!
            if check(U_init, V_init, W_init):
                best_rank = 23
                best_U, best_V, best_W = U_init, V_init, W_init
                break
            continue
        
        # Only run local search if error is small (meaning we're close)
        if quick_err > 20:
            continue
        
        # Run local search with time budget
        time_per = max(0.1, (deadline - 10 - time.time()) / max(1, len(all_combs) - idx))
        U_sol, V_sol, W_sol, err = fast_local_search(U_init, V_init, W_init, max_iters=2000, time_limit=time.time() + time_per)
        
        if err == 0:
            if check(U_sol, V_sol, W_sol):
                best_rank = 23
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break

    # Final verification
    recon = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
    if not np.array_equal(recon, T):
        # Fallback to rank-26
        best_U, best_V, best_W = U26, V26, W26
        best_rank = 26
    
    # Convert to lists
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
