import time
import numpy as np

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
    
    def get_rank(U, V, W):
        """Effective rank: number of columns where all three are non-zero."""
        mask = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
        return int(np.sum(mask))
    
    # F4 field representation approach:
    # Represent 3x3 matrices as 2x2 block matrices over F4 = {0, 1, alpha, alpha+1}
    # where alpha^2 = alpha + 1 in F4.
    # 
    # The key idea: a 3x3 matrix can be decomposed using the F4 field structure.
    # Let's use the representation where each "element" is a pair of bits.
    #
    # Actually, let me use a different approach: direct construction of low-rank decompositions.
    #
    # Known results: The tensor rank of 3x3 matrix multiplication is known to be 23.
    # Let's try to construct a rank-23 decomposition.
    #
    # Strategy: Start with rank-26 and try to reduce via local search.
    
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
    
    def get_rank27():
        """Standard schoolbook rank-27 decomposition."""
        rank = 27
        U = np.zeros((9, rank), dtype=np.int64)
        V = np.zeros((9, rank), dtype=np.int64)
        W = np.zeros((9, rank), dtype=np.int64)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U[i * 3 + j][r] = 1
                    V[j * 3 + k][r] = 1
                    W[k * 3 + i][r] = 1
        
        return U, V, W
    
    # Verify rank-26
    U26, V26, W26 = get_rank26()
    assert check(U26, V26, W26), "Rank-26 decomposition failed verification"
    
    best_rank = 26
    best_U, best_V, best_W = U26.copy(), V26.copy(), W26.copy()
    
    # Local search to reduce rank
    # Strategy: Try to find a rank-r decomposition for r < 26 using simulated annealing
    # with integer/half-integer entries.
    
    rng = np.random.RandomState(12345)
    
    def error(Ux, Vx, Wx):
        """Compute squared error of reconstruction."""
        recon = np.einsum('ir,jr,kr->ijk', Ux.astype(np.int64), Vx.astype(np.int64), Wx.astype(np.int64))
        return int(np.sum((recon - T) ** 2))
    
    def try_reduce_rank(start_rank, target_rank, max_iters, seed_offset=0):
        """Try to reduce from start_rank to target_rank using local search."""
        if time.time() > deadline - 5:
            return None
        
        U_c = best_U.copy()
        V_c = best_V.copy()
        W_c = best_W.copy()
        
        # Initialize by zeroing out random columns beyond target_rank
        # and adjusting to compensate
        for r in range(target_rank, start_rank):
            # Try to absorb column r into other columns
            pass
        
        # Alternative: Start fresh with random rank-target_rank initialization
        # and use simulated annealing
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize with sparse random structure
            U_t = np.zeros((9, target_rank), dtype=np.int64)
            V_t = np.zeros((9, target_rank), dtype=np.int64)
            W_t = np.zeros((9, target_rank), dtype=np.int64)
            
            # Initialize with random ±1 entries (sparse)
            for r in range(target_rank):
                # Each column has a few non-zero entries
                n_u = rng.randint(1, 4)
                n_v = rng.randint(1, 4)
                n_w = rng.randint(1, 4)
                
                u_rows = rng.choice(9, size=n_u, replace=False)
                v_rows = rng.choice(9, size=n_v, replace=False)
                w_rows = rng.choice(9, size=n_w, replace=False)
                
                for row in u_rows:
                    U_t[row, r] = rng.choice([-1, 1])
                for row in v_rows:
                    V_t[row, r] = rng.choice([-1, 1])
                for row in w_rows:
                    W_t[row, r] = rng.choice([-1, 1])
            
            U_c = U_t.copy()
            V_c = V_t.copy()
            W_c = W_t.copy()
            
            current_err = error(U_c, V_c, W_c)
            best_err = current_err
            best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
            
            temperature = 50.0
            cooling = 0.999
            no_improve = 0
            max_no_improve = 500
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                
                # Pick a random element to perturb
                mat_choice = rng.randint(3)
                row = rng.randint(9)
                col = rng.randint(target_rank)
                delta = rng.choice([-1, 1])
                
                if mat_choice == 0:
                    U_c[row, col] += delta
                elif mat_choice == 1:
                    V_c[row, col] += delta
                else:
                    W_c[row, col] += delta
                
                new_err = error(U_c, V_c, W_c)
                
                if new_err < current_err or (temperature > 1e-6 and rng.random() < np.exp(-(new_err - current_err) / temperature)):
                    current_err = new_err
                    no_improve = 0
                    if new_err < best_err:
                        best_err = new_err
                        best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
                else:
                    # Revert
                    if mat_choice == 0:
                        U_c[row, col] -= delta
                    elif mat_choice == 1:
                        V_c[row, col] -= delta
                    else:
                        W_c[row, col] -= delta
                
                temperature *= cooling
                no_improve += 1
                
                if best_err == 0:
                    break
                if no_improve > max_no_improve and temperature < 1.0:
                    break
            
            if best_err == 0:
                # Verify and return
                if check(best_Ux, best_Vx, best_Wx):
                    mask = np.any(best_Ux != 0, axis=0) & np.any(best_Vx != 0, axis=0) & np.any(best_Wx != 0, axis=0)
                    actual_rank = int(np.sum(mask))
                    if actual_rank <= target_rank:
                        return best_Ux[:, mask], best_Vx[:, mask], best_Wx[:, mask], actual_rank
        
        return None
    
    # Try to find rank 23, 24, 25 decompositions
    for target_rank in [23, 24, 25]:
        if time.time() > deadline - 10:
            break
        
        result = try_reduce_rank(26, target_rank, max_iters=2000)
        if result is not None:
            U_r, V_r, W_r, actual_rank = result
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                # Continue trying to go lower
                if target_rank > 23:
                    continue
                else:
                    break
    
    # Final verification
    recon = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
    if not np.array_equal(recon, T):
        # Fallback to rank-26
        U26, V26, W26 = get_rank26()
        best_U, best_V, best_W = U26, V26, W26
        best_rank = 26
    
    # Convert to lists
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
