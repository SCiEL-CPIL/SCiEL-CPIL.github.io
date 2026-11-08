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
    
    def get_rank(U, V, W):
        """Effective rank: number of columns where all three are non-zero."""
        mask = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
        return int(np.sum(mask))

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
    
    # Targeted search for rank 23
    # Strategy: Remove 3 columns from the rank-26 solution and try to repair the error
    # by adjusting the remaining 23 columns.
    
    def error(Ux, Vx, Wx):
        """Compute squared error of reconstruction."""
        recon = np.einsum('ir,jr,kr->ijk', Ux, Vx, Wx)
        return int(np.sum((recon - T) ** 2))

    def local_search(U_init, V_init, W_init, max_iters, time_limit):
        """Perform local search to minimize error."""
        U_c = U_init.copy()
        V_c = V_init.copy()
        W_c = W_init.copy()
        
        current_err = error(U_c, V_c, W_c)
        if current_err == 0:
            return U_c, V_c, W_c, 0
        
        best_err = current_err
        best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
        
        rng = np.random.RandomState(42)
        temperature = 10.0
        cooling = 0.999
        no_improve = 0
        max_no_improve = 2000
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random element to perturb
            mat_choice = rng.randint(3)
            row = rng.randint(9)
            col = rng.randint(U_c.shape[1])
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
                    if best_err == 0:
                        break
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
            
            if no_improve > max_no_improve and temperature < 1.0:
                break
        
        return best_Ux, best_Vx, best_Wx, best_err

    # Try to find rank 23 by removing 3 columns
    # We try all combinations of 3 columns to remove
    # There are C(26, 3) = 2600 combinations.
    # For each, we run a short local search.
    
    # To save time, we prioritize combinations that remove "similar" columns or columns with small norms.
    # However, random sampling of combinations might be faster if we have a time limit.
    
    # Let's try a subset of combinations first, then random if time permits.
    # We will iterate through a random sample of combinations.
    
    rng_comb = np.random.RandomState(123)
    num_combinations = 2600
    # Sample 500 combinations
    sampled_combinations = rng_comb.choice(num_combinations, size=500, replace=False)
    
    # Map combination index to actual column indices
    # We need a list of all combinations
    all_combs = list(combinations(range(26), 3))
    
    for idx in sampled_combinations:
        if time.time() > deadline - 5:
            break
        
        cols_to_remove = all_combs[idx]
        cols_to_keep = [c for c in range(26) if c not in cols_to_remove]
        
        # Create initial guess: keep the selected columns, zero out the rest
        U_init = np.zeros((9, 23), dtype=np.int64)
        V_init = np.zeros((9, 23), dtype=np.int64)
        W_init = np.zeros((9, 23), dtype=np.int64)
        
        for new_idx, old_idx in enumerate(cols_to_keep):
            U_init[:, new_idx] = U26[:, old_idx]
            V_init[:, new_idx] = V26[:, old_idx]
            W_init[:, new_idx] = W26[:, old_idx]
        
        # Run local search
        U_sol, V_sol, W_sol, err = local_search(U_init, V_init, W_init, max_iters=500, time_limit=deadline - 2)
        
        if err == 0:
            # Verify
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
