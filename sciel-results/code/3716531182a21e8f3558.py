"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _schoolbook_2x4x4():
    """Return schoolbook rank-32 decomposition for 2x4 * 4x4 -> 2x4."""
    n, m, p = 2, 4, 4
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # 16 x 32
    W = [[0] * rank for _ in range(p * n)]  # 8 x 32
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=4, trials=30, atol=1e-9):
    """Verify decomposition correctness with random inputs."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # Compute sum_c (U[:,c]·A_flat) * (V[:,c]·B_flat) * W[:,c]
        u_a = U_arr.T @ A_flat  # shape (rank,)
        v_b = V_arr.T @ B_flat  # shape (rank,)
        C_flat = W_arr.T @ (u_a * v_b)  # shape (p*n,)
        
        # Reshape C_flat: W index is k*n+i, so C[i,k] is at index k*n+i
        C_computed = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_computed[i, k] = C_flat[k * n + i]
        
        if not np.allclose(C_true, C_computed, atol=atol):
            return False
    return True

def _try_rank_reduction(U, V, W, n=2, m=4, p=4, max_iters=5000, seed=42):
    """
    Try to reduce rank by randomly combining columns.
    Strategy: pick two columns, try random linear combination, check validity.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    for it in range(max_iters):
        if best_rank <= 1:
            break
        
        # Try to merge two random columns
        c1, c2 = rng.sample(range(best_rank), 2)
        
        # Try random coefficients: new_col = alpha * col1 + beta * col2
        # We need to find alpha, beta such that the decomposition remains valid
        # after removing one column.
        
        # Simpler approach: try to express one column as a linear combination of others
        # and remove it if possible.
        
        # Even simpler: try random linear combinations and check
        for attempt in range(20):
            alpha = np_rng.uniform(-2, 2)
            beta = np_rng.uniform(-2, 2)
            
            # Create new matrices with columns c1 and c2 replaced by their combination
            # Actually, let's try: remove column c2, and modify column c1
            # new_U[:, c1] = U[:, c1] + alpha * U[:, c2]
            # new_V[:, c2] = V[:, c2]  (unchanged, but we'll remove it)
            # This doesn't directly work...
            
            # Better approach: try to find if column c2 is in the span of others
            # For a valid decomposition, we need:
            # sum_c U[:,c] * V[:,c] * W[:,c] = T (the multiplication tensor)
            # If we can write col_c2 = sum_{c != c2} gamma_c * col_c for all three matrices
            # simultaneously, we can remove c2.
            
            # This is a system of linear equations. Let's try a simpler heuristic:
            # Randomly perturb and check.
            
            # Actually, let me try a different approach: random linear recombination
            # Pick a random vector of coefficients for all columns, create a new
            # decomposition, and check if it's valid.
            
            pass
        
        # Alternative: try to find linear dependencies
        # Check if any column is a linear combination of others in U, V, W simultaneously
        if best_rank > 2:
            # Try to find if removing column c2 and adjusting c1 works
            # This requires solving a system. Let's use a simpler approach:
            # Randomly try to merge and verify.
            
            for merge_attempt in range(5):
                a = np_rng.uniform(-3, 3)
                b = np_rng.uniform(-3, 3)
                
                # New decomposition: replace cols c1, c2 with single col
                # U_new[:, c1] = U[:, c1] + a * U[:, c2]
                # V_new[:, c1] = V[:, c1] + b * V[:, c2]  
                # W_new[:, c1] = W[:, c1]  (or some combination)
                # Then remove column c2.
                # But this only works if the cross terms cancel, which they won't in general.
                
                # The correct approach: we need
                # (U1 + a*U2)(V1 + b*V2)W1' = U1V1W1 + U2V2W2 + ...
                # This is getting complex. Let me use a different strategy.
                pass
        
        # Let me use a completely different approach: 
        # Try random rank-r decompositions directly.
        # For each candidate rank r, try to find U, V, W of that rank.
        
        if it % 100 == 0 and best_rank > 24:
            # Try to find a lower rank decomposition via random search
            target_rank = best_rank - 1
            if target_rank < 2:
                continue
            
            # Random initialization
            for trial in range(50):
                U_rand = np_rng.randn(n * m, target_rank)
                V_rand = np_rng.randn(m * p, target_rank)
                W_rand = np_rng.randn(p * n, target_rank)
                
                # Check if this decomposes the multiplication tensor
                # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # Check a few random entries
                valid = True
                for _ in range(20):
                    i = rng.randint(0, n-1)
                    j = rng.randint(0, m-1)
                    k = rng.randint(0, p-1)
                    expected = 1.0  # T[i,j,k] = 1 if this is a valid mult tensor entry
                    # Actually T[i,j,k] = 1 for the multiplication tensor
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should = 1
                    val = np.sum(U_rand[i*m+j, :] * V_rand[j*p+k, :] * W_rand[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                
                if valid:
                    # Verify fully
                    U_list = U_rand.tolist()
                    V_list = V_rand.tolist()
                    W_list = W_rand.tolist()
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p, trials=20):
                        best_U, best_V, best_W = U_rand, V_rand, W_rand
                        best_rank = target_rank
                        break
    
    # Convert back to lists
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return best_U_list, best_V_list, best_W_list, best_rank

def _strassen_2x2_rank7():
    """Returns (U, V, W) for 2x2 * 2x2 -> 2x2 with rank 7 (Strassen)."""
    U = [
        [1,  1,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1, -1],
        [0,  0,  1,  0,  1,  0,  0],
        [0,  0,  1,  1,  0, -1, 0],
    ]
    V = [
        [1,  0,  1,  0,  1,  0,  1],
        [0,  0,  0,  1,  0,  0, -1],
        [0,  0,  0, -1, 1,  1, 0],
        [1,  1,  0,  0,  0,  1, 0],
    ]
    W = [
        [ 1,  0,  0,  1, -1,  0,  1],
        [ 0,  1,  0,  1,  0, -1,  1],
        [-1,  0,  1,  0,  1,  1,  0],
        [ 1, -1,  1,  1,  0,  0,  0],
    ]
    return U, V, W

def _block_strassen_2x4x4():
    """
    Build rank-28 decomposition using block Strassen.
    A is 2x4, B is 4x4, C is 2x4.
    Split into 2x2 blocks.
    """
    U2, V2, W2 = _strassen_2x2_rank7()
    
    rank = 28
    n, m, p = 2, 4, 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    products = [
        (0, 0, 0, 0),  # A1 * B11 -> C1
        (1, 1, 0, 0),  # A2 * B21 -> C1
        (0, 0, 1, 1),  # A1 * B12 -> C2
        (1, 1, 1, 1),  # A2 * B22 -> C2
    ]
    
    for prod_idx, (a_col, b_row, b_col, c_col) in enumerate(products):
        offset = prod_idx * 7
        
        a_indices = [a_col*2, a_col*2+1, 4 + a_col*2, 4 + a_col*2+1]
        b_indices = [
            (b_row*2)*4 + b_col*2,
            (b_row*2)*4 + b_col*2+1,
            (b_row*2+1)*4 + b_col*2,
            (b_row*2+1)*4 + b_col*2+1
        ]
        c_indices = [
            (c_col*2)*2 + 0,
            (c_col*2+1)*2 + 0,
            (c_col*2)*2 + 1,
            (c_col*2+1)*2 + 1
        ]
        
        for c in range(7):
            rank_idx = offset + c
            for a_idx, a_flat_idx in enumerate(a_indices):
                U[a_flat_idx][rank_idx] = U2[a_idx][c]
            for b_idx, b_flat_idx in enumerate(b_indices):
                V[b_flat_idx][rank_idx] = V2[b_idx][c]
            for c_idx, c_flat_idx in enumerate(c_indices):
                W[c_flat_idx][rank_idx] = W2[c_idx][c]
    
    return U, V, W

def construct():
    """
    Construct a low-rank exact tensor decomposition for 2x4 * 4x4 -> 2x4.
    Uses block Strassen (rank 28) as baseline, then attempts rank reduction.
    """
    deadline = time.time() + 140
    n, m, p = 2, 4, 4
    
    # Start with block Strassen (rank 28)
    U, V, W = _block_strassen_2x4x4()
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, trials=10):
        # Fallback to schoolbook
        U, V, W = _schoolbook_2x4x4()
    
    best_rank = len(U[0])
    
    # Try to reduce rank using algebraic methods
    if time.time() < deadline - 20:
        U, V, W, best_rank = _try_rank_reduction(U, V, W, n, m, p, 
                                                   max_iters=3000, seed=42)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p, trials=20):
        # Fallback to schoolbook
        U, V, W = _schoolbook_2x4x4()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
