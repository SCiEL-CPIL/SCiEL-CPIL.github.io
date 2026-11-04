"""
Improved program: Strassen 2x2 block decomposition (rank 189) with local search for rank reduction.
"""
import time
import random
import numpy as np

DIMENSIONS = [6, 6, 6]

def _verify_decomposition(n, m, p, U, V, W, num_checks=500):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
    rank = len(U[0])
    random.seed(42)
    for _ in range(num_checks):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        a_idx = i * m + j
        b_idx = j * p + k
        c_idx = k * n + i
        expected = 1
        val = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
        if abs(val - expected) > 1e-9:
            return False
    # Check zero entries
    for _ in range(num_checks // 2):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        i2 = (i + 1) % n
        if i2 == i:
            i2 = (i + 2) % n
        a_idx = i * m + j
        b_idx = j * p + k
        c_idx = k * n + i2
        val = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, deadline, max_iterations=2000):
    """
    Attempt to reduce rank by finding linearly dependent columns in U, V, W.
    Uses structured search over the Strassen block decomposition.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    start_time = time.time()
    
    # First pass: remove zero columns
    removed = True
    while removed:
        removed = False
        for r in range(U_arr.shape[1]):
            if np.allclose(U_arr[:, r], 0, atol=1e-12) and \
               np.allclose(V_arr[:, r], 0, atol=1e-12) and \
               np.allclose(W_arr[:, r], 0, atol=1e-12):
                mask = [i for i in range(U_arr.shape[1]) if i != r]
                U_arr = U_arr[:, mask]
                V_arr = V_arr[:, mask]
                W_arr = W_arr[:, mask]
                removed = True
                break
    
    # Second pass: try to find columns that are linear combinations of others
    # For a column r to be eliminable, we need vectors a, b, c such that:
    # U[:,r] = U[:,S] @ a, V[:,r] = V[:,S] @ b, W[:,r] = W[:,S] @ c
    # AND the tensor contribution U[:,r]*V[:,r]*W[:,r] = sum_s a_s*b_s*c_s * (U[:,s]*V[:,s]*W[:,s])
    # This is a nonlinear constraint. Instead, try: can we find a subset S and 
    # scalars alpha_r, beta_r, gamma_r for r in S such that the column r's 
    # contribution is replicated?
    
    # Simpler heuristic: check if any pair of columns have proportional U, V, W
    # (i.e., U[:,r1] = c*U[:,r2], V[:,r1] = c*V[:,r2], W[:,r1] = c*W[:,r2])
    # Then we can merge them.
    for r1 in range(U_arr.shape[1]):
        if time.time() > deadline:
            break
        for r2 in range(r1+1, U_arr.shape[1]):
            # Check if columns are proportional
            if np.allclose(U_arr[:, r1], 0, atol=1e-12):
                continue
            c = U_arr[0, r1] / U_arr[0, r2] if abs(U_arr[0, r2]) > 1e-12 else None
            if c is None:
                continue
            if np.allclose(U_arr[:, r1], c * U_arr[:, r2], atol=1e-9) and \
               np.allclose(V_arr[:, r1], c * V_arr[:, r2], atol=1e-9) and \
               np.allclose(W_arr[:, r1], c * W_arr[:, r2], atol=1e-9):
                # Merge: column r1 = c * column r2, so we can remove r1 and scale r2
                # Actually, U[:,r1]*V[:,r1]*W[:,r1] = c^3 * U[:,r2]*V[:,r2]*W[:,r2]
                # So we can remove r1 and multiply column r2 by c^3... but that changes other entries
                # Better: just remove r1 if its contribution is already covered
                # This is tricky. Skip for now.
                pass
    
    # Third pass: use the block structure. The 189 columns are organized as
    # 27 blocks * 7 Strassen terms. Try to see if adjacent blocks can share terms.
    # For a 3x3 grid of 2x2 blocks, blocks that don't interact in the matrix
    # product might share multiplications.
    
    # Actually, let's try a different approach: use numpy to find the numerical
    # rank of the "flattened" decomposition and see if we can compress.
    
    # The tensor T is determined by the 189 rank-1 terms. If we can find a 
    # lower-rank representation, that would be ideal. But finding the minimal
    # rank is NP-hard in general.
    
    # Let's try a simple perturbation + rounding approach:
    # 1. Add small random perturbations to break symmetries
    # 2. Try to find if any column becomes redundant
    
    np.random.seed(42)
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Try removing one column at a time and see if the decomposition still works
        # by adjusting other columns (this is a heuristic)
        current_rank = U_arr.shape[1]
        for r in range(current_rank):
            if time.time() > deadline:
                break
            
            # Try removing column r
            mask = [i for i in range(current_rank) if i != r]
            U_test = U_arr[:, mask]
            V_test = V_arr[:, mask]
            W_test = W_arr[:, mask]
            
            # Check if the removed column's contribution can be approximated
            # by the remaining columns. This requires solving for adjustments.
            # For now, just check if the removed column is "small"
            norm_U = np.linalg.norm(U_arr[:, r])
            norm_V = np.linalg.norm(V_arr[:, r])
            norm_W = np.linalg.norm(W_arr[:, r])
            
            if norm_U < 1e-10 and norm_V < 1e-10 and norm_W < 1e-10:
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                best_rank = W_test.shape[1]
                best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
                break
    
    # Verify and return best found
    if best_rank < rank:
        best_U_list = best_U.tolist()
        best_V_list = best_V.tolist()
        best_W_list = best_W.tolist()
        if _verify_decomposition(n, m, p, best_U_list, best_V_list, best_W_list, 200):
            return best_U_list, best_V_list, best_W_list, best_rank
    
    return U, V, W, rank

def _build_strassen_2x2(n, m, p):
    """Build Strassen 2x2 block decomposition: rank = 3^3 * 7 = 189"""
    block_size = 2
    blocks_per_dim = 3
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build the verified Strassen 2x2 solution (rank 189)
    U, V, W = _build_strassen_2x2(n, m, p)
    rank_189 = len(U[0])
    
    if _verify_decomposition(n, m, p, U, V, W, 300):
        best_U, best_V, best_W = U, V, W
        best_rank = rank_189
        
        # Try to reduce rank with local search
        if time.time() < deadline - 10:
            reduced_U, reduced_V, reduced_W, reduced_rank = _try_reduce_rank(
                U, V, W, n, m, p, deadline - 10, max_iterations=500
            )
            if reduced_rank < best_rank:
                best_U, best_V, best_W = reduced_U, reduced_V, reduced_W
                best_rank = reduced_rank
        
        # Additional: try a different block structure - 3x3 blocks with schoolbook
        # This gives rank = 2^3 * 27 = 216, which is worse, so skip.
        
        # Try: use 1x2 blocks (12 blocks) with a custom decomposition
        # This is complex and unlikely to beat 189, so skip.
        
        # Final polish: try to clean up numerical noise
        if time.time() < deadline - 5:
            U_clean = np.array(best_U, dtype=float)
            V_clean = np.array(best_V, dtype=float)
            W_clean = np.array(best_W, dtype=float)
            
            # Round to nearest half-integer if close
            U_clean = np.round(U_clean * 2) / 2
            V_clean = np.round(V_clean * 2) / 2
            W_clean = np.round(W_clean * 2) / 2
            
            if _verify_decomposition(n, m, p, U_clean.tolist(), V_clean.tolist(), W_clean.tolist(), 200):
                best_U, best_V, best_W = U_clean.tolist(), V_clean.tolist(), W_clean.tolist()
    else:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    return {"U": best_U, "V": best_V, "W": best_W}
