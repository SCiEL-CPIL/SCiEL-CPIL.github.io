"""Improved program: top-level Strassen with inner Strassen+schoolbook for 4x4*4x6."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition using matrix operations for speed."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build target tensor and computed tensor
    # T_target[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # T_computed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # Efficient check: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Sum over r should give target tensor
    
    # Only check the 1-entries (n*m*p = 96 entries)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(s - 1.0) > tol:
                    return False
    
    # Spot-check some zero entries (not all, for speed)
    import random
    random.seed(42)
    checks = 200
    for _ in range(checks):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j1 == j2 and i1 == i2:
            continue  # This is a 1-entry, skip
        s = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(s) > tol:
            return False
    
    return True

def construct():
    """Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Top-level Strassen (7 terms) where each term is a (2x2) x (2x3) product.
    For each (2x2)x(2x3): use inner Strassen for the 2x2x2 part (7 mults) + 
    schoolbook for the 2x2x1 part (4 mults) = 11 mults per term.
    Total rank: 7 * 11 = 77.
    """
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 4, 4, 6
    
    # Top-level Strassen coefficients
    # A blocks indexed 0-3: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B blocks indexed 0-3: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Each block is 2x3 (since B is 4x6, split into 2x2 blocks of 2x3)
    
    # Strassen formulas:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # alpha[s][a] = coefficient of A-block a in Strassen term s
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # M0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},   # M1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},   # M2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},   # M3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},   # M4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0},  # M5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}   # M6: A01 - A11
    ]
    
    # beta[s][b] = coefficient of B-block b in Strassen term s
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # M0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},   # M1: B00
        {0: 0, 1: 1, 2: 0, 3: -1},  # M2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0},  # M3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},   # M4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},   # M5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}    # M6: B10 + B11
    ]
    
    # recon[s] = list of (C_block_row, C_block_col, coefficient)
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # M0 -> C00, C11
        [(1, 0, 1), (1, 1, -1)],    # M1 -> C10, -C11
        [(0, 1, 1), (1, 1, 1)],     # M2 -> C01, C11
        [(0, 0, 1), (1, 0, 1)],     # M3 -> C00, C10
        [(0, 0, -1), (0, 1, 1)],    # M4 -> -C00, C01
        [(1, 1, 1)],                # M5 -> C11
        [(0, 0, 1)]                 # M6 -> C00
    ]
    
    # Inner Strassen for 2x2x2 (same structure)
    inner_alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    inner_beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    inner_recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for s in range(7):
        # For each top-level Strassen term s:
        # X_s is a 2x2 matrix (linear combo of A blocks)
        # Y_s is a 2x3 matrix (linear combo of B blocks)
        # Compute X_s * Y_s using: inner Strassen for cols 0-1, schoolbook for col 2
        
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # Inner Strassen term si
            for a in range(4):
                coeff_a = inner_alpha[si].get(a, 0)
                if coeff_a != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        coeff_r = alpha[s].get(r, 0)
                        if coeff_r != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += coeff_a * coeff_r
            
            for b in range(4):
                coeff_b = inner_beta[si].get(b, 0)
                if coeff_b != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        coeff_c = beta[s].get(c, 0)
                        if coeff_c != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += coeff_b * coeff_c
            
            for a_idx, b_idx, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    w_idx = (b_idx * 3 + zb) * 4 + (a_idx * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    coeff_r = alpha[s].get(r, 0)
                    if coeff_r != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += coeff_r
                
                for c in range(4):
                    coeff_c = beta[s].get(c, 0)
                    if coeff_c != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += coeff_c
                
                for a_idx, b_idx, gamma in recon[s]:
                    w_idx = (b_idx * 3 + 2) * 4 + (a_idx * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Try to reduce rank by removing zero or redundant columns
    keep = []
    for c in range(R):
        if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0):
            keep.append(c)
    
    if len(keep) < R:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        R = len(keep)
    
    # Try to reduce rank by finding linearly dependent columns
    # A column r is redundant if there exist scalars lambda_r' for r' != r
    # such that for all (a,b,c): U[a,r]*V[b,r]*W[c,r] = sum_{r'!=r} lambda_{r'} * U[a,r']*V[b,r']*W[c,r']
    # This is equivalent to: the rank-1 tensor U[:,r]⊗V[:,r]⊗W[:,r] lies in the span
    # of the other rank-1 tensors.
    # 
    # A simpler check: if two columns have proportional U, V, AND W vectors,
    # we can merge them. More generally, we can try to eliminate columns by
    # checking if removing column r and adjusting others preserves the tensor.
    #
    # Strategy: try removing each column and see if the remaining columns
    # can be adjusted (scaled) to still produce the correct tensor.
    # This is expensive, so we'll use a heuristic: check for proportional columns.
    
    def _try_reduce_rank(U_mat, V_mat, W_mat, max_time=60):
        """Try to reduce rank by finding and removing redundant columns."""
        U_mat = U_mat.copy()
        V_mat = V_mat.copy()
        W_mat = W_mat.copy()
        current_rank = U_mat.shape[1]
        start = time.time()
        
        # First pass: remove zero columns
        keep = [c for c in range(current_rank) 
                if np.any(U_mat[:, c] != 0) and np.any(V_mat[:, c] != 0) and np.any(W_mat[:, c] != 0)]
        if len(keep) < current_rank:
            U_mat = U_mat[:, keep]
            V_mat = V_mat[:, keep]
            W_mat = W_mat[:, keep]
            current_rank = len(keep)
        
        # Second pass: try to find columns that can be eliminated
        # For each pair of columns, check if one is a scalar multiple of the other
        # in all three matrices simultaneously
        eliminated = set()
        for c1 in range(current_rank):
            if c1 in eliminated:
                continue
            if time.time() - start > max_time:
                break
            for c2 in range(c1 + 1, current_rank):
                if c2 in eliminated:
                    continue
                # Check if column c2 is proportional to column c1 in U, V, W
                # Find scalar from U
                nonzero_u = U_mat[:, c1] != 0
                if not np.any(nonzero_u):
                    continue
                u_vals = U_mat[nonzero_u, c1]
                v_vals = U_mat[nonzero_u, c2]
                # Check proportionality in U
                ratios_u = v_vals / u_vals
                if not np.allclose(ratios_u, ratios_u[0], atol=1e-10):
                    continue
                scalar_u = ratios_u[0]
                
                # Check in V
                nonzero_v = V_mat[:, c1] != 0
                if np.any(nonzero_v):
                    v_r = V_mat[nonzero_v, c2] / V_mat[nonzero_v, c1]
                    if not np.allclose(v_r, scalar_u, atol=1e-10):
                        continue
                
                # Check in W
                nonzero_w = W_mat[:, c1] != 0
                if np.any(nonzero_w):
                    w_r = W_mat[nonzero_w, c2] / W_mat[nonzero_w, c1]
                    if not np.allclose(w_r, scalar_u, atol=1e-10):
                        continue
                
                # Column c2 is proportional to c1; merge by adding c2 to c1
                U_mat[:, c1] += U_mat[:, c2]
                V_mat[:, c1] += V_mat[:, c2]
                W_mat[:, c1] += W_mat[:, c2]
                eliminated.add(c2)
        
        if eliminated:
            keep = [c for c in range(current_rank) if c not in eliminated]
            U_mat = U_mat[:, keep]
            V_mat = V_mat[:, keep]
            W_mat = W_mat[:, keep]
        
        return U_mat, V_mat, W_mat
    
    if time.time() < deadline - 20:
        U, V, W = _try_reduce_rank(U, V, W, max_time=40)
        R = U.shape[1]
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    col_fb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col_fb] = 1
                V_fb[j * p + k][col_fb] = 1
                W_fb[k * n + i][col_fb] = 1
                col_fb += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
