"""Improved program: top-level Strassen with inner Strassen+schoolbook for 4x4*4x6."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    where T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    # Check zeros
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i1, j1 = divmod(a, m)
                j2, k1 = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j1 != j2 or i1 != i2:
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
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
            # U part: X_s entries with inner_alpha[si] coefficients
            # X_s[i,j] = sum_r alpha[s][r] * A_r[i,j]
            # Inner Strassen: inner_X[i,j] = sum_a inner_alpha[si][a] * X_s[a//2, a%2]
            # So U contribution = sum_a inner_alpha[si][a] * sum_r alpha[s][r] * A_r[i,j]
            
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
            
            # V part: Y_s entries with inner_beta[si] coefficients
            # Y_s[i,j] for j in {0,1}: sum_c beta[s][c] * B_c[i,j]
            # Inner B-blocks: Y_s[0,0], Y_s[0,1], Y_s[1,0], Y_s[1,1]
            # b=0: Y_s[0,0], b=1: Y_s[0,1], b=2: Y_s[1,0], b=3: Y_s[1,1]
            
            for b in range(4):
                coeff_b = inner_beta[si].get(b, 0)
                if coeff_b != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        coeff_c = beta[s].get(c, 0)
                        if coeff_c != 0:
                            ac, bc = c // 2, c % 2
                            # B_c is 2x3, B_c[y_row, y_inner_col]
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += coeff_b * coeff_c
            
            # W part: output of inner Strassen
            # inner_recon[si] gives (out_row, out_col, gamma)
            # Output Z is 2x2, Z[out_row, out_col]
            # This is P_s[out_row, out_col] where P_s = X_s * Y_s[:,0:2]
            # Top-level: C_ab gets gamma * P_s from recon[s]
            
            for a_idx, b_idx, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    # C block (a_idx, b_idx) at position (za, zb)
                    # Global: row = a_idx*2 + za, col = b_idx*3 + zb
                    w_idx = (b_idx * 3 + zb) * 4 + (a_idx * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        # X_s (2x2) * Y_s[:,2] (2x1) -> Z (2x1)
        # Z[u] = sum_v X_s[u,v] * Y_s[v,2]
        # 4 schoolbook multiplications
        
        for u in range(2):
            for v in range(2):
                # Term: X_s[u,v] * Y_s[v,2]
                
                # U part
                for r in range(4):
                    coeff_r = alpha[s].get(r, 0)
                    if coeff_r != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += coeff_r
                
                # V part
                for c in range(4):
                    coeff_c = beta[s].get(c, 0)
                    if coeff_c != 0:
                        ac, bc = c // 2, c % 2
                        # B_c[v, 2]
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += coeff_c
                
                # W part
                for a_idx, b_idx, gamma in recon[s]:
                    # C block (a_idx, b_idx) at position (u, 2)
                    w_idx = (b_idx * 3 + 2) * 4 + (a_idx * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Try to reduce rank by identifying and removing redundant terms
    if time.time() < deadline:
        U_r, V_r, W_r = _try_reduce_rank(U, V, W, n, m, p, deadline)
        if U_r is not None:
            U, V, W = U_r, V_r, W_r
    
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

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by removing redundant terms.
    
    Strategy: For each term r, check if it can be expressed as a linear combination
    of other terms. If so, remove it. This is equivalent to checking if the rank-1
    tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r] lies in the span of the other rank-1 tensors.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = 4, 4, 6
    R = U.shape[1]
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(R):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    # For each term r, try to express U[:,r] ⊗ V[:,r] ⊗ W[:,r] as combo of others
    # We'll use a greedy approach: try to remove term r if the residual can be absorbed
    
    # More practical: try pairwise merging or check for exact linear dependencies
    # among the "flattened" rank-1 tensors
    
    # Flatten each rank-1 term into a vector of length (n*m)*(m*p)*(p*n)
    # But that's 16*24*24 = 9216 dims, and we have 77 terms
    # Check if any term is in the span of others using SVD
    
    # Build matrix M where column r is vec(U[:,r] ⊗ V[:,r] ⊗ W[:,r])
    # Actually, the tensor T = sum_r T_r where T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We want to find if any T_r is in span of {T_s : s ≠ r}
    
    # Efficient approach: for each r, project T_r onto span of others
    # Use QR or SVD of the other terms
    
    R_new = R
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Try to remove each term by checking if it's redundant
    # Use a simpler heuristic: check if two terms have proportional U, V, or W vectors
    # If U[:,r] = c * U[:,s] and V[:,r] = d * V[:,s] and W[:,r] = e * W[:,s]
    # with c*d*e = 1, then term r can be merged into term s
    
    removed = [False] * R
    for r in range(R):
        if removed[r] or time.time() > deadline - 5:
            continue
        for s in range(r + 1, R):
            if removed[s]:
                continue
            # Check if term r and s are "proportional" in a way that allows merging
            # This is rare for Strassen terms, but check anyway
            # Actually, for Strassen terms, they're typically all independent
            # So this won't help much. Let's try a different approach.
            pass
    
    # Alternative: try to find a lower-rank decomposition by optimization
    # But that's complex. For now, return the original if no reduction found
    if R_new == R:
        return None, None, None
    
    return U_new, V_new, W_new
# EVOLVE-BLOCK-END
