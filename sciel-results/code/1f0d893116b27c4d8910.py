import numpy as np
import time

DIMENSIONS = (4, 4, 7)

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x3 using schoolbook (rank 48)
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # Strassen for 2x2: 7 products
    # Each product: (linear_form_of_X, linear_form_of_Y, contribution_to_Z)
    # X, Y are 2x2 matrices; Z = XY
    #
    # M0 = (X00+X11)(Y00+Y11)  -> Z00 += M0, Z11 += M0
    # M1 = (X10+X11)Y00        -> Z10 += M1, Z11 -= M1
    # M2 = X00(Y01-Y11)        -> Z01 += M2, Z11 += M2
    # M3 = X11(Y10-Y00)        -> Z00 += M3, Z10 += M3
    # M4 = (X00+X01)Y11        -> Z00 -= M4, Z01 += M4
    # M5 = (X10-X00)(Y00+Y01)  -> Z11 += M5
    # M6 = (X01-X11)(Y10+Y11)  -> Z00 += M6
    
    strassen = [
        # (X_terms, Y_terms, Z_terms)
        # Each term is (row, col, coeff)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],           [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],           [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)],           [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)],  [(1,1,1)],           [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)],  [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)],  [(0,0,1)]),
    ]
    
    # Part 1: 4x4x4 via nested Strassen
    # A is 4x4, B is 4x4 (first 4 cols), C is 4x4 (first 4 rows of output)
    # Block structure: A = [[A00,A01],[A10,A11]], each block 2x2
    # Same for B and C
    # Strassen at block level: 7 products of 2x2 blocks
    # Each 2x2 block product uses Strassen again: 7 scalar products
    # Total: 49 scalar products
    
    for s_idx in range(7):
        X_def, Y_def, Z_def = strassen[s_idx]
        for w_idx in range(7):
            u_def, v_def, out_def = strassen[w_idx]
            r = s_idx * 7 + w_idx
            
            # Build U column: coefficients of A entries
            # A is 4x4, indexed as a_idx = i*m + j = i*4 + j
            # Block (bx, by) of A contains A[2*bx+ii, 2*by+jj]
            # X[ii, jj] = sum over block terms: cx * A[2*bx+ii, 2*by+jj]
            # u_val = sum over u terms: cu * X[ii, jj]
            # So U[a_idx, r] = sum cx * cu where a_idx = (2*bx+ii)*4 + (2*by+jj)
            
            U_col = np.zeros(nm)
            for (bx, by, cx) in X_def:
                for (ii, jj, cu) in u_def:
                    i = 2 * bx + ii
                    j = 2 * by + jj
                    a_idx = i * m + j
                    U_col[a_idx] += cx * cu
            U[:, r] = U_col
            
            # Build V column: coefficients of B entries
            # B is 4x4 (first 4 cols of the 4x7 matrix)
            # B indexed as b_idx = row*p + col = row*7 + col
            # Block (bx, by) of B contains B[2*bx+ii, 2*by+jj]
            
            V_col = np.zeros(mp)
            for (bx, by, cy) in Y_def:
                for (ii, jj, cv) in v_def:
                    row_b = 2 * bx + ii
                    col_b = 2 * by + jj
                    b_idx = row_b * p + col_b
                    V_col[b_idx] += cy * cv
            V[:, r] = V_col
            
            # Build W column: coefficients of C entries
            # C is 4x4 (first 4 cols of output), indexed as c_idx = col*n + row
            # Block (dx, dy) of C contains C[2*dx+ii, 2*dy+jj]
            
            W_col = np.zeros(pn)
            for (dx, dy, cd) in Z_def:
                for (ii, jj, co) in out_def:
                    row_c = 2 * dx + ii
                    col_c = 2 * dy + jj
                    c_idx = col_c * n + row_c
                    W_col[c_idx] += cd * co
            W[:, r] = W_col
    
    # Part 2: 4x4x3 schoolbook for columns 4,5,6 of B
    # C[i, k] = sum_j A[i,j] * B[j, k]
    # A index: i*m + j
    # B index: j*p + (4+k) for k in 0,1,2
    # C index: (4+k)*n + i
    
    offset = rank_444
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                a_idx = i * m + j
                b_idx = j * p + (4 + k)
                c_idx = (4 + k) * n + i
                
                U[a_idx, offset + col] = 1.0
                V[b_idx, offset + col] = 1.0
                W[c_idx, offset + col] = 1.0
                col += 1
    
    # Fast validation using matrix operations
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For each pair (a,b): T[a,b,:] = (U[a,:] * V[b,:]) @ W.T
    
    # Check all target entries (112 of them)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                if abs(val - 1.0) > 1e-9:
                    raise ValueError(f"Target entry ({a},{b},{c}) = {val}, expected 1.0")
    
    # Check all non-target entries efficiently using batched computation
    # For each (a,b), compute full vector across c, then check non-target positions
    # 16*28 = 448 pairs, each with 28 c-values = 12544 total
    # Build the full tensor via: for each r, outer(U[:,r], V[:,r]) then contract with W
    # T = sum_r np.einsum('ar,br,cr->abc', U, V, W)
    # But that's 16*28*28*97 which is fine
    
    # More efficient: T[a,b,:] = sum_r U[a,r]*V[b,r]*W[:,r] = (U*V.T) @ W.T per (a,b)
    # Actually: for fixed a, T[a,b,:] = sum_r U[a,r]*V[b,r]*W[:,r]
    # Let M_ab = U[a,:] * V[b,:] (element-wise, shape (rank,))
    # Then T[a,b,:] = M_ab @ W.T (shape (pn,))
    
    # Build target mask
    target_mask = np.zeros((nm, mp, pn), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_mask[a, b, c] = True
    
    # Compute full tensor in chunks for memory
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Use: T = np.einsum('ar,br,cr->abc', U, V, W)
    T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    # Check non-target entries
    non_target_mask = ~target_mask
    if np.any(non_target_mask):
        max_non_target = np.abs(T[non_target_mask]).max()
        if max_non_target > 1e-9:
            # Find worst entry
            worst = np.unravel_index(np.argmax(np.abs(T[non_target_mask])), (nm, mp, pn))
            raise ValueError(f"Non-target entry {worst} = {T[worst]}, expected 0.0")
    
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
