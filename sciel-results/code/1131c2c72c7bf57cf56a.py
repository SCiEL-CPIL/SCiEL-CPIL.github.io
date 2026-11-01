"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    # Check target entries
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    # Check non-target entries
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))  # 6 x 44
    V = np.zeros((m*p, 44))  # 24 x 44
    W = np.zeros((p*n, 44))  # 16 x 44
    
    col_offset = 0
    
    for block in range(4):
        # Block handles columns 2*block and 2*block+1 of B
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for the 2x2 * 2x2 part (A cols 0,1 times B rows 0,1)
        
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1   # a00
        U[4, t] += 1   # a11
        V[k0, t] += 1  # b00
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += 1  # C00
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1   # a10
        U[4, t] += 1   # a11
        V[k0, t] += 1  # b00
        W[2*k0+1, t] += 1  # C10
        W[2*k1+1, t] += -1  # C11
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1   # a00
        V[k1, t] += 1  # b01
        V[8+k1, t] += -1  # b11
        W[2*k1, t] += 1  # C01
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1   # a11
        V[8+k0, t] += 1  # b10
        V[k0, t] += -1  # b00
        W[2*k0, t] += 1  # C00
        W[2*k0+1, t] += 1  # C10
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1   # a00
        U[1, t] += 1   # a01
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += -1  # C00
        W[2*k1, t] += 1  # C01
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1   # a10
        U[0, t] += -1  # a00
        V[k0, t] += 1  # b00
        V[k1, t] += 1  # b01
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1   # a01
        U[4, t] += -1  # a11
        V[8+k0, t] += 1  # b10
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += 1  # C00
        col_offset += 1
        
        # Now the 4 rank-1 terms for A_right * B_bottom
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1   # A[ii, 2]
                V[16+k_val, t] = 1  # B[2, k_val]
                W[k_val*2+ii, t] = 1  # C[ii, k_val]
                col_offset += 1
    
    assert col_offset == 44, f"Expected 44 columns, got {col_offset}"
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank43_search(deadline):
    """
    Try to find a rank-43 decomposition by modifying the rank-44 structure.
    Idea: The 4 standard terms per block for A_col2 * B_row2 might be reducible.
    Try merging terms across blocks using the fact that A_col2 is shared.
    """
    n, m, p = 2, 3, 8
    
    # Start from rank-44 and try to zero out one column by adjusting others
    U44, V44, W44 = _build_rank44_decomposition()
    U = np.array(U44, dtype=float)
    V = np.array(V44, dtype=float)
    W = np.array(W44, dtype=float)
    
    # Try: can we express one of the 16 "standard" columns as a combination of others?
    # The standard columns are at positions 7+4*b through 10+4*b for each block b
    # Let's try to see if column 43 (last standard col of last block) is redundant
    
    # For each candidate column to remove, try to solve for it as linear combo of others
    # This is a linear algebra problem: can we find c such that
    # U[:,r] = sum_s c_s U[:,s], V[:,r] = sum_s c_s V[:,s], W[:,r] = sum_s c_s W[:,s]?
    # No, that's not right. The tensor decomposition is:
    # T = sum_r U[:,r] V[:,r] W[:,r]
    # If we remove column r, we need the remaining columns to still sum to T.
    # So we need: U[:,r] V[:,r] W[:,r] = sum_{s != r} delta_s U[:,s] V[:,s] W[:,s]
    # This is a rank-1 tensor in the difference, which is generally not possible.
    
    # Instead, try a different approach: use the structure to find a rank-43.
    # The 2x3x8 tensor has 2*3*8 = 48 nonzero entries.
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # Actually, a better lower bound is from the border rank or other invariants.
    
    # Let's try a more systematic approach: use the fact that we can potentially
    # share the A_col2 factor across blocks.
    
    # New idea: pack the problem differently.
    # A is 2x3, B is 3x8. 
    # Think of it as: C = A * B where A = [A_left | A_right] with A_left 2x2, A_right 2x1
    # C = A_left * B_top + A_right * B_bottom
    # where B_top is 2x8, B_bottom is 1x8
    
    # For A_left * B_top: this is 2x2 * 2x8. We can use Strassen per 2x2 block of B_top.
    # B_top has 4 blocks of 2x2, so 4 * 7 = 28 terms.
    # For A_right * B_bottom: this is 2x1 * 1x8 = 8 terms.
    # Total: 36 terms! But wait, that's not right because Strassen gives 2x2 output per block,
    # and we need the full 2x8 output.
    
    # Actually: A_left (2x2) * B_top (2x8):
    # Split B_top into 4 blocks of 2x2: B_top = [B0 | B1 | B2 | B3]
    # C_left = A_left * B_top = [A_left*B0 | A_left*B1 | A_left*B2 | A_left*B3]
    # Each A_left*Bk is 2x2, computed with Strassen rank 7.
    # Total for this part: 4 * 7 = 28.
    
    # A_right (2x1) * B_bottom (1x8):
    # C_right = A_right * B_bottom, which is 2x8.
    # This is just 8 scalar multiplications: C_right[i,j] = A_right[i,0] * B_bottom[0,j]
    # So 8 rank-1 terms.
    
    # Total: 28 + 8 = 36!
    
    # Let me verify this construction.
    U36 = np.zeros((6, 36))
    V36 = np.zeros((24, 36))
    W36 = np.zeros((16, 36))
    
    col = 0
    # A_left * B_top part: 4 Strassen blocks
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # m1 = (a00+a11)(b00+b11)
        t = col
        U36[0, t] += 1
        U36[4, t] += 1
        V36[k0, t] += 1
        V36[8+k1, t] += 1
        W36[2*k0, t] += 1
        W36[2*k1+1, t] += 1
        col += 1
        
        # m2 = (a10+a11)b00
        t = col
        U36[3, t] += 1
        U36[4, t] += 1
        V36[k0, t] += 1
        W36[2*k0+1, t] += 1
        W36[2*k1+1, t] += -1
        col += 1
        
        # m3 = a00(b01-b11)
        t = col
        U36[0, t] += 1
        V36[k1, t] += 1
        V36[8+k1, t] += -1
        W36[2*k1, t] += 1
        W36[2*k1+1, t] += 1
        col += 1
        
        # m4 = a11(b10-b00)
        t = col
        U36[4, t] += 1
        V36[8+k0, t] += 1
        V36[k0, t] += -1
        W36[2*k0, t] += 1
        W36[2*k0+1, t] += 1
        col += 1
        
        # m5 = (a00+a01)b11
        t = col
        U36[0, t] += 1
        U36[1, t] += 1
        V36[8+k1, t] += 1
        W36[2*k0, t] += -1
        W36[2*k1, t] += 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col
        U36[3, t] += 1
        U36[0, t] += -1
        V36[k0, t] += 1
        V36[k1, t] += 1
        W36[2*k1+1, t] += 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col
        U36[1, t] += 1
        U36[4, t] += -1
        V36[8+k0, t] += 1
        V36[8+k1, t] += 1
        W36[2*k0, t] += 1
        col += 1
    
    # A_right * B_bottom part: 8 standard terms
    for k in range(8):
        t = col
        U36[2, t] = 1   # A[0,2]
        V36[16+k, t] = 1  # B[2,k]
        W36[k*2+0, t] = 1  # C[0,k]
        col += 1
        # Wait, this only gives C[0,k]. We also need C[1,k].
        # Let me redo this.
    
    # Hmm, I made an error. Let me reconsider.
    # A_right is 2x1: [a02; a12]
    # B_bottom is 1x8: [b20 b21 ... b27]
    # C_right = A_right * B_bottom is 2x8:
    # C_right[0,k] = a02 * b2k
    # C_right[1,k] = a12 * b2k
    # So we need 16 rank-1 terms, not 8!
    # Total: 28 + 16 = 44. Same as before.
    
    # OK so that doesn't help. Let me think differently.
    
    # What if we use a different split? 
    # A is 2x3. Split into 1x3 and 1x3 (two rows).
    # B is 3x8.
    # C[0,:] = A[0,:] * B (1x3 * 3x8 = 1x8)
    # C[1,:] = A[1,:] * B (1x3 * 3x8 = 1x8)
    # Each is a 1x3 * 3x8 multiplication, which is just 3*8 = 24 scalar mults each.
    # Total: 48. Worse.
    
    # What about splitting B differently?
    # B is 3x8. Split into 3x4 and 3x4.
    # C = A * [B1 | B2] = [A*B1 | A*B2]
    # Each A*Bk is 2x3 * 3x4.
    # For 2x3 * 3x4: schoolbook is 24. Can we do better?
    # Lower bound for 2x3*3x4: at least 2*4=8, but realistically...
    # Using the same Strassen idea: split 3x4 into two 3x2 blocks.
    # 2x3 * 3x2: schoolbook 12. 
    # Hmm, this is getting complicated.
    
    # Let me just return the rank-44 result.
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 140
    
    # First, try the explicit rank-44 decomposition
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6):
            rounded = _round_and_verify(U44, V44, W44)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 44
            else:
                best_U, best_V, best_W = U44, V44, W44
                best_rank = 44
    except Exception as e:
        pass
    
    # Try rank-43 search (limited time)
    if best_rank == 44 and time.time() < deadline - 30:
        result = _try_rank43_search(deadline)
        if result:
            U, V, W = result
            if _verify(U, V, W, tol=1e-6):
                rounded = _round_and_verify(U, V, W)
                if rounded:
                    best_U, best_V, best_W = rounded
                    best_rank = 43
                else:
                    best_U, best_V, best_W = U, V, W
                    best_rank = 43
    
    return {"U": best_U, "V": best_V, "W": best_W}
