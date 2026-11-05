import time
import numpy as np

def construct():
    """
    Returns factor matrices U, V, W that decompose the tensor for (3, 4, 4) matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 sub-blocks and schoolbook for 1x2x2 sub-blocks.
    Total rank: 44 (4*7 + 2*8).
    
    Returns:
    Dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    rank = 44
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    
    def add_strassen(X_u, Y_v, Z_w, col):
        """Add 7 Strassen terms for X(2x2) * Y(2x2) = Z(2x2)."""
        # M1 = (X00+X11)(Y00+Y11)
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        # M2 = (X10+X11)Y00
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        # M3 = X00(Y01-Y11)
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        # M4 = X11(Y10-Y00)
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        # M5 = (X00+X01)Y11
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        # M6 = (X10-X00)(Y00+Y01)
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        # M7 = (X01-X11)(Y10+Y11)
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col
    
    # Define index mappings for the 2x2 blocks
    # A is 3x4. Indices: row*m + col
    # B is 4x4. Indices: row*p + col
    # C is 3x4 (W is p*n, so indices: col*n + row? No, W[k*n+i]. k is col of C, i is row of C)
    # Wait, the task says W[k*n+i, r]. k is the column index of C (0..p-1), i is the row index of C (0..n-1).
    # In Inspiration 1: Z_w indices are [0, 3, 1, 4] etc.
    # Let's verify W index mapping.
    # W shape is (p*n, rank).
    # Element W[idx, r] contributes to C[i, k] where idx = k*n + i.
    # In add_strassen, Z_w[0] corresponds to C[0,0]?
    # Let's check Inspiration 1's first call:
    # Z_w = [0, 3, 1, 4]
    # M1: W[Z_w[0]][col] += 1 => W[0][col] += 1 => idx=0 => k=0, i=0 => C[0,0].
    # M1: W[Z_w[3]][col] += 1 => W[4][col] += 1 => idx=4 => k=1, i=0? No, n=3.
    # idx = k*n + i.
    # If idx=4, and n=3, then 4 = 1*3 + 1 => k=1, i=1 => C[1,1].
    # So Z_w[3] is C[1,1].
    # Z_w[1] is idx=1 => k=0, i=1 => C[1,0].
    # Z_w[2] is idx=3 => k=1, i=0 => C[0,1].
    # So the mapping for a 2x2 block of C (rows i0, i1, cols k0, k1) is:
    # C[i0, k0] -> idx = k0*n + i0
    # C[i1, k0] -> idx = k0*n + i1
    # C[i0, k1] -> idx = k1*n + i0
    # C[i1, k1] -> idx = k1*n + i1
    
    # In Inspiration 1:
    # Strassen 1: C[0:2, 0:2] (Rows 0,1; Cols 0,1)
    # i0=0, i1=1, k0=0, k1=1.
    # Z_w should be [k0*n+i0, k0*n+i1, k1*n+i0, k1*n+i1]
    # = [0*3+0, 0*3+1, 1*3+0, 1*3+1] = [0, 1, 3, 4].
    # Inspiration 1 used Z_w = [0, 3, 1, 4].
    # Let's check the Strassen formula mapping in Inspiration 1.
    # M1: W[Z_w[0]] and W[Z_w[3]].
    # Standard Strassen M1 = (A00+A11)(B00+B11) contributes to C00 and C11.
    # So Z_w[0] should be C00 and Z_w[3] should be C11.
    # In Insp 1, Z_w[0]=0 (C00), Z_w[3]=4 (C11). Correct.
    # M2: (A10+A11)B00 contributes to C10 and -C11.
    # So we need C10 and C11.
    # Insp 1: W[Z_w[2]] += 1, W[Z_w[3]] -= 1.
    # Z_w[2]=1 (C10), Z_w[3]=4 (C11). Correct.
    # M3: A00(B01-B11) contributes to C01 and C11.
    # Insp 1: W[Z_w[1]] += 1, W[Z_w[3]] += 1.
    # Z_w[1]=3 (C01), Z_w[3]=4 (C11). Correct.
    # M4: A11(B10-B00) contributes to C00 and C10.
    # Insp 1: W[Z_w[0]] += 1, W[Z_w[2]] += 1.
    # Z_w[0]=0 (C00), Z_w[2]=1 (C10). Correct.
    # M5: (A00+A01)B11 contributes to -C00 and C01.
    # Insp 1: W[Z_w[0]] -= 1, W[Z_w[1]] += 1.
    # Z_w[0]=0 (C00), Z_w[1]=3 (C01). Correct.
    # M6: (A10-A00)(B00+B01) contributes to C11.
    # Insp 1: W[Z_w[3]] += 1.
    # Z_w[3]=4 (C11). Correct.
    # M7: (A01-A11)(B10+B11) contributes to C00.
    # Insp 1: W[Z_w[0]] += 1.
    # Z_w[0]=0 (C00). Correct.
    
    # So the Z_w array order is [C00, C01, C10, C11] in terms of logical positions, but the indices are mapped as:
    # Z_w[0] = idx(C00)
    # Z_w[1] = idx(C01)
    # Z_w[2] = idx(C10)
    # Z_w[3] = idx(C11)
    
    # Let's re-verify the indices for the 4 blocks.
    
    # Block 1: C[0:2, 0:2]. i in {0,1}, k in {0,1}.
    # Z_w = [0*3+0, 1*3+0, 0*3+1, 1*3+1] = [0, 3, 1, 4].
    # Matches Insp 1.
    
    # Block 2: C[0:2, 0:2] contribution from A[0:2, 2:4] * B[2:4, 0:2].
    # Same C block, so same Z_w = [0, 3, 1, 4].
    
    # Block 3: C[0:2, 2:4]. i in {0,1}, k in {2,3}.
    # Z_w = [2*3+0, 3*3+0, 2*3+1, 3*3+1] = [6, 9, 7, 10].
    # Matches Insp 1.
    
    # Block 4: C[0:2, 2:4] contribution from A[0:2, 2:4] * B[2:4, 2:4].
    # Same C block, so same Z_w = [6, 9, 7, 10].
    
    # Bottom row: C[2, 0:4].
    # C[2,0] -> idx = 0*3+2 = 2.
    # C[2,1] -> idx = 1*3+2 = 5.
    # C[2,2] -> idx = 2*3+2 = 8.
    # C[2,3] -> idx = 3*3+2 = 11.
    
    # Insp 1 uses:
    # w_idx_0 = 2 (C[2,0]), w_idx_1 = 5 (C[2,1]).
    # w_idx_0 = 8 (C[2,2]), w_idx_1 = 11 (C[2,3]).
    # Matches.

    # Execute the construction
    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
    
    # Bottom left
    for j in range(4):
        u_idx = 8 + j
        v_idx_0 = j * 4 + 0
        v_idx_1 = j * 4 + 1
        w_idx_0 = 2
        w_idx_1 = 5
        U[u_idx][col] += 1; V[v_idx_0][col] += 1; W[w_idx_0][col] += 1
        col += 1
        U[u_idx][col] += 1; V[v_idx_1][col] += 1; W[w_idx_1][col] += 1
        col += 1
        
    # Bottom right
    for j in range(4):
        u_idx = 8 + j
        v_idx_0 = j * 4 + 2
        v_idx_1 = j * 4 + 3
        w_idx_0 = 8
        w_idx_1 = 11
        U[u_idx][col] += 1; V[v_idx_0][col] += 1; W[w_idx_0][col] += 1
        col += 1
        U[u_idx][col] += 1; V[v_idx_1][col] += 1; W[w_idx_1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
