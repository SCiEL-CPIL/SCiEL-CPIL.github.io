import time
import itertools

DIMENSIONS = [2, 4, 5]

def _strassen_222():
    """Return U, V, W for 2x2x2 Strassen multiplication.
    A (2x2), B (2x2), C = A*B (2x2).
    U indexed by (i,j) of A -> row i*2+j
    V indexed by (j,k) of B -> row j*2+k
    W indexed by (k,i) of C -> row k*2+i  (trace convention W[k*n+i])
    n=2, so W[k*2+i].
    """
    # 7 Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    n = 2
    rank = 7
    U = [[0]*rank for _ in range(n*2)]  # 4 rows
    V = [[0]*rank for _ in range(2*2)]  # 4 rows
    W = [[0]*rank for _ in range(2*n)]  # 4 rows, W[k*2+i]
    
    # A indices: A[0,0]=row0, A[0,1]=row1, A[1,0]=row2, A[1,1]=row3
    # B indices: B[0,0]=row0, B[0,1]=row1, B[1,0]=row2, B[1,1]=row3
    # C indices: C[0,0]=W[0*2+0]=W[0], C[0,1]=W[1*2+0]=W[2], C[1,0]=W[0*2+1]=W[1], C[1,1]=W[1*2+1]=W[3]
    
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[1][0] += 1; W[2][0] += 1; W[3][0] += 1
    
    # M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] += 1
    
    # M3
    U[0][2] += 1
    V[2][2] += 1; V[3][2] -= 1
    W[2][2] += 1; W[3][2] += 1
    
    # M4
    U[3][3] += 1
    V[1][3] += 1; V[3][3] -= 1
    W[0][3] += 1; W[3][3] += 1
    
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[2][4] += 1
    
    # M6
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[2][6] -= 1
    
    return U, V, W

def _schoolbook_221():
    """Return U, V, W for 2x2x1 multiplication.
    A (2x2), B (2x1), C = A*B (2x1).
    U indexed by (i,j) of A -> row i*2+j
    V indexed by (j,k) of B -> row j (since k only 0)
    W indexed by (k,i) of C -> row k*2+i, but k=0 so row i
    """
    n = 2
    rank = 4
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    
    for i in range(2):
        for j in range(2):
            col = i*2 + j
            U[i*2+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
    
    return U, V, W

def _combine_decomps(decomps):
    """Combine a list of (U, V, W) decompositions into one.
    Each U has shape (rows_U, rank_i), V has shape (rows_V, rank_i), W has shape (rows_W, rank_i).
    All must have the same rows_U, rows_V, rows_W.
    Returns U, V, W with total rank = sum of ranks.
    """
    total_rank = sum(d[0][0].__len__() if d[0] else 0 for d in decomps)
    # Better: get rank from length of first row of U
    ranks = []
    for U, V, W in decomps:
        if U:
            ranks.append(len(U[0]))
        else:
            ranks.append(0)
    total_rank = sum(ranks)
    
    rows_U = len(decomps[0][0])
    rows_V = len(decomps[0][1])
    rows_W = len(decomps[0][2])
    
    U_full = [[0]*total_rank for _ in range(rows_U)]
    V_full = [[0]*total_rank for _ in range(rows_V)]
    W_full = [[0]*total_rank for _ in range(rows_W)]
    
    offset = 0
    for (U, V, W), r in zip(decomps, ranks):
        for i in range(rows_U):
            for c in range(r):
                U_full[i][offset+c] = U[i][c]
        for i in range(rows_V):
            for c in range(r):
                V_full[i][offset+c] = V[i][c]
        for i in range(rows_W):
            for c in range(r):
                W_full[i][offset+c] = W[i][c]
        offset += r
    
    return U_full, V_full, W_full

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication.
    Uses block decomposition: split A(2x4) into A1(2x2), A2(2x2); B(4x5) into blocks.
    C = A1*B_top + A2*B_bottom, where B_top is 2x5, B_bottom is 2x5.
    For each 2x2 x 2x5 product, use Strassen on 2x2x2 parts and schoolbook on 2x2x1 part.
    Total rank = 2 * (7 + 7 + 4) = 36.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # U rows: n*m = 8 (indexed i*m+j)
    # V rows: m*p = 20 (indexed j*p+k)
    # W rows: p*n = 10 (indexed k*n+i)
    
    # Split: A = [A1; A2] where A1 is rows 0,1 (i=0) cols 0,1,2,3? 
    # Actually A is 2x4. Let's split columns of A (which is rows of B):
    # A (2x4) = [A_left (2x2) | A_right (2x2)]
    # B (4x5) = [B_top (2x5); B_bottom (2x5)]
    # C = A * B = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top (2x2 * 2x5):
    #   Split B_top = [B_top_left (2x2) | B_top_mid (2x2) | B_top_right (2x1)]
    #   A_left * B_top = A_left*B_top_left + A_left*B_top_mid + A_left*B_top_right
    #   Use Strassen for first two (each 7), schoolbook for last (4). Total 18.
    # Similarly for A_right * B_bottom: total 18.
    # Grand total: 36.
    
    def make_22x25_block(a_row_offset, b_row_offset):
        """Create decomposition for A_sub (2x2) * B_sub (2x5).
        a_row_offset: starting row index in A (0 or 2)
        b_row_offset: starting row index in B (0 or 2)
        Returns list of (U_block, V_block, W_block) where each has full-size rows.
        """
        decomps = []
        
        # B_sub columns: 0,1 -> B_top_left; 2,3 -> B_top_mid; 4 -> B_top_right
        # For Strassen 2x2x2:
        #   B_top_left has B rows b_row_offset, b_row_offset+1, cols 0,1
        #   B_top_mid has B rows b_row_offset, b_row_offset+1, cols 2,3
        
        # We need to map local 2x2x2 Strassen to global indices.
        # Let's build each component separately.
        
        # Helper: build a single rank-1 or Strassen-like term with global indices
        # For Strassen 2x2x2, we need to map local (i,j) of A to global, (j,k) of B to global, (k,i) of C to global.
        
        # A_sub is at rows a_row_offset, a_row_offset+1 of A (which is 2x4, so A row = i_local, col = j_local + col_offset)
        # Wait, A is 2x4. A_left is cols 0,1. So A_sub[i_local, j_local] = A[a_row_offset... no, A has 2 rows.
        # A is 2x4. A_left = A[:, 0:2], A_right = A[:, 2:4].
        # So for A_left: A[i, j] where i in {0,1}, j in {0,1} -> global A index i*4+j
        # For A_right: A[i, j] where i in {0,1}, j in {2,3} -> global A index i*4+(j+2)
        
        # B is 4x5. B_top = B[0:2, :], B_bottom = B[2:4, :].
        # B_top_left = B[0:2, 0:2], B_top_mid = B[0:2, 2:4], B_top_right = B[0:2, 4:5]
        # For B_sub at b_row_offset (0 or 2), col_offset (0, 2, or 4):
        #   B_local[j_local, k_local] = B[b_row_offset + j_local, col_offset + k_local]
        #   Global V index = (b_row_offset + j_local)*p + (col_offset + k_local)
        
        # C is 2x5. C[i, k] = global W index k*n + i = k*2 + i
        
        a_col_offset = 0 if a_row_offset == 0 else 2  # A_left vs A_right
        # Actually a_row_offset here refers to which A block: 0 for left, 2 for right? No.
        # Let me redefine: for A_left, a_col_offset=0; for A_right, a_col_offset=2.
        # The a_row_offset parameter was confusing. Let me use a_col_offset.
        
        # Actually, let's just pass a_col_offset and b_row_offset.
        pass
    
    # Let me just build the three 2x2x2x5 decompositions directly.
    
    def strassen_222_global(a_col_off, b_row_off, b_col_off):
        """Strassen 2x2x2 with A at A[:, a_col_off:a_col_off+2], B at B[b_row_off:b_row_off+2, b_col_off:b_col_off+2].
        C result goes to C[:, b_col_off:b_col_off+2].
        """
        U = [[0]*7 for _ in range(8)]
        V = [[0]*7 for _ in range(20)]
        W = [[0]*7 for _ in range(10)]
        
        # Local Strassen
        u_l, v_l, w_l = _strassen_222()
        
        for c in range(7):
            for i_local in range(2):
                for j_local in range(2):
                    u_idx = i_local*2 + j_local
                    if u_l[u_idx][c] != 0:
                        global_a_row = i_local  # A has 2 rows
                        global_a_col = a_col_off + j_local
                        U[global_a_row*4 + global_a_col][c] += u_l[u_idx][c]
            
            for j_local in range(2):
                for k_local in range(2):
                    v_idx = j_local*2 + k_local
                    if v_l[v_idx][c] != 0:
                        global_b_row = b_row_off + j_local
                        global_b_col = b_col_off + k_local
                        V[global_b_row*5 + global_b_col][c] += v_l[v_idx][c]
            
            for k_local in range(2):
                for i_local in range(2):
                    w_idx = k_local*2 + i_local
                    if w_l[w_idx][c] != 0:
                        global_c_row = i_local
                        global_c_col = b_col_off + k_local
                        W[global_c_col*2 + global_c_row][c] += w_l[w_idx][c]
        
        return U, V, W
    
    def schoolbook_221_global(a_col_off, b_row_off, b_col_off):
        """Schoolbook 2x2x1 with A at A[:, a_col_off:a_col_off+2], B at B[b_row_off:b_row_off+2, b_col_off:b_col_off+1].
        C result goes to C[:, b_col_off:b_col_off+1].
        """
        U = [[0]*4 for _ in range(8)]
        V = [[0]*4 for _ in range(20)]
        W = [[0]*4 for _ in range(10)]
        
        u_l, v_l, w_l = _schoolbook_221()
        
        for c in range(4):
            for i_local in range(2):
                for j_local in range(2):
                    u_idx = i_local*2 + j_local
                    if u_l[u_idx][c] != 0:
                        global_a_row = i_local
                        global_a_col = a_col_off + j_local
                        U[global_a_row*4 + global_a_col][c] += u_l[u_idx][c]
            
            for j_local in range(2):
                v_idx = j_local
                if v_l[v_idx][c] != 0:
                    global_b_row = b_row_off + j_local
                    global_b_col = b_col_off
                    V[global_b_row*5 + global_b_col][c] += v_l[v_idx][c]
            
            for i_local in range(2):
                w_idx = i_local
                if w_l[w_idx][c] != 0:
                    global_c_row = i_local
                    global_c_col = b_col_off
                    W[global_c_col*2 + global_c_row][c] += w_l[w_idx][c]
        
        return U, V, W
    
    # Now build the full decomposition
    decomps = []
    
    # A_left (cols 0,1) * B_top (rows 0,1)
    #   Strassen for B_top_left (cols 0,1)
    decomps.append(strassen_222_global(0, 0, 0))
    #   Strassen for B_top_mid (cols 2,3)
    decomps.append(strassen_222_global(0, 0, 2))
    #   Schoolbook for B_top_right (col 4)
    decomps.append(schoolbook_221_global(0, 0, 4))
    
    # A_right (cols 2,3) * B_bottom (rows 2,3)
    #   Strassen for B_bottom_left (cols 0,1)
    decomps.append(strassen_222_global(2, 2, 0))
    #   Strassen for B_bottom_mid (cols 2,3)
    decomps.append(strassen_222_global(2, 2, 2))
    #   Schoolbook for B_bottom_right (col 4)
    decomps.append(schoolbook_221_global(2, 2, 4))
    
    # Combine all
    U_full, V_full, W_full = _combine_decomps(decomps)
    
    return {"U": U_full, "V": V_full, "W": W_full}
