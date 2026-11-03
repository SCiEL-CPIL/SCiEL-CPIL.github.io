"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return (U, V, W) each of shape (2*2, 7) for 2x2x2 Strassen."""
    # Strassen decomposition of 2x2x2 tensor.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    #
    # U rows indexed by i*2+j (i=row of A, j=col of A)
    # V rows indexed by j*2+k (j=row of B, k=col of B)
    # W rows indexed by k*2+i (k=col of B, i=row of C) -- wait, W[k*n+i, c]
    # With n=2: W rows are k*2+i where k is col index of B (0,1) and i is row index of C (0,1)
    
    # Let me define each bilinear form M_r = sum_{i,j} u_{ij} A_{ij} * sum_{j,k} v_{jk} B_{jk}
    # Actually the tensor decomposition is: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    # where T_{ijk} = 1 if (i,j,k) corresponds to a valid product term.
    # For matrix multiplication C = AB, the tensor is T_{i,j,k} = 1 for all i,j,k (it's the full tensor).
    # Wait no. The matrix multiplication tensor T has T[i,j,k] = 1 where C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For Strassen 2x2x2:
    # n=m=p=2
    # U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    
    # M1: (A11+A22)(B11+B22) -> u: A11=1,A22=1; v: B11=1,B22=1; w: C11+=1,C22+=1
    # U[0,0]=1 (A[0,0]), U[3,0]=1 (A[1,1])
    # V[0,0]=1 (B[0,0]), V[3,0]=1 (B[1,1])
    # W[0,0]=1 (C[0,0] via k=0,i=0), W[3,0]=1 (C[1,1] via k=1,i=1)
    
    # M2: (A21+A22)B11 -> u: A21=1,A22=1; v: B11=1; w: C21+=1,C22-=1
    # U[2,1]=1 (A[1,0]), U[3,1]=1 (A[1,1])
    # V[0,1]=1 (B[0,0])
    # W[2,1]=1 (C[1,0] via k=0,i=1), W[3,1]=-1 (C[1,1] via k=1,i=1)
    
    # M3: A11(B21-B22) -> u: A11=1; v: B21=1,B22=-1; w: C12+=1,C22+=1
    # U[0,2]=1 (A[0,0])
    # V[2,2]=1 (B[1,0]), V[3,2]=-1 (B[1,1])
    # W[1,2]=1 (C[0,1] via k=1,i=0), W[3,2]=1 (C[1,1] via k=1,i=1)
    
    # M4: A22(B12-B11) -> u: A22=1; v: B12=1,B11=-1; w: C11-=1,C12+=1
    # U[3,3]=1 (A[1,1])
    # V[1,3]=1 (B[0,1]), V[0,3]=-1 (B[0,0])
    # W[0,3]=-1 (C[0,0] via k=0,i=0), W[1,3]=1 (C[0,1] via k=1,i=0)
    
    # M5: (A11+A12)B22 -> u: A11=1,A12=1; v: B22=1; w: C11+=1,C21+=1
    # U[0,4]=1 (A[0,0]), U[1,4]=1 (A[0,1])
    # V[3,4]=1 (B[1,1])
    # W[0,4]=1 (C[0,0] via k=0,i=0), W[2,4]=1 (C[1,0] via k=0,i=1)
    
    # M6: (A21-A11)(B11+B12) -> u: A21=1,A11=-1; v: B11=1,B12=1; w: C22+=1
    # U[2,5]=1 (A[1,0]), U[0,5]=-1 (A[0,0])
    # V[0,5]=1 (B[0,0]), V[1,5]=1 (B[0,1])
    # W[3,5]=1 (C[1,1] via k=1,i=1)
    
    # M7: (A12-A22)(B21+B22) -> u: A12=1,A22=-1; v: B21=1,B22=1; w: C11+=1
    # U[1,6]=1 (A[0,1]), U[3,6]=-1 (A[1,1])
    # V[2,6]=1 (B[1,0]), V[3,6]=1 (B[1,1])
    # W[0,6]=1 (C[0,0] via k=0,i=0)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return (U, V, W) for schoolbook decomposition of n x m times m x p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _embed_block(U_local, V_local, W_local, i_start, j_start, k_start, n, m, p, global_U, global_V, global_W, rank_start):
    """Embed a local block decomposition into global matrices."""
    n_block = U_local[0].__len__()  # this is wrong, let me use len(U_local[0])
    local_nm = len(U_local)
    local_mp = len(V_local)
    local_pn = len(W_local)
    local_rank = len(U_local[0])
    
    # U_local row index = local_i * m + local_j, embed at (i_start+local_i)*m + (j_start+local_j)
    # But only for local_i in range, local_j in range where the block exists
    # Actually the block covers i_start..i_start+n_block_rows-1, j_start..j_start+m_block_cols-1, etc.
    # Let me compute block dimensions from local shapes:
    # U_local has shape (n_b * m_b, r), V_local has shape (m_b * p_b, r), W_local has shape (p_b * n_b, r)
    
    # Find n_b, m_b, p_b
    # local_nm = n_b * m_b, local_mp = m_b * p_b, local_pn = p_b * n_b
    # We need to figure out the block dimensions. Let's pass them explicitly.
    pass

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 130  # safety margin
    
    # Strategy: split the 3x5x5 problem into blocks and apply Strassen where possible.
    # 
    # The 3x5 matrix A and 5x5 matrix B:
    # A is 3x5, B is 5x5, C = AB is 3x5.
    #
    # Split A into row blocks: [A1; A2; A3] where A1 is 1x5, A2 is 1x5, A3 is 1x5
    # Or better: split into [A_top (2x5); A_bot (1x5)]
    # Split B into col blocks: [B_left (5x3); B_right (5x2)]
    # Then C = [A_top * B_left, A_top * B_right; A_bot * B_left, A_bot * B_right]
    # C_top_left = A_top * B_left: 2x3
    # C_top_right = A_top * B_right: 2x2
    # C_bot_left = A_bot * B_left: 1x3
    # C_bot_right = A_bot * B_right: 1x2
    #
    # For each sub-product, further split if beneficial.
    #
    # Let's use a hierarchical decomposition:
    # 
    # Level 1: Split A as [A1 (2x5); A2 (1x5)], B as [B1 (5x3); B2 (5x2)]
    # Sub-problems:
    #   (2,5,3): A1*B1
    #   (2,5,2): A1*B2
    #   (1,5,3): A2*B1
    #   (1,5,2): A2*B2
    #
    # For (2,5,3): split B1 as [B1a (5x1); B1b (5x2)] -> (2,5,1) + (2,5,2)
    #   (2,5,1): schoolbook rank 10, or split A1 as [A1a (1x5); A1b (1x5)] -> two (1,5,1) = 5+5=10
    #   (2,5,2): split A1 into [1x5; 1x5] -> two (1,5,2) = 5+5=10, OR use Strassen-like
    #     Actually (2,5,2): split A1 as [A1a(1x5); A1b(1x5)], B2 as is -> (1,5,2)+(1,5,2) = 10
    #     Or keep as is: schoolbook 20.
    #     Better: further split. (1,5,2): schoolbook 10.
    #   So (2,5,3) -> (2,5,1)+(2,5,2) -> (1,5,1)+(1,5,1) + (1,5,2)+(1,5,2) = 5+5+10+10 = 30
    #   Schoolbook for (2,5,3) = 30. No gain.
    #
    #   Alternative for (2,5,2): Use Strassen if we can reshape. (2,5,2) doesn't have a 2x2x2 block directly.
    #   But we can split B2 (5x2) into [B2a(5x1); B2b(5x1)] -> two (2,5,1) = 10+10=20. Same as schoolbook.
    #
    #   Hmm, the key is to find 2x2x2 sub-blocks. Let me think differently.
    #
    #   For (2,5,2): Split A1 (2x5) as [A1_left (2x3); A1_right (2x2)], B2 (5x2) as [B2_top (3x2); B2_bot (2x2)]
    #   Then (2,5,2) = A1_left*B2_top + A1_right*B2_bot = (2,3,2) + (2,2,2)
    #   (2,2,2) -> Strassen rank 7!
    #   (2,3,2): Split A1_left as [1x3; 1x3] -> two (1,3,2) = 6+6=12. Or split B2_top...
    #   (1,3,2): schoolbook 6. No Strassen possible (need at least 2 in each dim).
    #   So (2,3,2) = 12, (2,2,2) = 7, total = 19 vs schoolbook 20. Gain of 1.
    #
    # Let me reconsider the whole decomposition more carefully.
    #
    # n=3, m=5, p=5. Schoolbook rank = 75.
    #
    # Approach: partition the 5 intermediate columns into groups and the 3 rows into groups,
    # and the 5 output columns into groups, to create 2x2x2 blocks.
    #
    # Let's partition:
    # Rows of A (n=3): [1, 2] i.e., row 0 alone, rows 1-2 together -> sizes [1, 2]
    # Cols of A / rows of B (m=5): [2, 3] i.e., cols 0-1, cols 2-4 -> sizes [2, 3]
    # Cols of B / cols of C (p=5): [2, 3] i.e., cols 0-1, cols 2-4 -> sizes [2, 3]
    #
    # This gives 2*2*2 = 8 sub-problems:
    # (1,2,2), (1,2,3), (1,3,2), (1,3,3)
    # (2,2,2), (2,2,3), (2,3,2), (2,3,3)
    #
    # Ranks:
    # (1,2,2): schoolbook 4
    # (1,2,3): schoolbook 6
    # (1,3,2): schoolbook 6
    # (1,3,3): schoolbook 9
    # (2,2,2): Strassen 7
    # (2,2,3): ? Split into (2,2,1)+(2,2,2) = 4+7=11, schoolbook 12. Or (2,1,3)+(2,1,3)=6+6=12. Best: 11
    #   Actually (2,2,3): partition m=2 into [1,1], p=3 into [1,2]: (2,1,1)+(2,1,2)+(2,1,1)+(2,1,2) = 2+4+2+4=12
    #   Or partition p=3 into [2,1]: (2,2,2)+(2,2,1) = 7+4=11
    #   So (2,2,3) = 11
    # (2,3,2): partition m=3 into [2,1]: (2,2,2)+(2,1,2) = 7+4=11. Schoolbook 12.
    #   Or partition m=3 into [1,2]: (2,1,2)+(2,2,2) = 4+7=11
    #   So (2,3,2) = 11
    # (2,3,3): partition m=3 into [2,1], p=3 into [2,1]:
    #   (2,2,2)+(2,2,1)+(2,1,2)+(2,1,1) = 7+4+4+2 = 17. Schoolbook 18.
    #   Or other splits... (2,3,3): partition m=[1,2], p=[1,2]:
    #   (2,1,1)+(2,1,2)+(2,2,1)+(2,2,2) = 2+4+4+7 = 17
    #   So (2,3,3) = 17
    #
    # Total: 4+6+6+9+7+11+11+17 = 71. Gain of 4 from schoolbook 75.
    #
    # Let me verify: 4+6=10, +6=16, +9=25, +7=32, +11=43, +11=54, +17=71. Yes, 71.
    #
    # Now I need to construct the actual U, V, W matrices for rank 71.
    #
    # Let me build this step by step, embedding each sub-problem's decomposition.
    
    # Define partition boundaries
    # n=3: row partition [0:1, 1:3] -> sizes [1, 2], starts [0, 1]
    # m=5: col partition [0:2, 2:5] -> sizes [2, 3], starts [0, 2]
    # p=5: col partition [0:2, 2:5] -> sizes [2, 3], starts [0, 2]
    
    n_row_splits = [1, 2]  # sizes
    m_col_splits = [2, 3]  # sizes
    p_col_splits = [2, 3]  # sizes
    
    # Compute offsets
    n_starts = [0]
    for s in n_row_splits[:-1]:
        n_starts.append(n_starts[-1] + s)
    # n_starts = [0, 1]
    
    m_starts = [0]
    for s in m_col_splits[:-1]:
        m_starts.append(m_starts[-1] + s)
    # m_starts = [0, 2]
    
    p_starts = [0]
    for s in p_col_splits[:-1]:
        p_starts.append(p_starts[-1] + s)
    # p_starts = [0, 2]
    
    # For each sub-problem (ni, mi, pi), compute its rank and decomposition
    def get_decomp(ni, mi, pi):
        """Return (U, V, W) for ni x mi x pi with minimal rank using known decompositions."""
        if ni == 1 and mi == 1 and pi == 1:
            return [[[1]], [[1]], [[1]]]
        if ni == 2 and mi == 2 and pi == 2:
            return _strassen_222()
        # Schoolbook fallback
        return _schoolbook_block(ni, mi, pi)
    
    def get_rank(ni, mi, pi):
        if ni == 1 and mi == 1 and pi == 1:
            return 1
        if ni == 2 and mi == 2 and pi == 2:
            return 7
        # Use recursive splitting
        return _compute_min_rank(ni, mi, pi)
    
    def _compute_min_rank(ni, mi, pi):
        """Compute minimum rank using splitting."""
        if ni == 2 and mi == 2 and pi == 2:
            return 7
        if ni == 1 and mi == 1 and pi == 1:
            return 1
        
        best = ni * mi * pi  # schoolbook
        
        # Try all possible splits along each dimension
        # Split ni
        for i1 in range(1, ni):
            i2 = ni - i1
            r = _compute_min_rank(i1, mi, pi) + _compute_min_rank(i2, mi, pi)
            if r < best:
                best = r
        # Split mi
        for j1 in range(1, mi):
            j2 = mi - j1
            r = _compute_min_rank(ni, j1, pi) + _compute_min_rank(ni, j2, pi)
            if r < best:
                best = r
        # Split pi
        for k1 in range(1, pi):
            k2 = pi - k1
            r = _compute_min_rank(ni, mi, k1) + _compute_min_rank(ni, mi, k2)
            if r < best:
                best = r
        return best
    
    # Compute total rank
    total_rank = 0
    for ni in n_row_splits:
        for mi in m_col_splits:
            for pi in p_col_splits:
                total_rank += _compute_min_rank(ni, mi, pi)
    
    # Now build global U, V, W
    global_rank = total_rank
    n_global = n * m  # 15
    m_global = m * p  # 25
    p_global = p * n  # 15
    
    U = [[0]*global_rank for _ in range(n_global)]
    V = [[0]*global_rank for _ in range(m_global)]
    W = [[0]*global_rank for _ in range(p_global)]
    
    rank_offset = 0
    
    for ni_idx, ni in enumerate(n_row_splits):
        i_start = n_starts[ni_idx]
        for mi_idx, mi in enumerate(m_col_splits):
            j_start = m_starts[mi_idx]
            for pi_idx, pi in enumerate(p_col_splits):
                k_start = p_starts[pi_idx]
                
                # Get decomposition for this sub-problem
                ni_r, mi_r, pi_r = ni, mi, pi
                local_rank = _compute_min_rank(ni_r, mi_r, pi_r)
                
                # Get the local U, V, W
                local_U, local_V, local_W = _build_decomp(ni_r, mi_r, pi_r)
                
                # Embed into global
                # local_U has shape (ni*mi, local_rank)
                # local row (li, lj) -> global row (i_start+li)*m + (j_start+lj)
                for li in range(ni_r):
                    for lj in range(mi_r):
                        local_row = li * mi_r + lj
                        global_row = (i_start + li) * m + (j_start + lj)
                        for c in range(local_rank):
                            U[global_row][rank_offset + c] += local_U[local_row][c]
                
                # local_V has shape (mi*pi, local_rank)
                # local row (lj, lk) -> global row (j_start+lj)*p + (k_start+lk)
                for lj in range(mi_r):
                    for lk in range(pi_r):
                        local_row = lj * pi_r + lk
                        global_row = (j_start + lj) * p + (k_start + lk)
                        for c in range(local_rank):
                            V[global_row][rank_offset + c] += local_V[local_row][c]
                
                # local_W has shape (pi*ni, local_rank)
                # local row (lk, li) -> global row (k_start+lk)*n + (i_start+li)
                for lk in range(pi_r):
                    for li in range(ni_r):
                        local_row = lk * ni_r + li
                        global_row = (k_start + lk) * n + (i_start + li)
                        for c in range(local_rank):
                            W[global_row][rank_offset + c] += local_W[local_row][c]
                
                rank_offset += local_rank
    
    return {"U": U, "V": V, "W": W}

def _build_decomp(ni, mi, pi):
    """Build decomposition for ni x mi x pi using recursive splitting and Strassen."""
    if ni == 2 and mi == 2 and pi == 2:
        return _strassen_222()
    if ni == 1 and mi == 1 and pi == 1:
        return [[[1]], [[1]], [[1]]]
    
    # Try to find a beneficial split
    best_rank = ni * mi * pi
    best_split = None  # (dim, pos)
    
    # Try splitting each dimension
    for i1 in range(1, ni):
        r = _compute_min_rank(i1, mi, pi) + _compute_min_rank(ni - i1, mi, pi)
        if r < best_rank:
            best_rank = r
            best_split = (0, i1)
    for j1 in range(1, mi):
        r = _compute_min_rank(ni, j1, pi) + _compute_min_rank(ni, mi - j1, pi)
        if r < best_rank:
            best_rank = r
            best_split = (1, j1)
    for k1 in range(1, pi):
        r = _compute_min_rank(ni, mi, k1) + _compute_min_rank(ni, mi, pi - k1)
        if r < best_rank:
            best_rank = r
            best_split = (2, k1)
    
    if best_split is None:
        return _schoolbook_block(ni, mi, pi)
    
    dim, pos = best_split
    if dim == 0:
        # Split rows
        U1, V1, W1 = _build_decomp(pos, mi, pi)
        U2, V2, W2 = _build_decomp(ni - pos, mi, pi)
        r1 = len(U1[0])
        r2 = len(U2[0])
        # U: (pos*mi, r1) + ((ni-pos)*mi, r2) -> (ni*mi, r1+r2)
        U = [[0]*(r1+r2) for _ in range(ni*mi)]
        V = [[0]*(r1+r2) for _ in range(mi*pi)]
        W = [[0]*(r1+r2) for _ in range(pi*ni)]
        for i in range(pos):
            for j in range(mi):
                lr = i*mi+j
                for c in range(r1):
                    U[lr][c] = U1[lr][c]
        for i in range(ni-pos):
            for j in range(mi):
                lr = (pos+i)*mi+j
                lr1 = i*mi+j
                for c in range(r2):
                    U[lr][r1+c] = U2[lr1][c]
        for c in range(r1):
            for row in range(mi*pi):
                V[row][c] = V1[row][c]
            for row in range(pi*ni):
                W[row][c] = W1[row][c]
        for c in range(r2):
            for row in range(mi*pi):
                V[row][r1+c] = V2[row][c]
            for row in range(pi*ni):
                W[row][r1+c] = W2[row][c]
        return U, V, W
    elif dim == 1:
        # Split columns of A / rows of B
        U1, V1, W1 = _build_decomp(ni, pos, pi)
        U2, V2, W2 = _build_decomp(ni, mi - pos, pi)
        r1 = len(U1[0])
        r2 = len(U2[0])
        U = [[0]*(r1+r2) for _ in range(ni*mi)]
        V = [[0]*(r1+r2) for _ in range(mi*pi)]
        W = [[0]*(r1+r2) for _ in range(pi*ni)]
        for i in range(ni):
            for j in range(pos):
                lr = i*mi+j
                lr1 = i*pos+j
                for c in range(r1):
                    U[lr][c] = U1[lr1][c]
            for j in range(mi-pos):
                lr = i*mi+(pos+j)
                lr2 = i*(mi-pos)+j
                for c in range(r2):
                    U[lr][r1+c] = U2[lr2][c]
        for j in range(pos):
            for k in range(pi):
                lr = j*pi+k
                lr1 = j*pi+k
                for c in range(r1):
                    V[lr][c] = V1[lr1][c]
        for j in range(mi-pos):
            for k in range(pi):
                lr = (pos+j)*pi+k
                lr2 = j*pi+k
                for c in range(r2):
                    V[lr][r1+c] = V2[lr2][c]
        for c in range(r1):
            for row in range(pi*ni):
                W[row][c] = W1[row][c]
        for c in range(r2):
            for row in range(pi*ni):
                W[row][r1+c] = W2[row][c]
        return U, V, W
    else:
        # Split columns of B / columns of C
        U1, V1, W1 = _build_decomp(ni, mi, pos)
        U2, V2, W2 = _build_decomp(ni, mi, pi - pos)
        r1 = len(U1[0])
        r2 = len(U2[0])
        U = [[0]*(r1+r2) for _ in range(ni*mi)]
        V = [[0]*(r1+r2) for _ in range(mi*pi)]
        W = [[0]*(r1+r2) for _ in range(pi*ni)]
        for c in range(r1):
            for row in range(ni*mi):
                U[row][c] = U1[row][c]
        for c in range(r2):
            for row in range(ni*mi):
                U[row][r1+c] = U2[row][c]
        for j in range(mi):
            for k in range(pos):
                lr = j*pi+k
                lr1 = j*pos+k
                for c in range(r1):
                    V[lr][c] = V1[lr1][c]
            for k in range(pi-pos):
                lr = j*pi+(pos+k)
                lr2 = j*(pi-pos)+k
                for c in range(r2):
                    V[lr][r1+c] = V2[lr2][c]
        for k in range(pos):
            for i in range(ni):
                lr = k*ni+i
                lr1 = k*ni+i
                for c in range(r1):
                    W[lr][c] = W1[lr1][c]
        for k in range(pi-pos):
            for i in range(ni):
                lr = (pos+k)*ni+i
                lr2 = k*ni+i
                for c in range(r2):
                    W[lr][r1+c] = W2[lr2][c]
        return U, V, W
# EVOLVE-BLOCK-END
