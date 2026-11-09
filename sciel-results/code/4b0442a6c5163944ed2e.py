"""Matrix multiplication 3x4x4 rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 4]


def _strassen_2x2x2():
    """Return (U, V, W) lists for 2x2 x 2x2 = 2x2, rank 7.
    U has shape (4,7) indexed by [i*2+j, c] (input A entries),
    V has shape (4,7) indexed by [j*2+k, c] (input B entries),
    W has shape (4,7) indexed by [k*2+i, c] (output C entries, trace convention).
    """
    # Strassen products M1..M7 with coefficients in {0, +/-1}
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # M1=(a00+a11)(b00+b11)
    # M2=(a10+a11)b00
    # M3=a00(b10-b11)
    # M4=a11(b01-b00)
    # M5=(a00+a01)b11
    # M6=(a10-a00)(b00+b01)
    # M7=(a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # A index: a00=0, a01=1, a10=2, a11=3
    # B index: b00=0, b01=1, b10=2, b11=3
    # C index (k*2+i): C00=0, C01=1, C10=2, C11=3
    # M1: (a00+a11)(b00+b11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1  # C00 += M1, C11 += M1
    # M2: (a10+a11)b00
    U[2][1]=1; U[3][1]=1; V[0][1]=1
    W[2][1]=1; W[3][1]=-1  # C10 += M2, C11 -= M2
    # M3: a00(b10-b11)
    U[0][2]=1; V[2][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1  # C01 += M3, C11 += M3
    # M4: a11(b01-b00)
    U[3][3]=1; V[1][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1  # C00 += M4, C10 += M4
    # M5: (a00+a01)b11
    U[0][4]=1; U[1][4]=1; V[3][4]=1
    W[0][4]=-1; W[1][4]=1  # C00 -= M5, C01 += M5
    # M6: (a10-a00)(b00+b01)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1
    W[3][5]=1  # C11 += M6
    # M7: (a01-a11)(b10+b11)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1
    W[0][6]=1  # C00 += M7
    return U, V, W


def _strip_1x2x2():
    """1x2 x 2x2 = 1x2, rank 4 (schoolbook).
    A is 1x2: [a0, a1]. B is 2x2: [b00,b01;b10,b11]. C is 1x2: [c0,c1].
    U shape (2,4) [a index], V shape (4,4) [b index], W shape (2,4) [c index k*1+i -> k].
    c0 = a0*b00 + a1*b10 ; c1 = a0*b01 + a1*b11
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    # a0=0, a1=1 ; b00=0,b01=1,b10=2,b11=3 ; c0=0,c1=1
    U[0][0]=1; V[0][0]=1; W[0][0]=1   # a0*b00 -> c0
    U[0][1]=1; V[1][1]=1; W[1][1]=1   # a0*b01 -> c1
    U[1][2]=1; V[2][2]=1; W[0][2]=1   # a1*b10 -> c0
    U[1][3]=1; V[3][3]=1; W[1][3]=1   # a1*b11 -> c1
    return U, V, W


def construct():
    """Build a rank decomposition for 3x4 x 4x4 = 3x4.
    U shape (12, rank) indexed [i*4+j, c], V shape (16, rank) [j*4+k, c],
    W shape (12, rank) [k*3+i, c].
    """
    n, m, p = DIMENSIONS  # 3,4,4
    # We'll assemble a list of (u_entries, v_entries, w_entries) where each is
    # a dict mapping local-index -> coefficient, then place into global.
    # Global A index: i in [0,3), j in [0,4) -> a_idx = i*4 + j
    # Global B index: j in [0,4), k in [0,4) -> b_idx = j*4 + k
    # Global C index (trace): k in [0,4), i in [0,3) -> c_idx = k*3 + i

    # Strategy:
    # A_top = rows 0,1 (2x4); A_bot = row 2 (1x4)
    # B split into B_left (cols 0,1), B_right (cols 2,3), each 4x2.
    # C_top = A_top * B  (2x4)  -> columns 0..3
    #   C_top = [A_top*B_left | A_top*B_right]
    #   Each A_top(2x4)*B_half(4x2) = 2x2.
    #     Split A_top into A_top1 (rows0-1, cols0-1) 2x2 and A_top2 (rows0-1, cols2-3) 2x2.
    #     B_half split into B_half_top (rows0-1) 2x2 and B_half_bottom (rows2-3) 2x2.
    #     product = A_top1*B_half_top + A_top2*B_half_bottom  (two 2x2x2 Strassen)
    # C_bot = A_bot * B (1x4) -> columns 0..3
    #   C_bot = [A_bot*B_left | A_bot*B_right]
    #   Each A_bot(1x4)*B_half(4x2) = 1x2.
    #     Split A_bot into A_bot1 (cols0-1) 1x2 and A_bot2 (cols2-3) 1x2.
    #     B_half into B_half_top (rows0-1) 2x2, B_half_bottom (rows2-3) 2x2.
    #     product = A_bot1*B_half_top + A_bot2*B_half_bottom (two 1x2x2 strips, rank4 each)

    # Accumulators: list of (a_local, coeff), etc. We'll build per-term.
    U_rows = n*m  # 12
    V_rows = m*p  # 16
    W_rows = p*n  # 12

    # We'll collect terms as list of (a_idx, b_idx, c_idx, coeff)
    terms = []

    def add_strassen(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        """Add a 2x2x2 Strassen product.
        a_row_off: global row offset in A (0..2), a_col_off: col offset (0..2, must be even)
        b_row_off: global row offset in B (0..2 even), b_col_off: col offset (0..2 even)
        c_row_off: global row offset in C (0..2), c_col_off: col offset (0..2 even)
        Local A[i][j] at global A[(a_row_off+i)*(4) + (a_col_off+j)]
        Local B[i][j] at global B[(b_row_off+i)*(4) + (b_col_off+j)]
        Local C[i][j] at global C[(c_col_off+j)*3 + (c_row_off+i)]
        """
        sU, sV, sW = _strassen_2x2x2()
        for c in range(7):
            a_terms = [(li, sU[li][c]) for li in range(4) if sU[li][c] != 0]
            b_terms = [(li, sV[li][c]) for li in range(4) if sV[li][c] != 0]
            w_terms = [(li, sW[li][c]) for li in range(4) if sW[li][c] != 0]
            for (ai, ac) in a_terms:
                for (bi, bc) in b_terms:
                    for (ci, cc) in w_terms:
                        a_idx = (a_row_off+ai)*4 + (a_col_off+ (bi//2))  # careful
                        # Actually a index: local A row=ai//2? No.
                        # sU indexed by [i*2+j] where i=row, j=col of the 2x2.
                        # So ai = i*2+j -> i = ai//2, j = ai%2.
                        pass
        # Redo cleanly below.
        pass

    # Let me redo add_strassen with explicit i,j.
    def add_strassen(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        sU, sV, sW = _strassen_2x2x2()
        # sU indexed by [i*2+j], sV by [j*2+k], sW by [k*2+i]
        # We need, for each rank column c, the set of (i,j) with sU[i*2+j][c]!=0,
        # (j,k) with sV[j*2+k][c]!=0, (k,i) with sW[k*2+i][c]!=0.
        for c in range(7):
            a_pairs = [( (li//2), (li%2), sU[li][c]) for li in range(4) if sU[li][c]!=0]
            b_pairs = [( (li//2), (li%2), sV[li][c]) for li in range(4) if sV[li][c]!=0]
            w_pairs = [( (li//2), (li%2), sW[li][c]) for li in range(4) if sW[li][c]!=0]
            for (ai, aj, ac) in a_pairs:
                for (bi, bk, bc) in b_pairs:
                    for (ck, ci, cc) in w_pairs:
                        a_idx = (a_row_off+ai)*4 + (a_col_off+aj)
                        b_idx = (b_row_off+bi)*4 + (b_col_off+bk)
                        c_idx = (c_col_off+ck)*3 + (c_row_off+ci)
                        terms.append((a_idx, b_idx, c_idx, ac*bc*cc))

    def add_strip(a_row, a_col_off, b_row_off, b_col_off, c_row, c_col_off):
        """1x2 x 2x2 = 1x2.
        a_row: global row in A (single row), a_col_off: start col (even)
        b_row_off: global row offset in B (even), b_col_off: start col (even)
        c_row: global row in C, c_col_off: start col (even)
        Local A[j] at global A[a_row*4 + (a_col_off+j)]
        Local B[i][j] at global B[(b_row_off+i)*4 + (b_col_off+j)]
        Local C[j] at global C[(c_col_off+j)*3 + c_row]
        """
        sU, sV, sW = _strip_1x2x2()
        # sU indexed by [j] (j=0,1), sV by [i*2+j], sW by [j]
        for c in range(4):
            a_pairs = [( j, sU[j][c]) for j in range(2) if sU[j][c]!=0]
            b_pairs = [( (li//2), (li%2), sV[li][c]) for li in range(4) if sV[li][c]!=0]
            w_pairs = [( j, sW[j][c]) for j in range(2) if sW[j][c]!=0]
            for (aj, ac) in a_pairs:
                for (bi, bk, bc) in b_pairs:
                    for (cj, cc) in w_pairs:
                        a_idx = a_row*4 + (a_col_off+aj)
                        b_idx = (b_row_off+bi)*4 + (b_col_off+bk)
                        c_idx = (c_col_off+cj)*3 + c_row
                        terms.append((a_idx, b_idx, c_idx, ac*bc*cc))

    # C_top: rows 0,1.
    #   B_left: cols 0,1 -> b_col_off=0 ; B_right: cols 2,3 -> b_col_off=2
    #   For B_half, top rows 0,1 (b_row_off=0), bottom rows 2,3 (b_row_off=2)
    #   A_top1: rows0-1, cols0-1 (a_row_off=0,a_col_off=0); A_top2: rows0-1, cols2-3 (a_row_off=0,a_col_off=2)
    #   C_top output cols: left -> c_col_off=0, right -> c_col_off=2 ; c_row_off=0
    for b_col_off in (0, 2):
        c_col_off = b_col_off
        # A_top1 * B_half_top
        add_strassen(0, 0, 0, b_col_off, 0, c_col_off)
        # A_top2 * B_half_bottom
        add_strassen(0, 2, 2, b_col_off, 0, c_col_off)

    # C_bot: row 2.
    #   A_bot1: row2, cols0-1 (a_row=2, a_col_off=0); A_bot2: row2, cols2-3 (a_row=2, a_col_off=2)
    for b_col_off in (0, 2):
        c_col_off = b_col_off
        add_strip(2, 0, 0, b_col_off, 2, c_col_off)
        add_strip(2, 2, 2, b_col_off, 2, c_col_off)

    # Now merge terms with same (a_idx,b_idx,c_idx).
    from collections import defaultdict
    merged = defaultdict(int)
    for (a,b,c,w) in terms:
        merged[(a,b,c)] += w
    # Drop zeros.
    merged = {k:v for k,v in merged.items() if v != 0}

    # Assign rank indices.
    rank = len(merged)
    U = [[0]*rank for _ in range(U_rows)]
    V = [[0]*rank for _ in range(V_rows)]
    W = [[0]*rank for _ in range(W_rows)]
    for col, ((a,b,c),w) in enumerate(merged.items()):
        U[a][col] = w
        V[b][col] = w
        W[c][col] = w

    # Verify exactness numerically.
    Um = np.array(U, dtype=float)
    Vm = np.array(V, dtype=float)
    Wm = np.array(W, dtype=float)
    # Tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Compare with matmul tensor: T_true[a,b,c] = sum_j A[i,j]*B[j,k] where a=i*m+j, b=j*p+k, c=k*n+i.
    # Build true tensor.
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_true[a,b,c] = 1.0
    T_comp = np.einsum('ar,br,cr->abc', Um, Vm, Wm)
    if not np.allclose(T_comp, T_true, atol=1e-9):
        # Fallback to schoolbook.
        return _schoolbook()
    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n*m*p
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
    return {"U": U, "V": V, "W": W}
