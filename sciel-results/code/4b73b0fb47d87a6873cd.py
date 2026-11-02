"""Matrix multiplication 3,4,4 - hybrid Strassen/schoolbook decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _strassen_2x2x2():
    """Return U,V,W factors (each 4x7) for 2x2x2 matrix multiplication using Strassen's algorithm.
    
    Strassen's 7 products for C = A*B where A,B,C are 2x2:
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    p = M1 + M4 - M5 + M7
    q = M3 + M5
    r = M2 + M4
    s = M1 - M2 + M3 + M6
    """
    # A indices: a=0, b=1, c=2, d=3 (row-major: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3)
    # B indices: e=0, f=1, g=2, h=3
    # C indices: p=0, q=1, r=2, s=3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h): U has a=1,d=1; V has e=1,h=1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    # p += M1, s += M1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (c+d)e: U has c=1,d=1; V has e=1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    # r += M2, s -= M2
    W[2,1] = 1; W[3,1] = -1
    
    # M3 = a(f-h): U has a=1; V has f=1,h=-1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    # q += M3, s += M3
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = d(g-e): U has d=1; V has g=1,e=-1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    # p += M4, r += M4
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (a+b)h: U has a=1,b=1; V has h=1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    # p -= M5, q += M5
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (c-a)(e+f): U has c=1,a=-1; V has e=1,f=1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    # s += M6
    W[3,5] = 1
    
    # M7 = (b-d)(g+h): U has b=1,d=-1; V has g=1,h=1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    # p += M7
    W[0,6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    """Construct rank-minimal decomposition for 3x4 x 4x4 matrix multiplication."""
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4  # A is n x m, B is m x p, C is n x p
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (16, rank)  
    # W shape: (p*n, rank) = (12, rank), indexed as W[k*n + i, r]
    
    # Strategy: Partition A into top 2 rows and bottom 1 row.
    # Partition B into left 2 cols and right 2 cols.
    # 
    # A_top (2x4) * B_left (4x2) -> C_top_left (2x2): Strassen, rank 7
    # A_top (2x4) * B_right (4x2) -> C_top_right (2x2): Strassen, rank 7
    # A_bot (1x4) * B_left (4x2) -> C_bot_left (1x2): trivial, rank 2
    # A_bot (1x4) * B_right (4x2) -> C_bot_right (1x2): trivial, rank 2
    #
    # But wait - we need to be more careful. The full multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let me index: A has rows i=0,1,2 and cols j=0,1,2,3
    # B has rows j=0,1,2,3 and cols k=0,1,2,3
    # C has rows i=0,1,2 and cols k=0,1,2,3
    #
    # For Strassen on 2x2x2, we need 2x2 submatrices.
    # Let's partition:
    # A: rows {0,1} as "top", row {2} as "bot"
    #    cols {0,1} as "left", cols {2,3} as "right"
    # B: rows {0,1} as "top", rows {2,3} as "bot"
    #    cols {0,1} as "left", cols {2,3} as "right"
    #
    # This gives 2x2 blocks:
    # A_tl = A[0:2, 0:2], A_tr = A[0:2, 2:4]
    # A_bl = A[2:3, 0:2],  A_br = A[2:3, 2:4]
    # B_tl = B[0:2, 0:2], B_tr = B[0:2, 2:4]
    # B_bl = B[2:4, 0:2], B_br = B[2:4, 2:4]
    #
    # C[0:2, 0:2] = A_tl * B_tl + A_tr * B_bl  (2x2 = 2x2*2x2 + 2x2*2x2)
    # C[0:2, 2:4] = A_tl * B_tr + A_tr * B_br  (2x2 = 2x2*2x2 + 2x2*2x2)
    # C[2:3, 0:2] = A_bl * B_tl + A_br * B_bl  (1x2 = 1x2*2x2 + 1x2*2x2)
    # C[2:3, 2:4] = A_bl * B_tr + A_br * B_br  (1x2 = 1x2*2x2 + 1x2*2x2)
    #
    # For the 2x2*2x2 -> 2x2 products, use Strassen (rank 7 each)
    # For the 1x2*2x2 -> 1x2 products, use trivial (rank 2 each, since it's just dot products)
    #
    # Total rank: 4 * 7 + 4 * 2 = 28 + 8 = 36? 
    # Hmm, but the knowledge says 44. Let me reconsider.
    #
    # Actually, I think the issue is that Strassen's algorithm applies to the 2x2x2 tensor,
    # but here we have products like A_tl * B_tl where A_tl is 2x2 and B_tl is 2x2.
    # Each such product contributes 7 rank-1 terms. There are 4 such products (for C[0:2,0:2] 
    # and C[0:2,2:4]), so 4*7 = 28.
    # For the bottom row: C[2:3,0:2] = A_bl*B_tl + A_br*B_bl. 
    # A_bl is 1x2, B_tl is 2x2. The product is 1x2. Each element is a dot product of 
    # length 2, so 2 terms per element, 2 elements = 4 terms? No wait...
    #
    # C[2,0] = A_bl[0]*B_tl[0,0] + A_bl[1]*B_tl[1,0]  -> 2 terms
    # C[2,1] = A_bl[0]*B_tl[0,1] + A_bl[1]*B_tl[1,1]  -> 2 terms
    # Similarly for C[2:3,2:4]: 4 terms
    # Total for bottom: 8 terms
    # Grand total: 28 + 8 = 36
    #
    # But the knowledge says 44. Maybe my partitioning is suboptimal or I'm missing something.
    # Let me just implement rank 36 and verify it's correct. If it works, great.
    # If not, fall back to schoolbook.
    
    # Let me build the decomposition step by step.
    # First, get the Strassen factors for 2x2x2.
    Su, Sv, Sw = _strassen_2x2x2()
    # Su, Sv, Sw are 4x7 matrices (as lists)
    # Su[a_idx, t] = coefficient of A[a_idx] in product t
    # Sv[b_idx, t] = coefficient of B[b_idx] in product t  
    # Sw[c_idx, t] = coefficient in C[c_idx] = sum_t Sw[c_idx,t] * (Su * Sv)_t
    
    # Now map to our problem.
    # A is 3x4: A[i,j] for i in 0..2, j in 0..3
    # B is 4x4: B[j,k] for j in 0..3, k in 0..3
    # C is 3x4: C[i,k] for i in 0..2, k in 0..3
    # U is 12xR: U[i*4+j, r]
    # V is 16xR: V[j*4+k, r]
    # W is 12xR: W[k*3+i, r]
    
    R = 36  # expected rank
    U = [[0]*R for _ in range(12)]
    V = [[0]*R for _ in range(16)]
    W = [[0]*R for _ in range(12)]
    
    col = 0
    
    # Helper: add a Strassen block for A_block (2x2) * B_block (2x2) -> C_block (2x2)
    # A_block rows: i_off, i_off+1; cols: j_off, j_off+1
    # B_block rows: j_off, j_off+1; cols: k_off, k_off+1
    # C_block rows: i_off, i_off+1; cols: k_off, k_off+1
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        for t in range(7):
            for a_i in range(2):
                for a_j in range(2):
                    val = Su[a_i*2 + a_j][t]
                    if val != 0:
                        U[(i_off + a_i)*4 + (j_off + a_j)][col] += val
            for b_i in range(2):
                for b_j in range(2):
                    val = Sv[b_i*2 + b_j][t]
                    if val != 0:
                        V[(j_off + b_i)*4 + (k_off + b_j)][col] += val
            for c_i in range(2):
                for c_k in range(2):
                    val = Sw[c_i*2 + c_k][t]
                    if val != 0:
                        W[(k_off + c_k)*3 + (i_off + c_i)][col] += val
            col += 1
    
    # Helper: add trivial 1x2 * 2x2 -> 1x2 block
    # A_block: row i, cols j_off, j_off+1 (1x2)
    # B_block: rows j_off, j_off+1; cols k_off, k_off+1 (2x2)
    # C_block: row i, cols k_off, k_off+1 (1x2)
    # C[i, k_off]   = A[i,j_off]*B[j_off,k_off]   + A[i,j_off+1]*B[j_off+1,k_off]
    # C[i, k_off+1] = A[i,j_off]*B[j_off,k_off+1] + A[i,j_off+1]*B[j_off+1,k_off+1]
    def add_trivial_1x2x2(i, j_off, k_off):
        nonlocal col
        # Term 1: A[i,j_off] * B[j_off, k_off] -> C[i, k_off]
        U[i*4 + j_off][col] = 1
        V[j_off*4 + k_off][col] = 1
        W[k_off*3 + i][col] = 1
        col += 1
        # Term 2: A[i,j_off+1] * B[j_off+1, k_off] -> C[i, k_off]
        U[i*4 + (j_off+1)][col] = 1
        V[(j_off+1)*4 + k_off][col] = 1
        W[k_off*3 + i][col] = 1
        col += 1
        # Term 3: A[i,j_off] * B[j_off, k_off+1] -> C[i, k_off+1]
        U[i*4 + j_off][col] = 1
        V[j_off*4 + (k_off+1)][col] = 1
        W[(k_off+1)*3 + i][col] = 1
        col += 1
        # Term 4: A[i,j_off+1] * B[j_off+1, k_off+1] -> C[i, k_off+1]
        U[i*4 + (j_off+1)][col] = 1
        V[(j_off+1)*4 + (k_off+1)][col] = 1
        W[(k_off+1)*3 + i][col] = 1
        col += 1
    
    # Four Strassen blocks for the top 2 rows of A:
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    add_strassen(0, 0, 0)   # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 2, 0)   # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen(0, 0, 2)   # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(0, 2, 2)   # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    
    # Four trivial blocks for the bottom row of A:
    # C[2, 0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2]
    add_trivial_1x2x2(2, 0, 0)  # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    add_trivial_1x2x2(2, 2, 0)  # A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
    add_trivial_1x2x2(2, 0, 2)  # A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    add_trivial_1x2x2(2, 2, 2)  # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta
    # The tensor T[i,j,k] = 1 for all i,j,k (the multiplication tensor)
    # Actually, the decomposition should satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid term
    # Wait, the tensor for matrix multiplication is T[(i,j),(j,k),(k,i)] = 1
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify by reconstructing
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(R):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1) > 1e-10:
                    error += 1
    
    if error > 0:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Check deadline
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

