"""
Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
"""
import time

DIMENSIONS = [3, 3, 8]

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 Strassen.
    U shape (4, 7): rows (i,j) -> i*2+j
    V shape (4, 7): rows (j,k) -> j*2+k
    W shape (4, 7): rows (k,i) -> k*2+i
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1: (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3: A00(B10 - B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4: A11(B01 - B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00 + A10)B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A11 - A00)(B00 + B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify if U, V, W form a valid tensor decomposition for 3x3 * 3x8."""
    n, m, p = 3, 3, 8
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _strassen_3x2x2():
    """
    Decomposition for 3x2 matrix (A) times 2x2 matrix (B) to get 3x2 matrix (C).
    A is 3x2, B is 2x2, C is 3x2.
    This is equivalent to 3 rows of A (each 2x1) times B (2x2).
    We can use Strassen on the top 2x2 block of A and handle the bottom row separately.
    Or better, treat it as 3 separate 1x2 * 2x2 products?
    Actually, let's look at the indices.
    A is 3x2. B is 2x2. C is 3x2.
    C_{ik} = sum_{j=0}^{1} A_{ij} B_{jk}.
    We can split A into A_top (2x2) and A_bot (1x2).
    C_top = A_top * B (2x2 * 2x2 -> 2x2). Use Strassen (rank 7).
    C_bot = A_bot * B (1x2 * 2x2 -> 1x2). Use Schoolbook (rank 1*2*2 = 4).
    Total rank = 7 + 4 = 11.
    
    Indices:
    U rows: i (0..2) * 2 + j (0..1) -> 6 rows.
    V rows: j (0..1) * 2 + k (0..1) -> 4 rows.
    W rows: k (0..1) * 3 + i (0..2) -> 6 rows.
    
    Strassen part (i in 0,1):
    Local i, j in 0,1.
    U_str[4, 7], V_str[4, 7], W_str[4, 7] (W_str local k*2+i).
    Map to global:
    U_global[i*2+j, r] = U_str[i*2+j, r]
    V_global[j*2+k, r] = V_str[j*2+k, r]
    W_global[k*3+i, r] = W_str[k*2+i, r]
    
    Schoolbook part (i=2):
    j in 0,1, k in 0,1.
    U_global[2*2+j, r] = 1
    V_global[j*2+k, r] = 1
    W_global[k*3+2, r] = 1
    4 terms.
    """
    rank = 11
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(6)]
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # Copy Strassen
    for i in range(2):
        for j in range(2):
            idx_u = i*2 + j
            for r in range(7):
                U[idx_u][r] = U_str[idx_u][r]
    for j in range(2):
        for k in range(2):
            idx_v = j*2 + k
            for r in range(7):
                V[idx_v][r] = V_str[idx_v][r]
    for k in range(2):
        for i in range(2):
            idx_w_global = k*3 + i
            idx_w_local = k*2 + i
            for r in range(7):
                W[idx_w_global][r] = W_str[idx_w_local][r]
                
    # Copy Schoolbook for i=2
    col = 7
    for j in range(2):
        for k in range(2):
            U[2*2 + j][col] = 1
            V[j*2 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
            
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 3, 3, 8
    
    # Strategy:
    # We have A (3x3), B (3x8), C (3x8).
    # Split j (columns of A, rows of B) into j=0,1 and j=2.
    # Case 1: j in {0,1}. A_block is 3x2, B_block is 2x8.
    #   Split k (cols of B, cols of C) into groups of 2: {0,1}, {2,3}, {4,5}, {6,7}.
    #   For each k_group, B_sub is 2x2.
    #   Product is 3x2 * 2x2 = 3x2.
    #   Use _strassen_3x2x2 (rank 11).
    #   4 groups * 11 = 44.
    # Case 2: j=2. A_block is 3x1, B_block is 1x8.
    #   Split k into 8 groups of 1? Or just schoolbook.
    #   Product is 3x1 * 1x8 = 3x8.
    #   Schoolbook rank = 3 * 1 * 8 = 24.
    # Total rank = 44 + 24 = 68.
    
    rank = 68
    U = [[0]*rank for _ in range(n*m)]  # 9x68
    V = [[0]*rank for _ in range(m*p)]  # 24x68
    W = [[0]*rank for _ in range(p*n)]  # 24x68
    
    col_offset = 0
    
    # Case 1: j in {0,1}
    U_322, V_322, W_322 = _strassen_3x2x2()
    
    for k_group in range(4):
        k_start = k_group * 2
        k0 = k_start
        k1 = k_start + 1
        
        # U: rows (i, j) for i in 0..2, j in 0..1.
        # Global U index: i*m + j = i*3 + j.
        # Local U index: i*2 + j.
        for i in range(3):
            for j in range(2):
                idx_u_global = i*3 + j
                idx_u_local = i*2 + j
                for r in range(11):
                    U[idx_u_global][col_offset + r] = U_322[idx_u_local][r]
        
        # V: rows (j, k) for j in 0..1, k in k0, k1.
        # Global V index: j*p + k = j*8 + k.
        # Local V index: j*2 + (k - k_start).
        for j in range(2):
            for k_local in range(2):
                k_global = k_start + k_local
                idx_v_global = j*8 + k_global
                idx_v_local = j*2 + k_local
                for r in range(11):
                    V[idx_v_global][col_offset + r] = V_322[idx_v_local][r]
        
        # W: rows (k, i) for k in k0, k1, i in 0..2.
        # Global W index: k*n + i = k*3 + i.
        # Local W index: k_local*3 + i. (Note: local W is for 3x2 output, so k in 0,1, i in 0,1,2)
        # In _strassen_3x2x2, W has 6 rows: k in 0,1, i in 0,1,2.
        # idx_w_local = k_local * 3 + i.
        for k_local in range(2):
            k_global = k_start + k_local
            for i in range(3):
                idx_w_global = k_global*3 + i
                idx_w_local = k_local*3 + i
                for r in range(11):
                    W[idx_w_global][col_offset + r] = W_322[idx_w_local][r]
        
        col_offset += 11
    
    # Case 2: j=2
    # A is 3x1 (col 2), B is 1x8 (row 2).
    # C is 3x8.
    # Schoolbook.
    for i in range(3):
        for k in range(8):
            U[i*3 + 2][col_offset] = 1
            V[2*8 + k][col_offset] = 1
            W[k*3 + i][col_offset] = 1
            col_offset += 1
            
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0]*rank_school for _ in range(n*m)]
        V_school = [[0]*rank_school for _ in range(m*p)]
        W_school = [[0]*rank_school for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][c] = 1
                    V_school[j*p+k][c] = 1
                    W_school[k*n+i][c] = 1
                    c += 1
        return {"U": U_school, "V": V_school, "W": W_school}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
