import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Split m into m1=2, m2=3 and p into p1=2, p2=3
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    # Global dimensions
    U_global = [[0] * 0 for _ in range(n * m)] # Placeholder, we will build columns
    # Actually, let's determine total rank first to allocate arrays
    # Or just append to lists of lists
    
    # We will collect contributions in lists
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomposition(n_loc, m_loc, p_loc, U_loc, V_loc, W_loc, 
                          i_off, j_off, k_off):
        """
        Add a local decomposition to the global accumulation.
        n_loc, m_loc, p_loc: local dimensions
        U_loc: list of lists (n_loc*m_loc x rank_loc)
        V_loc: list of lists (m_loc*p_loc x rank_loc)
        W_loc: list of lists (p_loc*n_loc x rank_loc)
        i_off, j_off, k_off: offsets for A (rows, cols), B (rows, cols), C (rows, cols)
        Note: A is n x m. B is m x p. C is p x n? No, C is n x p.
        W index is k*n + i. k is col of C (0..p-1), i is row of C (0..n-1).
        """
        rank_loc = len(U_loc[0]) if U_loc else 0
        for c in range(rank_loc):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Fill U
            for i in range(n_loc):
                for j in range(m_loc):
                    val = U_loc[i * m_loc + j][c]
                    if val != 0:
                        u_col[(i + i_off) * m + (j + j_off)] = val
                        
            # Fill V
            for j in range(m_loc):
                for k in range(p_loc):
                    val = V_loc[j * p_loc + k][c]
                    if val != 0:
                        v_col[(j + j_off) * p + (k + k_off)] = val
                        
            # Fill W
            # W index: k*n + i. 
            # Local k is 0..p_loc-1, local i is 0..n_loc-1.
            # Global k is k + k_off, global i is i + i_off.
            for k in range(p_loc):
                for i in range(n_loc):
                    val = W_loc[k * n_loc + i][c]
                    if val != 0:
                        w_col[(k + k_off) * n + (i + i_off)] = val
                        
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    def get_schoolbook(n_loc, m_loc, p_loc):
        rank = n_loc * m_loc * p_loc
        U = [[0] * rank for _ in range(n_loc * m_loc)]
        V = [[0] * rank for _ in range(m_loc * p_loc)]
        W = [[0] * rank for _ in range(p_loc * n_loc)]
        col = 0
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    U[i * m_loc + j][col] = 1
                    V[j * p_loc + k][col] = 1
                    W[k * n_loc + i][col] = 1
                    col += 1
        return U, V, W

    def get_strassen_222():
        # Rank 7 decomposition for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices for A: 00, 01, 10, 11 (i*m+j)
        # Indices for B: 00, 01, 10, 11 (j*p+k)
        # Indices for C: 00, 01, 10, 11 (k*n+i)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1
        # A: 00(1), 11(1). B: 00(1), 11(1).
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C11 = M1 + M4 - M5 + M7
        W[0][0] = 1 # C11
        # C12 = M3 + M5 (M1 not in C12)
        # C21 = M2 + M4 (M1 not in C21)
        # C22 = M1 - M2 + M3 + M6
        W[3][0] = 1 # C22 (k=1, i=1 -> index 1*2+1=3)
        
        # M2
        # A: 10(1), 11(1). B: 00(1).
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21 = M2 + M4
        W[2][1] = 1 # C21 (k=1, i=0 -> index 1*2+0=2)
        
        # M3
        # A: 00(1). B: 01(1), 11(-1).
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C12 = M3 + M5
        W[1][2] = 1 # C12 (k=1, i=0? No. C is 2x2.
        # C indices:
        # C00 (i=0, k=0) -> W[0*2+0]=W[0]
        # C01 (i=0, k=1) -> W[1*2+0]=W[2]
        # C10 (i=1, k=0) -> W[0*2+1]=W[1]
        # C11 (i=1, k=1) -> W[1*2+1]=W[3]
        # Wait, W index is k*n + i.
        # n=2.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # Let's re-map W for Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: C00(1), C11(1)
        W[0][0] = 1
        W[3][0] = 1
        
        # M2: C10(1)
        W[1][1] = 1
        
        # M3: C01(1), C11(1)
        W[2][2] = 1
        W[3][2] = 1
        
        # M4: C00(1), C10(1)
        W[0][3] = 1
        W[1][3] = 1
        
        # M5: C00(-1), C01(1)
        W[0][4] = -1
        W[2][4] = 1
        
        # M6: C11(1)
        W[3][5] = 1
        
        # M7: C00(1)
        W[0][6] = 1
        
        return U, V, W

    # Block 1: A1(2x2) * B11(2x2) -> C11(2x2)
    # A1: i_off=0, j_off=0. B11: j_off=0, k_off=0. C11: i_off=0, k_off=0.
    U1, V1, W1 = get_strassen_222()
    add_decomposition(2, 2, 2, U1, V1, W1, 0, 0, 0)
    
    # Block 2: A2(2x3) * B21(3x2) -> C12(2x2)
    # A2: i_off=0, j_off=2. B21: j_off=2, k_off=0. C12: i_off=0, k_off=0.
    # Note: B21 is rows 2-4, cols 0-1.
    U2, V2, W2 = get_schoolbook(2, 3, 2)
    add_decomposition(2, 3, 2, U2, V2, W2, 0, 2, 0)
    
    # Block 3: A1(2x2) * B12(2x3) -> C21(2x3)
    # A1: i_off=0, j_off=0. B12: j_off=0, k_off=2. C21: i_off=0, k_off=2.
    U3, V3, W3 = get_schoolbook(2, 2, 3)
    add_decomposition(2, 2, 3, U3, V3, W3, 0, 0, 2)
    
    # Block 4: A2(2x3) * B22(3x3) -> C22(2x3)
    # A2: i_off=0, j_off=2. B22: j_off=2, k_off=2. C22: i_off=0, k_off=2.
    U4, V4, W4 = get_schoolbook(2, 3, 3)
    add_decomposition(2, 3, 3, U4, V4, W4, 0, 2, 2)
    
    # Convert lists of columns to lists of rows (as required by interface? 
    # Interface: U (n*m, rank). So U[row][col].
    # We have U_cols which is a list of columns.
    # We need to transpose.
    
    rank = len(U_cols)
    U_final = [[0]*rank for _ in range(n*m)]
    V_final = [[0]*rank for _ in range(m*p)]
    W_final = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for i in range(n*m):
            U_final[i][c] = U_cols[c][i]
        for i in range(m*p):
            V_final[i][c] = V_cols[c][i]
        for i in range(p*n):
            W_final[i][c] = W_cols[c][i]
            
    if time.time() > deadline:
         # Fallback to schoolbook if somehow slow (unlikely)
         return _construct_schoolbook()
         
    return {"U": U_final, "V": V_final, "W": W_final}

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

# EVOLVE-BLOCK-END
