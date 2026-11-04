import numpy as np
import time

DIMENSIONS = [4, 5, 6]
DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Total rank
    total_rank = 114
    
    # Initialize matrices
    # U: (n*m, rank) = (20, 114)
    # V: (m*p, rank) = (30, 114)
    # W: (p*n, rank) = (24, 114)
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    def add_strassen_2x2x2(i_start, j_start, k_start):
        # i_start, j_start, k_start are the starting indices in the global A, B, C
        # A block is 2x2 at (i_start, j_start)
        # B block is 2x2 at (j_start, k_start)
        # C block is 2x2 at (i_start, k_start)
        
        # Strassen 2x2x2 factors
        # U_local (4, 7), V_local (4, 7), W_local (4, 7)
        # Indices:
        # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # Standard Strassen 2x2x2 tensor
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U_loc = np.zeros((4, 7), dtype=np.float64)
        V_loc = np.zeros((4, 7), dtype=np.float64)
        W_loc = np.zeros((4, 7), dtype=np.float64)
        
        # M1
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1; W_loc[1, 0] = -1; W_loc[2, 0] = 1
        # Wait, W indices for C11, C12, C21, C22 are 0,1,2,3.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the contributions to W for each M
        # M1: C11 += 1, C22 += 1
        # M2: C21 += 1, C22 -= 1
        # M3: C12 += 1, C22 += 1
        # M4: C11 += 1, C21 += 1
        # M5: C11 -= 1, C12 += 1
        # M6: C22 += 1
        # M7: C11 += 1
        
        U_loc[:, 0] = [1, 0, 0, 1] # A11, A22
        V_loc[:, 0] = [1, 0, 0, 1] # B11, B22
        W_loc[:, 0] = [1, 0, 0, 1] # C11, C22
        
        U_loc[:, 1] = [0, 0, 1, 1] # A21, A22
        V_loc[:, 1] = [1, 0, 0, 0] # B11
        W_loc[:, 1] = [0, 0, 1, -1] # C21, C22
        
        U_loc[:, 2] = [1, 0, 0, 0] # A11
        V_loc[:, 2] = [0, 1, 1, 0] # B21, B11 (B21 - B11) -> V[1]=1, V[0]=-1?
        # V_loc[:, 2] corresponds to B21 - B11.
        # V rows: 0:(0,0)=B11, 1:(0,1)=B12, 2:(1,0)=B21, 3:(1,1)=B22.
        # So B21 is index 2, B11 is index 0.
        V_loc[:, 2] = [-1, 0, 1, 0]
        W_loc[:, 2] = [0, 1, 0, 1] # C12, C22
        
        U_loc[:, 3] = [0, 0, 0, 1] # A22
        V_loc[:, 3] = [0, 1, 0, -1] # B12 - B22. B12 is idx 1, B22 is idx 3.
        W_loc[:, 3] = [1, 0, 1, 0] # C11, C21
        
        U_loc[:, 4] = [1, 1, 0, 0] # A11, A12
        V_loc[:, 4] = [0, 0, 0, 1] # B22
        W_loc[:, 4] = [-1, 1, 0, 0] # C11, C12
        
        U_loc[:, 5] = [1, 0, -1, 0] # A21 - A11. A21 idx 2, A11 idx 0.
        V_loc[:, 5] = [1, 1, 0, 0] # B11, B12
        W_loc[:, 5] = [0, 0, 0, 1] # C22
        
        U_loc[:, 6] = [0, 1, 0, -1] # A12 - A22. A12 idx 1, A22 idx 3.
        V_loc[:, 6] = [0, 0, 1, 1] # B21, B22
        W_loc[:, 6] = [1, 0, 0, 0] # C11
        
        # Map local indices to global
        # U global row: i*m + j
        # i in [i_start, i_start+1], j in [j_start, j_start+1]
        # Local U row 0: i=0, j=0 -> Global i_start, j_start
        # Local U row 1: i=0, j=1 -> Global i_start, j_start+1
        # Local U row 2: i=1, j=0 -> Global i_start+1, j_start
        # Local U row 3: i=1, j=1 -> Global i_start+1, j_start+1
        
        for r in range(7):
            for lr in range(4):
                if U_loc[lr, r] != 0:
                    gi = i_start + (lr // 2)
                    gj = j_start + (lr % 2)
                    U[gi * m + gj, col_offset + r] = U_loc[lr, r]
                    
        for r in range(7):
            for lr in range(4):
                if V_loc[lr, r] != 0:
                    gj = j_start + (lr // 2)
                    gk = k_start + (lr % 2)
                    V[gj * p + gk, col_offset + r] = V_loc[lr, r]
                    
        for r in range(7):
            for lr in range(4):
                if W_loc[lr, r] != 0:
                    gk = k_start + (lr // 2)
                    gi = i_start + (lr % 2)
                    # W index is k*n + i
                    W[gk * n + gi, col_offset + r] = W_loc[lr, r]

    # Helper to add a 2x3x2 Schoolbook decomposition
    def add_schoolbook_2x3x2(i_start, j_start, k_start):
        # A block 2x3 at (i_start, j_start)
        # B block 3x2 at (j_start, k_start)
        # C block 2x2 at (i_start, k_start)
        
        # Schoolbook: 2*3*2 = 12 terms
        U_loc = np.zeros((6, 12), dtype=np.float64)
        V_loc = np.zeros((6, 12), dtype=np.float64)
        W_loc = np.zeros((4, 12), dtype=np.float64)
        
        col = 0
        for li in range(2):
            for lj in range(3):
                for lk in range(2):
                    # A[li, lj] * B[lj, lk] -> C[li, lk]
                    U_loc[li * 3 + lj, col] = 1
                    V_loc[lj * 2 + lk, col] = 1
                    W_loc[lk * 2 + li, col] = 1 # W index k*n + i -> lk * 2 + li?
                    # Wait, W local index for 2x2 C is k*n + i.
                    # n=2. So index is k*2 + i.
                    # Here li is i (0,1), lk is k (0,1).
                    # So index is lk * 2 + li.
                    col += 1
                    
        # Map local to global
        # U local row: li * 3 + lj. li in 0,1. lj in 0,1,2.
        # Global i = i_start + li. Global j = j_start + lj.
        for r in range(12):
            for lr in range(6):
                if U_loc[lr, r] != 0:
                    li = lr // 3
                    lj = lr % 3
                    gi = i_start + li
                    gj = j_start + lj
                    U[gi * m + gj, col_offset + r] = U_loc[lr, r]
                    
        # V local row: lj * 2 + lk. lj in 0,1,2. lk in 0,1.
        # Global j = j_start + lj. Global k = k_start + lk.
        for r in range(12):
            for lr in range(6):
                if V_loc[lr, r] != 0:
                    lj = lr // 2
                    lk = lr % 2
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk, col_offset + r] = V_loc[lr, r]
                    
        # W local row: lk * 2 + li. lk in 0,1. li in 0,1.
        # Global k = k_start + lk. Global i = i_start + li.
        for r in range(12):
            for lr in range(4):
                if W_loc[lr, r] != 0:
                    lk = lr // 2
                    li = lr % 2
                    gk = k_start + lk
                    gi = i_start + li
                    W[gk * n + gi, col_offset + r] = W_loc[lr, r]

    # Part 1: A1 B1 (4x2 * 2x6)
    # A1 rows 0,1 (i=0,1) and 2,3 (i=2,3). j=0,1.
    # B1 rows 0,1 (j=0,1). k=0,1 (k1), 2,3 (k2), 4,5 (k3).
    
    # 6 blocks of 2x2x2
    # i_start, j_start, k_start
    add_strassen_2x2x2(0, 0, 0) # A1t B1l
    col_offset += 7
    add_strassen_2x2x2(0, 0, 2) # A1t B1m
    col_offset += 7
    add_strassen_2x2x2(0, 0, 4) # A1t B1r
    col_offset += 7
    add_strassen_2x2x2(2, 0, 0) # A1b B1l
    col_offset += 7
    add_strassen_2x2x2(2, 0, 2) # A1b B1m
    col_offset += 7
    add_strassen_2x2x2(2, 0, 4) # A1b B1r
    col_offset += 7
    
    # Part 2: A2 B2 (4x3 * 3x6)
    # A2 rows 0,1 (i=0,1) and 2,3 (i=2,3). j=2,3,4.
    # B2 rows 2,3,4 (j=2,3,4). k=0,1 (k1), 2,3 (k2), 4,5 (k3).
    
    # 6 blocks of 2x3x2
    add_schoolbook_2x3x2(0, 2, 0) # A2t B2l
    col_offset += 12
    add_schoolbook_2x3x2(0, 2, 2) # A2t B2m
    col_offset += 12
    add_schoolbook_2x3x2(0, 2, 4) # A2t B2r
    col_offset += 12
    add_schoolbook_2x3x2(2, 2, 0) # A2b B2l
    col_offset += 12
    add_schoolbook_2x3x2(2, 2, 2) # A2b B2m
    col_offset += 12
    add_schoolbook_2x3x2(2, 2, 4) # A2b B2r
    col_offset += 12
    
    # Convert to lists of lists for output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
