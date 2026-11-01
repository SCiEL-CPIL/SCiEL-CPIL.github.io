"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Total rank: 5 blocks * 11 per block = 55
    num_blocks = p // 2  # 5 blocks
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]   # (6, 55)
    V = [[0] * total_rank for _ in range(m * p)]   # (30, 55)
    W = [[0] * total_rank for _ in range(p * n)]   # (20, 55)
    
    for block in range(num_blocks):
        base_rank = block * rank_per_block
        k0 = 2 * block      # first column of B block
        k1 = 2 * block + 1  # second column of B block
        
        # Part 1: Schoolbook for A_left(2x1) * B_top(1x2), rank 4
        # A_left uses j=0 (column 0 of A), B_top uses j=0 (row 0 of B block)
        
        # Component 0: a[0,0] * b[0, k0] -> C[0, k0]
        r = base_rank + 0
        U[0 * m + 0][r] = 1   # i=0, j=0
        V[0 * p + k0][r] = 1  # j=0, k=k0
        W[k0 * n + 0][r] = 1  # k=k0, i=0
        
        # Component 1: a[0,0] * b[0, k1] -> C[0, k1]
        r = base_rank + 1
        U[0 * m + 0][r] = 1
        V[0 * p + k1][r] = 1
        W[k1 * n + 0][r] = 1
        
        # Component 2: a[1,0] * b[0, k0] -> C[1, k0]
        r = base_rank + 2
        U[1 * m + 0][r] = 1
        V[0 * p + k0][r] = 1
        W[k0 * n + 1][r] = 1
        
        # Component 3: a[1,0] * b[0, k1] -> C[1, k1]
        r = base_rank + 3
        U[1 * m + 0][r] = 1
        V[0 * p + k1][r] = 1
        W[k1 * n + 1][r] = 1
        
        # Part 2: Strassen for A_right(2x2) * B_bottom(2x2), rank 7
        # A_right: rows i=0,1; cols j=1,2 => a[0,1]=x00, a[0,2]=x01, a[1,1]=x10, a[1,2]=x11
        # B_bottom: rows j=1,2; cols k0,k1 => b[1,k0]=y00, b[1,k1]=y01, b[2,k0]=y10, b[2,k1]=y11
        
        # M1 = (x00 + x11)(y00 + y11)
        r = base_rank + 4
        U[0 * m + 1][r] = 1   # a[0,1] = x00
        U[1 * m + 2][r] = 1   # a[1,2] = x11
        V[1 * p + k0][r] = 1  # b[1,k0] = y00
        V[2 * p + k0][r] = 1  # b[2,k0] = y10
        V[1 * p + k1][r] = 1  # b[1,k1] = y01
        V[2 * p + k1][r] = 1  # b[2,k1] = y11
        # C11 = M1 - M5 + M4 + M7: M1 coeff +1
        # C12 = M3 + M5: M1 coeff 0
        # C21 = M2 + M6: M1 coeff 0
        # C22 = M1 - M2 + M3 + M4: M1 coeff +1
        W[k0 * n + 0][r] = 1   # C[0, k0]
        W[k1 * n + 0][r] = 0   # C[0, k1]
        W[k0 * n + 1][r] = 0   # C[1, k0]
        W[k1 * n + 1][r] = 1   # C[1, k1]
        
        # M2 = (x10 + x11)(y00)
        r = base_rank + 5
        U[1 * m + 1][r] = 1   # a[1,1] = x10
        U[1 * m + 2][r] = 1   # a[1,2] = x11
        V[1 * p + k0][r] = 1  # b[1,k0] = y00
        # C11: 0, C12: 0, C21 = M2 + M6: +1, C22: -1
        W[k0 * n + 0][r] = 0
        W[k1 * n + 0][r] = 0
        W[k0 * n + 1][r] = 1
        W[k1 * n + 1][r] = -1
        
        # M3 = x00(y01 - y11)
        r = base_rank + 6
        U[0 * m + 1][r] = 1   # a[0,1] = x00
        V[1 * p + k1][r] = 1  # b[1,k1] = y01
        V[2 * p + k1][r] = -1 # b[2,k1] = -y11
        # C11: 0, C12 = M3 + M5: +1, C21: 0, C22: +1
        W[k0 * n + 0][r] = 0
        W[k1 * n + 0][r] = 1
        W[k0 * n + 1][r] = 0
        W[k1 * n + 1][r] = 1
        
        # M4 = x11(y00 - y01)
        r = base_rank + 7
        U[1 * m + 2][r] = 1   # a[1,2] = x11
        V[1 * p + k0][r] = 1  # b[1,k0] = y00
        V[1 * p + k1][r] = -1 # b[1,k1] = -y01
        # C11 = M1 - M5 + M4 + M7: +1, C12: 0, C21: 0, C22: +1
        W[k0 * n + 0][r] = 1
        W[k1 * n + 0][r] = 0
        W[k0 * n + 1][r] = 0
        W[k1 * n + 1][r] = 1
        
        # M5 = (x00 + x01)(y11)
        r = base_rank + 8
        U[0 * m + 1][r] = 1   # a[0,1] = x00
        U[0 * m + 2][r] = 1   # a[0,2] = x01
        V[2 * p + k1][r] = 1  # b[2,k1] = y11
        # C11: -1, C12 = M3 + M5: +1, C21: 0, C22: 0
        W[k0 * n + 0][r] = -1
        W[k1 * n + 0][r] = 1
        W[k0 * n + 1][r] = 0
        W[k1 * n + 1][r] = 0
        
        # M6 = (x10 - x00)(y00 + y01)
        r = base_rank + 9
        U[1 * m + 1][r] = 1   # a[1,1] = x10
        U[0 * m + 1][r] = -1  # a[0,1] = -x00
        V[1 * p + k0][r] = 1  # b[1,k0] = y00
        V[1 * p + k1][r] = 1  # b[1,k1] = y01
        # C11: 0, C12: 0, C21 = M2 + M6: +1, C22: 0
        W[k0 * n + 0][r] = 0
        W[k1 * n + 0][r] = 0
        W[k0 * n + 1][r] = 1
        W[k1 * n + 1][r] = 0
        
        # M7 = (x01 - x11)(y10 + y11)
        r = base_rank + 10
        U[0 * m + 2][r] = 1   # a[0,2] = x01
        U[1 * m + 2][r] = -1  # a[1,2] = -x11
        V[2 * p + k0][r] = 1  # b[2,k0] = y10
        V[2 * p + k1][r] = 1  # b[2,k1] = y11
        # C11 = M1 - M5 + M4 + M7: +1, C12: 0, C21: 0, C22: 0
        W[k0 * n + 0][r] = 1
        W[k1 * n + 0][r] = 0
        W[k0 * n + 1][r] = 0
        W[k1 * n + 1][r] = 0
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
