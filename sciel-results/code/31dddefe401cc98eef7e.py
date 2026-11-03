import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose C = A * B
    # A is 3x4, B is 4x7
    # Split A rows: 1, 2
    # Split B cols: 3, 4
    
    # A1: 1x4, A2: 2x4
    # B1: 4x3, B2: 4x4
    
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    
    # 1. A1*B1 (1x4 * 4x3) -> Rank 12 (Schoolbook)
    # 2. A1*B2 (1x4 * 4x4) -> Rank 16 (Schoolbook)
    # 3. A2*B1 (2x4 * 4x3) -> Rank 24 (Schoolbook)
    # 4. A2*B2 (2x4 * 4x4) -> Rank 28 (Strassen on 2x2x2 blocks)
    
    rank_total = 12 + 16 + 24 + 28
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook(n_blk, m_blk, p_blk, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_blk):
            for j in range(m_blk):
                for k in range(p_blk):
                    U[(i_off + i) * m + (j_off + j)][col] = 1
                    V[(j_off + j) * p + (k_off + k)][col] = 1
                    W[(k_off + k) * n + (i_off + i)][col] = 1
                    col += 1

    # Helper to add a Strassen block for 2x2x2
    def add_strassen(n_blk, m_blk, p_blk, i_off, j_off, k_off):
        nonlocal col
        # Strassen for 2x2x2
        # Indices local: i in 0..1, j in 0..1, k in 0..1
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y10-Y11)
        # M4 = X11(Y00-Y01)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # We need to map these to the global U, V, W indices.
        # U index: (i_off+i)*m + (j_off+j)
        # V index: (j_off+j)*p + (k_off+k)
        # W index: (k_off+k)*n + (i_off+i)
        
        def get_u(i, j): return (i_off + i) * m + (j_off + j)
        def get_v(j, k): return (j_off + j) * p + (k_off + k)
        def get_w(k, i): return (k_off + k) * n + (i_off + i)
        
        # M1
        U[get_u(0,0)][col] = 1
        U[get_u(1,1)][col] = 1
        V[get_v(0,0)][col] = 1
        V[get_v(1,1)][col] = 1
        W[get_w(0,0)][col] = 1
        W[get_w(1,1)][col] = 1
        col += 1
        
        # M2
        U[get_u(1,0)][col] = 1
        U[get_u(1,1)][col] = 1
        V[get_v(0,0)][col] = 1
        W[get_w(0,1)][col] = 1
        col += 1
        
        # M3
        U[get_u(0,0)][col] = 1
        V[get_v(1,0)][col] = 1
        V[get_v(1,1)][col] = -1
        W[get_w(0,0)][col] = 1
        col += 1
        
        # M4
        U[get_u(1,1)][col] = 1
        V[get_v(0,0)][col] = 1
        V[get_v(0,1)][col] = -1
        W[get_w(1,1)][col] = 1
        col += 1
        
        # M5
        U[get_u(0,0)][col] = 1
        U[get_u(0,1)][col] = 1
        V[get_v(1,1)][col] = 1
        W[get_w(1,0)][col] = 1
        col += 1
        
        # M6
        U[get_u(1,0)][col] = 1
        U[get_u(0,0)][col] = -1
        V[get_v(0,0)][col] = 1
        V[get_v(0,1)][col] = 1
        W[get_w(0,1)][col] = 1
        col += 1
        
        # M7
        U[get_u(0,1)][col] = 1
        U[get_u(1,1)][col] = -1
        V[get_v(1,0)][col] = 1
        V[get_v(1,1)][col] = 1
        W[get_w(1,0)][col] = 1
        col += 1

    # 1. A1*B1: i_off=0, j_off=0, k_off=0. n=1, m=4, p=3
    # Split B1 cols 3 into 1+1+1? No, just schoolbook.
    add_schoolbook(1, 4, 3, 0, 0, 0)
    
    # 2. A1*B2: i_off=0, j_off=0, k_off=3. n=1, m=4, p=4
    add_schoolbook(1, 4, 4, 0, 0, 3)
    
    # 3. A2*B1: i_off=1, j_off=0, k_off=0. n=2, m=4, p=3
    # Split A2 rows 2 into 1+1.
    # A2a (1x4) * B1 (4x3)
    add_schoolbook(1, 4, 3, 1, 0, 0)
    add_schoolbook(1, 4, 3, 2, 0, 0)
    
    # 4. A2*B2: i_off=1, j_off=0, k_off=3. n=2, m=4, p=4
    # Split A2 cols 4 into 2+2.
    # A2L (2x2) * B2T (2x4)
    # B2T cols 4 into 2+2.
    # A2L * B2T1 (2x2x2)
    add_strassen(2, 2, 2, 1, 0, 3)
    # A2L * B2T2 (2x2x2)
    add_strassen(2, 2, 2, 1, 0, 5)
    # A2R (2x2) * B2B (2x4)
    # B2B cols 4 into 2+2.
    # A2R * B2B1 (2x2x2)
    add_strassen(2, 2, 2, 1, 2, 3)
    # A2R * B2B2 (2x2x2)
    add_strassen(2, 2, 2, 1, 2, 5)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
