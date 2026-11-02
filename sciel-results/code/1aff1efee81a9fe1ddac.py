import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Returns a valid tensor decomposition for 4x5 by 5x6 matrix multiplication.
    Uses block Kronecker structure with Strassen for square blocks.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    rank_full = n * m * p  # 120
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    # Schoolbook decomposition for small dimensions
    def schoolbook(n1, m1, p1):
        r = n1 * m1 * p1
        U = [[Fraction(0)] * r for _ in range(n1 * m1)]
        V = [[Fraction(0)] * r for _ in range(m1 * p1)]
        W = [[Fraction(0)] * r for _ in range(p1 * n1)]
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[i * m1 + j][col] = Fraction(1)
                    V[j * p1 + k][col] = Fraction(1)
                    W[k * n1 + i][col] = Fraction(1)
                    col += 1
        return U, V, W
    
    # Strassen decomposition for 2x2 * 2x2 -> 2x2 (rank 7)
    def strassen_222():
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # C = A*B
        # U: (4, 7) for A entries, V: (4, 7) for B entries, W: (4, 7) for C entries
        # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        # i,j in {0,1} for A; j,k in {0,1} for B; k,i in {0,1} for C
        
        # Strassen:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        U = [[Fraction(0)] * 7 for _ in range(4)]
        V = [[Fraction(0)] * 7 for _ in range(4)]
        W = [[Fraction(0)] * 7 for _ in range(4)]
        
        # m1 = (a11+a22)(b11+b22)
        U[0][0] = Fraction(1)  # a11
        U[3][0] = Fraction(1)  # a22
        V[0][0] = Fraction(1)  # b11
        V[3][0] = Fraction(1)  # b22
        W[0][0] = Fraction(1)  # c11 = k=0,i=0
        W[1][0] = Fraction(1)  # c22 = k=1,i=1
        
        # m2 = (a21+a22)b11
        U[2][1] = Fraction(1)  # a21
        U[3][1] = Fraction(1)  # a22
        V[0][1] = Fraction(1)  # b11
        W[2][1] = Fraction(1)  # c21 = k=1,i=0
        
        # m3 = a11(b12-b22)
        U[0][2] = Fraction(1)  # a11
        V[1][2] = Fraction(1)  # b12
        V[3][2] = Fraction(-1)  # b22
        W[1][2] = Fraction(1)  # c12 = k=0,i=1
        W[3][2] = Fraction(1)  # c22 = k=1,i=1
        
        # m4 = a22(b21-b11)
        U[3][3] = Fraction(1)  # a22
        V[2][3] = Fraction(1)  # b21
        V[0][3] = Fraction(-1)  # b11
        W[0][3] = Fraction(1)  # c11
        W[2][3] = Fraction(1)  # c21
        
        # m5 = (a11+a12)b22
        U[0][4] = Fraction(1)  # a11
        U[1][4] = Fraction(1)  # a12
        V[3][4] = Fraction(1)  # b22
        W[0][4] = Fraction(-1)  # c11
        W[1][4] = Fraction(1)  # c12
        
        # m6 = (a21-a11)(b11+b12)
        U[2][5] = Fraction(1)  # a21
        U[0][5] = Fraction(-1)  # a11
        V[0][5] = Fraction(1)  # b11
        V[1][5] = Fraction(1)  # b12
        W[3][5] = Fraction(1)  # c22
        
        # m7 = (a12-a22)(b21+b22)
        U[1][6] = Fraction(1)  # a12
        U[3][6] = Fraction(-1)  # a22
        V[2][6] = Fraction(1)  # b21
        V[3][6] = Fraction(1)  # b22
        W[0][6] = Fraction(1)  # c11
        
        return U, V, W
    
    # Verify decomposition using NumPy for speed
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j]
                    v_val = V_arr[j * p + k]
                    w_val = W_arr[k * n + i]
                    prod_sum = np.dot(u_val, v_val * w_val)
                    if abs(prod_sum - 1.0) > 1e-9:
                        return False
        return True
    
    # Try different block partitions
    
    # Strategy 1: Partition 4x5*5x6 into blocks
    # Split A (4x5) into rows: [0,1,2,3], cols: [0,1,2,3,4]
    # Split B (5x6) into rows: [0,1,2,3,4], cols: [0,1,2,3,4,5]
    # Split C (6x4) into rows: [0,1,2,3,4,5], cols: [0,1,2,3]
    
    # Try: A rows into {0,1}, {2,3}; A cols into {0,1,2}, {3,4}
    # B rows into {0,1,2}, {3,4}; B cols into {0,1,2}, {3,4,5}
    # C rows into {0,1,2}, {3,4,5}; C cols into {0,1}, {2,3}
    
    # Block 1: A[0:2, 0:3] * B[0:3, 0:3] -> C[0:3, 0:2]
    # 2x3 * 3x3 -> 3x2, schoolbook rank = 18
    # Block 2: A[0:2, 3:5] * B[3:5, 0:3] -> C[0:3, 0:2]
    # 2x2 * 2x3 -> 3x2, schoolbook rank = 12
    # Block 3: A[0:2, 0:3] * B[0:3, 3:6] -> C[3:6, 0:2]
    # 2x3 * 3x3 -> 3x2, schoolbook rank = 18
    # Block 4: A[0:2, 3:5] * B[3:5, 3:6] -> C[3:6, 0:2]
    # 2x2 * 2x3 -> 3x2, schoolbook rank = 12
    # Block 5: A[2:4, 0:3] * B[0:3, 0:3] -> C[0:3, 2:4]
    # 2x3 * 3x3 -> 3x2, schoolbook rank = 18
    # Block 6: A[2:4, 3:5] * B[3:5, 0:3] -> C[0:3, 2:4]
    # 2x2 * 2x3 -> 3x2, schoolbook rank = 12
    # Block 7: A[2:4, 0:3] * B[0:3, 3:6] -> C[3:6, 2:4]
    # 2x3 * 3x3 -> 3x2, schoolbook rank = 18
    # Block 8: A[2:4, 3:5] * B[3:5, 3:6] -> C[3:6, 2:4]
    # 2x2 * 2x3 -> 3x2, schoolbook rank = 12
    # Total: 18*4 + 12*4 = 72 + 48 = 120 (same as schoolbook)
    
    # Better: Use Strassen for square blocks
    # Try partition: A rows {0,1},{2,3}; A cols {0,1},{2,3},{4}
    # B rows {0,1},{2,3},{4}; B cols {0,1},{2,3},{4,5}
    # C rows {0,1},{2,3},{4,5}; C cols {0,1},{2,3}
    
    # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] 2x2*2x2->2x2, Strassen rank 7
    # Block (0,1): A[0:2,0:2] * B[0:2,2:4] -> C[2:4,0:2] 2x2*2x2->2x2, Strassen rank 7
    # Block (0,2): A[0:2,0:2] * B[0:2,4:6] -> C[4:6,0:2] 2x2*2x2->2x2, Strassen rank 7
    # Block (1,0): A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] 2x2*2x2->2x2, Strassen rank 7
    # Block (1,1): A[0:2,2:4] * B[2:4,2:4] -> C[2:4,0:2] 2x2*2x2->2x2, Strassen rank 7
    # Block (1,2): A[0:2,2:4] * B[2:4,4:6] -> C[4:6,0:2] 2x2*2x2->2x2, Strassen rank 7
    # Block (2,0): A[0:2,4:5] * B[4:5,0:2] -> C[0:2,0:2] 2x1*1x2->2x2, schoolbook rank 4
    # Block (2,1): A[0:2,4:5] * B[4:5,2:4] -> C[2:4,0:2] 2x1*1x2->2x2, schoolbook rank 4
    # Block (2,2): A[0:2,4:5] * B[4:5,4:6] -> C[4:6,0:2] 2x1*1x2->2x2, schoolbook rank 4
    # Block (3,0): A[2:4,0:2] * B[0:2,0:2] -> C[0:2,2:4] 2x2*2x2->2x2, Strassen rank 7
    # Block (3,1): A[2:4,0:2] * B[0:2,2:4] -> C[2:4,2:4] 2x2*2x2->2x2, Strassen rank 7
    # Block (3,2): A[2:4,0:2] * B[0:2,4:6] -> C[4:6,2:4] 2x2*2x2->2x2, Strassen rank 7
    # Block (4,0): A[2:4,2:4] * B[2:4,0:2] -> C[0:2,2:4] 2x2*2x2->2x2, Strassen rank 7
    # Block (4,1): A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4] 2x2*2x2->2x2, Strassen rank 7
    # Block (4,2): A[2:4,2:4] * B[2:4,4:6] -> C[4:6,2:4] 2x2*2x2->2x2, Strassen rank 7
    # Block (5,0): A[2:4,4:5] * B[4:5,0:2] -> C[0:2,2:4] 2x1*1x2->2x2, schoolbook rank 4
    # Block (5,1): A[2:4,4:5] * B[4:5,2:4] -> C[2:4,2:4] 2x1*1x2->2x2, schoolbook rank 4
    # Block (5,2): A[2:4,4:5] * B[4:5,4:6] -> C[4:6,2:4] 2x1*1x2->2x2, schoolbook rank 4
    
    # Total: 12*7 + 6*4 = 84 + 24 = 108
    
    # Build this decomposition
    def build_block_decomposition():
        # Total rank = 108
        total_rank = 108
        U = [[Fraction(0)] * total_rank for _ in range(n * m)]
        V = [[Fraction(0)] * total_rank for _ in range(m * p)]
        W = [[Fraction(0)] * total_rank for _ in range(p * n)]
        
        offset = 0
        
        # Helper to add a sub-decomposition
        def add_sub(U_sub, V_sub, W_sub, u_map, v_map, w_map, off):
            for r in range(len(U_sub[0])):
                for u_idx in u_map:
                    if U_sub[u_idx[0]][r] != 0:
                        U[u_idx[1]][off + r] = U_sub[u_idx[0]][r]
                for v_idx in v_map:
                    if V_sub[v_idx[0]][r] != 0:
                        V[v_idx[1]][off + r] = V_sub[v_idx[0]][r]
                for w_idx in w_map:
                    if W_sub[w_idx[0]][r] != 0:
                        W[w_idx[1]][off + r] = W_sub[w_idx[0]][r]
            return off + len(U_sub[0])
        
        # Strassen 2x2 blocks
        S_U, S_V, S_W = strassen_222()
        
        # Schoolbook 2x1*1x2->2x2
        def schoolbook_212():
            # A: 2x1, B: 1x2, C: 2x2
            # U: (2, 4), V: (2, 4), W: (4, 4)
            U_s = [[Fraction(0)] * 4 for _ in range(2)]
            V_s = [[Fraction(0)] * 4 for _ in range(2)]
            W_s = [[Fraction(0)] * 4 for _ in range(4)]
            col = 0
            for i in range(2):
                for k in range(2):
                    U_s[i][col] = Fraction(1)
                    V_s[0][col] = Fraction(1)
                    W_s[k * 2 + i][col] = Fraction(1)
                    col += 1
            return U_s, V_s, W_s
        
        SB_U, SB_V, SB_W = schoolbook_212()
        
        # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # A rows 0,1 cols 0,1; B rows 0,1 cols 0,1; C rows 0,1 cols 0,1
        u_map = [(0, 0), (1, 1), (2, 2), (3, 3)]  # A[0,0]->U[0], A[0,1]->U[1], A[1,0]->U[2], A[1,1]->U[3]
        v_map = [(0, 0), (1, 1), (2, 2), (3, 3)]  # B[0,0]->V[0], B[0,1]->V[1], B[1,0]->V[2], B[1,1]->V[3]
        w_map = [(0, 0), (1, 1), (2, 2), (3, 3)]  # C[0,0]->W[0], C[1,0]->W[1], C[0,1]->W[2], C[1,1]->W[3]
        # U index: i*m+j = i*5+j for A, so A[i,j] -> U[i*5+j]
        # V index: j*p+k = j*6+k for B
        # W index: k*n+i = k*4+i for C
        u_map = [(0*5+0, 0), (0*5+1, 1), (1*5+0, 2), (1*5+1, 3)]
        v_map = [(0*6+0, 0), (0*6+1, 1), (1*6+0, 2), (1*6+1, 3)]
        w_map = [(0*4+0, 0), (1*4+0, 1), (0*4+1, 2), (1*4+1, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (0,1): A[0:2,0:2] * B[0:2,2:4] -> C[2:4,0:2]
        u_map = [(0*5+0, 0), (0*5+1, 1), (1*5+0, 2), (1*5+1, 3)]
        v_map = [(0*6+2, 0), (0*6+3, 1), (1*6+2, 2), (1*6+3, 3)]
        w_map = [(2*4+0, 0), (3*4+0, 1), (2*4+1, 2), (3*4+1, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (0,2): A[0:2,0:2] * B[0:2,4:6] -> C[4:6,0:2]
        u_map = [(0*5+0, 0), (0*5+1, 1), (1*5+0, 2), (1*5+1, 3)]
        v_map = [(0*6+4, 0), (0*6+5, 1), (1*6+4, 2), (1*6+5, 3)]
        w_map = [(4*4+0, 0), (5*4+0, 1), (4*4+1, 2), (5*4+1, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (1,0): A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        u_map = [(0*5+2, 0), (0*5+3, 1), (1*5+2, 2), (1*5+3, 3)]
        v_map = [(2*6+0, 0), (2*6+1, 1), (3*6+0, 2), (3*6+1, 3)]
        w_map = [(0*4+0, 0), (1*4+0, 1), (0*4+1, 2), (1*4+1, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (1,1): A[0:2,2:4] * B[2:4,2:4] -> C[2:4,0:2]
        u_map = [(0*5+2, 0), (0*5+3, 1), (1*5+2, 2), (1*5+3, 3)]
        v_map = [(2*6+2, 0), (2*6+3, 1), (3*6+2, 2), (3*6+3, 3)]
        w_map = [(2*4+0, 0), (3*4+0, 1), (2*4+1, 2), (3*4+1, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (1,2): A[0:2,2:4] * B[2:4,4:6] -> C[4:6,0:2]
        u_map = [(0*5+2, 0), (0*5+3, 1), (1*5+2, 2), (1*5+3, 3)]
        v_map = [(2*6+4, 0), (2*6+5, 1), (3*6+4, 2), (3*6+5, 3)]
        w_map = [(4*4+0, 0), (5*4+0, 1), (4*4+1, 2), (5*4+1, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (2,0): A[0:2,4:5] * B[4:5,0:2] -> C[0:2,0:2]
        u_map = [(0*5+4, 0), (1*5+4, 1)]
        v_map = [(4*6+0, 0), (4*6+1, 1)]
        w_map = [(0*4+0, 0), (1*4+0, 1), (0*4+1, 2), (1*4+1, 3)]
        offset = add_sub(SB_U, SB_V, SB_W, u_map, v_map, w_map, offset)
        
        # Block (2,1): A[0:2,4:5] * B[4:5,2:4] -> C[2:4,0:2]
        u_map = [(0*5+4, 0), (1*5+4, 1)]
        v_map = [(4*6+2, 0), (4*6+3, 1)]
        w_map = [(2*4+0, 0), (3*4+0, 1), (2*4+1, 2), (3*4+1, 3)]
        offset = add_sub(SB_U, SB_V, SB_W, u_map, v_map, w_map, offset)
        
        # Block (2,2): A[0:2,4:5] * B[4:5,4:6] -> C[4:6,0:2]
        u_map = [(0*5+4, 0), (1*5+4, 1)]
        v_map = [(4*6+4, 0), (4*6+5, 1)]
        w_map = [(4*4+0, 0), (5*4+0, 1), (4*4+1, 2), (5*4+1, 3)]
        offset = add_sub(SB_U, SB_V, SB_W, u_map, v_map, w_map, offset)
        
        # Block (3,0): A[2:4,0:2] * B[0:2,0:2] -> C[0:2,2:4]
        u_map = [(2*5+0, 0), (2*5+1, 1), (3*5+0, 2), (3*5+1, 3)]
        v_map = [(0*6+0, 0), (0*6+1, 1), (1*6+0, 2), (1*6+1, 3)]
        w_map = [(0*4+2, 0), (1*4+2, 1), (0*4+3, 2), (1*4+3, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (3,1): A[2:4,0:2] * B[0:2,2:4] -> C[2:4,2:4]
        u_map = [(2*5+0, 0), (2*5+1, 1), (3*5+0, 2), (3*5+1, 3)]
        v_map = [(0*6+2, 0), (0*6+3, 1), (1*6+2, 2), (1*6+3, 3)]
        w_map = [(2*4+2, 0), (3*4+2, 1), (2*4+3, 2), (3*4+3, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (3,2): A[2:4,0:2] * B[0:2,4:6] -> C[4:6,2:4]
        u_map = [(2*5+0, 0), (2*5+1, 1), (3*5+0, 2), (3*5+1, 3)]
        v_map = [(0*6+4, 0), (0*6+5, 1), (1*6+4, 2), (1*6+5, 3)]
        w_map = [(4*4+2, 0), (5*4+2, 1), (4*4+3, 2), (5*4+3, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (4,0): A[2:4,2:4] * B[2:4,0:2] -> C[0:2,2:4]
        u_map = [(2*5+2, 0), (2*5+3, 1), (3*5+2, 2), (3*5+3, 3)]
        v_map = [(2*6+0, 0), (2*6+1, 1), (3*6+0, 2), (3*6+1, 3)]
        w_map = [(0*4+2, 0), (1*4+2, 1), (0*4+3, 2), (1*4+3, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (4,1): A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4]
        u_map = [(2*5+2, 0), (2*5+3, 1), (3*5+2, 2), (3*5+3, 3)]
        v_map = [(2*6+2, 0), (2*6+3, 1), (3*6+2, 2), (3*6+3, 3)]
        w_map = [(2*4+2, 0), (3*4+2, 1), (2*4+3, 2), (3*4+3, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (4,2): A[2:4,2:4] * B[2:4,4:6] -> C[4:6,2:4]
        u_map = [(2*5+2, 0), (2*5+3, 1), (3*5+2, 2), (3*5+3, 3)]
        v_map = [(2*6+4, 0), (2*6+5, 1), (3*6+4, 2), (3*6+5, 3)]
        w_map = [(4*4+2, 0), (5*4+2, 1), (4*4+3, 2), (5*4+3, 3)]
        offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        
        # Block (5,0): A[2:4,4:5] * B[4:5,0:2] -> C[0:2,2:4]
        u_map = [(2*5+4, 0), (3*5+4, 1)]
        v_map = [(4*6+0, 0), (4*6+1, 1)]
        w_map = [(0*4+2, 0), (1*4+2, 1), (0*4+3, 2), (1*4+3, 3)]
        offset = add_sub(SB_U, SB_V, SB_W, u_map, v_map, w_map, offset)
        
        # Block (5,1): A[2:4,4:5] * B[4:5,2:4] -> C[2:4,2:4]
        u_map = [(2*5+4, 0), (3*5+4, 1)]
        v_map = [(4*6+2, 0), (4*6+3, 1)]
        w_map = [(2*4+2, 0), (3*4+2, 1), (2*4+3, 2), (3*4+3, 3)]
        offset = add_sub(SB_U, SB_V, SB_W, u_map, v_map, w_map, offset)
        
        # Block (5,2): A[2:4,4:5] * B[4:5,4:6] -> C[4:6,2:4]
        u_map = [(2*5+4, 0), (3*5+4, 1)]
        v_map = [(4*6+4, 0), (4*6+5, 1)]
        w_map = [(4*4+2, 0), (5*4+2, 1), (4*4+3, 2), (5*4+3, 3)]
        offset = add_sub(SB_U, SB_V, SB_W, u_map, v_map, w_map, offset)
        
        # Check offset == total_rank
        if offset != total_rank:
            # Fallback to schoolbook
            U, V, W = schoolbook(n, m, p)
            return {"U": [[float(x) for x in row] for row in U], 
                    "V": [[float(x) for x in row] for row in V], 
                    "W": [[float(x) for x in row] for row in W]}
        
        # Verify
        if verify(U, V, W):
            return {"U": [[float(x) for x in row] for row in U], 
                    "V": [[float(x) for x in row] for row in V], 
                    "W": [[float(x) for x in row] for row in W]}
        
        # Fallback
        U, V, W = schoolbook(n, m, p)
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}
    
    try:
        result = build_block_decomposition()
        return result
    except Exception as e:
        # Fallback to schoolbook
        U, V, W = schoolbook(n, m, p)
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}
