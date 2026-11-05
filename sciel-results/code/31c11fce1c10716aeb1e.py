import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False
        return True

    def verify_half_integer(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    doubled = val * 2
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return verify(U, V, W)

    # Build Strassen M_2,2,2
    # U: 4x7, V: 4x7, W: 4x7
    U_222 = [[0]*7 for _ in range(4)]
    V_222 = [[0]*7 for _ in range(4)]
    W_222 = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U_222[0][0] = 1; U_222[3][0] = 1
    V_222[0][0] = 1; V_222[3][0] = 1
    W_222[0][0] = 1; W_222[3][0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U_222[2][1] = 1; U_222[3][1] = 1
    V_222[0][1] = 1
    W_222[2][1] = 1; W_222[3][1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U_222[0][2] = 1
    V_222[1][2] = 1; V_222[3][2] = -1
    W_222[1][2] = 1; W_222[3][2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U_222[3][3] = 1
    V_222[2][3] = 1; V_222[0][3] = -1
    W_222[0][3] = 1; W_222[2][3] = 1
    
    # M5: (A00+A10)B11 -> -C00, C01
    U_222[0][4] = 1; U_222[2][4] = 1
    V_222[3][4] = 1
    W_222[0][4] = -1; W_222[1][4] = 1
    
    # M6: (A11-A00)(B00+B01) -> C11
    U_222[3][5] = 1; U_222[0][5] = -1
    V_222[0][5] = 1; V_222[1][5] = 1
    W_222[3][5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U_222[1][6] = 1; U_222[3][6] = -1
    V_222[2][6] = 1; V_222[3][6] = 1
    W_222[0][6] = 1

    # Build M_2,2,8 (Rank 28)
    # U_228: 4x28 (4 copies of U_222)
    # V_228: 16x28
    # W_228: 16x28
    
    U_228 = [[0]*28 for _ in range(4)]
    V_228 = [[0]*28 for _ in range(16)]
    W_228 = [[0]*28 for _ in range(16)]
    
    for q in range(4):
        col_offset = q * 7
        # U
        for r in range(4):
            for c in range(7):
                U_228[r][col_offset + c] = U_222[r][c]
        # V
        # B block q: rows 0,1 of B, cols 2q, 2q+1
        # V_222 rows: 0:B00, 1:B01, 2:B10, 3:B11
        # Global V rows: 0:B0, 1:B1, 2:B2, 3:B3, 4:B4, 5:B5, 6:B6, 7:B7, 8:B8..
        # Block q uses B cols 2q, 2q+1.
        # V_222[0] corresponds to B[0, 2q] -> Global V[0*8 + 2q]
        # V_222[1] corresponds to B[0, 2q+1] -> Global V[0*8 + 2q+1]
        # V_222[2] corresponds to B[1, 2q] -> Global V[1*8 + 2q]
        # V_222[3] corresponds to B[1, 2q+1] -> Global V[1*8 + 2q+1]
        for c in range(7):
            V_228[0*8 + 2*q][col_offset + c] = V_222[0][c]
            V_228[0*8 + 2*q + 1][col_offset + c] = V_222[1][c]
            V_228[1*8 + 2*q][col_offset + c] = V_222[2][c]
            V_228[1*8 + 2*q + 1][col_offset + c] = V_222[3][c]
        # W
        # C block q: rows 0,1 of C, cols 2q, 2q+1
        # W_222 rows: 0:C00, 1:C01, 2:C10, 3:C11
        # Global W rows: k*2 + i.
        # C[0, 2q] -> W[2q*2 + 0]
        # C[0, 2q+1] -> W[(2q+1)*2 + 0]
        # C[1, 2q] -> W[2q*2 + 1]
        # C[1, 2q+1] -> W[(2q+1)*2 + 1]
        for c in range(7):
            W_228[2*q*2 + 0][col_offset + c] = W_222[0][c]
            W_228[(2*q+1)*2 + 0][col_offset + c] = W_222[1][c]
            W_228[2*q*2 + 1][col_offset + c] = W_222[2][c]
            W_228[(2*q+1)*2 + 1][col_offset + c] = W_222[3][c]

    # Build M_2,1,8 (Rank 16)
    U_218 = [[0]*16 for _ in range(2)]
    V_218 = [[0]*16 for _ in range(8)]
    W_218 = [[0]*16 for _ in range(16)]
    
    for i in range(2):
        for k in range(8):
            c = i * 8 + k
            U_218[i][c] = 1
            V_218[k][c] = 1
            W_218[k*2 + i][c] = 1

    # Build M_1,2,8 (Rank 16)
    U_128 = [[0]*16 for _ in range(2)]
    V_128 = [[0]*16 for _ in range(16)]
    W_128 = [[0]*16 for _ in range(8)]
    
    for j in range(2):
        for k in range(8):
            c = j * 8 + k
            U_128[j][c] = 1
            V_128[j*8 + k][c] = 1
            W_128[k][c] = 1

    # Build M_1,1,8 (Rank 8)
    U_118 = [[0]*8 for _ in range(1)]
    V_118 = [[0]*8 for _ in range(8)]
    W_118 = [[0]*8 for _ in range(8)]
    
    for k in range(8):
        c = k
        U_118[0][c] = 1
        V_118[k][c] = 1
        W_118[k][c] = 1

    # Assemble Global
    # n=3, m=3, p=8
    # U: 9x68
    # V: 24x68
    # W: 24x68
    
    total_rank = 28 + 16 + 16 + 8
    U = [[0]*total_rank for _ in range(9)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Sub 1: M_2,2,8
    # U indices: 0,1,3,4
    u_map_1 = [0, 1, 3, 4]
    for r_local, r_global in enumerate(u_map_1):
        for c in range(28):
            U[r_global][c] = U_228[r_local][c]
            
    # V indices: 0..15
    for r in range(16):
        for c in range(28):
            V[r][c] = V_228[r][c]
            
    # W indices: k*3 + i for i in {0,1}
    for r_local in range(16):
        # r_local = k*2 + i
        k = r_local // 2
        i = r_local % 2
        r_global = k * 3 + i
        for c in range(28):
            W[r_global][c] = W_228[r_local][c]
            
    # Sub 2: M_2,1,8
    # U indices: 2, 5
    u_map_2 = [2, 5]
    for r_local, r_global in enumerate(u_map_2):
        for c in range(16):
            U[r_global][28 + c] = U_218[r_local][c]
            
    # V indices: 16..23
    for r in range(8):
        for c in range(16):
            V[16 + r][28 + c] = V_218[r][c]
            
    # W indices: k*3 + i for i in {0,1}
    for r_local in range(16):
        k = r_local // 2
        i = r_local % 2
        r_global = k * 3 + i
        for c in range(16):
            W[r_global][28 + c] = W_218[r_local][c]
            
    # Sub 3: M_1,2,8
    # U indices: 6, 7
    u_map_3 = [6, 7]
    for r_local, r_global in enumerate(u_map_3):
        for c in range(16):
            U[r_global][44 + c] = U_128[r_local][c]
            
    # V indices: 0..15 (Add to existing)
    for r in range(16):
        for c in range(16):
            V[r][44 + c] = V_128[r][c]
            
    # W indices: k*3 + 2
    for r in range(8):
        r_global = r * 3 + 2
        for c in range(16):
            W[r_global][44 + c] = W_128[r][c]
            
    # Sub 4: M_1,1,8
    # U indices: 8
    U[8][60] = 1 # U_118[0][0] is 1, etc.
    for c in range(8):
        U[8][60 + c] = U_118[0][c]
        
    # V indices: 16..23 (Add to existing)
    for r in range(8):
        for c in range(8):
            V[16 + r][60 + c] = V_118[r][c]
            
    # W indices: k*3 + 2 (Add to existing)
    for r in range(8):
        r_global = r * 3 + 2
        for c in range(8):
            W[r_global][60 + c] = W_118[r][c]
            
    if verify_half_integer(U, V, W):
        return {"U": U, "V": V, "W": W}
        
    return schoolbook()
# EVOLVE-BLOCK-END
