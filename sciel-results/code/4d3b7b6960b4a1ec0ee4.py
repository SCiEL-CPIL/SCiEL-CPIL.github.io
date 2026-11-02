# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # Split m into m1=1, m2=2
    # Part 1: 2x1 * 1x8 -> rank 16
    # Part 2: 2x2 * 2x8 -> 4 blocks of 2x2*2x2, each Strassen rank 7 -> rank 28
    # Total rank = 44
    
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    # Part 1: 2x1 * 1x8
    # A rows: i=0,1; B col: j=0; C: k=0..7
    for i in range(n):       # 2
        for k in range(p):   # 8
            U[i * m + 0][col] = 1.0   # A[i, 0]
            V[0 * p + k][col] = 1.0   # B[0, k]
            W[k * n + i][col] = 1.0   # C[k, i]
            col += 1
    # col = 16
    
    # Part 2: 2x2 * 2x8
    # A: rows i=0,1; cols j=1,2 (offset by m1=1)
    # B: rows j=1,2; cols k=0..7
    # Split B into 4 blocks of 2x2: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    # For each block, use Strassen 2x2x2
    
    # Strassen for 2x2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # A indices in U: row i*m + j where j=1,2 for part 2
    # So A[0,1] -> U[0*3+1]=U[1], A[0,2]->U[2], A[1,1]->U[4], A[1,2]->U[5]
    # B indices in V: row j*p + k where j=1,2
    # So B[1,k] -> V[1*8+k]=V[8+k], B[2,k]->V[16+k]
    # C indices in W: row k*n + i
    # C[0,0]->W[0], C[0,1]->W[1], C[1,0]->W[2], C[1,1]->W[3], etc.
    
    # For block b (0,1,2,3), columns k0=2*b, k1=2*b+1
    # B[1,k0] -> V[8+k0], B[1,k1] -> V[8+k1]
    # B[2,k0] -> V[16+k0], B[2,k1] -> V[16+k1]
    # C[k0,0]->W[k0*2+0]=W[2*k0], C[k0,1]->W[2*k0+1]
    # C[k1,0]->W[2*k1], C[k1,1]->W[2*k1+1]
    
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # B indices
        b00_idx = 8 + k0    # B[1,k0]
        b01_idx = 8 + k1    # B[1,k1]
        b10_idx = 16 + k0   # B[2,k0]
        b11_idx = 16 + k1   # B[2,k1]
        
        # C indices
        c00_idx = 2 * k0    # C[k0,0]
        c01_idx = 2 * k0 + 1  # C[k0,1]
        c10_idx = 2 * k1    # C[k1,0]
        c11_idx = 2 * k1 + 1  # C[k1,1]
        
        # A indices
        a00 = 1  # U[0*3+1] = A[0,1]
        a01 = 2  # U[0*3+2] = A[0,2]
        a10 = 4  # U[1*3+1] = A[1,1]
        a11 = 5  # U[1*3+2] = A[1,2]
        
        # M1 = (a00+a11)(b00+b11)
        U[a00][col] += 1.0
        U[a11][col] += 1.0
        V[b00_idx][col] += 1.0
        V[b11_idx][col] += 1.0
        W[c00_idx][col] += 1.0
        W[c11_idx][col] += 1.0
        col += 1
        
        # M2 = (a10+a11)b00
        U[a10][col] += 1.0
        U[a11][col] += 1.0
        V[b00_idx][col] += 1.0
        W[c10_idx][col] += 1.0
        col += 1
        
        # M3 = a00(b10-b11)
        U[a00][col] += 1.0
        V[b10_idx][col] += 1.0
        V[b11_idx][col] += -1.0
        W[c01_idx][col] += 1.0
        col += 1
        
        # M4 = a11(b01-b00)
        U[a11][col] += 1.0
        V[b01_idx][col] += 1.0
        V[b00_idx][col] += -1.0
        W[c00_idx][col] += 1.0
        W[c10_idx][col] += 1.0
        col += 1
        
        # M5 = (a00+a01)b11
        U[a00][col] += 1.0
        U[a01][col] += 1.0
        V[b11_idx][col] += 1.0
        W[c01_idx][col] += 1.0
        W[c00_idx][col] += -1.0
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10][col] += 1.0
        U[a00][col] += -1.0
        V[b00_idx][col] += 1.0
        V[b01_idx][col] += 1.0
        W[c11_idx][col] += 1.0
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01][col] += 1.0
        U[a11][col] += -1.0
        V[b10_idx][col] += 1.0
        V[b11_idx][col] += 1.0
        W[c00_idx][col] += 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
