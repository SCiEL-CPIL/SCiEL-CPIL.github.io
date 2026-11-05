"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 construction
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block for 2x2x2
    def add_strassen_block(a_rows, b_rows, c_rows, start_col):
        # a_rows: [a00, a01, a10, a11] global U indices
        # b_rows: [b00, b01, b10, b11] global V indices
        # c_rows: [c00, c01, c10, c11] global W indices
        # c_rows order: c[i, k] -> W[k*n + i]
        # c00 is W[0*n+0], c01 is W[1*n+0], c10 is W[0*n+1], c11 is W[1*n+1]
        
        # P1 = (a00+a11)(b00+b11)
        U[a_rows[0]][start_col] += 1
        U[a_rows[3]][start_col] += 1
        V[b_rows[0]][start_col] += 1
        V[b_rows[3]][start_col] += 1
        W[c_rows[0]][start_col] += 1 # c00
        W[c_rows[3]][start_col] += 1 # c11
        col = start_col + 1
        
        # P2 = (a10+a11)b00
        U[a_rows[2]][col] += 1
        U[a_rows[3]][col] += 1
        V[b_rows[0]][col] += 1
        W[c_rows[2]][col] += 1 # c10
        W[c_rows[3]][col] += 1 # c11
        col += 1
        
        # P3 = a00(b01-b11)
        U[a_rows[0]][col] += 1
        V[b_rows[1]][col] += 1
        V[b_rows[3]][col] -= 1
        W[c_rows[1]][col] += 1 # c01
        W[c_rows[3]][col] += 1 # c11
        col += 1
        
        # P4 = a11(b10-b00)
        U[a_rows[3]][col] += 1
        V[b_rows[2]][col] += 1
        V[b_rows[0]][col] -= 1
        W[c_rows[0]][col] -= 1 # c00
        W[c_rows[2]][col] += 1 # c10
        col += 1
        
        # P5 = (a00+a01)b11
        U[a_rows[0]][col] += 1
        U[a_rows[1]][col] += 1
        V[b_rows[3]][col] += 1
        W[c_rows[1]][col] += 1 # c01
        W[c_rows[3]][col] += 1 # c11
        col += 1
        
        # P6 = (a10-a00)(b00+b01)
        U[a_rows[2]][col] += 1
        U[a_rows[0]][col] -= 1
        V[b_rows[0]][col] += 1
        V[b_rows[1]][col] += 1
        W[c_rows[2]][col] += 1 # c10
        W[c_rows[3]][col] += 1 # c11
        col += 1
        
        # P7 = (a01-a11)(b10+b11)
        U[a_rows[1]][col] += 1
        U[a_rows[3]][col] -= 1
        V[b_rows[2]][col] += 1
        V[b_rows[3]][col] += 1
        W[c_rows[0]][col] += 1 # c00
        W[c_rows[3]][col] += 1 # c11
        col += 1
        
        return col

    def add_schoolbook_2x2x1_block(a_rows, b_rows, c_rows, start_col):
        # a_rows: [a00, a01, a10, a11]
        # b_rows: [b00, b10] (only k=0 for this sub-block, mapped to global k)
        # c_rows: [c00, c10]
        
        # P1 = (a00+a10)b00
        U[a_rows[0]][start_col] += 1
        U[a_rows[2]][start_col] += 1
        V[b_rows[0]][start_col] += 1
        W[c_rows[0]][start_col] += 1
        W[c_rows[1]][start_col] += 1
        col = start_col + 1
        
        # P2 = (a01+a11)b10
        U[a_rows[1]][col] += 1
        U[a_rows[3]][col] += 1
        V[b_rows[1]][col] += 1
        W[c_rows[0]][col] += 1
        W[c_rows[1]][col] += 1
        col += 1
        
        return col

    # Block 1: A1 (j=0,1) * B1 (j=0,1)
    # A1 rows in U:
    # A[0,0]=U[0], A[0,1]=U[1]
    # A[1,0]=U[4], A[1,1]=U[5]
    a1_rows = [0, 1, 4, 5]
    
    # B1 is B[0:2, :]
    # B[0,0]=V[0], B[0,1]=V[1]
    # B[1,0]=V[5], B[1,1]=V[6]
    # B[0,2]=V[2], B[0,3]=V[3]
    # B[1,2]=V[7], B[1,3]=V[8]
    # B[0,4]=V[4], B[1,4]=V[9]
    
    # Sub-block 1.1: B1[0:2, 0:2] (k=0,1)
    b11_rows = [0, 1, 5, 6] # b00, b01, b10, b11
    # C indices for k=0,1
    # c00=W[0], c01=W[2]
    # c10=W[1], c11=W[3]
    c11_rows = [0, 2, 1, 3]
    col = add_strassen_block(a1_rows, b11_rows, c11_rows, col)
    
    # Sub-block 1.2: B1[0:2, 2:4] (k=2,3)
    b12_rows = [2, 3, 7, 8] # b00(k=2), b01(k=3), b10(k=2), b11(k=3)
    # C indices for k=2,3
    # c00(k=2)=W[4], c01(k=3)=W[6]
    # c10(k=2)=W[5], c11(k=3)=W[7]
    c12_rows = [4, 6, 5, 7]
    col = add_strassen_block(a1_rows, b12_rows, c12_rows, col)
    
    # Sub-block 1.3: B1[0:2, 4:5] (k=4)
    b13_rows = [4, 9] # b00(k=4), b10(k=4)
    # C indices for k=4
    # c00(k=4)=W[8], c10(k=4)=W[9]
    c13_rows = [8, 9]
    col = add_schoolbook_2x2x1_block(a1_rows, b13_rows, c13_rows, col)
    
    # Block 2: A2 (j=2,3) * B2 (j=2,3)
    # A2 rows in U:
    # A[0,2]=U[2], A[0,3]=U[3]
    # A[1,2]=U[6], A[1,3]=U[7]
    a2_rows = [2, 3, 6, 7]
    
    # B2 is B[2:4, :]
    # B[2,0]=V[10], B[2,1]=V[11]
    # B[3,0]=V[15], B[3,1]=V[16]
    # B[2,2]=V[12], B[2,3]=V[13]
    # B[3,2]=V[17], B[3,3]=V[18]
    # B[2,4]=V[14], B[3,4]=V[19]
    
    # Sub-block 2.1: B2[0:2, 0:2] (k=0,1)
    b21_rows = [10, 11, 15, 16]
    c21_rows = [0, 2, 1, 3] # Same C indices as block 1
    col = add_strassen_block(a2_rows, b21_rows, c21_rows, col)
    
    # Sub-block 2.2: B2[0:2, 2:4] (k=2,3)
    b22_rows = [12, 13, 17, 18]
    c22_rows = [4, 6, 5, 7]
    col = add_strassen_block(a2_rows, b22_rows, c22_rows, col)
    
    # Sub-block 2.3: B2[0:2, 4:5] (k=4)
    b23_rows = [14, 19]
    c23_rows = [8, 9]
    col = add_schoolbook_2x2x1_block(a2_rows, b23_rows, c23_rows, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
