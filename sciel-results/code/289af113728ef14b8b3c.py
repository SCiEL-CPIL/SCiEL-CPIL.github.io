import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def validate(U, V, W):
        """Check if decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # val = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    prod = V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    val = np.dot(U_arr[i*m+j, :], prod)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def build_laderman_3x3x3x3():
        """
        Build Laderman's rank-20 decomposition for 3x3 * 3x3.
        Returns U (9x20), V (9x20), W (9x20) for the 3x3x3 case.
        
        Convention: U row index = i*m+j = i*3+j (A[i,j])
                   V row index = j*p+k = j*3+k (B[j,k])  
                   W row index = k*n+i = k*3+i (C[i,k])
        
        For each rank term r, we need:
        sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
        """
        R = 20
        U = [[0.0]*R for _ in range(9)]
        V = [[0.0]*R for _ in range(9)]
        W = [[0.0]*R for _ in range(9)]
        
        # Laderman's 20 bilinear forms for 3x3 matrix multiplication
        # Each form: (linear combo of A) * (linear combo of B) * (coefficients for C)
        # A indices: a00=0, a01=1, a02=2, a10=3, a11=4, a12=5, a20=6, a21=7, a22=8
        # B indices: b00=0, b01=1, b02=2, b10=3, b11=4, b12=5, b20=6, b21=7, b22=8
        # C indices (trace convention W[k*3+i]): 
        #   C[0,0]: k=0,i=0 -> idx 0
        #   C[1,0]: k=1,i=0 -> idx 3
        #   C[2,0]: k=2,i=0 -> idx 6
        #   C[0,1]: k=0,i=1 -> idx 1
        #   C[1,1]: k=1,i=1 -> idx 4
        #   C[2,1]: k=2,i=1 -> idx 7
        #   C[0,2]: k=0,i=2 -> idx 2
        #   C[1,2]: k=1,i=2 -> idx 5
        #   C[2,2]: k=2,i=2 -> idx 8
        
        # Laderman's algorithm (from Laderman 1976):
        # The 20 products are:
        
        # 1: (a00+a11+a22)(b00+b11+b22)
        # 2: (a00+a11)(b00+b11)
        # 3: (a00+a22)(b00+b22)
        # 4: (a00+a11+a22)(b01+b12+b20)
        # 5: (a00+a11)(b01+b12)
        # 6: (a00+a22)(b01+b20)
        # 7: (a01+a12+a20)(b00+b11+b22)
        # 8: (a01+a12)(b00+b11)
        # 9: (a01+a20)(b00+b22)
        # 10: (a01+a12+a20)(b02+b10+b21)
        # 11: (a01+a12)(b02+b10)
        # 12: (a01+a20)(b02+b21)
        # 13: (a10+a21)(b00+b11+b22)
        # 14: (a10+a21)(b01+b12)
        # 15: (a10+a21)(b02+b20)
        # 16: (a00+a11)(b02+b10+b21)
        # 17: (a00+a11)(b02+b10)
        # 18: (a00+a22)(b02+b21)
        # 19: (a01+a12)(b01+b20)
        # 20: (a01+a20)(b01+b12)
        
        # Actually, let me use the standard Laderman formulation more carefully.
        # Let me use a known correct set.
        
        # I'll use the formulation where we define:
        # x0 = a00, x1 = a01, x2 = a02, x3 = a10, x4 = a11, x5 = a12, x6 = a20, x7 = a21, x8 = a22
        # y0 = b00, y1 = b01, y2 = b02, y3 = b10, y4 = b11, y5 = b12, y6 = b20, y7 = b21, y8 = b22
        
        # Laderman's 20 products (using standard references):
        
        # Form 1: (x0+x4+x8)(y0+y4+y8)
        U[0][0] += 1; U[4][0] += 1; U[8][0] += 1
        V[0][0] += 1; V[4][0] += 1; V[8][0] += 1
        # This contributes to C[i,k] when the indices align
        # (x0+x4+x8)(y0+y4+y8) = x0y0+x0y4+x0y8+x4y0+x4y4+x4y8+x8y0+x8y4+x8y8
        # x0y0 = a00*b00 -> C[0,0], W[0*3+0][0] = W[0][0] += 1
        # x0y4 = a00*b11 -> doesn't directly contribute (a00 is row 0 col 0, b11 is row 1 col 1)
        # Wait, in matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
        # a00*b11: i=0,j=0 from A; j=1,k=1 from B. j mismatch (0 vs 1), so no direct contribution.
        # But in Laderman, the W coefficients handle this through cancellation.
        
        # The key insight: for each product term x_a * y_b, it contributes to C[i,k] 
        # where i is the row of A (from x_a) and k is the col of B (from y_b),
        # BUT ONLY IF the column index of A (from x_a) equals the row index of B (from y_b).
        # Otherwise, it contributes with appropriate W coefficients that will cancel in the sum.
        
        # This is getting complex. Let me use a different approach:
        # I'll construct the decomposition by solving the system directly.
        
        # Actually, let me just use the schoolbook for now and try to find a better approach.
        
        # Let me try a different strategy: use the known Laderman matrices from literature.
        
        # Clear and rebuild
        U = [[0.0]*R for _ in range(9)]
        V = [[0.0]*R for _ in range(9)]
        W = [[0.0]*R for _ in range(9)]
        
        # I'll use a systematic approach: define the 20 bilinear forms explicitly
        # based on Laderman's paper.
        
        # Laderman's algorithm uses these 20 products:
        # Let me denote A elements as a[i,j] and B elements as b[i,j]
        
        # Product 1: (a00+a11+a22)(b00+b11+b22)
        U[0][0] += 1; U[4][0] += 1; U[8][0] += 1
        V[0][0] += 1; V[4][0] += 1; V[8][0] += 1
        # W contributions: for each (a_ij, b_kl) pair, if j==k, contributes to C[i,l]
        # a00*b00: i=0,j=0; k=0,l=0; j==k=0 -> C[0,0], W[0*3+0][0]=W[0][0]+=1
        # a00*b11: i=0,j=0; k=1,l=1; j!=k -> no direct, but in Laderman this is handled
        # a00*b22: i=0,j=0; k=2,l=2; j!=k -> no direct
        # a11*b00: i=1,j=1; k=0,l=0; j!=k -> no direct
        # a11*b11: i=1,j=1; k=1,l=1; j==k=1 -> C[1,1], W[1*3+1][0]=W[4][0]+=1
        # a11*b22: i=1,j=1; k=2,l=2; j!=k -> no direct
        # a22*b00: i=2,j=2; k=0,l=0; j!=k -> no direct
        # a22*b11: i=2,j=2; k=1,l=1; j!=k -> no direct
        # a22*b22: i=2,j=2; k=2,l=2; j==k=2 -> C[2,2], W[2*3+2][0]=W[8][0]+=1
        W[0][0] += 1
        W[4][0] += 1
        W[8][0] += 1
        
        # Product 2: (a00+a11)(b00+b11)
        U[0][1] += 1; U[4][1] += 1
        V[0][1] += 1; V[4][1] += 1
        # a00*b00: C[0,0], W[0][1]+=1
        # a00*b11: j=0,k=1, no
        # a11*b00: j=1,k=0, no
        # a11*b11: C[1,1], W[4][1]+=1
        W[0][1] += 1
        W[4][1] += 1
        
        # Product 3: (a00+a22)(b00+b22)
        U[0][2] += 1; U[8][2] += 1
        V[0][2] += 1; V[8][2] += 1
        # a00*b00: C[0,0], W[0][2]+=1
        # a00*b22: j=0,k=2, no
        # a22*b00: j=2,k=0, no
        # a22*b22: C[2,2], W[8][2]+=1
        W[0][2] += 1
        W[8][2] += 1
        
        # Product 4: (a00+a11+a22)(b01+b12+b20)
        U[0][3] += 1; U[4][3] += 1; U[8][3] += 1
        V[1][3] += 1; V[5][3] += 1; V[6][3] += 1
        # a00*b01: j=0,k=0? b01 is b[0,1], so k=0,l=1. j=0==k=0 -> C[0,1], W[1*3+0][3]=W[1][3]+=1
        # Wait, b01 means B[0,1], so row=0, col=1. In V index j*p+k, j=0,k=1 -> V[0*3+1]=V[1]
        # a00 is A[0,0], i=0,j=0. b01 is B[0,1], j=0,k=1. j==k=0 -> C[0,1], W[1*3+0][3]=W[1][3]+=1
        # a00*b12: b12=B[1,2], j=1,k=2. j=0!=1, no
        # a00*b20: b20=B[2,0], j=2,k=0. j=0!=2, no
        # a11*b01: a11=A[1,1], i=1,j=1. b01=B[0,1], j=0,k=1. j=1!=0, no
        # a11*b12: a11=A[1,1], b12=B[1,2], j=1,k=2. j=1!=2, no
        # a11*b20: a11=A[1,1], b20=B[2,0], j=2,k=0. j=1!=2, no
        # a22*b01: a22=A[2,2], b01=B[0,1], j=0,k=1. j=2!=0, no
        # a22*b12: a22=A[2,2], b12=B[1,2], j=1,k=2. j=2!=1, no
        # a22*b20: a22=A[2,2], b20=B[2,0], j=2,k=0. j=2==k=2? No, k=0. j=2!=0, no
        # Hmm, only one contribution? That seems wrong for Laderman.
        
        # I think I'm confusing myself. Let me restart with a cleaner approach.
        
        # Let me just use schoolbook for the full 3x3x3x8 to ensure correctness,
        # and then try to optimize if time permits.
        
        U = [[0.0]*R for _ in range(9)]
        V = [[0.0]*R for _ in range(9)]
        W = [[0.0]*R for _ in range(9)]
        
        # Schoolbook for 3x3x3: rank 27
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1.0
                    V[j*3+k][col] = 1.0
                    W[k*3+i][col] = 1.0
                    col += 1
        
        return U, V, W, 27

    def build_schoolbook_3x3x3xP(p_val):
        """Schoolbook decomposition for 3x3 * 3xP."""
        rank = 3 * 3 * p_val
        U = [[0.0]*rank for _ in range(9)]
        V = [[0.0]*rank for _ in range(3*p_val)]
        W = [[0.0]*rank for _ in range(p_val*3)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(p_val):
                    U[i*3+j][col] = 1.0
                    V[j*p_val+k][col] = 1.0
                    W[k*3+i][col] = 1.0
                    col += 1
        return U, V, W, rank

    # Strategy: Use Laderman for the first 3 columns (3x3 * 3x3) and schoolbook for remaining 5 columns
    # Laderman rank for 3x3*3x3 is 20, schoolbook for 3x3*3x5 is 45, total = 65
    
    # But first, let me verify if I can get a valid Laderman decomposition.
    
    # For now, use full schoolbook: rank 72
    U_full, V_full, W_full, rank_full = build_schoolbook_3x3x3xP(8)
    
    # Try to construct a better decomposition using Laderman for first 3 cols
    # and schoolbook for last 5 cols
    
    # Let me try to build Laderman properly by solving the linear system
    # For 3x3*3x3, we need 27 equations (one for each i,j,k) and rank R unknowns
    # Each rank term r has U_r (9-dim), V_r (9-dim), W_r (9-dim)
    # Constraint: for all i,j,k: sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = 1
    
    # This is a nonlinear system (trilinear). Hard to solve directly.
    
    # Instead, let me use the known Laderman result. I'll hardcode the matrices.
    
    # Laderman's 20 products (from standard references):
    # I'll use the formulation where:
    # U_r[i,j] = coefficient of A[i,j] in r-th linear form
    # V_r[j,k] = coefficient of B[j,k] in r-th linear form  
    # W_r[k,i] = coefficient for C[i,k] in r-th product
    
    # Known Laderman decomposition (rank 20 for 3x3*3x3):
    
    U_lad = [[0.0]*20 for _ in range(9)]
    V_lad = [[0.0]*20 for _ in range(9)]
    W_lad = [[0.0]*20 for _ in range(9)]
    
    # I'll define the 20 bilinear forms based on Laderman's paper
    # Let me use a verified set:
    
    # Form 0: (a00+a11+a22)(b00+b11+b22)
    U_lad[0][0] = 1; U_lad[4][0] = 1; U_lad[8][0] = 1
    V_lad[0][0] = 1; V_lad[4][0] = 1; V_lad[8][0] = 1
    W_lad[0][0] = 1; W_lad[4][0] = 1; W_lad[8][0] = 1
    
    # Form 1: (a00+a11)(b00+b11)
    U_lad[0][1] = 1; U_lad[4][1] = 1
    V_lad[0][1] = 1; V_lad[4][1] = 1
    W_lad[0][1] = -1; W_lad[4][1] = -1
    
    # Form 2: (a00+a22)(b00+b22)
    U_lad[0][2] = 1; U_lad[8][2] = 1
    V_lad[0][2] = 1; V_lad[8][2] = 1
    W_lad[0][2] = -1; W_lad[8][2] = -1
    
    # Form 3: (a00+a11+a22)(b01+b12+b20)
    U_lad[0][3] = 1; U_lad[4][3] = 1; U_lad[8][3] = 1
    V_lad[1][3] = 1; V_lad[5][3] = 1; V_lad[6][3] = 1
    W_lad[1][3] = 1; W_lad[5][3] = 1; W_lad[6][3] = 1
    
    # Form 4: (a00+a11)(b01+b12)
    U_lad[0][4] = 1; U_lad[4][4] = 1
    V_lad[1][4] = 1; V_lad[5][4] = 1
    W_lad[1][4] = -1; W_lad[5][4] = -1
    
    # Form 5: (a00+a22)(b01+b20)
    U_lad[0][5] = 1; U_lad[8][5] = 1
    V_lad[1][5] = 1; V_lad[6][5] = 1
    W_lad[1][5] = -1; W_lad[6][5] = -1
    
    # Form 6: (a01+a12+a20)(b00+b11+b22)
    U_lad[1][6] = 1; U_lad[5][6] = 1; U_lad[6][6] = 1
    V_lad[0][6] = 1; V_lad[4][6] = 1; V_lad[8][6] = 1
    W_lad[0][6] = 1; W_lad[4][6] = 1; W_lad[8][6] = 1
    
    # Form 7: (a01+a12)(b00+b11)
    U_lad[1][7] = 1; U_lad[5][7] = 1
    V_lad[0][7] = 1; V_lad[4][7] = 1
    W_lad[0][7] = -1; W_lad[4][7] = -1
    
    # Form 8: (a01+a20)(b00+b22)
    U_lad[1][8] = 1; U_lad[6][8] = 1
    V_lad[0][8] = 1; V_lad[8][8] = 1
    W_lad[0][8] = -1; W_lad[8][8] = -1
    
    # Form 9: (a01+a12+a20)(b02+b10+b21)
    U_lad[1][9] = 1; U_lad[5][9] = 1; U_lad[6][9] = 1
    V_lad[2][9] = 1; V_lad[3][9] = 1; V_lad[7][9] = 1
    W_lad[2][9] = 1; W_lad[3][9] = 1; W_lad[7][9] = 1
    
    # Form 10: (a01+a12)(b02+b10)
    U_lad[1][10] = 1; U_lad[5][10] = 1
    V_lad[2][10] = 1; V_lad[3][10] = 1
    W_lad[2][10] = -1; W_lad[3][10] = -1
    
    # Form 11: (a01+a20)(b02+b21)
    U_lad[1][11] = 1; U_lad[6][11] = 1
    V_lad[2][11] = 1; V_lad[7][11] = 1
    W_lad[2][11] = -1; W_lad[7][11] = -1
    
    # Form 12: (a10+a21)(b00+b11+b22)
    U_lad[3][12] = 1; U_lad[7][12] = 1
    V_lad[0][12] = 1; V_lad[4][12] = 1; V_lad[8][12] = 1
    W_lad[0][12] = 1; W_lad[4][12] = 1; W_lad[8][12] = 1
    
    # Form 13: (a10+a21)(b01+b12)
    U_lad[3][13] = 1; U_lad[7][13] = 1
    V_lad[1][13] = 1; V_lad[5][13] = 1
    W_lad[1][13] = -1; W_lad[5][13] = -1
    
    # Form 14: (a10+a21)(b02+b20)
    U_lad[3][14] = 1; U_lad[7][14] = 1
    V_lad[2][14] = 1; V_lad[6][14] = 1
    W_lad[2][14] = -1; W_lad[6][14] = -1
    
    # Form 15: (a00+a11)(b02+b10+b21)
    U_lad[0][15] = 1; U_lad[4][15] = 1
    V_lad[2][15] = 1; V_lad[3][15] = 1; V_lad[7][15] = 1
    W_lad[2][15] = 1; W_lad[3][15] = 1; W_lad[7][15] = 1
    
    # Form 16: (a00+a11)(b02+b10)
    U_lad[0][16] = 1; U_lad[4][16] = 1
    V_lad[2][16] = 1; V_lad[3][16] = 1
    W_lad[2][16] = -1; W_lad[3][16] = -1
    
    # Form 17: (a00+a22)(b02+b21)
    U_lad[0][17] = 1; U_lad[8][17] = 1
    V_lad[2][17] = 1; V_lad[7][17] = 1
    W_lad[2][17] = -1; W_lad[7][17] = -1
    
    # Form 18: (a01+a12)(b01+b20)
    U_lad[1][18] = 1; U_lad[5][18] = 1
    V_lad[1][18] = 1; V_lad[6][18] = 1
    W_lad[1][18] = -1; W_lad[6][18] = -1
    
    # Form 19: (a01+a20)(b01+b12)
    U_lad[1][19] = 1; U_lad[6][19] = 1
    V_lad[1][19] = 1; V_lad[5][19] = 1
    W_lad[1][19] = -1; W_lad[5][19] = -1
    
    # Validate Laderman
    if validate(U_lad, V_lad, W_lad):
        # Use Laderman for first 3 cols, schoolbook for last 5 cols
        # Total rank = 20 + 3*3*5 = 20 + 45 = 65
        U_comb = [[0.0]*65 for _ in range(9)]
        V_comb = [[0.0]*65 for _ in range(24)]  # 3*8 = 24
        W_comb = [[0.0]*65 for _ in range(24)]  # 8*3 = 24
        
        # Copy Laderman for first 3 columns (k=0,1,2)
        for r in range(20):
            for idx in range(9):
                U_comb[idx][r] = U_lad[idx][r]
            for j in range(3):
                for k in range(3):
                    V_comb[j*8+k][r] = V_lad[j*3+k][r]
            for k in range(3):
                for i in range(3):
                    W_comb[k*3+i][r] = W_lad[k*3+i][r]
        
        # Schoolbook for last 5 columns (k=3,4,5,6,7)
        col = 20
        for i in range(3):
            for j in range(3):
                for k in range(3, 8):
                    U_comb[i*3+j][col] = 1.0
                    V_comb[j*8+k][col] = 1.0
                    W_comb[k*3+i][col] = 1.0
                    col += 1
        
        if validate(U_comb, V_comb, W_comb):
            return {"U": U_comb, "V": V_comb, "W": W_comb}
    
    # Fallback to schoolbook
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
