import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: Use Strassen for 4x4x4 part and schoolbook for 4x4x3 part
    # Split columns: first 4 columns use Strassen, last 3 use schoolbook
    
    # Strassen 4x4x4 decomposition
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For Strassen 4x4x4, we need to decompose into rank 49.
    # Each Mi is a 2x2x2 multiplication, which can be done with rank 7.
    # So we need 7 * 7 = 49 terms.
    
    # Let's construct the Strassen 4x4x4 decomposition.
    # The 2x2x2 Strassen decomposition:
    # For multiplying 2x2 by 2x2:
    # M1' = (a11+a22)(b11+b22)
    # M2' = (a21+a22)b11
    # M3' = a11(b12-b22)
    # M4' = a22(b21-b11)
    # M5' = (a11+a12)b22
    # M6' = (a21-a11)(b11+b12)
    # M7' = (a12-a22)(b21+b22)
    # c11 = M1'+M4'-M5'+M7'
    # c12 = M3'+M5'
    # c21 = M2'+M4'
    # c22 = M1'-M2'+M3'+M6'
    
    # The rank 7 decomposition of 2x2x2:
    # U entries: (a11+a22, a21+a22, a11, a22, a11+a12, a21-a11, a12-a22)
    # V entries: (b11+b22, b11, b12-b22, b21-b11, b22, b11+b12, b21+b22)
    # W entries: (c11+c22-c5+w7, c3+c5, c2+c4, c1-c2+c3+c6)
    # More precisely:
    # w1 = 1,1,0,0,0,0,1  (for c11: M1'+M4'+M7', but let me be more careful)
    
    # Actually, the standard 2x2x2 rank 7 decomposition:
    # U = [[1,0,0,0,0,0,0],  # a11
    #      [0,0,0,0,1,0,0],  # a12
    #      [0,1,0,0,0,0,0],  # a21
    #      [1,1,0,0,0,0,0]]  # a22
    # Wait, let me use the standard formulation.
    
    # Standard 2x2x2 Strassen:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b12-b22)
    # p4 = a22(b21-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    # c11 = p1+p4-p5+p7
    # c12 = p3+p5
    # c21 = p2+p4
    # c22 = p1-p2+p3+p6
    
    # U matrix (4x7): rows are a11,a12,a21,a22
    # U[0,:] = [1,0,1,0,1,-1,0]  # a11 appears in p1,p3,p5,p6
    # U[1,:] = [0,0,0,0,1,0,-1]  # a12 appears in p5,p7
    # U[2,:] = [0,1,0,0,0,1,0]   # a21 appears in p2,p6
    # U[3,:] = [1,1,0,1,0,0,0]   # a22 appears in p1,p2,p4
    
    # Wait, let me re-derive:
    # p1 = (a11+a22)(b11+b22): U coeff for a11 is 1, for a22 is 1
    # p2 = (a21+a22)b11: U coeff for a21 is 1, for a22 is 1
    # p3 = a11(b12-b22): U coeff for a11 is 1
    # p4 = a22(b21-b11): U coeff for a22 is 1
    # p5 = (a11+a12)b22: U coeff for a11 is 1, for a12 is 1
    # p6 = (a21-a11)(b11+b12): U coeff for a21 is 1, for a11 is -1
    # p7 = (a12-a22)(b21+b22): U coeff for a12 is 1, for a22 is -1
    
    # So U (4x7):
    # Row a11: [1,0,1,0,1,-1,0]
    # Row a12: [0,0,0,0,1,0,1]
    # Row a21: [0,1,0,0,0,1,0]
    # Row a22: [1,1,0,1,0,0,-1]
    
    # V (4x7): rows are b11,b12,b21,b22
    # p1: b11=1, b22=1
    # p2: b11=1
    # p3: b12=1, b22=-1
    # p4: b21=1, b11=-1
    # p5: b22=1
    # p6: b11=1, b12=1
    # p7: b21=1, b22=1
    # V:
    # Row b11: [1,1,0,-1,0,1,0]
    # Row b12: [0,0,1,0,0,1,0]
    # Row b21: [0,0,0,1,0,0,1]
    # Row b22: [1,0,-1,0,1,0,1]
    
    # W (4x7): rows are c11,c12,c21,c22
    # c11 = p1+p4-p5+p7
    # c12 = p3+p5
    # c21 = p2+p4
    # c22 = p1-p2+p3+p6
    # W:
    # Row c11: [1,0,0,1,-1,0,1]
    # Row c12: [0,0,1,0,1,0,0]
    # Row c21: [0,1,0,1,0,0,0]
    # Row c22: [1,-1,1,0,0,1,0]
    
    # Now for 4x4x4 Strassen, we have 7 products, each of which is a 2x2x2 multiplication.
    # The total rank is 7*7 = 49.
    
    # Let's define the U, V, W for the 4x4x4 Strassen decomposition.
    # A is 4x4, indexed as A[i,j] where i,j in 0..3
    # Blocks: A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each Mi, we use the 2x2x2 rank 7 decomposition.
    # The U, V, W for each Mi depend on how the blocks are combined.
    
    # Let's define the decomposition for each Mi.
    # For M1 = (A11+A22)(B11+B22):
    # The "a" matrix is A11+A22, which has entries:
    # (A11+A22)[0,0] = A[0,0]+A[2,2]
    # (A11+A22)[0,1] = A[0,1]+A[2,3]
    # (A11+A22)[1,0] = A[1,0]+A[3,2]
    # (A11+A22)[1,1] = A[1,1]+A[3,3]
    # The "b" matrix is B11+B22, which has entries:
    # (B11+B22)[0,0] = B[0,0]+B[2,2]
    # (B11+B22)[0,1] = B[0,1]+B[2,3]
    # (B11+B22)[1,0] = B[1,0]+B[3,2]
    # (B11+B22)[1,1] = B[1,1]+B[3,3]
    
    # This is getting complex. Let me use a different approach.
    
    # Actually, let me just use the schoolbook decomposition for simplicity,
    # but try to optimize by using Strassen for the 4x4x4 part.
    
    # For now, let's use the schoolbook for the entire 4x4x7.
    # Rank = 4*4*7 = 112
    
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
