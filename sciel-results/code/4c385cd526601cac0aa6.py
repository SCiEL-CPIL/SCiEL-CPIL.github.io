import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition based on Strassen for 2x2x2 blocks and schoolbook for the remainder.
    # 11 = 4 + 4 + 3 = (2+2) + (2+2) + (2+1)
    # Rank = 2*7 (for first 4) + 2*7 (for next 4) + 7 (for next 2) + 4 (for last 1) = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # k_offset is the starting k index for this block (size 2)
    def add_strassen_block(k1, k2):
        nonlocal col
        # Strassen products for A (2x2) and B (2x2, cols k1, k2)
        # B is indexed by j (0,1) and k (k1, k2)
        # Local B indices: b00=B[0,k1], b01=B[0,k2], b10=B[1,k1], b11=B[1,k2]
        
        # 1. (a+d)(b00+b11)
        # U: a+d -> i=0,j=0 (1), i=1,j=1 (1)
        # V: b00+b11 -> j=0,k1 (1), j=1,k2 (1)
        # W: C00, C11 -> k1*n+0, k2*n+1
        # C00 = M1 - M4 + M5 - M7
        # C11 = M1 - M2 + M6 + M7
        
        # We need to add contributions to U, V, W for each of the 7 terms.
        # Term 1: M1
        U[0*2+0][col] += 1; U[1*2+1][col] += 1
        V[0*p+k1][col] += 1; V[1*p+k2][col] += 1
        W[k1*n+0][col] += 1; W[k2*n+1][col] += 1
        col += 1
        
        # Term 2: M2 = (c+d)b01
        U[1*2+0][col] += 1; U[1*2+1][col] += 1
        V[0*p+k2][col] += 1
        W[k2*n+1][col] += -1 # C11 has -M2
        col += 1
        
        # Term 3: M3 = a(b10-b11)
        U[0*2+0][col] += 1
        V[1*p+k1][col] += 1; V[1*p+k2][col] += -1
        W[k2*n+1][col] += 1 # C11 has +M3? No, C12 = M3+M7. C11 is k2?
        # Wait, C indices:
        # C[0, k1] = M1 - M4 + M5 - M7
        # C[0, k2] = M3 + M7
        # C[1, k1] = M2 + M4
        # C[1, k2] = M1 - M2 + M6 + M7
        
        # Let's re-verify the mapping.
        # W index is k*n + i.
        # C[0, k1] -> W[k1*n + 0]
        # C[0, k2] -> W[k2*n + 0]
        # C[1, k1] -> W[k1*n + 1]
        # C[1, k2] -> W[k2*n + 1]
        
        # Correcting Term 1 (M1):
        # M1 contributes to C[0, k1] (+1), C[1, k2] (+1)
        # My previous W update was: W[k1*n+0] += 1, W[k2*n+1] += 1. This is correct.
        
        # Correcting Term 2 (M2):
        # M2 contributes to C[1, k1] (+1), C[1, k2] (-1)
        # V: b01 -> V[0*p+k2]
        # U: c+d -> U[1*2+0], U[1*2+1]
        # W: W[k1*n+1] += 1, W[k2*n+1] += -1
        # I had W[k2*n+1] += -1. Missing W[k1*n+1] += 1.
        
        # Let's restart the helper carefully.
        
        # I will rewrite the helper to be safe.
        pass

    # Let's just implement the 39 terms explicitly to avoid errors.
    
    # Block 1: k=0,1 (Strassen)
    # M1: (A00+A11)(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[11+1][col] = 1 # V[0,0]=1, V[1,1]=1
    W[0][col] = 1; W[2*1+1][col] = 1 # W[0,0]=1, W[1,1]=1
    col += 1
    
    # M2: (A10+A11)B01
    U[2][col] = 1; U[3][col] = 1
    V[11+1][col] = 1 # V[0,1]
    W[1][col] = 1; W[2*1+1][col] = -1 # W[0,1] is C[1,0]? No.
    # C[1,0] index: k=0, i=1 -> 0*2+1 = 1.
    # C[1,1] index: k=1, i=1 -> 1*2+1 = 3.
    # M2 contributes to C[1,0] (+1) and C[1,1] (-1).
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M3: A00(B10-B11)
    U[0][col] = 1
    V[11][col] = 1; V[11+1+11][col] = -1 # V[1,0]=1, V[1,1]=-1
    W[2*1+0][col] = 1 # C[0,1] index: k=1, i=0 -> 2.
    # M3 contributes to C[0,1] (+1).
    W[2][col] = 1
    col += 1
    
    # M4: A11(B00-B10)
    U[3][col] = 1
    V[0][col] = 1; V[11][col] = -1
    W[0][col] = -1; W[1][col] = 1 # C[0,0] (-1), C[1,0] (+1)
    col += 1
    
    # M5: (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[11+1+11][col] = 1 # V[1,1]
    W[0][col] = 1; W[3][col] = 1 # C[0,0] (+1), C[1,1] (+1)
    col += 1
    
    # M6: (A10-A00)(B00+B10)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[11][col] = 1
    W[3][col] = 1 # C[1,1] (+1)
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][col] = 1; U[3][col] = -1
    V[11][col] = 1; V[11+1+11][col] = 1
    W[0][col] = -1; W[2][col] = 1; W[3][col] = 1 # C[0,0](-1), C[0,1](+1), C[1,1](+1)
    col += 1
    
    # Block 2: k=2,3 (Strassen)
    # Same structure, shift k by 2.
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[0+2][col] = 1; V[11+1+3][col] = 1
    W[2*2+0][col] = 1; W[2*3+1][col] = 1
    col += 1
    # M2
    U[2][col] = 1; U[3][col] = 1
    V[0+3][col] = 1
    W[2*2+1][col] = 1; W[2*3+1][col] = -1
    col += 1
    # M3
    U[0][col] = 1
    V[11+2][col] = 1; V[11+1+3][col] = -1
    W[2*3+0][col] = 1
    col += 1
    # M4
    U[3][col] = 1
    V[0+2][col] = 1; V[11+2][col] = -1
    W[2*2+0][col] = -1; W[2*2+1][col] = 1
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[11+1+3][col] = 1
    W[2*2+0][col] = 1; W[2*3+1][col] = 1
    col += 1
    # M6
    U[2][col] = 1; U[0][col] = -1
    V[0+2][col] = 1; V[11+2][col] = 1
    W[2*3+1][col] = 1
    col += 1
    # M7
    U[1][col] = 1; U[3][col] = -1
    V[11+2][col] = 1; V[11+1+3][col] = 1
    W[2*2+0][col] = -1; W[2*3+0][col] = 1; W[2*3+1][col] = 1
    col += 1
    
    # Block 3: k=4,5 (Strassen)
    # Shift k by 4.
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[0+4][col] = 1; V[11+1+5][col] = 1
    W[2*4+0][col] = 1; W[2*5+1][col] = 1
    col += 1
    # M2
    U[2][col] = 1; U[3][col] = 1
    V[0+5][col] = 1
    W[2*4+1][col] = 1; W[2*5+1][col] = -1
    col += 1
    # M3
    U[0][col] = 1
    V[11+4][col] = 1; V[11+1+5][col] = -1
    W[2*5+0][col] = 1
    col += 1
    # M4
    U[3][col] = 1
    V[0+4][col] = 1; V[11+4][col] = -1
    W[2*4+0][col] = -1; W[2*4+1][col] = 1
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[11+1+5][col] = 1
    W[2*4+0][col] = 1; W[2*5+1][col] = 1
    col += 1
    # M6
    U[2][col] = 1; U[0][col] = -1
    V[0+4][col] = 1; V[11+4][col] = 1
    W[2*5+1][col] = 1
    col += 1
    # M7
    U[1][col] = 1; U[3][col] = -1
    V[11+4][col] = 1; V[11+1+5][col] = 1
    W[2*4+0][col] = -1; W[2*5+0][col] = 1; W[2*5+1][col] = 1
    col += 1
    
    # Block 4: k=6,7 (Strassen)
    # Shift k by 6.
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[0+6][col] = 1; V[11+1+7][col] = 1
    W[2*6+0][col] = 1; W[2*7+1][col] = 1
    col += 1
    # M2
    U[2][col] = 1; U[3][col] = 1
    V[0+7][col] = 1
    W[2*6+1][col] = 1; W[2*7+1][col] = -1
    col += 1
    # M3
    U[0][col] = 1
    V[11+6][col] = 1; V[11+1+7][col] = -1
    W[2*7+0][col] = 1
    col += 1
    # M4
    U[3][col] = 1
    V[0+6][col] = 1; V[11+6][col] = -1
    W[2*6+0][col] = -1; W[2*6+1][col] = 1
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[11+1+7][col] = 1
    W[2*6+0][col] = 1; W[2*7+1][col] = 1
    col += 1
    # M6
    U[2][col] = 1; U[0][col] = -1
    V[0+6][col] = 1; V[11+6][col] = 1
    W[2*7+1][col] = 1
    col += 1
    # M7
    U[1][col] = 1; U[3][col] = -1
    V[11+6][col] = 1; V[11+1+7][col] = 1
    W[2*6+0][col] = -1; W[2*7+0][col] = 1; W[2*7+1][col] = 1
    col += 1
    
    # Block 5: k=8,9 (Strassen)
    # Shift k by 8.
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[0+8][col] = 1; V[11+1+9][col] = 1
    W[2*8+0][col] = 1; W[2*9+1][col] = 1
    col += 1
    # M2
    U[2][col] = 1; U[3][col] = 1
    V[0+9][col] = 1
    W[2*8+1][col] = 1; W[2*9+1][col] = -1
    col += 1
    # M3
    U[0][col] = 1
    V[11+8][col] = 1; V[11+1+9][col] = -1
    W[2*9+0][col] = 1
    col += 1
    # M4
    U[3][col] = 1
    V[0+8][col] = 1; V[11+8][col] = -1
    W[2*8+0][col] = -1; W[2*8+1][col] = 1
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[11+1+9][col] = 1
    W[2*8+0][col] = 1; W[2*9+1][col] = 1
    col += 1
    # M6
    U[2][col] = 1; U[0][col] = -1
    V[0+8][col] = 1; V[11+8][col] = 1
    W[2*9+1][col] = 1
    col += 1
    # M7
    U[1][col] = 1; U[3][col] = -1
    V[11+8][col] = 1; V[11+1+9][col] = 1
    W[2*8+0][col] = -1; W[2*9+0][col] = 1; W[2*9+1][col] = 1
    col += 1
    
    # Block 6: k=10 (Schoolbook)
    # C[0,10] = A[0,0]B[0,10] + A[0,1]B[1,10]
    # C[1,10] = A[1,0]B[0,10] + A[1,1]B[1,10]
    
    # Term 1: A[0,0]B[0,10] -> C[0,10]
    U[0][col] = 1
    V[0+10][col] = 1
    W[2*10+0][col] = 1
    col += 1
    
    # Term 2: A[0,1]B[1,10] -> C[0,10]
    U[1][col] = 1
    V[11+10][col] = 1
    W[2*10+0][col] = 1
    col += 1
    
    # Term 3: A[1,0]B[0,10] -> C[1,10]
    U[2][col] = 1
    V[0+10][col] = 1
    W[2*10+1][col] = 1
    col += 1
    
    # Term 4: A[1,1]B[1,10] -> C[1,10]
    U[3][col] = 1
    V[11+10][col] = 1
    W[2*10+1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
