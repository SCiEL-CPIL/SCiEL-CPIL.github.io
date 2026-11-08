import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 4, 4, 8
    
    # Use 2x2x2 Strassen (rank 7) applied to 4x4x8
    # Partition A (4x4) into four 2x2 blocks
    # Partition B (4x8) into four 2x4 blocks  
    # Output C (4x8) into four 2x4 blocks
    # Each 2x2 * 2x4 = 2x4 can use Strassen-like with rank 7 per pair
    # But 2x2 * 2x4 is not square, need different approach
    
    # Alternative: use 4x4x4 decomposition (rank 49) for two halves
    # Total rank = 2 * 49 = 98
    
    # Try to improve: use a 4x4x4 decomposition with rank < 49
    # Known: 4x4x4 tensor rank is at most 49 (Strassen) and at least 36
    # Let's use the standard Strassen 4x4x4 (rank 49)
    
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 2x2x2 Strassen base
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # m0: (A00+A11)(B00+B11) -> C00+C11
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    # m1: (A10+A11)B00 -> C10-C11
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1
    # m2: A00(B01-B11) -> C01+C11
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1
    # m3: A11(B10-B00) -> C00+C10
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[2,3]=1
    # m4: (A00+A01)B11 -> -C00+C01
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    # m5: (A10-A00)(B00+B01) -> C11
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    # m6: (A01-A11)(B10+B11) -> C00
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1
    
    # 4x4x4 Strassen: 7x7 = 49 terms
    # A block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B block indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C block indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # Strassen 4x4x4 specs: (A_coef, A_block), (B_coef, B_block), (C_coef, C_block)
    strassen_specs = [
        # P0 = (A00+A11)(B00+B11) -> C00+C11
        ([((1,0),(1,3))], [((1,0),(1,3))], [((1,0),(1,3))]),
        # P1 = (A10+A11)B00 -> C10-C11
        ([((1,2),(1,3))], [((1,0))], [((1,2),(-1,3))]),
        # P2 = A00(B01-B11) -> C01+C11
        ([((1,0))], [((1,1),(-1,3))], [((1,1),(1,3))]),
        # P3 = A11(B10-B00) -> C00+C10
        ([((1,3))], [((1,2),(-1,0))], [((1,0),(1,2))]),
        # P4 = (A00+A01)B11 -> -C00+C01
        ([((1,0),(1,1))], [((1,3))], [((-1,0),(1,1))]),
        # P5 = (A10-A00)(B00+B01) -> C11
        ([((1,2),(-1,0))], [((1,0),(1,1))], [((1,3))]),
        # P6 = (A01-A11)(B10+B11) -> C00
        ([((1,1),(-1,3))], [((1,2),(1,3))], [((1,0))]),
    ]
    
    for half in range(2):
        base_col = half * 49
        for i in range(7):
            for j in range(7):
                col = base_col + i*7 + j
                a_specs, b_specs, c_specs = strassen_specs[i]
                
                # Fill U (A side): 4x4 matrix, 16 entries
                for coeff, block in a_specs:
                    br, bc = divmod(block, 2)
                    for r in range(2):
                        for c in range(2):
                            a_row = br*2 + r
                            a_col = bc*2 + c
                            U[a_row*4 + a_col, col] += coeff * U2[r*2+c, j]
                
                # Fill V (B side): 4x8 matrix, 32 entries
                for coeff, block in b_specs:
                    br, bc = divmod(block, 2)
                    for r in range(2):
                        for c in range(2):
                            b_row = br*2 + r
                            b_col = bc*2 + c + half*4
                            V[b_row*8 + b_col, col] += coeff * V2[r*2+c, j]
                
                # Fill W (C side): 4x8 matrix, 32 entries, W[k*n+i, r]
                for coeff, block in c_specs:
                    br, bc = divmod(block, 2)
                    for r in range(2):
                        for c in range(2):
                            c_row = br*2 + r
                            c_col = bc*2 + c + half*4
                            # W[k*n+i, r] where k=c_col, i=c_row, r is output row
                            W[c_col*4 + c_row, col] += coeff * W2[r*2+c, j]
    
    if time.time() > deadline:
        # Return what we have
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
