import numpy as np

def construct():
        n, m, p = 2, 3, 10
        rank = 55
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # A indices: 0..5
        # A[0,0]=0, A[0,1]=1, A[0,2]=2
        # A[1,0]=3, A[1,1]=4, A[1,2]=5
        
        # B indices: 0..29
        # B[r,c] = r*10 + c
        
        # C indices: 0..19
        # C[r,c] = r*2 + c? No, "output factor follows the official trace convention W[k*n+i,r]".
        # Usually W is indexed by [c, r] or similar?
        # "W[k*n+i,r]" where k is column index of C?
        # C is 2x10.
        # W shape is (p*n, rank) = (20, rank).
        # Index in W: (c_idx * n + r_idx, rank_idx)?
        # "k*n+i" -> k is likely the column index (0..9), i is row index (0..1).
        # So W[k*2+i, :] corresponds to C[i,k].
        
        col = 0
        for b in range(5):
            # B block b: columns 2b, 2b+1
            # B indices for this block:
            # B[0, 2b], B[0, 2b+1]
            # B[1, 2b], B[1, 2b+1]
            # B[2, 2b], B[2, 2b+1]
            
            # C indices for this block:
            # C[0, 2b], C[0, 2b+1]
            # C[1, 2b], C[1, 2b+1]
            
            # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> 0, 1, 3, 4
            # A_R: A[0,2], A[1,2] -> 2, 5
            
            # B_L: B[0, 2b], B[0, 2b+1], B[1, 2b], B[1, 2b+1]
            # B_R: B[2, 2b], B[2, 2b+1]
            
            b00 = 0 * 10 + 2 * b
            b01 = 0 * 10 + 2 * b + 1
            b10 = 1 * 10 + 2 * b
            b11 = 1 * 10 + 2 * b + 1
            b20 = 2 * 10 + 2 * b
            b21 = 2 * 10 + 2 * b + 1
            
            c00 = 2 * b * 2 + 0 # k=2b, i=0 -> 2b*2 + 0
            c01 = 2 * b * 2 + 1 # k=2b+1, i=0 -> (2b+1)*2 + 0? No.
            # W[k*n+i, r].
            # C[i,k].
            # If k=2b, i=0 -> index = 2b*2 + 0 = 4b.
            # If k=2b+1, i=0 -> index = (2b+1)*2 + 0 = 4b+2.
            # If k=2b, i=1 -> index = 2b*2 + 1 = 4b+1.
            # If k=2b+1, i=1 -> index = (2b+1)*2 + 1 = 4b+3.
            
            c_0_2b = 4 * b
            c_1_2b = 4 * b + 1
            c_0_2b1 = 4 * b + 2
            c_1_2b1 = 4 * b + 3
            
            # Strassen for A_L B_L
            # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            U[0, col] = 1; U[4, col] = 1
            V[b00, col] = 1; V[b11, col] = 1
            W[c_0_2b, col] = 1; W[c_1_2b1, col] = 1
            col += 1
            
            # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
            U[3, col] = 1; U[4, col] = 1
            V[b00, col] = 1
            W[c_1_2b, col] = 1; W[c_1_2b1, col] = -1
            col += 1
            
            # M3: A00(B01-B11) -> C01 += M3, C11 += M3
            U[0, col] = 1
            V[b01, col] = 1; V[b11, col] = -1
            W[c_0_2b1, col] = 1; W[c_1_2b1, col] = 1
            col += 1
            
            # M4: A11(B10-B00) -> C00 += M4, C10 += M4
            U[4, col] = 1
            V[b10, col] = 1; V[b00, col] = -1
            W[c_0_2b, col] = 1; W[c_1_2b, col] = 1
            col += 1
            
            # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
            U[0, col] = 1; U[1, col] = 1
            V[b11, col] = 1
            W[c_0_2b, col] = -1; W[c_0_2b1, col] = 1
            col += 1
            
            # M6: (A10-A00)(B00+B01) -> C11 += M6
            U[3, col] = 1; U[0, col] = -1
            V[b00, col] = 1; V[b01, col] = 1
            W[c_1_2b1, col] = 1
            col += 1
            
            # M7: (A01-A11)(B10+B11) -> C00 += M7
            U[1, col] = 1; U[4, col] = -1
            V[b10, col] = 1; V[b11, col] = 1
            W[c_0_2b, col] = 1
            col += 1
            
            # Cross terms A_R B_R
            # P1: A02 B20 -> C00 += P1
            U[2, col] = 1; V[b20, col] = 1; W[c_0_2b, col] = 1
            col += 1
            
            # P2: A02 B21 -> C01 += P2
            U[2, col] = 1; V[b21, col] = 1; W[c_0_2b1, col] = 1
            col += 1
            
            # P3: A12 B20 -> C10 += P3
            U[5, col] = 1; V[b20, col] = 1; W[c_1_2b, col] = 1
            col += 1
            
            # P4: A12 B21 -> C11 += P4
            U[5, col] = 1; V[b21, col] = 1; W[c_1_2b1, col] = 1
            col += 1
            
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
