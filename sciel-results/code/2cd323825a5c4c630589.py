import time
import numpy as np

def construct():
    t0 = time.time()
    limit = t0 + 110.0

    # Dimensions: multiply (2x2) by (2x16) -> (2x16)
    # n=2 (rows of A, cols of C), m=2 (cols of A, rows of B), p=16 (cols of B, cols of C)
    n, m, p = 2, 2, 16
    
    # Rank 7 Strassen decomposition for 2x2 x 2x2
    # We define the 7 terms for a single 2x2x2 block.
    # U_local: (4, 7), V_local: (4, 7), W_local: (4, 7)
    # Indices:
    # A: i in {0,1}, j in {0,1} -> idx = i*2 + j
    # B: j in {0,1}, k in {0,1} -> idx = j*2 + k
    # C: i in {0,1}, k in {0,1} -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)(B00)     -> C10 += M2, C11 -= M2
    # M3 = (A00)(B10-B11)     -> C01 += M3, C11 += M3
    # M4 = (A11)(B00-B10)     -> C00 += M4, C01 += M4
    # M5 = (A00+A01)(B11)     -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C10 += M6
    # M7 = (A01-A11)(B10+B11) -> C01 += M7
    
    # Let's verify the reconstruction:
    # C00 = M1 + M4 - M5
    # C01 = M3 + M4 + M5 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3
    
    # Check C00: (A00+A11)(B00+B11) + A11(B00-B10) - (A00+A01)B11
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B00 - A11B10 - A00B11 - A01B11
    # = A00B00 + 2A11B00 + A11B11 - A11B10 - A01B11
    # Target C00 = A00B00 + A01B10. 
    # Wait, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21-B22)
    # M4 = (A22)(B11-B21)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3
    
    # My indices:
    # A11->A00, A12->A01, A21->A10, A22->A11
    # B11->B00, B12->B01, B21->B10, B22->B11
    # C11->C00, C12->C01, C21->C10, C22->C11
    
    # Let's map the standard Strassen to 0-indexed:
    # M1: U: (0,0):1, (1,1):1. V: (0,0):1, (1,1):1. W: (0,0):1, (1,1):1. (C00, C11)
    # M2: U: (1,0):1, (1,1):1. V: (0,0):1. W: (0,1):1, (1,1):-1. (C10, C11) -> C10+=M2, C11-=M2
    # M3: U: (0,0):1. V: (0,1):-1, (1,1):1. W: (1,0):1, (1,1):1. (C01, C11) -> C01+=M3, C11+=M3
    #      Wait, standard M3 = A11(B21-B22) = A00(B10-B11).
    #      V for B10-B11: B10 is (1,0), B11 is (1,1). So V: (1,0):1, (1,1):-1.
    #      Let's re-check standard M3.
    #      M3 = A11(B21 - B22).
    #      A11 is A00. B21 is B10. B22 is B11.
    #      So V: (1,0):1, (1,1):-1.
    #      C12 = M3 + M5. C22 = M1 - M2 + M3.
    #      So W for M3 should contribute to C01 and C11.
    #      W: (1,0):1 (C01), (1,1):1 (C11).
    
    # M4: A22(B11-B21) = A11(B00-B10).
    #      U: (1,1):1.
    #      V: (0,0):1, (1,0):-1.
    #      C11 = ... + M4. C12 = ... + M4.
    #      W: (0,0):1 (C00? No, C11 is (1,1) in 1-based, which is (1,1) in 0-based? No.
    #      C indices: C11->(0,0), C12->(0,1), C21->(1,0), C22->(1,1).
    #      Trace convention W[k*n+i, r].
    #      C[i,k].
    #      C00 -> i=0, k=0 -> W[0*2+0, r] = W[0, r].
    #      C01 -> i=0, k=1 -> W[1*2+0, r] = W[2, r].
    #      C10 -> i=1, k=0 -> W[0*2+1, r] = W[1, r].
    #      C11 -> i=1, k=1 -> W[1*2+1, r] = W[3, r].
    
    # Let's re-verify W indices for Strassen.
    # C00 (i=0, k=0) -> W[0, l]
    # C01 (i=0, k=1) -> W[2, l]
    # C10 (i=1, k=0) -> W[1, l]
    # C11 (i=1, k=1) -> W[3, l]
    
    # M1: C00+=M1, C11+=M1.
    #      W: W[0,0]=1, W[3,0]=1.
    # M2: C10+=M2, C11-=M2.
    #      W: W[1,1]=1, W[3,1]=-1.
    # M3: C01+=M3, C11+=M3.
    #      W: W[2,2]=1, W[3,2]=1.
    # M4: C00+=M4, C01+=M4.
    #      W: W[0,3]=1, W[2,3]=1.
    # M5: C00-=M5, C01+=M5.
    #      W: W[0,4]=-1, W[2,4]=1.
    # M6: C10+=M6.
    #      W: W[1,5]=1.
    # M7: C01+=M7.
    #      W: W[2,6]=1.
    
    # Let's check U and V.
    # M1: A00+A11, B00+B11.
    #      U: U[0,0]=1, U[3,0]=1. (A00 is idx 0, A11 is idx 3)
    #      V: V[0,0]=1, V[3,0]=1. (B00 is idx 0, B11 is idx 3)
    # M2: A10+A11, B00.
    #      U: U[2,1]=1, U[3,1]=1. (A10 is idx 2, A11 is idx 3)
    #      V: V[0,1]=1.
    # M3: A00, B10-B11.
    #      U: U[0,2]=1.
    #      V: V[2,2]=1, V[3,2]=-1. (B10 is idx 2, B11 is idx 3)
    # M4: A11, B00-B10.
    #      U: U[3,3]=1.
    #      V: V[0,3]=1, V[2,3]=-1.
    # M5: A00+A01, B11.
    #      U: U[0,4]=1, U[1,4]=1. (A00 idx 0, A01 idx 1)
    #      V: V[3,4]=1.
    # M6: A10-A00, B00+B01.
    #      U: U[2,5]=1, U[0,5]=-1.
    #      V: V[0,5]=1, V[1,5]=1. (B00 idx 0, B01 idx 1)
    # M7: A01-A11, B10+B11.
    #      U: U[1,6]=1, U[3,6]=-1.
    #      V: V[2,6]=1, V[3,6]=1.

    # Construct base matrices
    base_U = np.zeros((4, 7), dtype=np.int8)
    base_V = np.zeros((4, 7), dtype=np.int8)
    base_W = np.zeros((4, 7), dtype=np.int8)

    # M1
    base_U[0, 0] = 1; base_U[3, 0] = 1
    base_V[0, 0] = 1; base_V[3, 0] = 1
    base_W[0, 0] = 1; base_W[3, 0] = 1
    
    # M2
    base_U[2, 1] = 1; base_U[3, 1] = 1
    base_V[0, 1] = 1
    base_W[1, 1] = 1; base_W[3, 1] = -1
    
    # M3
    base_U[0, 2] = 1
    base_V[2, 2] = 1; base_V[3, 2] = -1
    base_W[2, 2] = 1; base_W[3, 2] = 1
    
    # M4
    base_U[3, 3] = 1
    base_V[0, 3] = 1; base_V[2, 3] = -1
    base_W[0, 3] = 1; base_W[2, 3] = 1
    
    # M5
    base_U[0, 4] = 1; base_U[1, 4] = 1
    base_V[3, 4] = 1
    base_W[0, 4] = -1; base_W[2, 4] = 1
    
    # M6
    base_U[2, 5] = 1; base_U[0, 5] = -1
    base_V[0, 5] = 1; base_V[1, 5] = 1
    base_W[1, 5] = 1
    
    # M7
    base_U[1, 6] = 1; base_U[3, 6] = -1
    base_V[2, 6] = 1; base_V[3, 6] = 1
    base_W[2, 6] = 1

    # Expand to 2x2x16
    # Rank = 56
    rank = 56
    U_full = np.zeros((4, 56), dtype=np.int8)
    V_full = np.zeros((32, 56), dtype=np.int8) # m*p = 2*16
    W_full = np.zeros((32, 56), dtype=np.int8) # p*n = 16*2

    for q in range(8):
        r0 = 7 * q
        # U is identical for all blocks
        U_full[:, r0:r0+7] = base_U
        
        # V: Block q uses B columns 2q, 2q+1
        # V_full[j*16 + k, r]
        # base_V[j*2 + kl, l] corresponds to B[j, kl]
        # Global k = 2q + kl
        for j in range(2):
            for kl in range(2):
                val = base_V[j*2 + kl, :]
                if np.any(val != 0):
                    V_full[j*16 + (2*q + kl), r0:r0+7] = val

        # W: Block q uses C columns 2q, 2q+1
        # W_full[k*2 + i, r]
        # base_W[kl*2 + i, l] corresponds to C[i, kl]
        # Global k = 2q + kl
        for kl in range(2):
            for i in range(2):
                val = base_W[kl*2 + i, :]
                if np.any(val != 0):
                    W_full[(2*q + kl)*2 + i, r0:r0+7] = val

    # Convert to list of lists of ints
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
