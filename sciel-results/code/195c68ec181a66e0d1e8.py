import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    n, m, p = 6, 6, 6
    rank = 189
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2 factors
    # Indices: 0:00, 1:01, 2:10, 3:11 for U and V (row*2+col)
    # W indices: 0:00, 1:01, 2:10, 3:11
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1
    
    # M3 = A00(B10-B00)
    # C10 += M3, C11 += M3
    U2[0,2]=1
    V2[2,2]=1; V2[0,2]=-1
    W2[2,2]=1; W2[3,2]=1
    
    # M4 = A11(B01-B11)
    # C00 -= M4, C01 += M4
    # C11 coeff is 0
    U2[3,3]=1
    V2[1,3]=1; V2[3,3]=-1
    W2[0,3]=-1; W2[1,3]=1
    # W2[3,3] remains 0
    
    # M5 = (A00+A01)B11
    # C00 += M5, C01 += M5
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=1; W2[1,4]=1
    
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # M7 = (A01-A11)(B10+B11)
    # C00 -= M7
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=-1
    
    # Schoolbook 3x3 factors (Rank 27)
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U3[i*3+k, col] = 1
                V3[k*3+j, col] = 1
                W3[j*3+i, col] = 1
                col += 1
                
    # Kronecker Product
    # Total Rank = 27 * 7 = 189
    # n1=3, m1=3, p1=3, R1=27
    # n2=2, m2=2, p2=2, R2=7
    for r3 in range(27):
        for r2 in range(7):
            r_total = r3 * 7 + r2
            
            # Check deadline
            if time.time() > deadline:
                # Return partial result or fallback? 
                # For now, break and return what we have (zeros for rest)
                # But this might be invalid. 
                # Given the speed, this is unlikely to happen.
                break
                
            # U: (i1*2+i2)*6 + (k1*2+k2)
            for i1 in range(3):
                for k1 in range(3):
                    val1 = U3[i1*3+k1, r3]
                    if val1 == 0: continue
                    for i2 in range(2):
                        for k2 in range(2):
                            val2 = U2[i2*2+k2, r2]
                            if val2 == 0: continue
                            idx_u = (i1*2+i2)*6 + (k1*2+k2)
                            U[idx_u, r_total] = val1 * val2
                            
            # V: (k1*2+k2)*6 + (j1*2+j2)
            for k1 in range(3):
                for j1 in range(3):
                    val1 = V3[k1*3+j1, r3]
                    if val1 == 0: continue
                    for k2 in range(2):
                        for j2 in range(2):
                            val2 = V2[k2*2+j2, r2]
                            if val2 == 0: continue
                            idx_v = (k1*2+k2)*6 + (j1*2+j2)
                            V[idx_v, r_total] = val1 * val2
                            
            # W: (j1*2+j2)*6 + (i1*2+i2)
            for j1 in range(3):
                for i1 in range(3):
                    val1 = W3[j1*3+i1, r3]
                    if val1 == 0: continue
                    for j2 in range(2):
                        for i2 in range(2):
                            val2 = W2[j2*2+i2, r2]
                            if val2 == 0: continue
                            idx_w = (j1*2+j2)*6 + (i1*2+i2)
                            W[idx_w, r_total] = val1 * val2
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
