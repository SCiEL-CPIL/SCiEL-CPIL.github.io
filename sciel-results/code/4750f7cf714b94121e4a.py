import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 3, 6
    
    # We will construct M_{2,3,6} as 3 blocks of M_{2,3,2}
    # Each M_{2,3,2} has rank 11.
    # Total rank = 33.
    
    # First, define the rank 11 decomposition for M_{2,3,2}
    # M_{2,3,2} multiplies A (2x3) by B (3x2).
    # Indices: a in {0,1}, b in {0,1,2}, c in {0,1,2}, d in {0,1}, i in {0,1}, k in {0,1}
    # U: 6x11, V: 6x11, W: 4x11
    
    U_232 = np.zeros((6, 11))
    V_232 = np.zeros((6, 11))
    W_232 = np.zeros((4, 11))
    
    # Block 1: b,c in {0,1}. This is M_{2,2,2} (Strassen)
    # a in {0,1}, b in {0,1} -> U rows 0,1,2,3 (a*m+b = 2a+b)
    # c in {0,1}, d in {0,1} -> V rows 0,1,2,3 (c*p+d = 2c+d)
    # k in {0,1}, i in {0,1} -> W rows 0,1,2,3 (k*n+i = 2k+i)
    
    # Strassen's 7 terms for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Map to our indices:
    # A is 2x2 (rows a=0,1; cols b=0,1).
    # B is 2x2 (rows c=0,1; cols d=0,1).
    # C is 2x2 (rows i=0,1; cols k=0,1).
    
    # Term 0 (M1): (A00+A11)(B00+B11)
    U_232[0, 0] = 1 # A00
    U_232[3, 0] = 1 # A11
    V_232[0, 0] = 1 # B00
    V_232[3, 0] = 1 # B11
    W_232[0, 0] = 1 # C00 (k=0,i=0)
    W_232[3, 0] = 1 # C11 (k=1,i=1)
    
    # Term 1 (M2): (A10+A11)B00
    U_232[2, 1] = 1 # A10
    U_232[3, 1] = 1 # A11
    V_232[0, 1] = 1 # B00
    W_232[2, 1] = 1 # C10 (k=0,i=1)
    
    # Term 2 (M3): A00(B10-B11)
    U_232[0, 2] = 1 # A00
    V_232[2, 2] = 1 # B10
    V_232[3, 2] = -1 # B11
    W_232[1, 2] = 1 # C01 (k=1,i=0)
    W_232[2, 2] = 1 # C10 (k=0,i=1)
    
    # Term 3 (M4): A11(B01-B00)
    U_232[3, 3] = 1 # A11
    V_232[1, 3] = 1 # B01
    V_232[0, 3] = -1 # B00
    W_232[0, 3] = 1 # C00
    W_232[2, 3] = 1 # C10
    
    # Term 4 (M5): (A00+A01)B11
    U_232[0, 4] = 1 # A00
    U_232[1, 4] = 1 # A01
    V_232[3, 4] = 1 # B11
    W_232[0, 4] = -1 # C00
    W_232[1, 4] = 1 # C01
    
    # Term 5 (M6): (A10-A00)(B00+B10)
    U_232[2, 5] = 1 # A10
    U_232[0, 5] = -1 # A00
    V_232[0, 5] = 1 # B00
    V_232[2, 5] = 1 # B10
    W_232[2, 5] = 1 # C10
    W_232[3, 5] = 1 # C11
    
    # Term 6 (M7): (A01-A11)(B01+B11)
    U_232[1, 6] = 1 # A01
    U_232[3, 6] = -1 # A11
    V_232[1, 6] = 1 # B01
    V_232[3, 6] = 1 # B11
    W_232[0, 6] = 1 # C00
    W_232[1, 6] = 1 # C01
    
    # Block 2: b=2, c=2. This is M_{2,1,2} (Outer product)
    # A is 2x1 (rows a=0,1; col b=2).
    # B is 1x2 (row c=2; cols d=0,1).
    # C is 2x2.
    # C_{i,k} = A_{i,2} B_{2,k}.
    # We need 4 terms for the 4 diagonal elements of the outer product tensor.
    
    # Term 7 (C00 = A02 * B20):
    U_232[4, 7] = 1 # A02 (a=0, b=2 -> idx 2*0+2=4)
    V_232[4, 7] = 1 # B20 (c=2, d=0 -> idx 2*2+0=4)
    W_232[0, 7] = 1 # C00
    
    # Term 8 (C01 = A02 * B21):
    U_232[4, 8] = 1 # A02
    V_232[5, 8] = 1 # B21 (c=2, d=1 -> idx 2*2+1=5)
    W_232[1, 8] = 1 # C01
    
    # Term 9 (C10 = A12 * B20):
    U_232[5, 9] = 1 # A12 (a=1, b=2 -> idx 2*1+2=5)
    V_232[4, 9] = 1 # B20
    W_232[2, 9] = 1 # C10
    
    # Term 10 (C11 = A12 * B21):
    U_232[5, 10] = 1 # A12
    V_232[5, 10] = 1 # B21
    W_232[3, 10] = 1 # C11
    
    # Now construct M_{2,3,6} from 3 blocks of M_{2,3,2}
    # Block 0: d in {0,1}, k in {0,1}
    # Block 1: d in {2,3}, k in {2,3}
    # Block 2: d in {4,5}, k in {4,5}
    
    R = 33
    U_comb = np.zeros((6, R))
    V_comb = np.zeros((18, R))
    W_comb = np.zeros((12, R))
    
    for block in range(3):
        offset = block * 11
        
        # U is the same for all blocks (since A is the same)
        U_comb[:, offset:offset+11] = U_232
        
        # V: rows c*6+d. c in {0,1,2}. d in {2*block, 2*block+1}
        for c in range(3):
            for d_local in range(2):
                d_global = 2 * block + d_local
                v_row_global = c * 6 + d_global
                v_row_local = c * 2 + d_local # This matches the V_232 indexing (c*2+d)
                V_comb[v_row_global, offset:offset+11] = V_232[v_row_local]
                
        # W: rows k*2+i. k in {2*block, 2*block+1}. i in {0,1}
        for k_local in range(2):
            k_global = 2 * block + k_local
            for i in range(2):
                w_row_global = k_global * 2 + i
                w_row_local = k_local * 2 + i # This matches W_232 indexing (k*2+i)
                W_comb[w_row_global, offset:offset+11] = W_232[w_row_local]
                
    return {
        "U": U_comb.tolist(),
        "V": V_comb.tolist(),
        "W": W_comb.tolist()
    }

if __name__ == "__main__":
    construct()
