"""Matrix multiplication 4,4,4 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal (or near-minimal) exact tensor decomposition for 4x4 matrix multiplication.
    We use the Strassen-based decomposition of rank 49 (7x7).
    """
    n, m, p = 4, 4, 4
    
    # 2x2 Strassen coefficients
    # U2, V2, W2 are 4x7 matrices.
    # Rows of U2: i*2+j for i,j in 0..1
    # Rows of V2: j*2+k for j,k in 0..1
    # Rows of W2: k*2+i for k,i in 0..1
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    rank = 49
    U4 = np.zeros((16, rank), dtype=int)
    V4 = np.zeros((16, rank), dtype=int)
    W4 = np.zeros((16, rank), dtype=int)
    
    # Construct 4x4 factors using tensor product
    # i = 2*i1 + i0
    # j = 2*j1 + j0
    # k = 2*k1 + k0
    # r = 7*r1 + r0
    
    for i in range(4):
        i1 = i // 2
        i0 = i % 2
        for j in range(4):
            j1 = j // 2
            j0 = j % 2
            row_u = i * 4 + j
            
            for k in range(4):
                k1 = k // 2
                k0 = k % 2
                row_v = j * 4 + k
                row_w = k * 4 + i
                
                for r1 in range(7):
                    for r0 in range(7):
                        r = r1 * 7 + r0
                        
                        u_val = U2[i1*2+j1, r1] * U2[i0*2+j0, r0]
                        v_val = V2[j1*2+k1, r1] * V2[j0*2+k0, r0]
                        w_val = W2[k1*2+i1, r1] * W2[k0*2+i0, r0]
                        
                        U4[row_u, r] = u_val
                        V4[row_v, r] = v_val
                        W4[row_w, r] = w_val
                        
    # Optional: Verify for safety, but this construction is known to be correct.
    # If verification fails, fall back to rank 64.
    valid = True
    # Sample verification to save time? No, 4*4*4*49 = 3136 ops, very fast.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for r in range(rank):
                    s += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break

    if not valid:
        # Fallback to rank-64 trivial decomposition
        rank = 64
        U4 = np.zeros((16, rank), dtype=int)
        V4 = np.zeros((16, rank), dtype=int)
        W4 = np.zeros((16, rank), dtype=int)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r = i * 16 + j * 4 + k
                    U4[i*4+j, r] = 1
                    V4[j*4+k, r] = 1
                    W4[k*4+i, r] = 1

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
# EVOLVE-BLOCK-END
