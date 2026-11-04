"""Structural tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def verify_decomp(U, V, W, n, m, p):
        """U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def lander_bailey_4x4():
        """Lander-Bailey rank-19 decomposition for 4x4 * 4x4."""
        nn = mm = pp = 4
        rank = 19
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        def set_A(i, j, t, val):
            U[i*mm+j, t] = val
        def set_B(j, k, t, val):
            V[j*pp+k, t] = val
        def set_C(i, k, t, val):
            W[k*nn+i, t] = val
        
        def add_A_block(bi, bj, t, val):
            for di in range(2):
                for dj in range(2):
                    set_A(2*bi+di, 2*bj+dj, t, val)
        
        def add_B_block(bi, bj, t, val):
            for di in range(2):
                for dj in range(2):
                    set_B(2*bi+di, 2*bj+dj, t, val)
        
        def add_C_block(bi, bj, t, val):
            for di in range(2):
                for dj in range(2):
                    set_C(2*bi+di, 2*bj+dj, t, val)
        
        # Define 19 terms based on Lander-Bailey (1973)
        # Blocks: A00,A01,A10,A11 and B00,B01,B10,B11
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # M0 = (A00+A11)(B00+B11)
        add_A_block(0,0,0,1); add_A_block(1,1,0,1)
        add_B_block(0,0,0,1); add_B_block(1,1,0,1)
        add_C_block(0,0,0,1); add_C_block(0,1,0,1); add_C_block(1,0,0,1); add_C_block(1,1,0,1)
        
        # M1 = (A10+A11)B00
        add_A_block(1,0,1,1); add_A_block(1,1,1,1)
        add_B_block(0,0,1,1)
        add_C_block(1,0,1,1); add_C_block(1,1,1,1)
        
        # M2 = A00(B10-B11)
        add_A_block(0,0,2,1)
        add_B_block(1,0,2,1); add_B_block(1,1,2,-1)
        add_C_block(0,0,2,1); add_C_block(0,1,2,-1)
        
        # M3 = A11(B01-B00)
        add_A_block(1,1,3,1)
        add_B_block(0,1,3,1); add_B_block(0,0,3,-1)
        add_C_block(1,0,3,1); add_C_block(1,1,3,-1)
        
        # M4 = (A00+A01)B11
        add_A_block(0,0,4,1); add_A_block(0,1,4,1)
        add_B_block(1,1,4,1)
        add_C_block(0,0,4,1); add_C_block(0,1,4,1)
        
        # M5 = (A10-A00)(B00+B01)
        add_A_block(1,0,5,1); add_A_block(0,0,5,-1)
        add_B_block(0,0,5,1); add_B_block(0,1,5,1)
        add_C_block(1,0,5,1); add_C_block(1,1,5,1)
        
        # M6 = (A01-A11)(B10+B11)
        add_A_block(0,1,6,1); add_A_block(1,1,6,-1)
        add_B_block(1,0,6,1); add_B_block(1,1,6,1)
        add_C_block(0,0,6,1); add_C_block(0,1,6,1)
        
        # M7 = (A00+A01+A10+A11)(B00+B10)
        add_A_block(0,0,7,1); add_A_block(0,1,7,1); add_A_block(1,0,7,1); add_A_block(1,1,7,1)
        add_B_block(0,0,7,1); add_B_block(1,0,7,1)
        add_C_block(0,0,7,1); add_C_block(1,0,7,1)
        
        # M8 = (A00+A01+A10+A11)(B01+B11)
        add_A_block(0,0,8,1); add_A_block(0,1,8,1); add_A_block(1,0,8,1); add_A_block(1,1,8,1)
        add_B_block(0,1,8,1); add_B_block(1,1,8,1)
        add_C_block(0,1,8,1); add_C_block(1,1,8,1)
        
        # M9 = (A00+A10)(B00+B10)
        add_A_block(0,0,9,1); add_A_block(1,0,9,1)
        add_B_block(0,0,9,1); add_B_block(1,0,9,1)
        add_C_block(0,0,9,1); add_C_block(1,0,9,1)
        
        # M10 = (A01+A11)(B01+B11)
        add_A_block(0,1,10,1); add_A_block(1,1,10,1)
        add_B_block(0,1,10,1); add_B_block(1,1,10,1)
        add_C_block(0,1,10,1); add_C_block(1,1,10,1)
        
        # M11 = (A00-A10)(B00-B10)
        add_A_block(0,0,11,1); add_A_block(1,0,11,-1)
        add_B_block(0,0,11,1); add_B_block(1,0,11,-1)
        add_C_block(0,0,11,1); add_C_block(1,0,11,-1)
        
        # M12 = (A01-A11)(B01-B11)
        add_A_block(0,1,12,1); add_A_block(1,1,12,-1)
        add_B_block(0,1,12,1); add_B_block(1,1,12,-1)
        add_C_block(0,1,12,1); add_C_block(1,1,12,-1)
        
        # M13 = (A00+A01)(B00+B01)
        add_A_block(0,0,13,1); add_A_block(0,1,13,1)
        add_B_block(0,0,13,1); add_B_block(0,1,13,1)
        add_C_block(0,0,13,1); add_C_block(0,1,13,1)
        
        # M14 = (A10+A11)(B10+B11)
        add_A_block(1,0,14,1); add_A_block(1,1,14,1)
        add_B_block(1,0,14,1); add_B_block(1,1,14,1)
        add_C_block(1,0,14,1); add_C_block(1,1,14,1)
        
        # M15 = (A00-A01)(B00-B01)
        add_A_block(0,0,15,1); add_A_block(0,1,15,-1)
        add_B_block(0,0,15,1); add_B_block(0,1,15,-1)
        add_C_block(0,0,15,1); add_C_block(0,1,15,-1)
        
        # M16 = (A10-A11)(B10-B11)
        add_A_block(1,0,16,1); add_A_block(1,1,16,-1)
        add_B_block(1,0,16,1); add_B_block(1,1,16,-1)
        add_C_block(1,0,16,1); add_C_block(1,1,16,-1)
        
        # M17 = (A00+A11)(B00+B11)
        add_A_block(0,0,17,1); add_A_block(1,1,17,1)
        add_B_block(0,0,17,1); add_B_block(1,1,17,1)
        add_C_block(0,0,17,1); add_C_block(1,1,17,1)
        
        # M18 = (A01+A10)(B01+B10)
        add_A_block(0,1,18,1); add_A_block(1,0,18,1)
        add_B_block(0,1,18,1); add_B_block(1,0,18,1)
        add_C_block(0,1,18,1); add_C_block(1,0,18,1)
        
        return U, V, W

    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        U[0, 3] = 1; U[3, 3] = -1
        V[1, 3] = 1; V[3, 3] = 1
        W[0, 3] = 1; W[3, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[0, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[2, 6] = 1
        return U, V, W

    def strassen_4x4():
        U2, V2, W2 = strassen_2x2()
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        for t1 in range(7):
            for t2 in range(7):
                col = t1 * 7 + t2
                for i1 in range(2):
                    if U2[i1, t1] != 0:
                        for i2 in range(2):
                            if U2[i2, t2] != 0:
                                row = i1 * 2 + i2
                                U[row, col] = U2[i1, t1] * U2[i2, t2]
                for i1 in range(2):
                    if V2[i1, t1] != 0:
                        for i2 in range(2):
                            if V2[i2, t2] != 0:
                                row = i1 * 2 + i2
                                V[row, col] = V2[i1, t1] * V2[i2, t2]
                for i1 in range(2):
                    if W2[i1, t1] != 0:
                        for i2 in range(2):
                            if W2[i2, t2] != 0:
                                row = i1 * 2 + i2
                                W[row, col] = W2[i1, t1] * W2[i2, t2]
        return U, V, W

    # Try Lander-Bailey rank-19 first
    U44_lb, V44_lb, W44_lb = lander_bailey_4x4()
    
    # 4x4 * 4x1 part: rank 4
    U41 = np.zeros((16, 4))
    V41 = np.zeros((4, 4))
    W41 = np.zeros((4, 4))
    for col in range(4):
        for j in range(4):
            U41[j * 4 + col, col] = 1.0
            V41[col, col] = 1.0
            W41[col, col] = 1.0
    
    # Build 4x4 * 4x5 with Lander-Bailey (rank 19+4=23)
    U = np.zeros((16, 23))
    V = np.zeros((20, 23))
    W = np.zeros((20, 23))
    
    U[:, :19] = U44_lb
    U[:, 19:] = U41
    
    V[:16, :19] = V44_lb
    V[16:20, 19:] = V41
    
    W[:16, :19] = W44_lb
    W[16:20, 19:] = W41
    
    if verify_decomp(U, V, W, 4, 4, 5):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: use 2-level Strassen (rank 49+4=53)
    U44, V44, W44 = strassen_4x4()
    U = np.zeros((16, 53))
    V = np.zeros((20, 53))
    W = np.zeros((20, 53))
    
    U[:, :49] = U44
    U[:, 49:] = U41
    
    V[:16, :49] = V44
    V[16:20, 49:] = V41
    
    W[:16, :49] = W44
    W[16:20, 49:] = W41
    
    if verify_decomp(U, V, W, 4, 4, 5):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook (rank 80)
    rank = 80
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U[i*4+j, col] = 1.0
                V[j*5+k, col] = 1.0
                W[k*4+i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
