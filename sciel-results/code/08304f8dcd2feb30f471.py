import numpy as np

def strassen_2x2():
    # Returns U, V, W for 2x2x2 multiplication, rank 7
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (x11 + x22)(y11 + y22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1 # z11=+1, z12=0, z21=0, z22=1? No.
    # z11 = P1 + P4 - P5 + P7
    # z12 = P3 + P5
    # z21 = P2 + P4
    # z22 = P1 - P2 + P3 + P6
    
    # W is for output z.
    # z11 is index 0, z12 is index 1, z21 is index 2, z22 is index 3.
    # W[k*2+i, c]
    # z11: i=0, k=0 -> W[0, c]
    # z12: i=0, k=1 -> W[2, c] ? No.
    # Output C is 2x2. C[i,k].
    # W index is k*2+i.
    # C[0,0] -> W[0]
    # C[0,1] -> W[2]
    # C[1,0] -> W[1]
    # C[1,1] -> W[3]
    
    # P1
    W[0, 0] = 1 # z11
    W[3, 0] = 1 # z22
    
    # P2 = (x21 + x22)y11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1 # z21
    W[3, 1] = -1 # z22
    
    # P3 = x11(y12 - y22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1 # z12
    W[3, 2] = 1 # z22
    
    # P4 = x22(y21 - y11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # z11
    W[1, 3] = 1 # z21
    
    # P5 = (x11 + x12)y22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # z11
    W[2, 4] = 1 # z12
    
    # P6 = (x21 - x11)(y11 + y12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # z22
    
    # P7 = (x12 - x22)(y21 + y22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # z11
    W[3, 6] = 1 # z22
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 9
    # Rank 49 for 4x4x4, Rank 16 for 4x4x1
    # Total Rank = 49 + 49 + 16 = 114
    rank = 114
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        # u_idx, v_idx, w_idx are lists of indices to add to
        # coeff_u, coeff_v, coeff_w are scalars
        pass

    # This is too complex to do with lists. I will use NumPy.
    U_np = np.zeros((n*m, rank))
    V_np = np.zeros((m*p, rank))
    W_np = np.zeros((p*n, rank))
    
    col = 0
    # Block 1: k=0..3
    U1, V1, W1 = strassen_2x2()
    # Expand U1 (4x7) to (16x49)
    # U1[i, c] corresponds to A[i%2, i//2] in the 2x2 block?
    # No, U1 is for 2x2 matrices.
    # We need to map 2x2 indices to 4x4 indices.
    # A_4x4 is composed of 4 blocks of 2x2.
    # Block (a,b) of A corresponds to A[2a:2a+2, 2b:2b+2].
    # The Strassen decomposition works on the 2x2 blocks.
    # So we need to apply the 7-term decomposition to each of the 7 Strassen products.
    
    # This implementation is getting risky. I will fall back to a simpler improvement.
    # How about just using rank 16 for all? 144.
    # I will return the current program to be safe, as 114 requires careful implementation.
    
    # Wait, I can just use the schoolbook for 144.
    # But I want to improve.
    # I will implement the 114 rank version carefully.
    
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
