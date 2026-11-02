import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Construct the schoolbook solution (Rank = n*m*p)
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        W = np.zeros((p * n, R), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R
    
    # 2. Construct a hybrid Strassen/Schoolbook solution
    # Split inner dimension m=3 into 2+1
    # A (2x3) = [A_left (2x2) | A_right (2x1)]
    # B (3x7) = [B_top (2x7); B_bottom (1x7)]
    # C = A_left * B_top + A_right * B_bottom
    #
    # For A_left * B_top (2x2 * 2x7):
    #   Split B_top columns into pairs: cols 0-1, 2-3, 4-5, and col 6
    #   Use Strassen for the three 2x2x2 blocks (rank 7 each)
    #   Use schoolbook for the 2x2x1 block (rank 2)
    #
    # For A_right * B_bottom (2x1 * 1x7):
    #   This is a rank-2 operation (outer product structure)
    #
    # Total rank = 7 + 7 + 7 + 2 + 2 = 25
    
    # Rank-7 Strassen decomposition for 2x2x2 tensor
    # The tensor T[i,j,k] = 1 for all i,j,k in {0,1}
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Strassen's 7 products:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    
    # For the tensor decomposition, we need to express each output c[i,k] as sum of products
    # The rank-1 terms correspond to the 7 Strassen products
    
    # Let's define the U, V, W matrices for the 2x2x2 case:
    # U[i,j,r], V[j,k,r], W[k,i,r] for r in {0,...,6}
    
    # Term 0 (m1 = (a00+a11)*(b00+b11)):
    # U[0,0,0]=1, U[1,1,0]=1
    # V[0,0,0]=1, V[1,1,0]=1
    # W[0,0,0]=1, W[1,1,0]=1
    # This gives: T[0,0,0] += 1*1*1=1, T[0,0,1] += 1*1*0=0, T[0,1,0] += 0*0*0=0, T[0,1,1] += 0*1*0=0
    #             T[1,0,0] += 0*1*0=0, T[1,0,1] += 0*1*0=0, T[1,1,0] += 1*0*1=0, T[1,1,1] += 1*1*1=1
    
    # This is not correct. Let me think more carefully.
    
    # Actually, the tensor decomposition for matrix multiplication is:
    # T[i,j,k] = 1 for all valid (i,j,k)
    # We want: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # For 2x2x2, the rank-7 decomposition is:
    # Each rank-1 term corresponds to one Strassen product
    
    # Let me use a known correct decomposition. From literature:
    
    # I'll construct it by verifying against the tensor.
    
    def strassen_2x2x2():
        """Return U, V, W matrices (each 4x7) for the 2x2x2 tensor decomposition."""
        # U[i,j,r] for i,j in {0,1}, r in {0,...,6}
        # V[j,k,r] for j,k in {0,1}, r in {0,...,6}
        # W[k,i,r] for k,i in {0,1}, r in {0,...,6}
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        
        # Term 0: m1 = (a00+a11)*(b00+b11)
        # U[0,0]=1, U[1,1]=1
        # V[0,0]=1, V[1,1]=1
        # W[0,0]=1, W[1,1]=1
        U[0, 0] = 1; U[3, 0] = 1  # i=0,j=0 and i=1,j=1
        V[0, 0] = 1; V[3, 0] = 1  # j=0,k=0 and j=1,k=1
        W[0, 0] = 1; W[3, 0] = 1  # k=0,i=0 and k=1,i=1
        
        # Term 1: m2 = (a10+a11)*b00
        # U[1,0]=1, U[1,1]=1
        # V[0,0]=1
        # W[0,1]=1
        U[2, 1] = 1; U[3, 1] = 1  # i=1,j=0 and i=1,j=1
        V[0, 1] = 1  # j=0,k=0
        W[2, 1] = 1  # k=0,i=1
        
        # Term 2: m3 = a00*(b01-b11)
        # U[0,0]=1
        # V[0,1]=1, V[1,1]=-1
        # W[1,0]=1
        U[0, 2] = 1  # i=0,j=0
        V[1, 2] = 1; V[3, 2] = -1  # j=0,k=1 and j=1,k=1
        W[2, 2] = 1  # k=1,i=0
        
        # Term 3: m4 = a11*(b10-b00)
        # U[1,1]=1
        # V[1,0]=1, V[0,0]=-1
        # W[0,1]=1
        U[3, 3] = 1  # i=1,j=1
        V[2, 3] = 1; V[0, 3] = -1  # j=1,k=0 and j=0,k=0
        W[2, 3] = 1  # k=0,i=1
        
        # Term 4: m5 = (a00+a01)*b11
        # U[0,0]=1, U[0,1]=1
        # V[1,1]=1
        # W[1,0]=-1
        U[0, 4] = 1; U[1, 4] = 1  # i=0,j=0 and i=0,j=1
        V[3, 4] = 1  # j=1,k=1
        W[2, 4] = -1  # k=1,i=0
        
        # Term 5: m6 = (a10-a00)*(b00+b01)
        # U[1,0]=1, U[0,0]=-1
        # V[0,0]=1, V[0,1]=1
        # W[0,1]=1, W[1,1]=1
        U[2, 5] = 1; U[0, 5] = -1  # i=1,j=0 and i=0,j=0
        V[0, 5] = 1; V[1, 5] = 1  # j=0,k=0 and j=0,k=1
        W[2, 5] = 1; W[3, 5] = 1  # k=0,i=1 and k=1,i=1
        
        # Term 6: m7 = (a01-a11)*(b10+b11)
        # U[0,1]=1, U[1,1]=-1
        # V[1,0]=1, V[1,1]=1
        # W[0,0]=1, W[1,0]=1
        U[1, 6] = 1; U[3, 6] = -1  # i=0,j=1 and i=1,j=1
        V[2, 6] = 1; V[3, 6] = 1  # j=1,k=0 and j=1,k=1
        W[0, 6] = 1; W[2, 6] = 1  # k=0,i=0 and k=1,i=0
        
        return U, V, W
    
    def verify_strassen():
        """Verify the Strassen decomposition is correct."""
        U, V, W = strassen_2x2x2()
        # Check T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                    if abs(s - 1) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    # 3. Construct the hybrid decomposition
    def construct_hybrid():
        """Construct a hybrid Strassen/Schoolbook decomposition for 2x3x7."""
        # n=2, m=3, p=7
        # U: (6, R), V: (21, R), W: (14, R)
        
        # Split: A = [A12 | A2] where A12 is 2x2 (rows 0-1, cols 0-1), A2 is 2x1 (rows 0-1, col 2)
        # B = [B1; B2] where B1 is 2x7 (rows 0-1, all cols), B2 is 1x7 (row 2, all cols)
        
        # C = A12 * B1 + A2 * B2
        
        # For A12 * B1: B1 has 7 columns, split into [B1a | B1b | B1c | B1d]
        # where B1a = cols 0-1 (2x2), B1b = cols 2-3 (2x2), B1c = cols 4-5 (2x2), B1d = col 6 (2x1)
        
        # Use Strassen (rank 7) for each 2x2x2 block
        # Use schoolbook (rank 2) for the 2x2x1 block
        
        # For A2 * B2: This is (2x1) * (1x7) = 2x7
        # Rank 2: for each i, C[i,k] = A2[i,0] * B2[0,k]
        
        # Total rank = 7*3 + 2 + 2 = 25
        
        R = 25
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        
        col = 0
        
        # Get the Strassen decomposition for 2x2x2
        U_s, V_s, W_s = strassen_2x2x2()
        
        # Add Strassen blocks for B1a (cols 0-1), B1b (cols 2-3), B1c (cols 4-5)
        for block in range(3):
            k_start = block * 2
            for r in range(7):
                # U: A12[i, j] for i in {0,1}, j in {0,1}
                # Global U index: i*m + j = i*3 + j
                for i in range(2):
                    for j in range(2):
                        U[i*3 + j, col] = U_s[i*2+j, r]
                
                # V: B1[j, k] for j in {0,1}, k in {k_start, k_start+1}
                # Global V index: j*p + k = j*7 + k
                for j in range(2):
                    for k in range(2):
                        V[j*7 + (k_start+k), col] = V_s[j*2+k, r]
                
                # W: C[i, k] for i in {0,1}, k in {k_start, k_start+1}
                # Global W index: k*n + i = k*2 + i
                for k in range(2):
                    for i in range(2):
                        W[(k_start+k)*2 + i, col] = W_s[k*2+i, r]
                
                col += 1
        
        # Add schoolbook for B1d (col 6): 2x2x1 block
        # C[i, 6] = A12[i,0]*B1[0,6] + A12[i,1]*B1[1,6]
        for j in range(2):
            U[j, col] = 1  # A12[i,j] -> U[i*3+j, col] = 1 for all i
            # Actually, U[i*3+j, col] = 1 for all i
            for i in range(2):
                U[i*3 + j, col] = 1
            V[j*7 + 6, col] = 1  # B1[j, 6]
            W[6*2, col] = 1  # C[0, 6]
            W[6*2 + 1, col] = 1  # C[1, 6]
            col += 1
        
        # Add A2 * B2: (2x1) * (1x7)
        # T[i, 2, k] = 1 for i in {0,1}, k in {0,...,6}
        # Rank 2:
        # r=col: U[0*3+2, col]=1, V[2*7+k, col]=1 for all k, W[k*2+0, col]=1 for all k
        # r=col+1: U[1*3+2, col+1]=1, V[2*7+k, col+1]=1 for all k, W[k*2+1, col+1]=1 for all k
        
        # For i=0:
        U[0*3 + 2, col] = 1
        for k in range(7):
            V[2*7 + k, col] = 1
            W[k*2 + 0, col] = 1
        col += 1
        
        # For i=1:
        U[1*3 + 2, col] = 1
        for k in range(7):
            V[2*7 + k, col] = 1
            W[k*2 + 1, col] = 1
        col += 1
        
        return U, V, W, R
    
    # Verify the solution
    def verify(U, V, W, R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # First, try the hybrid decomposition
    try:
        hybrid_U, hybrid_V, hybrid_W, hybrid_R = construct_hybrid()
        if verify(hybrid_U, hybrid_V, hybrid_W, hybrid_R):
            best_U, best_V, best_W, best_R = hybrid_U, hybrid_V, hybrid_W, hybrid_R
        else:
            # Fallback to schoolbook
            best_U, best_V, best_W, best_R = schoolbook()
    except Exception:
        # Fallback to schoolbook
        best_U, best_V, best_W, best_R = schoolbook()
    
    # Verify the final solution
    if not verify(best_U, best_V, best_W, best_R):
        # Fallback: construct schoolbook again
        best_U, best_V, best_W, best_R = schoolbook()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
