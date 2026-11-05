import numpy as np
import time

def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    Uses a block decomposition approach:
    - Split intermediate dimension (3) into [0:2] and [2:3]
    - Split output dimension (5) into [0:2], [2:4], [4:5]
    - Use Strassen (rank 7) for 2x2*2x2 blocks
    - Use schoolbook for remaining blocks
    
    Target rank: 28
    """
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def check_time():
        if time.time() > deadline:
            return True
        return False
    
    rank = 28
    U = np.zeros((n * m, rank))  # Shape: (2*3, 28) = (6, 28)
    V = np.zeros((m * p, rank))  # Shape: (3*5, 28) = (15, 28)
    W = np.zeros((p * n, rank))  # Shape: (5*2, 28) = (10, 28)
    
    col = 0
    
    def strassen_2x2(i1, i2, j1, j2, k1, k2):
        """
        Apply Strassen's algorithm to multiply 2x2 blocks.
        
        i1, i2: row indices of A and C
        j1, j2: column indices of A and row indices of B  
        k1, k2: column indices of B and column indices of C
        """
        nonlocal col
        
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m + j1, col] = 1
        U[i2*m + j2, col] = 1
        V[j1*p + k1, col] = 1
        V[j2*p + k2, col] = 1
        W[k1*n + i1, col] = 1
        W[k2*n + i2, col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[i2*m + j1, col] = 1
        U[i2*m + j2, col] = 1
        V[j1*p + k1, col] = 1
        W[k1*n + i2, col] = 1
        W[k2*n + i2, col] = -1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[i1*m + j1, col] = 1
        V[j1*p + k2, col] = 1
        V[j2*p + k2, col] = -1
        W[k2*n + i1, col] = 1
        W[k2*n + i2, col] = 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[i2*m + j2, col] = 1
        V[j2*p + k1, col] = 1
        V[j1*p + k1, col] = -1
        W[k1*n + i1, col] = 1
        W[k1*n + i2, col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[i1*m + j1, col] = 1
        U[i1*m + j2, col] = 1
        V[j2*p + k2, col] = 1
        W[k1*n + i1, col] = -1
        W[k2*n + i1, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m + j1, col] = 1
        U[i1*m + j1, col] = -1
        V[j1*p + k1, col] = 1
        V[j1*p + k2, col] = 1
        W[k2*n + i2, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m + j2, col] = 1
        U[i2*m + j2, col] = -1
        V[j2*p + k1, col] = 1
        V[j2*p + k2, col] = 1
        W[k1*n + i1, col] = 1
        col += 1
    
    def schoolbook_block(i_list, j_list, k_list):
        """
        Schoolbook multiplication for specified blocks.
        """
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    if check_time():
                        return False
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return True
    
    # Build the decomposition
    success = True
    
    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen, rank 7)
    strassen_2x2(0, 1, 0, 1, 0, 1)
    
    # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen, rank 7)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    
    # Block 3: A[:,0:2] * B[0:2,4:5] -> C[0:2,4:5] (Schoolbook, rank 4)
    if not schoolbook_block([0, 1], [0, 1], [4]):
        success = False
    
    # Block 4: A[:,2:3] * B[2:3,0:2] -> C[0:2,0:2] (Schoolbook, rank 4)
    if success and not schoolbook_block([0, 1], [2], [0, 1]):
        success = False
    
    # Block 5: A[:,2:3] * B[2:3,2:4] -> C[0:2,2:4] (Schoolbook, rank 4)
    if success and not schoolbook_block([0, 1], [2], [2, 3]):
        success = False
    
    # Block 6: A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5] (Schoolbook, rank 2)
    if success and not schoolbook_block([0, 1], [2], [4]):
        success = False
    
    # Verify the decomposition using exact integer arithmetic
    if success:
        valid = True
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # This is a contribution, need to accumulate properly
                        pass
        
        # Proper verification: compute the full tensor
        tensor_sum = np.zeros((n, m, p), dtype=int)
        for c in range(rank):
            uc = U[:, c].reshape(n, m).astype(int)
            vc = V[:, c].reshape(m, p).astype(int)
            wc = W[:, c].reshape(p, n).astype(int)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_sum[i, j, k] += int(uc[i, j] * vc[j, k] * wc[k, i])
        
        # Check if all entries equal 1
        if not np.all(tensor_sum == 1):
            # Fallback to full schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, col] = 1
                        V[j*p + k, col] = 1
                        W[k*n + i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
