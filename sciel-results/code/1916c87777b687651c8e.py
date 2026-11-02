"""Matrix multiplication 4x5x5 rank-minimal decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 5, 5]

def _strassen_decomp(n):
    """
    Return Strassen decomposition for n x n matrix multiplication.
    For n=4, this gives rank 49.
    Returns U (n*n, rank), V (n*n, rank), W (n*n, rank) as numpy arrays.
    """
    if n == 1:
        rank = 1
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W
    if n == 2:
        # Strassen for 2x2: rank 7
        # U indices: (i,j) flattened as i*2+j
        # V indices: (j,k) flattened as j*2+k  
        # W indices: (k,i) flattened as k*2+i
        # The 7 Strassen multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Tensor decomposition: T[i,j,k] = sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r]
        # We need T[i,j,k] = 1 for all i,j,k (the matmul tensor for 2x2x2)
        # Actually, the matmul tensor has T[i,j,k] = 1 for all valid i,j,k.
        # With Strassen, we have rank 7, so we need 7 vectors U_r, V_r, W_r such that
        # sum_r U_r[i*2+j] * V_r[j*2+k] * W_r[k*2+i] = 1 for all i,j,k
        
        # Let me construct this explicitly.
        # U is 4x7, V is 4x7, W is 4x7
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # r=0: M1 = (A00+A11)(B00+B11)
        # U contribution: A00+A11 -> U[0,0]=1, U[3,0]=1 (indices 0*2+0=0, 1*2+1=3)
        # V contribution: B00+B11 -> V[0,0]=1, V[3,0]=1 (indices 0*2+0=0, 1*2+1=3)
        # W contribution: C00 gets +1, C11 gets +1 -> W[0,0]=1 (k=0,i=0 -> 0*2+0=0), W[3,0]=1 (k=1,i=1 -> 1*2+1=3)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # r=1: M2 = (A10+A11)B00
        # U: A10+A11 -> U[2,1]=1 (1*2+0=2), U[3,1]=1 (1*2+1=3)
        # V: B00 -> V[0,1]=1
        # W: C10 gets +1 -> W[2,1]=1 (k=0,i=1 -> 0*2+1=1? No, k*2+i, k=0,i=1 -> 1)
        # Wait, W index is k*n+i = k*2+i. C10 is i=1,k=0, so W[0*2+1, 1] = W[1,1]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # r=2: M3 = A00(B01-B11)
        # U: A00 -> U[0,2]=1
        # V: B01-B11 -> V[1,2]=1 (0*2+1=1), V[3,2]=-1 (1*2+1=3)
        # W: C01 gets +1 -> W[2,2]=1 (k=1,i=0 -> 1*2+0=2)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # r=3: M4 = A11(B10-B00)
        # U: A11 -> U[3,3]=1
        # V: B10-B00 -> V[2,3]=1 (1*2+0=2), V[0,3]=-1 (0*2+0=0)
        # W: C10 gets +1, C00 gets +1 -> W[1,3]=1 (k=0,i=1 -> 1), W[0,3]=1 (k=0,i=0 -> 0)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1; W[0, 3] = 1
        
        # r=4: M5 = (A00+A01)B11
        # U: A00+A01 -> U[0,4]=1, U[1,4]=1 (0*2+0=0, 0*2+1=1)
        # V: B11 -> V[3,4]=1 (1*2+1=3)
        # W: C00 gets -1, C01 gets +1 -> W[0,4]=-1, W[2,4]=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # r=5: M6 = (A10-A00)(B00+B01)
        # U: A10-A00 -> U[2,5]=1 (1*2+0=2), U[0,5]=-1 (0*2+0=0)
        # V: B00+B01 -> V[0,5]=1, V[1,5]=1 (0*2+0=0, 0*2+1=1)
        # W: C11 gets +1 -> W[3,5]=1 (k=1,i=1 -> 1*2+1=3)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # r=6: M7 = (A01-A11)(B10+B11)
        # U: A01-A11 -> U[1,6]=1 (0*2+1=1), U[3,6]=-1 (1*2+1=3)
        # V: B10+B11 -> V[2,6]=1 (1*2+0=2), V[3,6]=1 (1*2+1=3)
        # W: C00 gets +1 -> W[0,6]=1 (k=0,i=0 -> 0)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    if n == 4:
        # Use Strassen for 4x4: partition into 2x2 blocks and apply Strassen recursively
        # Actually, for 4x4, we can use the 2x2 Strassen on each 2x2 block
        # The 4x4 matmul tensor can be decomposed using the 2x2x2 Strassen decomposition
        # by treating the 4x4 as 2x2 blocks of 2x2 matrices
        # This gives rank 7 * 7 = 49
        
        # For 4x4, indices: U is (16, rank), V is (16, rank), W is (16, rank)
        # U index: i*4+j, V index: j*4+k, W index: k*4+i
        
        # We can compose the 2x2 Strassen decomposition.
        # Think of the 4x4 as 2x2 blocks. Let A be partitioned into A00, A01, A10, A11
        # where each is 2x2. Similarly B. Then C = AB is computed by 7 Strassen multiplications
        # of 2x2 matrices, each of which uses 7 Strassen multiplications of scalars.
        # Total: 7*7 = 49 scalar multiplications.
        
        # Let me construct this.
        U2, V2, W2 = _strassen_decomp(2)
        # U2 is (4, 7), V2 is (4, 7), W2 is (4, 7)
        # U2 index: i*2+j for 2x2
        # For 4x4, we need to handle the block structure
        
        # The 4x4 matrix A is partitioned as:
        # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
        # A00 covers rows 0-1, cols 0-1
        # A01 covers rows 0-1, cols 2-3
        # A10 covers rows 2-3, cols 0-1
        # A11 covers rows 2-3, cols 2-3
        
        # For the block Strassen, we have 7 block multiplications M0..M6
        # Each M_r is a 2x2 matmul, which itself has rank 7 via Strassen
        # So total rank = 7 * 7 = 49
        
        # U for 4x4: index i*4+j where i,j in 0..3
        # For block r (0..6) and sub-r (0..6):
        # The U contribution comes from the U of the block Strassen for the A-factor
        # and the U of the sub-Strassen for the 2x2 matmul
        
        # Block Strassen U factors (for A):
        # M0: (A00+A11) -> U_block[0,0]=1, U_block[3,3]=1 (in 2x2 block indices)
        # M1: (A10+A11) -> U_block[2,2]=1, U_block[3,3]=1
        # M2: A00 -> U_block[0,0]=1
        # M3: A11 -> U_block[3,3]=1
        # M4: (A00+A01) -> U_block[0,0]=1, U_block[1,1]=1
        # M5: (A10-A00) -> U_block[2,2]=1, U_block[0,0]=-1
        # M6: (A01-A11) -> U_block[1,1]=1, U_block[3,3]=-1
        
        # Similarly for V (B-factor) and W (C-factor)
        
        # Let me define the block-level factors
        # U_block: 4x7, where index is (block_row, block_col) flattened as block_row*2+block_col
        # For A:
        U_block = np.zeros((4, 7))
        U_block[0, 0] = 1; U_block[3, 0] = 1  # M0: A00+A11
        U_block[2, 1] = 1; U_block[3, 1] = 1  # M1: A10+A11
        U_block[0, 2] = 1                      # M2: A00
        U_block[3, 3] = 1                      # M3: A11
        U_block[0, 4] = 1; U_block[1, 4] = 1  # M4: A00+A01
        U_block[2, 5] = 1; U_block[0, 5] = -1 # M5: A10-A00
        U_block[1, 6] = 1; U_block[3, 6] = -1 # M6: A01-A11
        
        # For B:
        V_block = np.zeros((4, 7))
        V_block[0, 0] = 1; V_block[3, 0] = 1  # M0: B00+B11
        V_block[0, 1] = 1                      # M1: B00
        V_block[1, 2] = 1; V_block[3, 2] = -1 # M2: B01-B11
        V_block[2, 3] = 1; V_block[0, 3] = -1 # M3: B10-B00
        V_block[3, 4] = 1                      # M4: B11
        V_block[0, 5] = 1; V_block[1, 5] = 1  # M5: B00+B01
        V_block[2, 6] = 1; V_block[3, 6] = 1  # M6: B10+B11
        
        # For C:
        W_block = np.zeros((4, 7))
        W_block[0, 0] = 1; W_block[3, 0] = 1  # M0: C00+C11
        W_block[2, 1] = 1                      # M1: C10
        W_block[1, 2] = 1                      # M2: C01
        W_block[2, 3] = 1; W_block[0, 3] = 1  # M3: C10+C00
        W_block[0, 4] = -1; W_block[1, 4] = 1 # M4: C00-C01... wait
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        W_block[0, 0] = 1; W_block[3, 0] = 1  # M0: C00+C11
        W_block[2, 1] = 1                      # M1: C10
        W_block[1, 2] = 1                      # M2: C01
        W_block[2, 3] = 1; W_block[0, 3] = 1  # M3: C10+C00
        W_block[0, 4] = -1; W_block[1, 4] = 1 # M4: -C00+C01
        W_block[3, 5] = 1                      # M5: C11
        W_block[0, 6] = 1                      # M6: C00
        
        # Now for each block multiplication r (0..6) and sub-multiplication s (0..6):
        # U_4x4[i*4+j, r*7+s] = U_block[i//2*2+j//2, r] * U2[(i%2)*2+(j%2), s]
        # Wait, that's not quite right. Let me think more carefully.
        
        # For the 4x4 matrix, element (i,j) where i,j in 0..3:
        # Block row = i//2, block col = j//2
        # Within-block row = i%2, within-block col = j%2
        
        # U_4x4[i*4+j, r*7+s] = U_block[(i//2)*2 + (j//2), r] * U2[(i%2)*2 + (j%2), s]
        
        U_4x4 = np.zeros((16, 49))
        V_4x4 = np.zeros((16, 49))
        W_4x4 = np.zeros((16, 49))
        
        for i in range(4):
            for j in range(4):
                br = i // 2
                bc = j // 2
                wr = i % 2
                wc = j % 2
                for r in range(7):
                    for s in range(7):
                        U_4x4[i*4+j, r*7+s] = U_block[br*2+bc, r] * U2[wr*2+wc, s]
        
        for j in range(4):
            for k in range(4):
                br = j // 2
                bc = k // 2
                wr = j % 2
                wc = k % 2
                for r in range(7):
                    for s in range(7):
                        V_4x4[j*4+k, r*7+s] = V_block[br*2+bc, r] * V2[wr*2+wc, s]
        
        for k in range(4):
            for i in range(4):
                br = k // 2
                bc = i // 2
                wr = k % 2
                wc = i % 2
                for r in range(7):
                    for s in range(7):
                        W_4x4[k*4+i, r*7+s] = W_block[br*2+bc, r] * W2[wr*2+wc, s]
        
        return U_4x4, V_4x4, W_4x4
    
    raise ValueError(f"Unsupported n={n}")

def construct():
    """
    Construct a tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Uses Strassen for the 4x4x4 part (rank 49) and schoolbook for the 4x1x5 part (rank 5).
    Total rank: 49 + 5 = 54.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get Strassen decomposition for 4x4x4
    U_44, V_44, W_44 = _strassen_decomp(4)
    rank_44 = U_44.shape[1]  # Should be 49
    
    # Total rank: 49 (for 4x4x4) + 5 (for 4x1x5) = 54
    total_rank = rank_44 + 5
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Fill the 4x4x4 part (j in 0..3, k in 0..3, i in 0..3)
    # U_44 index: i*4+j, global U index: i*5+j
    for i in range(4):
        for j in range(4):
            sub_idx = i * 4 + j
            global_idx = i * 5 + j
            U[global_idx, :rank_44] = U_44[sub_idx, :rank_44]
    
    # V_44 index: j*4+k, global V index: j*5+k
    for j in range(4):
        for k in range(4):
            sub_idx = j * 4 + k
            global_idx = j * 5 + k
            V[global_idx, :rank_44] = V_44[sub_idx, :rank_44]
    
    # W_44 index: k*4+i, global W index: k*4+i (since n=4, p*n=20, k in 0..3, i in 0..3)
    for k in range(4):
        for i in range(4):
            sub_idx = k * 4 + i
            global_idx = k * 4 + i
            W[global_idx, :rank_44] = W_44[sub_idx, :rank_44]
    
    # Fill the 4x1x5 part (j=4)
    # This handles the column j=4 of A and row j=4 of B
    # U: i*5+4 for i in 0..3
    # V: 4*5+k = 20+k for k in 0..4
    # W: k*4+i for k in 0..4, i in 0..3
    
    # We need 5 additional columns (one for each k)
    for k in range(5):
        col = rank_44 + k
        # U: i*5+4
        for i in range(4):
            U[i * 5 + 4, col] = 1.0
        # V: 20+k
        V[20 + k, col] = 1.0
        # W: k*4+i
        for i in range(4):
            W[k * 4 + i, col] = 1.0
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    # Verify
    n, m, p = 4, 5, 5
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    rank = U.shape[1]
    
    # Check: for all i,j,k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
