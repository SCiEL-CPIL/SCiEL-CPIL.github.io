"""
Strategy:
The task is to decompose the tensor for C = A @ B where A is 2x2 and B is 2x13.
The schoolbook rank is 52.
We can improve this by grouping the 13 columns of B into 6 pairs and 1 singleton.
For each pair of columns (effectively a 2x2 matrix multiplication block), we can use the
rank-7 Strassen decomposition for 2x2 matrix multiplication.
For the remaining single column (2x1 multiplication), we use the rank-4 schoolbook decomposition.
Total rank = 6 * 7 + 1 * 4 = 42 + 4 = 46.
"""

import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # rank = 6 * 7 + 4 = 46
    total_rank = 46
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Indices in U: i*m + j (0..3)
    # Indices in V: j*p + k (0..25)
    # Indices in W: k*n + i (0..25) -> Note: k is column index, i is row index.
    
    col_offset = 0
    
    # Strassen coefficients for A, B, C
    # Z1 = (A00+A11)(B00+B11)
    # Z2 = (A10+A11)B00
    # Z3 = A00(B01-B11)
    # Z4 = A11(B10-B00)
    # Z5 = (A00+A01)B11
    # Z6 = (A10-A00)(B00+B01)
    # Z7 = (A01-A11)(B10+B11)
    
    # C00 = Z1 + Z4 - Z5 + Z7
    # C01 = Z3 + Z5
    # C10 = Z2 + Z4
    # C11 = Z1 - Z2 + Z3 + Z6
    
    # Map to indices
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B_block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block start
    # C_block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block start
    
    # U coefficients for each Strassen product
    # Z1: A00+A11 -> U[0]+=1, U[3]+=1
    # Z2: A10+A11 -> U[2]+=1, U[3]+=1
    # Z3: A00 -> U[0]+=1
    # Z4: A11 -> U[3]+=1
    # Z5: A00+A01 -> U[0]+=1, U[1]+=1
    # Z6: A10-A00 -> U[2]+=1, U[0]-=1
    # Z7: A01-A11 -> U[1]+=1, U[3]-=1
    
    U_strassen = [
        [1, 0, 0, 1], # Z1
        [0, 0, 1, 1], # Z2
        [1, 0, 0, 0], # Z3
        [0, 0, 0, 1], # Z4
        [1, 1, 0, 0], # Z5
        [-1, 0, 1, 0], # Z6
        [0, 1, 0, -1]  # Z7
    ]
    
    # V coefficients (B)
    # Z1: B00+B11
    # Z2: B00
    # Z3: B01-B11
    # Z4: B10-B00
    # Z5: B11
    # Z6: B00+B01
    # Z7: B10+B11
    
    # B indices in block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V_strassen = [
        [1, 0, 0, 1], # Z1
        [1, 0, 0, 0], # Z2
        [0, 1, 0, -1], # Z3
        [0, 0, 1, -1], # Z4  (Wait: B10-B00 -> [0,0,1,-1])
        [0, 0, 0, 1], # Z5
        [1, 1, 0, 0], # Z6
        [0, 0, 1, 1]  # Z7
    ]
    
    # W coefficients (C)
    # C00 = Z1 + Z4 - Z5 + Z7
    # C01 = Z3 + Z5
    # C10 = Z2 + Z4
    # C11 = Z1 - Z2 + Z3 + Z6
    
    # C indices in block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W row for C_ij corresponds to k*n+i? No, W is indexed by k*n+i.
    # C block cols: c0, c1.
    # C_00 is k=0, i=0 -> idx 0*n+0 = 0 (relative)
    # C_01 is k=1, i=0 -> idx 1*n+0 = 2? 
    # Wait. The global W index is k*n + i.
    # If block starts at column K:
    # C_0, K (i=0, k=K) -> idx K*2 + 0 = 2K
    # C_1, K (i=1, k=K) -> idx K*2 + 1 = 2K+1
    # C_0, K+1 (i=0, k=K+1) -> idx (K+1)*2 + 0 = 2K+2
    # C_1, K+1 (i=1, k=K+1) -> idx (K+1)*2 + 1 = 2K+3
    
    # So relative W indices: 0, 1, 2, 3.
    
    W_strassen = [
        [1, 0, 0, 0], # Z1 -> C00
        [0, 0, 1, 0], # Z4 -> C10 ? No.
        # Let's build W columns from C formulas
        # col 0 (Z1): C00+=1, C11+=1 -> W[0]+=1, W[3]+=1
        # col 1 (Z2): C10+=1, C11-=1 -> W[2]+=1, W[3]-=1
        # col 2 (Z3): C01+=1, C11+=1 -> W[1]+=1, W[3]+=1
        # col 3 (Z4): C00+=1, C10+=1 -> W[0]+=1, W[2]+=1
        # col 4 (Z5): C00-=1, C01+=1 -> W[0]-=1, W[1]+=1
        # col 5 (Z6): C11+=1 -> W[3]+=1
        # col 6 (Z7): C00+=1, C11+=1 ? No.
        # C00 = Z1+Z4-Z5+Z7
        # C01 = Z3+Z5
        # C10 = Z2+Z4
        # C11 = Z1-Z2+Z3+Z6
        
        # Z1: C00+1, C11+1
        # Z2: C10+1, C11-1
        # Z3: C01+1, C11+1
        # Z4: C00+1, C10+1
        # Z5: C00-1, C01+1
        # Z6: C11+1
        # Z7: C00+1
        
        # Let's re-verify Strassen formulas
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # W[0] (Z1): [1, 0, 0, 1]
        # W[1] (Z2): [0, 0, 1, -1]
        # W[2] (Z3): [0, 1, 0, 1]
        # W[3] (Z4): [1, 0, 1, 0]
        # W[4] (Z5): [-1, 1, 0, 0]
        # W[5] (Z6): [0, 0, 0, 1]
        # W[6] (Z7): [1, 0, 0, 0]
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # Z1
        [0, 0, 1, -1], # Z2
        [0, 1, 0, 1], # Z3
        [1, 0, 1, 0], # Z4
        [-1, 1, 0, 0], # Z5
        [0, 0, 0, 1], # Z6
        [1, 0, 0, 0]  # Z7
    ]

    # Process 6 blocks of 2 columns
    for b in range(6):
        K = b * 2 # Start column index in B and C
        for s in range(7):
            # U
            U[:, col_offset] = U_strassen[s]
            
            # V
            # B indices: j*p + k.
            # Block cols: K, K+1.
            # B[0, K] -> idx 0*13 + K = K
            # B[0, K+1] -> idx K+1
            # B[1, K] -> idx 1*13 + K = 13+K
            # B[1, K+1] -> idx 13+K+1
            
            V_indices = [K, K+1, 13+K, 13+K+1]
            V[col_offset] = 0 # already 0
            for idx, coeff in zip(V_indices, V_strassen[s]):
                V[idx, col_offset] += coeff
            
            # W
            # C indices: k*n + i.
            # C[0, K] -> idx K*2 + 0 = 2K
            # C[1, K] -> idx K*2 + 1 = 2K+1
            # C[0, K+1] -> idx (K+1)*2 + 0 = 2K+2
            # C[1, K+1] -> idx (K+1)*2 + 1 = 2K+3
            
            W_indices = [2*K, 2*K+1, 2*K+2, 2*K+3]
            for idx, coeff in zip(W_indices, W_strassen[s]):
                W[idx, col_offset] += coeff
                
            col_offset += 1
            
    # Process 1 block of 1 column (index 12)
    K = 12
    # Rank 4 for 2x1
    # C_0, K = A_00 B_0,K + A_01 B_1,K
    # C_1, K = A_10 B_0,K + A_11 B_1,K
    
    # Z1: A00 B0,K -> C0,K
    U[:, col_offset] = [1, 0, 0, 0]
    V[0*p + K, col_offset] = 1 # B[0,K]
    W[K*n + 0, col_offset] = 1 # C[0,K]
    col_offset += 1
    
    # Z2: A01 B1,K -> C0,K
    U[:, col_offset] = [0, 1, 0, 0]
    V[1*p + K, col_offset] = 1 # B[1,K]
    W[K*n + 0, col_offset] = 1
    col_offset += 1
    
    # Z3: A10 B0,K -> C1,K
    U[:, col_offset] = [0, 0, 1, 0]
    V[0*p + K, col_offset] = 1
    W[K*n + 1, col_offset] = 1
    col_offset += 1
    
    # Z4: A11 B1,K -> C1,K
    U[:, col_offset] = [0, 0, 0, 1]
    V[1*p + K, col_offset] = 1
    W[K*n + 1, col_offset] = 1
    col_offset += 1
    
    assert col_offset == total_rank
    
    return {"U": U, "V": V, "W": W}
