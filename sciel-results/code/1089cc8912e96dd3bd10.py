"""Improved program: Strassen-based decomposition for 5x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matmul.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    Indices: i in {0,1}, j in {0,1}, k in {0,1}
    """
    # Strassen formulas for C = A*B where A, B, C are 2x2:
    # C[0,0] = M1 - M2 + M4 - M5
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 - M7
    #
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])
    #
    # U[i*2+j, c] = coefficient of A[i,j] in M_c
    # V[j*2+k, c] = coefficient of B[j,k] in M_c
    # W[k*2+i, c] = coefficient of M_c in C[i,k]  (note: W index is k*n+i, for 2x2: k*2+i)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0] += M1
    W[3, 0] = 1  # C[1,1] += M1
    
    # M2: (A10+A11)*B00
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0] += M2
    
    # M3: A00*(B01-B11)
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1  # B[0,1]
    V[3, 2] = -1 # B[1,1]
    W[1, 2] = 1  # C[0,1] += M3
    W[3, 2] = -1 # C[1,1] -= M3
    
    # M4: A11*(B10-B00)
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1 # B[0,0]
    W[0, 3] = 1  # C[0,0] += M4
    W[2, 3] = 1  # C[1,0] += M4
    
    # M5: (A00+A01)*B11
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 4] = -1 # C[0,0] -= M5
    W[1, 4] = 1  # C[0,1] += M5
    
    # M6: (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1 # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    W[3, 5] = 1  # C[1,1] += M6
    
    # M7: (A11-A01)*(B10+B11)
    U[3, 6] = 1  # A[1,1]
    U[1, 6] = -1 # A[0,1]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    W[3, 6] = -1 # C[1,1] -= M7
    
    return U, V, W


def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for 5x5x5 matmul.
    
    Uses a 2+2+1 block partition of each dimension, applying Strassen (rank 7) to 2x2 blocks
    and schoolbook (rank 8) to 1x1 blocks.
    
    Total rank: 4 * 7 (2x2 Strassen blocks) + 4 * 8 (mixed 2x1 or 1x2 blocks) + 1 * 8 (1x1 block)
    Wait, let me reconsider the block structure.
    
    For 5x5x5 with partition 2+2+1 on each of the three dimensions:
    We have 3x3 = 9 blocks of size (a x b x c) where a, b, c are in {2, 2, 1}.
    
    Block (a, b, c) has rank a*b*c for schoolbook.
    For 2x2x2 blocks, we use Strassen with rank 7 instead of 8.
    
    Number of 2x2x2 blocks: 2*2 = 4 (positions where all three dims are size 2)
    Number of mixed blocks: 9 - 4 = 5
    
    Total rank = 4 * 7 + 5 * (a*b*c for each)
    The 5 mixed blocks:
    - (2,2,1): rank 4, there are 2 of these (i in {0,1}, j in {0,1}, k=2)
    - (2,1,2): rank 4, there are 2 of these
    - (1,2,2): rank 4, there is 1 of these
    - (1,1,1): rank 1, there is 1 of these
    
    Wait, I need to be more careful. The partition is on dimensions i (rows of A, 5), j (cols of A/rows of B, 5), k (cols of B, 5).
    
    Partition points for each dim: [0, 2, 4, 5] giving blocks of sizes [2, 2, 1].
    
    Block at (I, J, K) where I, J, K are block indices in {0, 1, 2}:
    - Block size: (size_I, size_J, size_K) where size = [2, 2, 1]
    
    For each block, we decompose the sub-matmul of that size.
    
    For 2x2x2: use Strassen (rank 7)
    For 2x2x1: schoolbook rank 4
    For 2x1x2: schoolbook rank 4  
    For 1x2x2: schoolbook rank 4
    For 2x1x1: schoolbook rank 2
    For 1x2x1: schoolbook rank 2
    For 1x1x2: schoolbook rank 2
    For 1x1x1: schoolbook rank 1
    
    Count of each type:
    (2,2,2): 2*2*1 = 4 blocks -> rank 7 each -> 28
    (2,2,1): 2*2*1 = 4 blocks -> rank 4 each -> 16
    (2,1,2): 2*1*2 = 4 blocks -> rank 4 each -> 16
    (2,1,1): 2*1*1 = 2 blocks -> rank 2 each -> 4
    (1,2,2): 1*2*2 = 4 blocks -> rank 4 each -> 16
    (1,2,1): 1*2*1 = 2 blocks -> rank 2 each -> 4
    (1,1,2): 1*1*2 = 2 blocks -> rank 2 each -> 4
    (1,1,1): 1*1*1 = 1 block -> rank 1 each -> 1
    
    Total rank = 28 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 89
    
    That's much better than 125!
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Partition sizes
    part_sizes = [2, 2, 1]
    part_offsets = [0, 2, 4]  # start index of each block
    
    # Get Strassen decomposition for 2x2x2
    S_U, S_V, S_W = _strassen_2x2()
    
    # Build the full decomposition
    # U: shape (n*m, rank) = (25, rank)
    # V: shape (m*p, rank) = (25, rank)  
    # W: shape (p*n, rank) = (25, rank)
    
    U_full = []
    V_full = []
    W_full = []
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                si, sj, sk = part_sizes[I], part_sizes[J], part_sizes[K]
                oi, oj, ok = part_offsets[I], part_offsets[J], part_offsets[K]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen
                    # Local indices: i in [0,1], j in [0,1], k in [0,1]
                    # Global: i_global = oi + i, j_global = oj + j, k_global = ok + k
                    for c in range(7):
                        # U column: for each (i,j) in local 2x2
                        u_col = [0] * (n * m)
                        v_col = [0] * (m * p)
                        w_col = [0] * (p * n)
                        
                        for i in range(2):
                            for j in range(2):
                                if S_U[i * 2 + j, c] != 0:
                                    u_col[(oi + i) * m + (oj + j)] = S_U[i * 2 + j, c]
                        
                        for j in range(2):
                            for k in range(2):
                                if S_V[j * 2 + k, c] != 0:
                                    v_col[(oj + j) * p + (ok + k)] = S_V[j * 2 + k, c]
                        
                        for k in range(2):
                            for i in range(2):
                                if S_W[k * 2 + i, c] != 0:
                                    w_col[(ok + k) * n + (oi + i)] = S_W[k * 2 + i, c]
                        
                        U_full.append(u_col)
                        V_full.append(v_col)
                        W_full.append(w_col)
                else:
                    # Schoolbook for this block
                    for i in range(si):
                        for j in range(sj):
                            for k in range(sk):
                                u_col = [0] * (n * m)
                                v_col = [0] * (m * p)
                                w_col = [0] * (p * n)
                                
                                u_col[(oi + i) * m + (oj + j)] = 1
                                v_col[(oj + j) * p + (ok + k)] = 1
                                w_col[(ok + k) * n + (oi + i)] = 1
                                
                                U_full.append(u_col)
                                V_full.append(v_col)
                                W_full.append(w_col)
    
    rank = len(U_full)
    
    # Convert to lists of lists (ensure integer entries)
    U = [[int(x) if x == int(x) else x for x in col] for col in U_full]
    V = [[int(x) if x == int(x) else x for x in col] for col in V_full]
    W = [[int(x) if x == int(x) else x for x in col] for col in W_full]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
