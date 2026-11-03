"""
Rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.

Uses two disjoint Strassen 2x2x2 blocks (rank 7 each) for the first 2 rows of A
and first 2 cols of B, combined with schoolbook for the remaining terms.
Total rank: 7 + 7 + 3*4*5 - 8 = 7 + 7 + 15 = 29... 
Wait, let me recalculate.

Schoolbook rank = 3*4*5 = 60.
Two Strassen blocks cover 2*2*2 = 8 terms each, but use rank 7 instead of 8.
So saving is 1 per block.
Total rank = 60 - 1 - 1 = 58.

The two Strassen blocks are:
Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> covers 8 schoolbook terms, uses rank 7
Block 2: i in {0,1}, j in {2,3}, k in {0,1} -> covers 8 schoolbook terms, uses rank 7

Wait, that's not right. Each Strassen block is for 2x2*2x2->2x2.
So Block 1: i in {0,1}, j in {0,1}, k in {0,1}
Block 2: i in {0,1}, j in {2,3}, k in {0,1}

But these two blocks share the same i and k indices, just different j.
Actually, let me think more carefully.

A Strassen 2x2x2 block handles:
A is 2x2, B is 2x2, C is 2x2.
It uses 7 multiplications instead of 8.

For our 3x4*4x5 problem:
We can partition A into blocks and B into blocks.

Let's use:
- A is 3x4, B is 4x5, C is 3x5.

One approach:
- Use Strassen for the top-left 2x2 block of A (rows 0-1, cols 0-1) 
  and top-left 2x2 block of B (rows 0-1, cols 0-1).
  This gives a 2x2 result for C (rows 0-1, cols 0-1) using rank 7.
  
- The remaining terms use schoolbook.

Total schoolbook terms: 3*4*5 = 60
Strassen block replaces 2*2*2 = 8 terms with rank 7, saving 1.
Total rank = 60 - 1 = 59.

Can we do better with two Strassen blocks?
If we use two independent Strassen blocks:
Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2], rank 7 (saves 1)
Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2], rank 7 (saves 1)

Wait, these two blocks both contribute to C[0:2,0:2]. They're not independent in the output.
Actually, each Strassen block is a separate tensor decomposition for a sub-problem.
The total rank is the sum of the ranks of each block.

Let me reconsider. The standard approach is:
- Partition the multiplication into independent sub-problems.
- Each sub-problem is a smaller matrix multiplication.
- Use Strassen for 2x2 sub-problems, schoolbook for others.

For 3x4 * 4x5:
We can write C = A*B where A is 3x4, B is 4x5.

One decomposition:
- C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
- C[0:2, 2:5] = A[0:2, 0:2]*B[0:2, 2:5] + A[0:2, 2:4]*B[2:4, 2:5]
- C[2:3, 0:2] = A[2:3, 0:2]*B[0:2, 0:2] + A[2:3, 2:4]*B[2:4, 0:2]
- C[2:3, 2:5] = A[2:3, 0:2]*B[0:2, 2:5] + A[2:3, 2:4]*B[2:4, 2:5]

This is getting complex. Let me just implement the known result:
Two disjoint Strassen 2x2x2 blocks give rank 58.

The two blocks are:
Block 1: i in {0,1}, j in {0,1}, k in {0,1}
Block 2: i in {2,2}, j in {0,1}, k in {0,1}  -- wait, this is 1x2*2x2, not 2x2*2x2.

Actually, for a 2x2*2x2 Strassen block, we need 2 rows of A, 2 cols of B, 
and the result is 2x2.

Let me just implement the standard two-Strassen-block approach:
Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> contributes to C[0:2, 0:2]
Block 2: A[0:2, 2:4] * B[2:4, 2:5] -> contributes to C[0:2, 2:5]

Wait, B[2:4, 2:5] is 2x3, not 2x2. So that's not a clean Strassen block.

Let me think differently. The known result from the inspirations is rank 58.
This comes from using two Strassen blocks, each saving 1 from schoolbook.

The simplest implementation:
- Start with schoolbook (rank 60).
- Replace one 2x2*2x2 sub-block with Strassen (rank 7 instead of 8), saving 1.
- Replace another independent 2x2*2x2 sub-block with Strassen, saving 1.
- Total rank: 60 - 2 = 58.

For the 3x4*4x5 case:
- Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2*2x2->2x2, Strassen rank 7)
- Block 2: A[1:3, 2:4] * B[2:4, 1:3] -> C[1:3, 1:3] (2x2*2x2->2x2, Strassen rank 7)

These two blocks are disjoint in their (i,j,k) index sets:
Block 1: i in {0,1}, j in {0,1}, k in {0,1}
Block 2: i in {1,2}, j in {2,3}, k in {1,2}

They overlap at i=1, but that's fine because they contribute to different (i,j,k) triples.
Actually, they share i=1, but j and k are different, so the (i,j,k) triples are disjoint.

Let me verify:
Block 1 triples: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
Block 2 triples: (1,2,1), (1,2,2), (1,3,1), (1,3,2), (2,2,1), (2,2,2), (2,3,1), (2,3,2)

Yes, these are disjoint! So we can use both Strassen blocks independently.

Total rank = 60 - 1 - 1 = 58.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U, V, W for the 2x2x2x2 tensor (rank 7).
    
    U has shape (4, 7) -- 4 = 2*2 entries of A
    V has shape (4, 7) -- 4 = 2*2 entries of B  
    W has shape (4, 7) -- 4 = 2*2 entries of C
    
    Indexing: A[i,j] -> U[i*2+j, :], B[j,k] -> V[j*2+k, :], C[i,k] -> W[i*2+k, :]
    """
    # Standard Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [
        [1, 0, 1, 0, 1, 1, 0],  # A00
        [0, 1, 0, 0, 0, 0, 1],  # A01
        [0, 1, 0, 0, 0, -1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 1, 1, 0],  # B01
        [1, 0, -1, -1, 0, 0, 1],# B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    return {"U": U, "V": V, "W": W}


def _build_decomposition():
    """
    Build the rank-58 decomposition using two Strassen blocks + schoolbook.
    
    Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen (rank 7)
    Block 2: i in {1,2}, j in {2,3}, k in {1,2} -> Strassen (rank 7)
    Rest: schoolbook (rank 60 - 8 - 8 = 44)
    
    Total rank: 7 + 7 + 44 = 58
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    total_rank = 58
    
    U = [[0] * total_rank for _ in range(n * m)]  # 12 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 20 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 15 rows
    
    strassen = _strassen_2x2_2x2()
    sU, sV, sW = strassen['U'], strassen['V'], strassen['W']
    
    col = 0
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Maps to Strassen indices:
    # A[i,j] -> U[i*2+j, :] in Strassen, which is U[i*m+j, :] in full
    # B[j,k] -> V[j*2+k, :] in Strassen, which is V[j*p+k, :] in full
    # C[i,k] -> W[i*2+k, :] in Strassen, which is W[k*n+i, :] in full (note: W[k*n+i] per convention)
    
    for r in range(7):
        # For each Strassen rank index r
        for si in range(2):  # Strassen i index
            for sj in range(2):  # Strassen j index
                # A[si, sj] in Strassen -> A[si, sj] in full
                # U[si*2+sj, r] in Strassen
                U[si * m + sj][col + r] += sU[si * 2 + sj][r]
        
        for sj in range(2):  # Strassen j index
            for sk in range(2):  # Strassen k index
                # B[sj, sk] in Strassen -> B[sj, sk] in full
                # V[sj*2+sk, r] in Strassen
                V[sj * p + sk][col + r] += sV[sj * 2 + sk][r]
        
        for si in range(2):  # Strassen i index
            for sk in range(2):  # Strassen k index
                # C[si, sk] in Strassen -> C[si, sk] in full
                # W[si*2+sk, r] in Strassen, but full convention is W[k*n+i]
                # So W[sk*n+si, r] in full
                W[sk * n + si][col + r] += sW[si * 2 + sk][r]
    
    col += 7  # 7 columns used for Block 1
    
    # Block 2: i in {1,2}, j in {2,3}, k in {1,2}
    # Strassen indices: si in {0,1} maps to i = 1+si, sj in {0,1} maps to j = 2+sj, sk in {0,1} maps to k = 1+sk
    
    for r in range(7):
        for si in range(2):
            for sj in range(2):
                # A[1+si, 2+sj] in full
                U[(1 + si) * m + (2 + sj)][col + r] += sU[si * 2 + sj][r]
        
        for sj in range(2):
            for sk in range(2):
                # B[2+sj, 1+sk] in full
                V[(2 + sj) * p + (1 + sk)][col + r] += sV[sj * 2 + sk][r]
        
        for si in range(2):
            for sk in range(2):
                # C[1+si, 1+sk] in full -> W[(1+sk)*n + (1+si), r]
                W[(1 + sk) * n + (1 + si)][col + r] += sW[si * 2 + sk][r]
    
    col += 7  # 7 columns used for Block 2
    
    # Schoolbook for remaining terms
    # All (i,j,k) triples not in Block 1 or Block 2
    block1_triples = set()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                block1_triples.add((i, j, k))
    
    block2_triples = set()
    for si in range(2):
        for sj in range(2):
            for sk in range(2):
                block2_triples.add((1 + si, 2 + sj, 1 + sk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in block1_triples or (i, j, k) in block2_triples:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    T_decomp = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        
        T_decomp += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat, optimize=True)
    
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_decomp, T_target, atol=tol)


def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    result = _build_decomposition()
    
    if _verify(result['U'], result['V'], result['W']):
        rank = len(result['U'][0]) if result['U'] else 0
        return result
    
    # Fallback to schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
