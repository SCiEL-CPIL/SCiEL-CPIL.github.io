"""Matrix multiplication tensor decomposition for 3x4 * 4x7.

We use a block decomposition strategy:
- Split the 3x4x7 multiplication into blocks where Strassen's algorithm can reduce rank.
- Specifically, split the output columns (p=7) into groups of 2 and 1.
- For each group of 2 output columns, we have a 3x4x2 sub-problem.
  - Split the 3 rows into 2+1.
  - The 2x4x2 part: split the 4 middle dimension into 2+2, giving two 2x2x2 sub-problems.
    Each 2x2x2 can use Strassen (rank 7) instead of schoolbook (rank 8).
    So 2x4x2 uses 7+7 = 14 rank.
  - The 1x4x2 part: this is just 4 multiplications, rank 4.
  - Total for 3x4x2: 14 + 4 = 18 (vs schoolbook 24).
- For the remaining 1 output column (p=7 = 2+2+2+1):
  - 3x4x1: just 3*4 = 12 multiplications, rank 12.
- Total rank: 3 * 18 + 12 = 54 + 12 = 66.

Wait, let me re-examine. The tensor T[i,j,k] = 1 for all i,j,k.
We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i in [0,3), j in [0,4), k in [0,7).

Let me construct this carefully.

For the 2x2x2 Strassen decomposition, we have:
  U_str: (4, 7), V_str: (4, 7), W_str: (4, 7)
  where U_str[i*2+j, c] corresponds to A[i,j], etc.

For embedding into 3x4x7:
  U: (12, rank), V: (28, rank), W: (21, rank)

Let me split k (output column) into groups:
  k in {0,1}: group 0
  k in {2,3}: group 1
  k in {4,5}: group 2
  k in {6}:   group 3

For each group with 2 k-values, we have a 3x4x2 sub-tensor.
  Split i into {0,1} and {2}.
  For i in {0,1}: 2x4x2 sub-tensor.
    Split j into {0,1} and {2,3}.
    For j in {0,1}: 2x2x2 -> Strassen, rank 7
    For j in {2,3}: 2x2x2 -> Strassen, rank 7
    Total: 14
  For i in {2}: 1x4x2 sub-tensor.
    This is just 4 terms (one for each j), rank 4.
  Total for 3x4x2: 14 + 4 = 18

For k in {6}: 3x4x1 sub-tensor, rank 12.

Total rank: 3*18 + 12 = 66.

Let me verify this is correct by constructing it.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U: (4,7) where row i*2+j is A[i,j]
    V: (4,7) where row i*2+j is B[i,j]  
    W: (4,7) where row i*2+j is C[i,j]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+b)(e+f) -> C11 += 1, C12 += 1
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    
    # M2 = (c+d)e -> C21 += 1, C22 += -1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h) -> C12 += 1, C22 += -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4 = d(g-e) -> C11 += -1, C22 += -1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[3, 3] = -1
    
    # M5 = (a+d)(e+h) -> C11 += 1, C21 += 1
    U[0, 4] = 1; U[3, 4] = 1
    V[0, 4] = 1; V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6 = (b-d)(g+h) -> C22 += 1
    U[1, 5] = 1; U[3, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # M7 = (c-a)(e+f) -> C11 += -1, C21 += 1
    U[2, 6] = 1; U[0, 6] = -1
    V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # We'll build the decomposition block by block.
    # Track the current column offset.
    col = 0
    
    # First, allocate maximum possible rank (schoolbook) and then trim.
    # Actually, let's just build it up.
    
    # We'll use lists of lists for U, V, W and append columns as we go.
    # But we need to know the total rank first. Let me compute it.
    
    # For each pair of k values (k0, k1) where k0 = 2*g, k1 = 2*g+1 for g in 0,1,2:
    #   For i in {0,1}, j in {0,1}: Strassen 2x2x2, rank 7
    #   For i in {0,1}, j in {2,3}: Strassen 2x2x2, rank 7
    #   For i = 2, j in {0,1,2,3}: schoolbook, rank 4
    #   Total per group: 18
    # For k = 6:
    #   For i in {0,1,2}, j in {0,1,2,3}: schoolbook, rank 12
    # Total: 3*18 + 12 = 66
    
    total_rank = 3 * 18 + 12  # = 66
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # (12, 66)
    V = [[0.0] * total_rank for _ in range(m * p)]  # (28, 66)
    W = [[0.0] * total_rank for _ in range(p * n)]  # (21, 66)
    
    U_str, V_str, W_str = _strassen_2x2()
    
    # Process groups of 2 k-values: k in {0,1}, {2,3}, {4,5}
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # Sub-block 1: i in {0,1}, j in {0,1}, k in {k0, k1}
        # This is a 2x2x2 sub-problem.
        # U rows: i*4+j for i in {0,1}, j in {0,1} -> rows 0,1,4,5
        # V rows: j*7+k for j in {0,1}, k in {k0,k1} -> rows k0, k0+1, 7+k0, 7+k0+1
        # W rows: k*3+i for k in {k0,k1}, i in {0,1} -> rows k0*3, k0*3+1, k1*3, k1*3+1
        
        for c in range(7):
            U_idx = [0, 1, 4, 5]  # i*4+j for (i,j) in [(0,0),(0,1),(1,0),(1,1)]
            V_idx = [k0, k0+1, 7+k0, 7+k0+1]  # j*7+k
            W_idx = [k0*3, k0*3+1, k1*3, k1*3+1]  # k*3+i
            
            for r in range(4):
                U[U_idx[r]][col] += U_str[r, c]
                V[V_idx[r]][col] += V_str[r, c]
                W[W_idx[r]][col] += W_str[r, c]
            col += 1
        
        # Sub-block 2: i in {0,1}, j in {2,3}, k in {k0, k1}
        # U rows: i*4+j for i in {0,1}, j in {2,3} -> rows 2,3,6,7
        # V rows: j*7+k for j in {2,3}, k in {k0,k1} -> rows 14+k0, 14+k0+1, 21+k0, 21+k0+1
        # W rows: k*3+i for k in {k0,k1}, i in {0,1} -> rows k0*3, k0*3+1, k1*3, k1*3+1
        
        for c in range(7):
            U_idx = [2, 3, 6, 7]
            V_idx = [14+k0, 14+k0+1, 21+k0, 21+k0+1]
            W_idx = [k0*3, k0*3+1, k1*3, k1*3+1]
            
            for r in range(4):
                U[U_idx[r]][col] += U_str[r, c]
                V[V_idx[r]][col] += V_str[r, c]
                W[W_idx[r]][col] += W_str[r, c]
            col += 1
        
        # Sub-block 3: i = 2, j in {0,1,2,3}, k in {k0, k1}
        # Schoolbook: 4 terms
        for j in range(4):
            U[2*4 + j][col] = 1.0  # i=2, j
            V[j*7 + k0][col] = 1.0  # j, k0
            W[k0*3 + 2][col] = 1.0  # k0, i=2
            
            U[2*4 + j][col] += 0.0  # already set
            V[j*7 + k1][col] = 1.0  # j, k1
            W[k1*3 + 2][col] = 1.0  # k1, i=2
            
            col += 1
    
    # Process k = 6: 3x4x1 sub-problem, schoolbook rank 12
    k = 6
    for i in range(3):
        for j in range(4):
            U[i*4 + j][col] = 1.0
            V[j*7 + k][col] = 1.0
            W[k*3 + i][col] = 1.0
            col += 1
    
    # Verify col == total_rank
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    # Convert to lists of lists (or keep as numpy? The interface says integer/half-integer entries)
    # Let's convert to lists for clarity.
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
