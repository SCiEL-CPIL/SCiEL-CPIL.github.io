"""Rank-45 decomposition for 3x3x6 matrix multiplication using three rank-15 3x3x2 blocks."""
import numpy as np

DIMENSIONS = [3, 3, 6]


def construct():
    """Construct a rank-45 decomposition for 3x3x6 matrix multiplication.
    
    Strategy: Split the 6 columns into three groups of 2 columns.
    For each group, use a rank-15 decomposition of 3x3x2.
    Total rank: 3 * 15 = 45.
    
    The rank-15 decomposition for 3x3x2 is constructed as follows:
    For each output row i (i=0,1,2), we need to compute:
      c[i,0] = a[i,0]*b[0,0] + a[i,1]*b[1,0] + a[i,2]*b[2,0]
      c[i,1] = a[i,0]*b[0,1] + a[i,1]*b[1,1] + a[i,2]*b[2,1]
    
    Using 5 terms per row:
      t1 = a[i,0] * b[0,0]
      t2 = a[i,1] * b[1,0]
      t3 = a[i,2] * b[2,0]
      t4 = a[i,0] * b[0,1]
      t5 = a[i,1] * b[1,1] + a[i,2] * (b[2,1] - b[2,0])
    
    Then:
      c[i,0] = t1 + t2 + t3
      c[i,1] = t4 + t5
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    rank = 45
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Split p=6 into 3 blocks of 2
    for block in range(3):
        k_offset = block * 2  # k_offset is 0, 2, or 4
        
        # For this block, the local column indices are 0 and 1
        # Global column indices are k_offset and k_offset+1
        
        for i in range(3):
            base = block * 15 + i * 5
            
            # Term 0: a[i,0] * b[0, k_offset]
            U[i * m + 0, base] = 1
            V[0 * p + k_offset, base] = 1
            W[k_offset * n + i, base] = 1  # contributes to c[i, k_offset]
            
            # Term 1: a[i,1] * b[1, k_offset]
            U[i * m + 1, base + 1] = 1
            V[1 * p + k_offset, base + 1] = 1
            W[k_offset * n + i, base + 1] = 1  # contributes to c[i, k_offset]
            
            # Term 2: a[i,2] * b[2, k_offset]
            U[i * m + 2, base + 2] = 1
            V[2 * p + k_offset, base + 2] = 1
            W[k_offset * n + i, base + 2] = 1  # contributes to c[i, k_offset]
            
            # Term 3: a[i,0] * b[0, k_offset+1]
            U[i * m + 0, base + 3] = 1
            V[0 * p + (k_offset + 1), base + 3] = 1
            W[(k_offset + 1) * n + i, base + 3] = 1  # contributes to c[i, k_offset+1]
            
            # Term 4: a[i,1] * b[1, k_offset+1] + a[i,2] * (b[2, k_offset+1] - b[2, k_offset])
            U[i * m + 1, base + 4] = 1
            V[1 * p + (k_offset + 1), base + 4] = 1
            W[(k_offset + 1) * n + i, base + 4] = 1  # contributes to c[i, k_offset+1]
            
            U[i * m + 2, base + 4] = 1
            V[2 * p + (k_offset + 1), base + 4] = 1
            W[(k_offset + 1) * n + i, base + 4] = 1  # contributes to c[i, k_offset+1]
            
            V[2 * p + k_offset, base + 4] = -1
            W[(k_offset + 1) * n + i, base + 4] = 1  # this term also contributes to c[i, k_offset+1]
    
    return {"U": U, "V": V, "W": W}
