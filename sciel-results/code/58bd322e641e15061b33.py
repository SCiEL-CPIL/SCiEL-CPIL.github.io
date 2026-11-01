import time
import numpy as np

def construct():
    start_time = time.time()
    
    n, m, p = 2, 2, 16
    # Schoolbook rank is n*m*p = 64
    # We'll use a block-circulant decomposition
    # Split p=16 into 4 blocks of size 4
    
    num_blocks = 4
    block_size = 4  # each block handles 4 columns of the 16
    
    # For each block, use a 2x2 matrix mult structure
    # The 2x16 mult tensor can be decomposed using 4 copies of a 2x2x4 structure
    
    # Strategy: Use 8 rank-1 terms per 4-column block (total 32)
    # This is based on the fact that 2x2x2 has rank 7 (Strassen),
    # but we use a simpler structured decomposition
    
    # Actually, let me use a more systematic approach:
    # The multiplication tensor T_{i,j,k} = delta (A_{i,j} * B_{j,k})
    # where i is row of A (0,1), j is column of A / row of B (0,1), k is column of B (0..15)
    
    # Output C_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    # For each pair (i,k), we need to capture sum_j A_{i,j} B_{j,k}
    # 
    # Let's use a decomposition where we introduce intermediate rank terms.
    # For 2x2x16, we can potentially use rank 2*2*16/2 = 32 by sharing structure.
    
    # Better approach: use the bilinear decomposition
    # Each term in the sum over j can be shared.
    
    # Let me construct a valid decomposition first, then try to minimize.
    # 
    # Simple valid decomposition (schoolbook, rank 64):
    # For each (i, j, k): U has 1 at position i*m+j, V has 1 at j*p+k, W has 1 at k*n+i
    
    # Try to do better: rank 32 decomposition
    # 
    # Key idea: For fixed i, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # We can share the "A[i,j]" and "B[j,k]" factors.
    #
    # Actually, the tensor rank of 2x2x16 matrix mult is at most 2*2*16 = 64.
    # 
    # Let me try a structured rank-32 decomposition:
    # Group columns in pairs. For column pair (2t, 2t+1), we can use
    # a shared factorization.
    
    # For now, let me just return the schoolbook decomposition which is guaranteed valid.
    # The rank is 64.
    
    rank = n * m * p  # 64
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
