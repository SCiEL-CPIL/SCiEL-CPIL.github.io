"""
Rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.

Strategy:
- Use a systematic search over candidate tensor decompositions.
- The key insight is that the 2x4 x 4x6 multiplication can be viewed as
  computing C = A @ B where A is 2x4 and B is 4x6.
- We can use the fact that this is equivalent to 3 separate 2x2 x 2x2 multiplications
  plus cross terms, but more efficiently we can use block structure.
- Actually, let's think about this differently:
  A (2x4) * B (4x6) = C (2x6)
  We can split A into two 2x2 blocks: A = [A0 | A1]
  B into two 4x3 blocks: B = [B0; B1] where B0, B1 are 2x3
  Then C = A0 * B0 + A1 * B1
  Each A0*B0 is 2x2 * 2x3, and A1*B1 is 2x2 * 2x3.
  
  For 2x2 * 2x3: we can view this as 2x2 * [C0 C1] where C0, C1 are 2x1 (columns)
  This is equivalent to computing two 2x2 * 2x1 products.
  
  Alternatively, 2x2 * 2x3 can be decomposed as:
  For each column of the result (2 columns in B become 3 columns in C),
  we need 2x2 * 2x1 which has rank 4 (schoolbook) or we can use Strassen-like.
  
  Actually, 2x2 * 2x1: result is 2x1, so we compute 2 outputs.
  c0 = a00*b0 + a01*b1
  c1 = a10*b0 + a11*b1
  This is rank 4 (each output is a dot product of length 2).
  
  But we can do better by sharing. Let's use the full structure.
  
  Better approach: Use the known rank-7 Strassen for 2x2 and apply it cleverly.
  
  Let's try a different decomposition:
  A (2x4) = [A0, A1] where A0, A1 are 2x2
  B (4x6) = [[B00, B01, B02], [B10, B11, B12]] where Bij are 2x2
  
  C (2x6) = [C0, C1, C2] where Cj = A0*Bj0 + A1*Bj1
  
  For each j, Cj = A0*Bj0 + A1*Bj1 is a 2x2 result computed from two 2x2 multiplications.
  
  We can use Strassen's rank-7 for each 2x2 multiplication, giving 7+7=14 per column block,
  total 42. But we can share the A-side Strassen factors across column blocks.
  
  Key insight: The U-factors for A0 and A1 are the same regardless of which column block
  we're computing. So we precompute the 7 Strassen combinations for A0 and 7 for A1,
  then for each column block j, we only need to compute the V-factors for Bj0 and Bj1
  (7 each) and combine.
  
  This gives: 7 (for A0) + 7 (for A1) + 3*(7+7) = 14 + 42 = 56? No, that's worse.
  
  Wait, the rank is the number of independent multiplications. Let me reconsider.
  
  Actually, the correct way: The tensor rank is the minimum number of rank-1 terms.
  For the full 2x4 x 4x6 multiplication, we need to find the minimal rank.
  
  Let's use a direct approach: try to find a decomposition with rank R by solving
  the system of equations. Start with R=42 (from block Strassen) and see if we can do better.
  
  Actually, let's just implement the rank-42 block Strassen decomposition correctly
  and verify it works. Then we can try to reduce further.
  
  The rank-42 decomposition:
  - For each of the 3 column blocks (j=0,1,2):
    - Compute A0*Bj0 using rank-7 Strassen: 7 multiplications
    - Compute A1*Bj1 using rank-7 Strassen: 7 multiplications
  - Total: 3 * 14 = 42 multiplications
  
  The U, V, W matrices:
  U is 8x42 (n*m = 2*4 = 8 rows)
  V is 24x42 (m*p = 4*6 = 24 rows)
  W is 12x42 (p*n = 6*2 = 12 rows)
  
  For column block j (j=0,1,2), columns 14*j to 14*j+13 of U,V,W:
  - Columns 14*j to 14*j+6: Strassen for A0*Bj0
  - Columns 14*j+7 to 14*j+13: Strassen for A1*Bj1
  
  A0 = A[:, 0:2], A1 = A[:, 2:4]
  Bj0 = B[0:2, 2*j:2*j+2], Bj1 = B[2:4, 2*j:2*j+2]
  Cj = C[:, 2*j:2*j+2]
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Construct a rank-42 exact tensor decomposition for 2x4 x 4x6 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Rank-7 Strassen decomposition for 2x2 matrix multiplication
    # X (2x2) * Y (2x2) = Z (2x2)
    # U22: 4x7, V22: 4x7, W22: 4x7
    # Indexing: U22[i*2+j, r], V22[j*2+k, r], W22[k*2+i, r]
    
    # Standard Strassen rank-7:
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*(y00)
    # M3 = (x00)*(y10-y00)
    # M4 = (x11)*(y01-y11)
    # M5 = (x00+x01)*(y11)
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01+x10)*(y10+y11)
    
    # Z00 = M1 + M3 - M5 - M6
    # Z01 = M4 + M5
    # Z10 = M2 + M3
    # Z11 = M1 - M2 + M4 + M7
    
    U22 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # x00
        [0, 0, 0, 0, 1, 0, 1],     # x01
        [0, 1, 0, 0, 0, 1, 1],     # x10
        [1, 1, 0, 1, 0, 0, 0],     # x11
    ], dtype=float)
    
    V22 = np.array([
        [1, 1, -1, 0, 0, 1, 0],    # y00
        [1, 0, 0, 1, 0, 1, 0],     # y01
        [1, 0, 1, 0, 0, 0, 1],     # y10
        [1, 0, 0, -1, 1, 0, 1],    # y11
    ], dtype=float)
    
    W22 = np.array([
        [1, 0, 1, 0, -1, -1, 0],   # Z00 (k=0, i=0)
        [0, 0, 0, 1, 1, 0, 0],     # Z01 (k=1, i=0)
        [0, 1, 1, 0, 0, 0, 0],     # Z10 (k=0, i=1)
        [1, -1, 0, 1, 0, 0, 1],    # Z11 (k=1, i=1)
    ], dtype=float)
    
    # Verify the 2x2 Strassen decomposition
    X_test = np.array([[1, 2], [3, 4]], dtype=float)
    Y_test = np.array([[5, 6], [7, 8]], dtype=float)
    Z_expected = X_test @ Y_test
    
    Z_computed = np.zeros((2, 2), dtype=float)
    for r in range(7):
        u_val = sum(U22[i*2+j, r] * X_test[i, j] for i in range(2) for j in range(2))
        v_val = sum(V22[j*2+k, r] * Y_test[j, k] for j in range(2) for k in range(2))
        for k in range(2):
            for i in range(2):
                Z_computed[i, k] += W22[k*2+i, r] * u_val * v_val
    
    strassen_valid = np.allclose(Z_computed, Z_expected, atol=1e-10)
    
    if not strassen_valid:
        # Fallback to schoolbook rank-48
        rank = n * m * p
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
    
    # Build rank-42 decomposition for 2x4 x 4x6
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=float)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=float)  # 12 x 42
    
    for j in range(3):  # j = 0, 1, 2 for the three column blocks
        base_col = 14 * j
        
        # Strassen for A0 * Bj0 (columns base_col to base_col+6)
        # A0: A[i, jj] for i=0,1; jj=0,1
        # Bj0: B[jj, 2*j+k] for jj=0,1; k=0,1
        # Cj: C[i, 2*j+k] for i=0,1; k=0,1
        
        for r in range(7):
            col = base_col + r
            # U factors for A0
            for i in range(2):
                for jj in range(2):
                    U[i * m + jj, col] = U22[i*2+jj, r]
            
            # V factors for Bj0
            for jj in range(2):
                for k in range(2):
                    V[jj * p + (2*j + k), col] = V22[jj*2+k, r]
            
            # W factors for Cj (from A0*Bj0)
            for k in range(2):
                for i in range(2):
                    W[(2*j + k) * n + i, col] = W22[k*2+i, r]
        
        # Strassen for A1 * Bj1 (columns base_col+7 to base_col+13)
        # A1: A[i, 2+jj] for i=0,1; jj=0,1
        # Bj1: B[2+jj, 2*j+k] for jj=0,1; k=0,1
        # Cj: C[i, 2*j+k] for i=0,1; k=0,1
        
        for r in range(7):
            col = base_col + 7 + r
            # U factors for A1
            for i in range(2):
                for jj in range(2):
                    U[i * m + (2 + jj), col] = U22[i*2+jj, r]
            
            # V factors for Bj1
            for jj in range(2):
                for k in range(2):
                    V[(2 + jj) * p + (2*j + k), col] = V22[jj*2+k, r]
            
            # W factors for Cj (from A1*Bj1)
            for k in range(2):
                for i in range(2):
                    W[(2*j + k) * n + i, col] = W22[k*2+i, r]
    
    # Verify the full decomposition
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p), dtype=float)
    for r in range(rank):
        u_val = sum(U[idx, r] * A_test[idx // m, idx % m] for idx in range(n * m))
        v_val = sum(V[idx, r] * B_test[idx // p, idx % p] for idx in range(m * p))
        for idx in range(p * n):
            k = idx // n
            i = idx % n
            C_computed[i, k] += W[idx, r] * u_val * v_val
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
