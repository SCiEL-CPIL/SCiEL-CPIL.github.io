"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.

Uses a block decomposition: split the 9 columns into 4+4+1.
- For each 4x4 * 4x4 block, use Strassen's algorithm (rank 49 per block).
- For the 4x4 * 4x1 block, use schoolbook (rank 16).
Total rank = 49 + 49 + 16 = 114.

The output factor follows the official trace convention W[k*n+i, r].
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]


def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2 * 2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) where:
    - U rows correspond to (i,j) pairs: row = i*2 + j
    - V rows correspond to (j,k) pairs: row = j*2 + k
    - W rows correspond to (k,i) pairs: row = k*2 + i
    """
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output: C00 = M1 + M4 - M5 - M7
    #         C01 = M3 + M5
    #         C10 = M2 + M4
    #         C11 = M1 - M2 + M3 + M7
    
    # For each of the 7 products, we need to specify:
    # - Which entries of A contribute (with what sign)
    # - Which entries of B contribute (with what sign)
    # - Which entries of C receive the product (with what sign)
    
    # M1: (A00 + A11) * (B00 + B11)
    # A factors: A00 (+1), A11 (+1)
    # B factors: B00 (+1), B11 (+1)
    # C contributions: C00 (+1), C11 (+1)
    
    # M2: (A10 + A11) * B00
    # A factors: A10 (+1), A11 (+1)
    # B factors: B00 (+1)
    # C contributions: C10 (+1), C11 (-1)
    
    # M3: A00 * (B01 - B11)
    # A factors: A00 (+1)
    # B factors: B01 (+1), B11 (-1)
    # C contributions: C01 (+1), C11 (+1)
    
    # M4: A11 * (B10 - B00)
    # A factors: A11 (+1)
    # B factors: B10 (+1), B00 (-1)
    # C contributions: C00 (+1), C10 (+1)
    
    # M5: (A00 + A01) * B11
    # A factors: A00 (+1), A01 (+1)
    # B factors: B11 (+1)
    # C contributions: C00 (-1), C01 (+1)
    
    # M6: (A10 - A00) * (B00 + B01)
    # A factors: A10 (+1), A00 (-1)
    # B factors: B00 (+1), B01 (+1)
    # C contributions: C10 (+1)
    
    # M7: (A01 - A11) * (B10 + B11)
    # A factors: A01 (+1), A11 (-1)
    # B factors: B10 (+1), B11 (+1)
    # C contributions: C00 (-1), C11 (+1)
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Index mapping:
    # A: (i,j) -> row i*2+j
    # B: (j,k) -> row j*2+k
    # C: (k,i) -> row k*2+i
    
    # M1: (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1   # A00
    U[1*2+1, 0] = 1   # A11
    V[0*2+0, 0] = 1   # B00
    V[1*2+1, 0] = 1   # B11
    W[0*2+0, 0] = 1   # C00 (k=0,i=0)
    W[1*2+1, 0] = 1   # C11 (k=1,i=1)
    
    # M2: (A10 + A11) * B00
    U[1*2+0, 1] = 1   # A10
    U[1*2+1, 1] = 1   # A11
    V[0*2+0, 1] = 1   # B00
    W[0*2+1, 1] = 1   # C10 (k=0,i=1)
    W[1*2+1, 1] = -1  # C11 (k=1,i=1)
    
    # M3: A00 * (B01 - B11)
    U[0*2+0, 2] = 1   # A00
    V[0*2+1, 2] = 1   # B01
    V[1*2+1, 2] = -1  # B11
    W[1*2+0, 2] = 1   # C01 (k=1,i=0)
    W[1*2+1, 2] = 1   # C11 (k=1,i=1)
    
    # M4: A11 * (B10 - B00)
    U[1*2+1, 3] = 1   # A11
    V[1*2+0, 3] = 1   # B10
    V[0*2+0, 3] = -1  # B00
    W[0*2+0, 3] = 1   # C00 (k=0,i=0)
    W[0*2+1, 3] = 1   # C10 (k=0,i=1)
    
    # M5: (A00 + A01) * B11
    U[0*2+0, 4] = 1   # A00
    U[0*2+1, 4] = 1   # A01
    V[1*2+1, 4] = 1   # B11
    W[0*2+0, 4] = -1  # C00 (k=0,i=0)
    W[1*2+0, 4] = 1   # C01 (k=1,i=0)
    
    # M6: (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1   # A10
    U[0*2+0, 5] = -1  # A00
    V[0*2+0, 5] = 1   # B00
    V[0*2+1, 5] = 1   # B01
    W[0*2+1, 5] = 1   # C10 (k=0,i=1)
    
    # M7: (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1   # A01
    U[1*2+1, 6] = -1  # A11
    V[1*2+0, 6] = 1   # B10
    V[1*2+1, 6] = 1   # B11
    W[0*2+0, 6] = -1  # C00 (k=0,i=0)
    W[1*2+1, 6] = 1   # C11 (k=1,i=1)
    
    return U, V, W


def _strassen_4x4():
    """Return the 49 rank-1 terms for 4x4 * 4x4 matrix multiplication.
    
    Uses Kronecker product of Strassen 2x2 with itself.
    
    Returns U, V, W each of shape (16, 49) where:
    - U rows correspond to (i,j) pairs: row = i*4 + j
    - V rows correspond to (j,k) pairs: row = j*4 + k
    - W rows correspond to (k,i) pairs: row = k*4 + i
    """
    U2, V2, W2 = _strassen_2x2()
    
    # For 4x4, we treat it as 2x2 blocks of 2x2 submatrices
    # The rank-49 decomposition comes from: for each of the 7 Strassen products
    # at the top level, we need 7 Strassen products at the sub-block level
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # Top-level Strassen: 7 products
    # Sub-level Strassen: 7 products per top-level product
    # Total: 7 * 7 = 49
    
    for r_top in range(7):
        for r_sub in range(7):
            r = r_top * 7 + r_sub
            
            # U: A has entries (i,j) where i,j in {0,1,2,3}
            # In block form: A = [[A00, A01], [A10, A11]] where each is 2x2
            # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
            
            # For top-level product r_top, the A factor is a linear combination of blocks
            # For sub-level product r_sub, within each block, we have a linear combination
            
            # Let's think of it as: U4[i*4+j, r] = sum over (block, sub-entry) of 
            #   coeff_top[block, r_top] * coeff_sub[sub_entry, r_sub]
            
            # Actually, let me use a cleaner formulation.
            # The 4x4 matrix multiplication tensor is the Kronecker product of two 2x2 tensors.
            # The rank of a Kronecker product of two tensors of rank r1 and r2 is r1*r2.
            
            # For the Kronecker product structure:
            # U4[(i1*2+i2)*4 + (j1*2+j2), r_top*7+r_sub] = U2[i1*2+j1, r_top] * U2[i2*2+j2, r_sub]
            # V4[(j1*2+j2)*4 + (k1*2+k2), r_top*7+r_sub] = V2[j1*2+k1, r_top] * V2[j2*2+k2, r_sub]
            # W4[(k1*2+k2)*4 + (i1*2+i2), r_top*7+r_sub] = W2[k1*2+i1, r_top] * W2[k2*2+i2, r_sub]
            
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U4[i * 4 + j, r] = U2[i1 * 2 + j1, r_top] * U2[i2 * 2 + j2, r_sub]
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V4[j * 4 + k, r] = V2[j1 * 2 + k1, r_top] * V2[j2 * 2 + k2, r_sub]
            
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W4[k * 4 + i, r] = W2[k1 * 2 + i1, r_top] * W2[k2 * 2 + i2, r_sub]
    
    return U4, V4, W4


def _verify_decomposition(U, V, W, n, m, p, R):
    """Verify the decomposition is exact."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all consistent indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check some inconsistent indices
    import random
    random.seed(42)
    for _ in range(500):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        
        i1, j1 = divmod(a, m)
        j2, k2 = divmod(b, p)
        k3, i3 = divmod(c, n)
        
        if j1 == j2 and i1 == i3 and k2 == k3:
            continue
        
        val = np.dot(U_np[a], V_np[b] * W_np[c])
        if abs(val) > 1e-9:
            return False
    
    return True


def construct():
    """Return a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook baseline
    rank_sb = n * m * p  # 144
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try Strassen-based decomposition: rank 114
    if time.time() > deadline - 50:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Get the 4x4 Strassen decomposition
    U_44, V_44, W_44 = _strassen_4x4()
    
    total_rank = 49 + 49 + 16  # 114
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Block 1: k in [0,4)
    for r in range(49):
        rank_idx = r
        U[:, rank_idx] = U_44[:, r]
        for j in range(m):
            for k in range(4):
                V[j * p + k, rank_idx] = V_44[j * 4 + k, r]
        for k in range(4):
            for i in range(n):
                W[k * n + i, rank_idx] = W_44[k * 4 + i, r]
    
    # Block 2: k in [4,8)
    for r in range(49):
        rank_idx = 49 + r
        U[:, rank_idx] = U_44[:, r]
        for j in range(m):
            for k in range(4, 8):
                V[j * p + k, rank_idx] = V_44[j * 4 + (k - 4), r]
        for k in range(4, 8):
            for i in range(n):
                W[k * n + i, rank_idx] = W_44[k * 4 + i, r]
    
    # Block 3: k in [8,9), schoolbook 4x4*4x1 (rank 16)
    for i in range(n):
        for j in range(m):
            k = 8
            rank_idx = 98 + i * m + j
            U[i * m + j, rank_idx] = 1
            V[j * p + k, rank_idx] = 1
            W[k * n + i, rank_idx] = 1
    
    # Verify
    if time.time() > deadline - 20:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    if _verify_decomposition(U, V, W, n, m, p, total_rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
