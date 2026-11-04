"""
Program for 5x5 Matrix Multiplication Tensor Decomposition.

Uses a hybrid approach:
- Strassen-based rank-49 decomposition for the 4x4x4 sub-tensor (indices 0..3 in each dimension).
- Schoolbook (rank-61) for the remaining terms involving index 4 in any position.
Total rank = 49 + 61 = 110 < 125.

Interface:
construct() -> {'U', 'V', 'W'}
U: shape (n*m, rank) = (25, 110)
V: shape (m*p, rank) = (25, 110)
W: shape (p*n, rank) = (25, 110)
Entries are integers.
"""

import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build Strassen 2x2x2 factors
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # U, V, W for 2x2x2 tensor (rank 7)
    # Indices: (i,j) for U, (j,k) for V, (k,i) for W
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen factors (well-known):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U matrix (4x7): rows (i,j) in order (0,0),(0,1),(1,0),(1,1)
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # A00: M1, M3, M5
        [0, 0, 0, 0, 1, 0, -1], # A01: M5, M7 (with -1 for A01-A11 in M6... wait)
        [0, 1, 0, 0, 0, -1, 0], # A10: M2, M6 (A21-A11, so A10 gets -1 for M6)
        [1, 1, 0, 1, 0, 0, 0],  # A11: M1, M2, M4
    ], dtype=object)
    
    # Let me be more careful with Strassen.
    # Standard Strassen:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B21 - B22)
    # M4 = A22 * (B11 - B21)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # U (rows are (i,j) pairs: 00, 01, 10, 11):
    # A00 appears in: M1(+1), M3(+1), M5(+1), M6(-1)
    # A01 appears in: M5(+1), M7(+1)
    # A10 appears in: M2(+1), M6(+1)
    # A11 appears in: M1(+1), M2(+1), M4(+1), M7(-1)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ], dtype=object)
    
    # V (rows are (j,k) pairs: 00, 01, 10, 11):
    # B00 appears in: M1(+1), M2(+1), M4(+1), M6(+1)
    # B01 appears in: M6(+1)
    # B10 appears in: M3(+1), M4(-1), M7(+1)
    # B11 appears in: M1(+1), M3(-1), M5(+1), M7(+1)
    
    V2 = np.array([
        [1, 1, 0, 1, 0, 1, 0],  # B00
        [0, 0, 0, 0, 0, 1, 0],  # B01
        [0, 0, 1, -1, 0, 0, 1], # B10
        [1, 0, -1, 0, 1, 0, 1], # B11
    ], dtype=object)
    
    # W (rows are (k,i) pairs: 00, 01, 10, 11):
    # C00 = M1 - M4 + M6 + M7
    # C01 = M5 + M7
    # C10 = M2 - M3 + M6
    # C11 = M1 + M3 - M5 + M6
    
    W2 = np.array([
        [1, 0, 0, -1, 0, 1, 1],  # C00 (k=0,i=0)
        [0, 0, 0, 0, 1, 0, 1],   # C01 (k=0,i=1)
        [0, 1, -1, 0, 0, 1, 0],  # C10 (k=1,i=0)
        [1, 0, 1, 0, -1, 0, 0],  # C11 (k=1,i=1)
    ], dtype=object)
    
    # Verify Strassen: check that U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] sums correctly
    # For each (i,j,k): sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] should be 1 if this is a valid term
    # Actually, the tensor decomposition means: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For matrix mult, T[i,j,k] = 1 if the term C[i,k] depends on A[i,j]*B[j,k], else 0
    # In standard MM: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (every combination appears exactly once in the sum)
    # Wait no: the tensor is T_{ijk} where the bilinear form is sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki} = 0
    # Or equivalently, the multiplication tensor has T_{ijk} = 1 for all i,j,k in {0..n-1}x{0..m-1}x{0..p-1}
    
    # Let's verify: for 2x2x2, every (i,j,k) in {0,1}^3 should give sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] for r in range(7))
                assert s == 1, f"Strassen verification failed at ({i},{j},{k}): sum={s}"
    
    # Now build 4x4x4 via tensor product of 2x2x2 with itself
    # 4x4 indices: i = i1*2 + i0, j = j1*2 + j0, k = k1*2 + k0
    # U4[i,j,r1,r0] = U2[i1,j1,r1] * U2[i0,j0,r0]
    # V4[j,k,r1,r0] = V2[j1,k1,r1] * V2[j0,k0,r0]
    # W4[k,i,r1,r0] = W2[k1,i1,r1] * W2[k0,i0,r0]
    
    # Flatten (r1, r0) into a single index r in 0..48
    U4 = np.zeros((16, 49), dtype=object)
    V4 = np.zeros((16, 49), dtype=object)
    W4 = np.zeros((16, 49), dtype=object)
    
    for i in range(4):
        i1, i0 = divmod(i, 2)
        for j in range(4):
            j1, j0 = divmod(j, 2)
            for r1 in range(7):
                for r0 in range(7):
                    r = r1 * 7 + r0
                    U4[i*4+j, r] = U2[i1*2+j1, r1] * U2[i0*2+j0, r0]
                    V4[j*4+k_, r] = V2[j1*2+k1, r1] * V2[j0*2+k0, r0] if False else V4[j*4+k_, r]
    
    # Let me redo this more carefully
    U4 = np.zeros((16, 49), dtype=object)
    V4 = np.zeros((16, 49), dtype=object)
    W4 = np.zeros((16, 49), dtype=object)
    
    for i in range(4):
        i1, i0 = divmod(i, 2)
        for j in range(4):
            j1, j0 = divmod(j, 2)
            for r1 in range(7):
                for r0 in range(7):
                    r = r1 * 7 + r0
                    U4[i*4+j, r] = U2[i1*2+j1, r1] * U2[i0*2+j0, r0]
    
    for j in range(4):
        j1, j0 = divmod(j, 2)
        for k in range(4):
            k1, k0 = divmod(k, 2)
            for r1 in range(7):
                for r0 in range(7):
                    r = r1 * 7 + r0
                    V4[j*4+k, r] = V2[j1*2+k1, r1] * V2[j0*2+k0, r0]
    
    for k in range(4):
        k1, k0 = divmod(k, 2)
        for i in range(4):
            i1, i0 = divmod(i, 2)
            for r1 in range(7):
                for r0 in range(7):
                    r = r1 * 7 + r0
                    W4[k*4+i, r] = W2[k1*2+i1, r1] * W2[k0*2+i0, r0]
    
    # Verify 4x4x4 Strassen product
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = sum(U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] for r in range(49))
                assert s == 1, f"4x4 Strassen verification failed at ({i},{j},{k}): sum={s}"
    
    # Now embed into 5x5x5
    # Total rank = 49 (for 0..3 sub-block) + 61 (schoolbook for terms involving index 4) = 110
    
    total_rank = 49 + 61
    U = np.zeros((25, total_rank), dtype=object)
    V = np.zeros((25, total_rank), dtype=object)
    W = np.zeros((25, total_rank), dtype=object)
    
    # Fill first 49 columns with 4x4 Strassen factors
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*5+j, r] = U4[i*4+j, r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*5+k, r] = V4[j*4+k, r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k*5+i, r] = W4[k*4+i, r]
    
    # Fill remaining 61 columns with schoolbook terms involving index 4
    col = 49
    for i in range(5):
        for j in range(5):
            for k in range(5):
                # Skip if all indices are in 0..3 (handled by Strassen)
                if i < 4 and j < 4 and k < 4:
                    continue
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*5+i, col] = 1
                col += 1
    
    assert col == total_rank, f"Column count mismatch: {col} vs {total_rank}"
    
    # Verify full 5x5x5 decomposition
    for i in range(5):
        for j in range(5):
            for k in range(5):
                s = sum(U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r] for r in range(total_rank))
                assert s == 1, f"5x5 verification failed at ({i},{j},{k}): sum={s}"
    
    # Convert to lists as expected by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
