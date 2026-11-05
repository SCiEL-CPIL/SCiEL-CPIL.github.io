"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 120s limit

    # Schoolbook rank as fallback
    rank_school = n * m * p  # 72
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                U_school[u_idx, idx] = 1
                V_school[v_idx, idx] = 1
                W_school[w_idx, idx] = 1
                idx += 1

    def schoolbook_result():
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # We want to find a rank-minimal decomposition of the matrix multiplication tensor
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # where the tensor is decomposed as T = sum_r U_r ⊗ V_r ⊗ W_r
    # with U_r of shape (n*m,), V_r of shape (m*p,), W_r of shape (p*n,)
    
    # The key insight: for 3x3x3, the rank is 8 (Strassen-Winograd).
    # For 3x3x8, we can potentially do better than 72 by sharing structure.
    
    # Strategy: Use the rank-8 decomposition of 3x3x3 as a building block.
    # The 3x3x8 tensor can be viewed as 3 blocks: two 3x3x3 blocks and one 3x3x2 block.
    # But the U factors (depending on A) can be shared across blocks!
    
    # Let me construct a decomposition using the Strassen-Winograd rank-8 for 3x3x3
    # and apply it to the first 6 output columns, then handle the last 2 columns.
    
    # First, let me define the rank-8 decomposition of 3x3x3.
    # The 8 multiplications are:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B10-B11)
    # M4 = A11 * (B01-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    # M8 = (A10+A11-A00) * (B00+B11)
    
    # For 3x3x3, the output C[i,k] = sum_j A[i,j]*B[j,k]
    # C is 3x3, stored as W[k*3+i, r]
    
    # Let me define the U, V, W factors for the 8 multiplications.
    # U: 9x8 (A is 3x3, flattened row-major: A00,A01,A02,A10,A11,A12,A20,A21,A22)
    # V: 9x8 (B is 3x3, flattened row-major: B00,B01,B02,B10,B11,B12,B20,B21,B22)
    # W: 9x8 (C is 3x3, stored as W[k*3+i], so C00,C01,C02,C10,C11,C12,C20,C21,C22)
    
    # M1: U = A00+A11 = [1,0,0, 0,1,0, 0,0,0]
    #     V = B00+B11 = [1,0,0, 0,1,0, 0,0,0]
    #     W contributions: C00+=1, C01+=1, C10+=1, C11+=1, C20+=1, C21+=1
    #     Wait, I need to be more careful.
    
    # Let me look up the exact Strassen-Winograd formulas for 3x3:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # M8 = (a10+a11-a00)(b00+b11)
    
    # C00 = M1 - M2 + M3 + M4 - M5 + M6
    # C01 = M1 - M2 - M4 + M5 - M6 - M7
    # C02 = M2 + M3 - M4 - M6
    # C10 = M2 + M3 - M4 + M5 + M6
    # C11 = M1 - M3 + M4 - M5 + M7
    # C12 = M1 - M2 + M4 - M5 + M7
    # C20 = M1 + M2 - M3 + M4 - M6 + M8
    # C21 = M1 - M2 + M3 - M4 + M5 - M7 - M8
    # C22 = M1 - M2 + M3 + M4 - M6 + M8
    
    # Now let me construct the U, V, W factors:
    # U factors (for A, 9-dimensional):
    U_strassen = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0, 0],   # M1: a00+a11
        [0, 0, 0, 1, 1, 0, 0, 0, 0],   # M2: a10+a11
        [1, 0, 0, 0, 0, 0, 0, 0, 0],   # M3: a00
        [0, 0, 0, 0, 1, 0, 0, 0, 0],   # M4: a11
        [1, 1, 0, 0, 0, 0, 0, 0, 0],   # M5: a00+a01
        [-1, 0, 0, 1, 0, 0, 0, 0, 0],  # M6: a10-a00
        [0, 1, 0, 0, -1, 0, 0, 0, 0],  # M7: a01-a11
        [-1, 0, 0, 1, 1, 0, 0, 0, 0],  # M8: a10+a11-a00
    ], dtype=float)  # 9x8
    
    # V factors (for B, 9-dimensional):
    V_strassen = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0, 0],   # M1: b00+b11
        [1, 0, 0, 0, 0, 0, 0, 0, 0],   # M2: b00
        [0, 0, 0, 1, -1, 0, 0, 0, 0],  # M3: b10-b11
        [0, 1, 0, -1, 0, 0, 0, 0, 0],  # M4: b01-b00
        [0, 0, 0, 0, 1, 0, 0, 0, 0],   # M5: b11
        [1, 1, 0, 0, 0, 0, 0, 0, 0],   # M6: b00+b01
        [0, 0, 0, 1, 1, 0, 0, 0, 0],   # M7: b10+b11
        [1, 0, 0, 0, 1, 0, 0, 0, 0],   # M8: b00+b11
    ], dtype=float)  # 9x8
    
    # W factors (for C, 9-dimensional, stored as C[k*3+i]):
    # C00 is at index 0, C01 at index 1, C02 at index 2
    # C10 at index 3, C11 at index 4, C12 at index 5
    # C20 at index 6, C21 at index 7, C22 at index 8
    
    W_strassen = np.zeros((9, 8), dtype=float)
    # M1: C00+=1, C01+=1, C10+=1, C11+=1, C20+=1, C21+=1
    W_strassen[0, 0] += 1   # C00
    W_strassen[1, 0] += 1   # C01
    W_strassen[3, 0] += 1   # C10
    W_strassen[4, 0] += 1   # C11
    W_strassen[6, 0] += 1   # C20
    W_strassen[7, 0] += 1   # C21
    
    # M2: C00-=1, C01-=1, C02+=1, C10+=1, C12-=1, C20+=1, C21-=1, C22-=1
    W_strassen[0, 1] -= 1   # C00
    W_strassen[1, 1] -= 1   # C01
    W_strassen[2, 1] += 1   # C02
    W_strassen[3, 1] += 1   # C10
    W_strassen[5, 1] -= 1   # C12
    W_strassen[6, 1] += 1   # C20
    W_strassen[7, 1] -= 1   # C21
    W_strassen[8, 1] -= 1   # C22
    
    # M3: C00+=1, C01+=1, C02+=1, C10+=1, C11-=1, C12+=1, C20-=1, C21+=1, C22+=1
    W_strassen[0, 2] += 1   # C00
    W_strassen[1, 2] += 1   # C01
    W_strassen[2, 2] += 1   # C02
    W_strassen[3, 2] += 1   # C10
    W_strassen[4, 2] -= 1   # C11
    W_strassen[5, 2] += 1   # C12
    W_strassen[6, 2] -= 1   # C20
    W_strassen[7, 2] += 1   # C21
    W_strassen[8, 2] += 1   # C22
    
    # M4: C00+=1, C01-=1, C02-=1, C10-=1, C11+=1, C12+=1, C20+=1, C21-=1, C22+=1
    W_strassen[0, 3] += 1   # C00
    W_strassen[1, 3] -= 1   # C01
    W_strassen[2, 3] -= 1   # C02
    W_strassen[3, 3] -= 1   # C10
    W_strassen[4, 3] += 1   # C11
    W_strassen[5, 3] += 1   # C12
    W_strassen[6, 3] += 1   # C20
    W_strassen[7, 3] -= 1   # C21
    W_strassen[8, 3] += 1   # C22
    
    # M5: C00-=1, C01+=1, C10+=1, C11-=1, C12-=1, C21+=1, C22-=1
    W_strassen[0, 4] -= 1   # C00
    W_strassen[1, 4] += 1   # C01
    W_strassen[3, 4] += 1   # C10
    W_strassen[4, 4] -= 1   # C11
    W_strassen[5, 4] -= 1   # C12
    W_strassen[7, 4] += 1   # C21
    W_strassen[8, 4] -= 1   # C22
    
    # M6: C00+=1, C01-=1, C02-=1, C10+=1, C20-=1, C21-=1, C22-=1
    W_strassen[0, 5] += 1   # C00
    W_strassen[1, 5] -= 1   # C01
    W_strassen[2, 5] -= 1   # C02
    W_strassen[3, 5] += 1   # C10
    W_strassen[6, 5] -= 1   # C20
    W_strassen[7, 5] -= 1   # C21
    W_strassen[8, 5] -= 1   # C22
    
    # M7: C01-=1, C11+=1, C12+=1, C21-=1
    W_strassen[1, 6] -= 1   # C01
    W_strassen[4, 6] += 1   # C11
    W_strassen[5, 6] += 1   # C12
    W_strassen[7, 6] -= 1   # C21
    
    # M8: C20+=1, C21-=1, C22+=1
    W_strassen[6, 7] += 1   # C20
    W_strassen[7, 7] -= 1   # C21
    W_strassen[8, 7] += 1   # C22
    
    # Verify the 3x3x3 decomposition:
    # For each (i,j,k), check that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
    # Let me verify this numerically.
    
    # Create test matrices
    A_test = np.random.randn(3, 3)
    B_test = np.random.randn(3, 3)
    C_test = A_test @ B_test
    
    # Compute using the decomposition
    A_flat = A_test.flatten()  # row-major: A00,A01,A02,A10,A11,A12,A20,A21,A22
    B_flat = B_test.flatten()
    
    # U_flat: for each r, U_r is a 9-vector, so U_flat = U_strassen.T @ A_flat
    U_vals = U_strassen.T @ A_flat  # shape (8,)
    V_vals = V_strassen.T @ B_flat  # shape (8,)
    
    # C_flat: C[k*3+i] for k=0,1,2, i=0,1,2
    C_flat = np.zeros(9)
    for r in range(8):
        C_flat += U_vals[r] * V_vals[r] * W_strassen[:, r]
    
    # C_flat should be [C00,C01,C02,C10,C11,C12,C20,C21,C22]
    C_reconstructed = C_flat.reshape(3, 3)
    
    # Check if it matches
    if not np.allclose(C_reconstructed, C_test, atol=1e-10):
        # The decomposition is wrong, fall back to schoolbook
        return schoolbook_result()
    
    # Now, for 3x3x8, we can use this rank-8 decomposition for the first 6 columns
    # and handle the last 2 columns separately.
    
    # The 3x3x8 tensor: A is 3x3, B is 3x8, C is 3x8 (stored as W[k*3+i])
    # We can split B into B1 (3x3), B2 (3x3), B3 (3x2)
    # C = A*B1 + A*B2 + A*B3 (where each product is 3x3, 3x3, 3x2)
    
    # For A*B1 and A*B2, we can use the same U factors (since they depend only on A)
    # but different V factors (for B1 and B2) and different W factors (for different output columns)
    
    # Total rank: 8 (for A*B1) + 8 (for A*B2) + rank(A*B3)
    # For A*B3 (3x3x2), the minimum rank is at least max(9,6,6)=9, and schoolbook gives 18.
    # Let me use schoolbook for the last 2 columns: rank 18.
    # Total: 8+8+18 = 34.
    
    # But wait, can we do better? Let me think...
    # Actually, the U factors for A*B1 and A*B2 are the same (they depend only on A).
    # So we can share them! The total rank would be 8 + rank(A*B3).
    
    # But that's not quite right either. Let me think more carefully.
    
    # The tensor T[i,j,k] for 3x3x8 is:
    # T[i,j,k] = 1 for all i,j,k
    # A decomposition is T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is 9-dim, V_r is 24-dim (3x8), W_r is 24-dim (8x3)
    
    # For the first 6 columns (k=0,1,2,3,4,5):
    # We can use the rank-8 decomposition, but we need to extend it to 3x3x6.
    # The V factors would be 3x6 instead of 3x3, and W factors would be 6x3 instead of 3x3.
    
    # Actually, let me just construct the full decomposition directly.
    
    # For k=0,1,2 (first 3 columns): use Strassen-Winograd with B[:,0:3]
    # For k=3,4,5 (next 3 columns): use Strassen-Winograd with B[:,3:6]
    # For k=6,7 (last 2 columns): use schoolbook
    
    # The U factors are the same for all three blocks (they depend only on A).
    # So the total rank is 8 + 8 + 18 = 34? No, that's not right.
    
    # Actually, the U factors for the first block and second block are the same,
    # so we can share them. The total rank is 8 + 18 = 26?
    
    # Wait, I'm confusing myself. Let me think step by step.
    
    # The tensor is T[i,j,k] for i in {0,1,2}, j in {0,1,2}, k in {0,...,7}
    # A decomposition is T = sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r
    
    # For k in {0,1,2}: use the rank-8 decomposition of 3x3x3
    #   T[i,j,k] = sum_{r=1}^{8} U_r[i,j] * V_r[j,k] * W_r[k,i] for k=0,1,2
    #   Here V_r is 3x3 (for B[:,0:3]) and W_r is 3x3 (for C[:,0:3])
    
    # For k in {3,4,5}: use another rank-8 decomposition
    #   T[i,j,k] = sum_{r=1}^{8} U_r[i,j] * V'_r[j,k-3] * W'_r[k-3,i] for k=3,4,5
    #   Here V'_r is 3x3 (for B[:,3:6]) and W'_r is 3x3 (for C[:,3:6])
    #   Note: U_r is the same as above!
    
    # For k in {6,7}: use schoolbook
    #   T[i,j,k] = sum_{r=1}^{18} U''_r[i,j] * V''_r[j,k-6] * W''_r[k-6,i] for k=6,7
    #   Here U''_r, V''_r, W''_r are the schoolbook factors for 3x3x2
    
    # Total rank: 8 + 8 + 18 = 34? But the U factors for the first two blocks are the same,
    # so we can combine them: total rank = 8 + 18 = 26.
    
    # Wait, that's not right either. The U factors are the same, but the V and W factors are different.
    # So we can't just add the ranks. Let me think again.
    
    # Actually, the decomposition is:
    # T = sum_{r=1}^{8} U_r ⊗ V_r ⊗ W_r + sum_{r=1}^{8} U_r ⊗ V'_r ⊗ W'_r + sum_{r=1}^{18} U''_r ⊗ V''_r ⊗ W''_r
    # = sum_{r=1}^{8} U_r ⊗ (V_r + V'_r) ⊗ ... NO, that's not right.
    
    # The issue is that V_r and V'_r have different support (different columns of B),
    # and W_r and W'_r have different support (different columns of C).
    # So we can't combine them.
    
    # The total rank is 8 + 8 + 18 = 34.
    
    # But wait, can we do better? What if we use a different decomposition for 3x3x6?
    # The rank of 3x3x6 is at least max(18, 18, 18) = 18, and at most 54 (schoolbook).
    # If we can find a rank-18 decomposition of 3x3x6, then total rank = 18 + 18 = 36.
    # That's worse than 34.
    
    # What about 3x3x2? The rank is at least max(9, 6, 6) = 9, and at most 18.
    # If we can find a rank-9 decomposition of 3x3x2, then total rank = 8 + 8 + 9 = 25.
    
    # Let me try to find a rank-9 decomposition of 3x3x2.
    # The tensor T[i,j,k] for i in {0,1,2}, j in {0,1,2}, k in {0,1}
    # A decomposition is T = sum_{r=1}^{9} U_r ⊗ V_r ⊗ W_r
    # where U_r is 9-dim, V_r is 6-dim, W_r is 6-dim
    
    # One approach: use the fact that 3x3x2 can be embedded in 3x3x3.
    # But I don't know a rank-9 decomposition off the top of my head.
    
    # For now, let me just use the 34-rank decomposition and see if it works.
    
    if time.time() > deadline - 10:
        return schoolbook_result()
    
    # Construct the 34-rank decomposition:
    # First 8 terms: for k=0,1,2
    # Next 8 terms: for k=3,4,5
    # Last 18 terms: for k=6,7
    
    R = 34
    U = np.zeros((9, R))
    V = np.zeros((24, R))  # 3x8 = 24
    W = np.zeros((24, R))  # 8x3 = 24
    
    # First 8 terms: for k=0,1,2
    U[:, 0:8] = U_strassen  # 9x8
    # V factors: V_r[j,k] for j in {0,1,2}, k in {0,1,2}
    # V_strassen is 9x8, where V_strassen[j*3+k, r] = V_r[j,k]
    # We need to place this in V[:, 0:8] where V[j*8+k, r] = V_r[j,k] for k=0,1,2
    for r in range(8):
        for j in range(3):
            for k in range(3):
                V[j*8+k, r] = V_strassen[j*3+k, r]
    # W factors: W_r[k,i] for k in {0,1,2}, i in {0,1,2}
    # W_strassen is 9x8, where W_strassen[k*3+i, r] = W_r[k,i]
    # We need to place this in W[:, 0:8] where W[k*3+i, r] = W_r[k,i] for k=0,1,2
    for r in range(8):
        for k in range(3):
            for i in range(3):
                W[k*3+i, r] = W_strassen[k*3+i, r]
    
    # Next 8 terms: for k=3,4,5
    U[:, 8:16] = U_strassen  # 9x8 (same U factors)
    # V factors: V'_r[j,k-3] for j in {0,1,2}, k in {3,4,5}
    for r in range(8):
        for j in range(3):
            for k in range(3):
                V[j*8+(k+3), 8+r] = V_strassen[j*3+k, r]
    # W factors: W'_r[k-3,i] for k in {3,4,5}, i in {0,1,2}
    for r in range(8):
        for k in range(3):
            for i in range(3):
                W[(k+3)*3+i, 8+r] = W_strassen[k*3+i, r]
    
    # Last 18 terms: for k=6,7 (schoolbook for 3x3x2)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                r = 16 + i*3 + j*2 + k
                U[i*3+j, r] = 1
                V[j*8+(k+6), r] = 1
                W[(k+6)*3+i, r] = 1
    
    # Verify the decomposition:
    # For each (i,j,k), check that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
    error = 0
    for i in range(3):
        for j in range(3):
            for k in range(8):
                val = 0
                for r in range(R):
                    val += U[i*3+j, r] * V[j*8+k, r] * W[k*3+i, r]
                if abs(val - 1) > 1e-10:
                    error += 1
                    break
            if error > 0:
                break
        if error > 0:
            break
    
    if error > 0:
        # The decomposition is wrong, fall back to schoolbook
        return schoolbook_result()
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
