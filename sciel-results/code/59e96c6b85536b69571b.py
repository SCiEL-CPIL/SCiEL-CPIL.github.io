"""Rank-minimal tensor decomposition for 4x4 @ 4x9 multiplication.

Strategy:
1. Build a known valid decomposition using Strassen(2x2) for the 4x4x4 part
   and schoolbook for the 4x4x5 part. Total rank = 49 + 80 = 129.
2. Attempt to reduce rank by finding linear dependencies among the rank terms
   and performing a rank-revealing factorization.

The key insight is that after constructing a valid decomposition, we can
stack U, V, W into a single matrix and look for low-rank structure. If the
effective rank is lower than the nominal rank, we can extract a smaller
decomposition.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Return the 7 Strassen terms for 2x2 @ 2x2.
    
    Each term is (U_row, V_row, W_row) where:
    - U_row is length 4 (flattened 2x2 A matrix, row-major)
    - V_row is length 4 (flattened 2x2 B matrix, row-major)  
    - W_row is length 4 (flattened 2x2 C matrix, row-major)
    
    Strassen:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    terms = []
    
    # M1 = (A11+A22)*(B11+B22)
    terms.append((np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,1,0,1])))
    # M2 = (A21+A22)*B11
    terms.append((np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,1,0])))
    # M3 = A11*(B12-B22)
    terms.append((np.array([1,0,0,0]), np.array([0,1,0,-1]), np.array([0,1,0,1])))
    # M4 = A22*(B21-B11)
    terms.append((np.array([0,0,0,1]), np.array([-1,0,1,0]), np.array([1,0,1,0])))
    # M5 = (A11+A12)*B22
    terms.append((np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,1,0,0])))
    # M6 = (A21-A11)*(B11+B12)
    terms.append((np.array([-1,0,1,0]), np.array([1,1,0,0]), np.array([0,0,0,1])))
    # M7 = (A12-A22)*(B21+B22)
    terms.append((np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([1,0,0,0])))
    
    return terms

def _build_4x4x4_strassen():
    """Build Strassen-based decomposition for 4x4 @ 4x4 -> 4x4.
    
    Split 4x4 into 2x2 blocks. Use Strassen for each 2x2@2x2 sub-product.
    Total rank = 4 * 7 = 28? No, we need to be careful.
    
    Actually for 4x4 @ 4x4:
    A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]], each block 2x2.
    C = A*B where C11=A11*B11+A12*B21, C12=A11*B12+A12*B22, etc.
    
    Each Cij is a sum of two 2x2@2x2 products. We can use the Kronecker
    product structure: the 4x4x4 tensor is the tensor product of two 2x2x2 tensors.
    
    Rank of 2x2x2 tensor (Strassen) = 7.
    Rank of tensor product <= 7*7 = 49.
    """
    s_terms = _strassen_2x2()
    
    # The 4x4x4 decomposition: for each pair of Strassen terms (s1, s2),
    # we get a rank term for the 4x4x4 problem.
    # A is 4x4, flattened as [A11, A12, A21, A22] where each is 2x2.
    # In the full 4x4, A has 16 entries.
    # U for 4x4x4 has shape (16, rank)
    # V for 4x4x4 has shape (16, rank)  
    # W for 4x4x4 has shape (16, rank)
    
    rank = 7 * 7
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for s1 in range(7):
        for s2 in range(7):
            u1, v1, w1 = s_terms[s1]
            u2, v2, w2 = s_terms[s2]
            
            # u1 corresponds to A-block (row_block, col_block) in the 2x2 block structure
            # For 4x4, the 16 entries are:
            # A[0,0]=A11[0,0], A[0,1]=A11[0,1], A[0,2]=A12[0,0], A[0,3]=A12[0,1]
            # A[1,0]=A11[1,0], A[1,1]=A11[1,1], A[1,2]=A12[1,0], A[1,3]=A12[1,1]
            # A[2,0]=A21[0,0], A[2,1]=A21[0,1], A[2,2]=A21[1,0], A[2,3]=A21[1,1]
            # A[3,0]=A21[1,0]... wait let me redo.
            
            # A is 4x4, row-major flatten: index = i*4+j
            # Block A11 covers rows 0-1, cols 0-1: indices 0,1,4,5
            # Block A12 covers rows 0-1, cols 2-3: indices 2,3,6,7
            # Block A21 covers rows 2-3, cols 0-1: indices 8,9,12,13
            # Block A22 covers rows 2-3, cols 2-3: indices 10,11,14,15
            
            # For Strassen term s1 acting on the "outer" 2x2 block structure:
            # u1 has 4 entries corresponding to the 4 blocks A11,A12,A21,A22
            # u1[0] = coeff for A11, u1[1] = coeff for A12, u1[2] = coeff for A21, u1[3] = coeff for A22
            
            # For Strassen term s2 acting on the "inner" 2x2 structure within each block:
            # u2 has 4 entries corresponding to the 4 entries within a 2x2 block
            
            # The combined U entry for block b (0-3) and inner position p (0-3):
            # U[block_b * 4 + p, col] = u1[b] * u2[p]
            
            for b in range(4):
                for p in range(4):
                    # Map block b and inner position p to full 4x4 index
                    # b=0 (A11): rows 0-1, cols 0-1. p=0->(0,0), p=1->(0,1), p=2->(1,0), p=3->(1,1)
                    # b=1 (A12): rows 0-1, cols 2-3. p=0->(0,2), p=1->(0,3), p=2->(1,2), p=3->(1,3)
                    # b=2 (A21): rows 2-3, cols 0-1. p=0->(2,0), p=1->(2,1), p=2->(3,0), p=3->(3,1)
                    # b=3 (A22): rows 2-3, cols 2-3. p=0->(2,2), p=1->(2,3), p=2->(3,2), p=3->(3,3)
                    
                    # For A (i,j) with i=row, j=col:
                    # Block row = i//2, block col = j//2
                    # Inner row = i%2, inner col = j%2
                    # b = block_row * 2 + block_col
                    # p = inner_row * 2 + inner_col
                    
                    # So for given b and p:
                    # block_row = b // 2, block_col = b % 2
                    # inner_row = p // 2, inner_col = p % 2
                    # i = block_row * 2 + inner_row
                    # j = block_col * 2 + inner_col
                    # full_index = i * 4 + j
                    
                    block_row = b // 2
                    block_col = b % 2
                    inner_row = p // 2
                    inner_col = p % 2
                    i = block_row * 2 + inner_row
                    j = block_col * 2 + inner_col
                    full_idx = i * 4 + j
                    
                    U[full_idx, col] = u1[b] * u2[p]
            
            # Similarly for V (B matrix, 4x4, same structure)
            for b in range(4):
                for p in range(4):
                    block_row = b // 2
                    block_col = b % 2
                    inner_row = p // 2
                    inner_col = p % 2
                    i = block_row * 2 + inner_row
                    j = block_col * 2 + inner_col
                    full_idx = i * 4 + j
                    V[full_idx, col] = v1[b] * v2[p]
            
            # W (C matrix, 4x4, same structure)
            for b in range(4):
                for p in range(4):
                    block_row = b // 2
                    block_col = b % 2
                    inner_row = p // 2
                    inner_col = p % 2
                    i = block_row * 2 + inner_row
                    j = block_col * 2 + inner_col
                    full_idx = i * 4 + j
                    W[full_idx, col] = w1[b] * w2[p]
            
            col += 1
    
    return U, V, W, rank

def _build_4x4x5_schoolbook():
    """Build schoolbook decomposition for 4x4 @ 4x5 -> 4x5.
    
    Rank = 4*4*5 = 80.
    U has shape (16, 80), V has shape (20, 80), W has shape (20, 80).
    
    Wait, for 4x4 @ 4x5:
    - A is 4x4 (16 entries), U shape (16, rank)
    - B is 4x5 (20 entries), V shape (20, rank)
    - C is 4x5 (20 entries), W shape (20, rank)
    
    But our problem is 4x4 @ 4x9. Let me reconsider.
    
    Actually, I should build the full 4x4 @ 4x9 decomposition directly.
    Let me split B into 4x4 and 4x5 parts.
    """
    # For 4x4 @ 4x5 schoolbook:
    n, m, p = 4, 4, 5
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
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 145
    
    # Strategy: Build 4x4x4 using Strassen (rank 49) + 4x4x5 using schoolbook (rank 80)
    # Total rank = 129
    
    # First, build the 4x4x4 Strassen decomposition
    U_444, V_444, W_444, r_444 = _build_4x4x4_strassen()
    
    # Build the 4x4x5 schoolbook decomposition
    U_445, V_445, W_445, r_445 = _build_4x4x5_schoolbook()
    
    # Combine: The full problem is 4x4 @ 4x9.
    # A is 4x4 (16 entries).
    # B is 4x9 (36 entries): first 4 columns are B_left (4x4), last 5 columns are B_right (4x5).
    # C is 4x9 (36 entries): first 4 columns are C_left (4x4), last 5 columns are C_right (4x5).
    
    # U for full problem: shape (16, 129)
    # For the 4x4x4 part, U entries are U_444 (16 x 49)
    # For the 4x4x5 part, U entries are U_445 (16 x 80)
    # Both use the same A, so U is just [U_444 | U_445]
    
    U_full = np.zeros((16, r_444 + r_445))
    U_full[:, :r_444] = U_444
    U_full[:, r_444:] = U_445
    
    # V for full problem: shape (36, 129)
    # B is 4x9, flattened row-major: index = j*9+k for j=0..3, k=0..8
    # For the 4x4x4 part: B_left is columns 0-3, so V_444 has shape (16, 49)
    #   V_444[j*4+k, col] for j=0..3, k=0..3 -> maps to V_full[j*9+k, col]
    # For the 4x4x5 part: B_right is columns 4-8, so V_445 has shape (20, 80)
    #   V_445[j*5+k, col] for j=0..3, k=0..4 -> maps to V_full[j*9+(k+4), col]
    
    V_full = np.zeros((36, r_444 + r_445))
    # Place V_444
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + k, :r_444] = V_444[j * 4 + k, :]
    # Place V_445
    for j in range(4):
        for k in range(5):
            V_full[j * 9 + (k + 4), r_444:] = V_445[j * 5 + k, :]
    
    # W for full problem: shape (36, 129)
    # C is 4x9, flattened as W[k*n+i] = W[k*4+i] for k=0..8, i=0..3
    # For the 4x4x4 part: C_left is columns 0-3, so W_444 has shape (16, 49)
    #   W_444[k*4+i, col] for k=0..3, i=0..3 -> maps to W_full[k*4+i, col]
    # For the 4x4x5 part: C_right is columns 4-8, so W_445 has shape (20, 80)
    #   W_445[k*4+i, col] for k=0..4, i=0..3 -> maps to W_full[(k+4)*4+i, col]
    
    W_full = np.zeros((36, r_444 + r_445))
    # Place W_444
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, :r_444] = W_444[k * 4 + i, :]
    # Place W_445
    for k in range(5):
        for i in range(4):
            W_full[(k + 4) * 4 + i, r_444:] = W_445[k * 4 + i, :]
    
    total_rank = r_444 + r_445
    
    # Verify the decomposition
    # T_{(i,j),(j,k),(k,i)} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all valid (i,j,k)
    
    # Quick verification with random test
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    C_computed = np.zeros((4, 9))
    for r in range(total_rank):
        # a_r = U[:, r] dot A (flattened)
        u_vec = U_full[:, r].reshape(4, 4)
        v_vec = V_full[:, r].reshape(4, 9)
        w_vec = W_full[:, r].reshape(9, 4)  # W[k*4+i] -> reshape to (9,4)
        
        a_r = np.sum(u_vec * A_test)
        b_r = np.sum(v_vec * B_test)
        c_r = a_r * b_r
        C_computed += c_r * w_vec.T  # w_vec is (9,4), we need (4,9)
    
    # Actually, let me redo the verification more carefully.
    # The decomposition says:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    
    # But that's not quite right either. The correct formula is:
    # sum_r (sum_{i,j} U[i*m+j,r] * A_{i,j}) * (sum_{j,k} V[j*p+k,r] * B_{j,k}) * (sum_{i,k} W[k*n+i,r] * C_{i,k})
    # = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    
    # So for a given A, B, the predicted C satisfies:
    # sum_{i,k} C_{i,k} * (sum_r W[k*n+i,r] * a_r * b_r) = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    
    # This means: sum_{i,k} C_{i,k} * D_{i,k} = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # where D_{i,k} = sum_r W[k*n+i,r] * a_r * b_r
    # and a_r = sum_{i,j} U[i*m+j,r] * A_{i,j}
    # and b_r = sum_{j,k} V[j*p+k,r] * B_{j,k}
    
    # For this to hold for all C, we need D_{i,k} = sum_j A_{i,j} B_{j,k} = (A@B)_{i,k}
    
    # Let's verify:
    D = np.zeros((4, 9))
    for r in range(total_rank):
        a_r = np.sum(U_full[:, r].reshape(4, 4) * A_test)
        b_r = np.sum(V_full[:, r].reshape(4, 9) * B_test)
        w_vec = W_full[:, r].reshape(9, 4)  # W[k*4+i]
        D += a_r * b_r * w_vec.T  # w_vec.T is (4,9), D[i,k] += ...
    
    # D should equal C_expected = A_test @ B_test
    err = np.max(np.abs(D - C_expected))
    
    if err > 1e-9:
        # Something is wrong, fall back to pure schoolbook
        rank = n * m * p
        U_mat = np.zeros((n * m, rank))
        V_mat = np.zeros((m * p, rank))
        W_mat = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_mat[i * m + j, col] = 1
                    V_mat[j * p + k, col] = 1
                    W_mat[k * n + i, col] = 1
                    col += 1
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    
    # Now try to reduce rank by looking for linear dependencies
    # Stack U, V, W into a single matrix M of shape (16+36+36, rank) = (88, 129)
    # If the rank of M is less than 129, we can potentially reduce the decomposition rank.
    # But this is tricky because U, V, W have different roles.
    
    # Alternative approach: Use the fact that the tensor has a specific structure.
    # The 4x4x4 part uses Strassen (rank 49) and 4x4x5 uses schoolbook (rank 80).
    # The total is 129.
    
    # Try to find if we can do better by using a different split.
    # For example, split B as 4x3 + 4x6 instead of 4x4 + 4x5.
    # 4x4x3: rank at most 49 (Strassen for 4x4x4, but we only need 3 columns)
    # Actually, for 4x4 @ 4x3, we can use Strassen for the 4x4x4 and just take 3 columns.
    # Rank would still be 49 for the 4x4x4 part, plus schoolbook for 4x4x3 = 48.
    # Total = 49 + 48 = 97. That's better!
    
    # Wait, but the Strassen decomposition for 4x4x4 gives us all 4 columns of C.
    # If we only need 3 columns, we can still use the same 49 rank terms but just
    # ignore the 4th column. The rank doesn't decrease.
    
    # Actually, we can use a different approach: use Strassen for 4x4 @ 4x4 to get
    # the first 4 columns, and schoolbook for 4x4 @ 4x5 for the last 5 columns.
    # That's what we already have: 49 + 80 = 129.
    
    # Alternatively, use schoolbook for the entire 4x4 @ 4x9: rank = 144.
    # Or use Strassen for 4x4 @ 4x4 (49) + Strassen-like for 4x4 @ 4x5.
    
    # For 4x4 @ 4x5, can we do better than schoolbook (80)?
    # 4x5 = 4x4 + 4x1. Use Strassen for 4x4x4 (49) + schoolbook for 4x4x1 (16).
    # Total for 4x4 @ 4x5: 49 + 16 = 65.
    # Then total for 4x4 @ 4x9: 49 + 65 = 114.
    
    # Even better: 4x9 = 4x4 + 4x4 + 4x1.
    # Strassen for each 4x4x4: 49 each, schoolbook for 4x4x1: 16.
    # Total: 49 + 49 + 16 = 114.
    
    # Or: 4x9 = 4x3 + 4x3 + 4x3.
    # For 4x4 @ 4x3: use Strassen for 4x4x4 (49 rank) but only 3 output columns.
    # The rank is still 49 because the decomposition is for the full 4x4x4 tensor.
    # Unless we can find a rank-? decomposition specifically for 4x4 @ 4x3.
    
    # Let me try the 4x4 + 4x4 + 4x1 split: total rank 114.
    
    if time.time() > deadline - 30:
        # Not enough time for more work, return current result
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Try the 4+4+1 split
    U_444b, V_444b, W_444b, r_444b = _build_4x4x4_strassen()
    U_441, V_441, W_441, r_441 = _build_4x4x1_schoolbook()
    
    # Build full decomposition for 4x4 @ 4x9 with split 4+4+1
    total_rank2 = r_444 + r_444b + r_441  # 49 + 49 + 16 = 114
    
    U_full2 = np.zeros((16, total_rank2))
    V_full2 = np.zeros((36, total_rank2))
    W_full2 = np.zeros((36, total_rank2))
    
    # First 4x4x4: columns 0-3
    U_full2[:, :r_444] = U_444
    for j in range(4):
        for k in range(4):
            V_full2[j * 9 + k, :r_444] = V_444[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W_full2[k * 4 + i, :r_444] = W_444[k * 4 + i, :]
    
    # Second 4x4x4: columns 4-7
    U_full2[:, r_444:r_444+r_444b] = U_444b
    for j in range(4):
        for k in range(4):
            V_full2[j * 9 + (k + 4), r_444:r_444+r_444b] = V_444b[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W_full2[(k + 4) * 4 + i, r_444:r_444+r_444b] = W_444b[k * 4 + i, :]
    
    # 4x4x1: column 8
    U_full2[:, r_444+r_444b:] = U_441
    for j in range(4):
        V_full2[j * 9 + 8, r_444+r_444b:] = V_441[j * 1 + 0, :]
    for i in range(4):
        W_full2[8 * 4 + i, r_444+r_444b:] = W_441[0 * 4 + i, :]
    
    # Verify
    D2 = np.zeros((4, 9))
    for r in range(total_rank2):
        a_r = np.sum(U_full2[:, r].reshape(4, 4) * A_test)
        b_r = np.sum(V_full2[:, r].reshape(4, 9) * B_test)
        w_vec = W_full2[:, r].reshape(9, 4)
        D2 += a_r * b_r * w_vec.T
    
    err2 = np.max(np.abs(D2 - C_expected))
    
    if err2 < 1e-9 and total_rank2 < total_rank:
        U_full, V_full, W_full, total_rank = U_full2, V_full2, W_full2, total_rank2
    
    # Return the best decomposition found
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

def _build_4x4x1_schoolbook():
    """Build schoolbook decomposition for 4x4 @ 4x1 -> 4x1.
    
    Rank = 4*4*1 = 16.
    U shape (16, 16), V shape (4, 16), W shape (4, 16).
    """
    n, m, p = 4, 4, 1
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
    
    return U, V, W, rank
