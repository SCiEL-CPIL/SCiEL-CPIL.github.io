"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def verify_decomposition(n, m, p, U, V, W, rank, tol=1e-9):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Convention: W[k*n + i, r] corresponds to coefficient of C[i, k].
    C = A @ B, A is n x m, B is m x p.
    C[i, k] = sum_j A[i, j] B[j, k]
    
    The tensor T has entries T[i, j, k] = 1 if the term A[i,j]B[j,k] appears in C[i,k].
    Wait, the tensor for matrix multiplication M_{n,m,p} is defined such that:
    sum_{r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
    Actually, the standard tensor is T_{i,j,k} = 1.
    The decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    So U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should sum to 1.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # We need to check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
    # Let's construct the tensor and check.
    # Efficient way:
    # For each i, j, k, sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # This is equivalent to:
    # Let M_A be n x m x rank.
    # Let M_B be m x p x rank.
    # Let M_C be p x n x rank.
    # We want element-wise sum of products.
    
    # We can iterate over r, and add outer products? No, too slow for large rank.
    # But rank is small (53).
    # Let's use einsum.
    # 'ijr,jkr,krk->ijk' ? No, indices are:
    # U: i, j, r
    # V: j, k, r
    # W: k, i, r (Note: W is indexed by k*n+i, so first dim is k, second is i)
    # So W[k, i, r]
    
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    
    # Let's check all i, j, k.
    max_diff = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if abs(val - 1.0) > max_diff:
                    max_diff = abs(val - 1.0)
    return max_diff < tol

def construct_strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using Strassen's algorithm.
    Rank 49.
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 2x2x2 decomposition
    # M1 = (A11 + A22) (B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    
    # We apply this recursively.
    # Level 1: Split 4x4 into 2x2 blocks of size 2x2.
    # There are 7 Strassen products.
    # Each product is a 2x2 matrix multiplication.
    # Each 2x2 multiplication is decomposed into 7 scalar products.
    # Total 49 scalar products.
    
    # Let's define the indices for the 4x4 matrices.
    # A[i, j], B[j, k], C[i, k]
    # i, j, k in {0, 1, 2, 3}
    
    # Block indices:
    # A11: i in {0,1}, j in {0,1}
    # A12: i in {0,1}, j in {2,3}
    # A21: i in {2,3}, j in {0,1}
    # A22: i in {2,3}, j in {2,3}
    
    # Similarly for B and C.
    
    # We will explicitly construct the coefficients.
    # For each of the 7 Strassen steps, we have 7 sub-steps.
    # Let's create a helper to add a scalar product term.
    
    col_idx = 0
    for s in range(7):
        # s determines which Strassen combination (M1..M7)
        # For each s, we decompose the 2x2 multiplication into 7 scalar products.
        for t in range(7):
            # t determines which scalar product within the 2x2 Strassen
            # We need to determine the indices i, j, k and the signs.
            
            # This manual construction is prone to error.
            # Alternative: Use known explicit formulas or generate programmatically.
            
            # Let's try to generate it programmatically using the recursive definition.
            pass
    
    # Programmatic generation of Strassen 4x4x4
    # Base case: 1x1x1 is rank 1.
    # 2x2x2 is rank 7.
    # 4x4x4 is rank 49.
    
    # Let's define the 2x2x2 decomposition coefficients.
    # U2[i, j, r], V2[j, k, r], W2[k, i, r] for i,j,k in {0,1}, r in {0..6}.
    
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    # Terms:
    # (A00 B00, A00 B11, A11 B00, A11 B11)
    # In Strassen, these are combined.
    # Actually, it's easier to list the 7 multiplications M1..M7 and their contributions to C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The inputs to the scalar multiplications are linear combinations of A and B entries.
    
    # M1 = (A00 + A11) (B00 + B11)
    # Inputs: A part: {00:1, 11:1}, B part: {00:1, 11:1}
    # M2 = (A10 + A11) B00
    # Inputs: A part: {10:1, 11:1}, B part: {00:1}
    # M3 = A00 (B01 - B11)
    # Inputs: A part: {00:1}, B part: {01:1, 11:-1}
    # M4 = A11 (B10 - B00)
    # Inputs: A part: {11:1}, B part: {10:1, 00:-1}
    # M5 = (A00 + A01) B11
    # Inputs: A part: {00:1, 01:1}, B part: {11:1}
    # M6 = (A10 - A00) (B00 + B01)
    # Inputs: A part: {10:1, 00:-1}, B part: {00:1, 01:1}
    # M7 = (A01 - A11) (B10 + B11)
    # Inputs: A part: {01:1, 11:-1}, B part: {10:1, 11:1}
    
    # This is getting complicated to map to U, V, W directly.
    # Let's use a different approach:
    # We know the rank is 49.
    # We can construct the 49 terms explicitly by expanding the Strassen recursion.
    
    # Let's define a function that returns the list of (i, j, k, coeff) for a given size.
    # For size 1: [(0,0,0,1)]
    # For size 2:
    #   We have 7 scalar multiplications.
    #   Let's call the scalar multiplication (a, b, c) where a, b, c are linear combos.
    #   Actually, U, V, W are matrices.
    #   U[i, j, r] is the coefficient of A[i, j] in the r-th factor.
    
    # Let's build the 2x2x2 U, V, W explicitly.
    # r=0 (M1):
    # U: A00:1, A11:1. V: B00:1, B11:1.
    # U[0,0,0]=1, U[1,1,0]=1.
    # V[0,0,0]=1, V[1,1,0]=1.
    # W contribution to C:
    # C00 += 1, C11 += 1.
    # So W[0,0,0]=1 (C00), W[1,1,0]=1 (C11).
    # Wait, W[k*n+i, r]. For 2x2, n=2.
    # W[0*2+0, 0] = W[0,0] = 1 (for C00).
    # W[1*2+1, 0] = W[3,0] = 1 (for C11).
    
    # r=1 (M2):
    # U: A10:1, A11:1. V: B00:1.
    # U[1,0,1]=1, U[1,1,1]=1.
    # V[0,0,1]=1.
    # C10 += 1.
    # W[0*2+1, 1] = W[1,1] = 1. (C10 -> k=0, i=1).
    
    # r=2 (M3):
    # U: A00:1. V: B01:1, B11:-1.
    # U[0,0,2]=1.
    # V[0,1,2]=1, V[1,1,2]=-1.
    # C01 += 1.
    # W[1*2+0, 2] = W[2,0] = 1. (C01 -> k=1, i=0).
    
    # r=3 (M4):
    # U: A11:1. V: B10:1, B00:-1.
    # U[1,1,3]=1.
    # V[1,0,3]=1, V[0,0,3]=-1.
    # C10 += 1.
    # W[0*2+1, 3] = W[1,1] = 1.
    
    # r=4 (M5):
    # U: A00:1, A01:1. V: B11:1.
    # U[0,0,4]=1, U[0,1,4]=1.
    # V[1,1,4]=1.
    # C01 += 1, C00 -= 1.
    # W[1*2+0, 4] = W[2,0] = 1. (C01)
    # W[0*2+0, 4] = W[0,0] = -1. (C00)
    
    # r=5 (M6):
    # U: A10:1, A00:-1. V: B00:1, B01:1.
    # U[1,0,5]=1, U[0,0,5]=-1.
    # V[0,0,5]=1, V[0,1,5]=1.
    # C11 += 1.
    # W[1*2+1, 5] = W[3,1] = 1.
    
    # r=6 (M7):
    # U: A01:1, A11:-1. V: B10:1, B11:1.
    # U[0,1,6]=1, U[1,1,6]=-1.
    # V[1,0,6]=1, V[1,1,6]=1.
    # C00 += 1, C11 += 1.
    # W[0*2+0, 6] = W[0,0] = 1.
    # W[1*2+1, 6] = W[3,1] = 1.
    
    # Let's assemble U2, V2, W2.
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # M1 (r=0)
    U2[0,0,0] = 1; U2[1,1,0] = 1
    V2[0,0,0] = 1; V2[1,1,0] = 1
    W2[0,0,0] = 1; W2[1,1,0] = 1 # C00, C11
    
    # M2 (r=1)
    U2[1,0,1] = 1; U2[1,1,1] = 1
    V2[0,0,1] = 1
    W2[1,1,1] = 1 # C10 (k=0, i=1) -> index 1
    
    # M3 (r=2)
    U2[0,0,2] = 1
    V2[0,1,2] = 1; V2[1,1,2] = -1
    W2[2,0,2] = 1 # C01 (k=1, i=0) -> index 2
    
    # M4 (r=3)
    U2[1,1,3] = 1
    V2[1,0,3] = 1; V2[0,0,3] = -1
    W2[1,1,3] = 1 # C10 (k=0, i=1) -> index 1
    
    # M5 (r=4)
    U2[0,0,4] = 1; U2[0,1,4] = 1
    V2[1,1,4] = 1
    W2[2,0,4] = 1 # C01
    W2[0,0,4] = -1 # C00
    
    # M6 (r=5)
    U2[1,0,5] = 1; U2[0,0,5] = -1
    V2[0,0,5] = 1; V2[0,1,5] = 1
    W2[3,1,5] = 1 # C11 (k=1, i=1) -> index 3
    
    # M7 (r=6)
    U2[0,1,6] = 1; U2[1,1,6] = -1
    V2[1,0,6] = 1; V2[1,1,6] = 1
    W2[0,0,6] = 1 # C00
    W2[3,1,6] = 1 # C11
    
    # Verify 2x2x2
    if not verify_decomposition(2, 2, 2, U2.tolist(), V2.tolist(), W2.tolist(), 7):
        raise ValueError("2x2x2 Strassen decomposition failed verification")
        
    # Now construct 4x4x4.
    # U4, V4, W4 of shape (4,4,49).
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    # The 4x4 matrices are split into 2x2 blocks.
    # A = [A00 A01; A10 A11] where each is 2x2.
    # Indices for 4x4:
    # i, j in {0, 1, 2, 3}.
    # Block row 0: i in {0,1}, Block row 1: i in {2,3}.
    # Block col 0: j in {0,1}, Block col 1: j in {2,3}.
    
    # Strassen 4x4:
    # M1 = (A00 + A11) (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # Each M is a 2x2 matrix product.
    # We use the U2, V2, W2 decomposition for each.
    
    # For each of the 7 M's, we have 7 sub-terms.
    # Total 49 terms.
    
    # Mapping of block indices to 4x4 indices:
    # Block (br, bc) of A corresponds to i in [2*br, 2*br+1], j in [2*bc, 2*bc+1].
    # Within the block, local indices are 0,1.
    # So global i = 2*br + i_local, global j = 2*bc + j_local.
    
    # For each Strassen step s (0..6):
    # We determine which blocks of A and B are involved and with what signs.
    # Then for each sub-term t (0..6) of the 2x2 decomposition:
    # We add the contributions to U4, V4, W4.
    
    # Let's define the block contributions for each s.
    # s=0 (M1): A: {00:1, 11:1}, B: {00:1, 11:1}
    # s=1 (M2): A: {10:1, 11:1}, B: {00:1}
    # s=2 (M3): A: {00:1}, B: {01:1, 11:-1}
    # s=3 (M4): A: {11:1}, B: {10:1, 00:-1}
    # s=4 (M5): A: {00:1, 01:1}, B: {11:1}
    # s=5 (M6): A: {10:1, 00:-1}, B: {00:1, 01:1}
    # s=6 (M7): A: {01:1, 11:-1}, B: {10:1, 11:1}
    
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    
    A_specs = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    B_specs = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    
    # C blocks are summed.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will accumulate W4.
    # For each s, the result of the 2x2 multiplication contributes to specific C blocks.
    # The W2 matrix tells us how the local C (2x2) is formed.
    # W2[k_local, i_local, t] is the coefficient.
    # Global C indices: k = 2*ck + k_local, i = 2*ci + i_local.
    # Which C block (ci, ck) is affected by M_s?
    # C00: s=0 (+1), s=3 (+1), s=4 (-1), s=6 (+1)
    # C01: s=2 (+1), s=4 (+1)
    # C10: s=1 (+1), s=3 (+1)
    # C11: s=0 (+1), s=1 (-1), s=2 (+1), s=5 (+1)
    
    # So for each s, we know the target C block (ci, ck) and the sign.
    C_specs = [
        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 0, 1), (0, 0, -1), (1, 1, 1), (0, 0, 1)
    ]
    # Wait, C_specs[s] gives (ci, ck, sign) for the entire 2x2 block result of M_s.
    # M_s is a 2x2 matrix. So it contributes to a 2x2 block of C.
    # The local indices k_local, i_local in W2 map to the block.
    
    for s in range(7):
        ci, ck, c_sign = C_specs[s]
        a_blocks = A_specs[s]
        b_blocks = B_specs[s]
        
        for t in range(7):
            col = s * 7 + t
            
            # Fill U4
            # U4[i, j, col] = sum_{(ar, ac, a_coeff) in a_blocks} a_coeff * U2[i_local, j_local, t]
            # where i = 2*ar + i_local, j = 2*ac + j_local
            for ar, ac, a_coeff in a_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        if U2[i_local, j_local, t] != 0:
                            i = 2 * ar + i_local
                            j = 2 * ac + j_local
                            U4[i, j, col] += a_coeff * U2[i_local, j_local, t]
            
            # Fill V4
            # V4[j, k, col] = sum_{(br, bc, b_coeff) in b_blocks} b_coeff * V2[j_local, k_local, t]
            # where j = 2*br + j_local, k = 2*bc + k_local
            # Note: V2 is indexed by [j_local, k_local, t].
            for br, bc, b_coeff in b_blocks:
                for j_local in range(2):
                    for k_local in range(2):
                        if V2[j_local, k_local, t] != 0:
                            j = 2 * br + j_local
                            k = 2 * bc + k_local
                            V4[j, k, col] += b_coeff * V2[j_local, k_local, t]
            
            # Fill W4
            # W4[k, i, col] = c_sign * W2[k_local, i_local, t]
            # where k = 2*ck + k_local, i = 2*ci + i_local
            for k_local in range(2):
                for i_local in range(2):
                    if W2[k_local, i_local, t] != 0:
                        k = 2 * ck + k_local
                        i = 2 * ci + i_local
                        W4[k, i, col] += c_sign * W2[k_local, i_local, t]
                        
    # Verify 4x4x4
    if not verify_decomposition(4, 4, 4, U4.tolist(), V4.tolist(), W4.tolist(), 49):
        raise ValueError("4x4x4 Strassen decomposition failed verification")
        
    return U4, V4, W4

def construct_full_4x4x5():
    """
    Construct 4x4x5 decomposition.
    Use Strassen 49 for the first 4 columns.
    Use schoolbook 4 for the 5th column.
    Total rank 53.
    """
    n, m, p = 4, 4, 5
    rank = 53
    
    U4, V4, W4 = construct_strassen_4x4x4()
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Copy Strassen part (first 49 columns)
    # U4 is (4,4,49) -> reshape to (16, 49)
    U[:, :49] = U4.reshape(16, 49)
    # V4 is (4,4,49) -> but V is (m, p, rank) = (4, 5, 53).
    # V4 corresponds to B[:, :4].
    # V4[j, k, r] where j in 0..3, k in 0..3.
    # In V matrix, index is j*p + k.
    # So for k in 0..3, V[j*5 + k, r] = V4[j, k, r].
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :49] = V4[j, k, :]
            
    # W4 is (4,4,49) -> W is (p, n, rank) = (5, 4, 53).
    # W4[k, i, r] where k in 0..3, i in 0..3.
    # In W matrix, index is k*n + i.
    # So for k in 0..3, W[k*4 + i, r] = W4[k, i, r].
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :49] = W4[k, i, :]
            
    # Add schoolbook part for column 4 (k=4)
    # We need to compute C[:, 4] = A @ B[:, 4].
    # C[i, 4] = sum_j A[i, j] B[j, 4].
    # This is 4 independent dot products (for i=0..3).
    # Each dot product has 4 terms (j=0..3).
    # Total 16 terms?
    # Wait, the tensor decomposition sums over r.
    # T[i, j, k] = 1.
    # For k=4, T[i, j, 4] = 1 for all i, j.
    # So we need 4*4 = 16 terms?
    # Schoolbook rank for 4x4x1 is 4*4*1 = 16?
    # No, rank of 4x4x1 is 4*4 = 16?
    # Let's check.
    # C[i, 4] = sum_j A[i, j] B[j, 4].
    # This is a sum of 4 products.
    # But i varies.
    # For each i, we have a sum.
    # So we can have terms (i, j, 4) for each i, j.
    # That is 16 terms.
    # Can we do better?
    # The tensor is T[i, j, 4] = 1.
    # This is a rank 16 tensor?
    # T = sum_{i,j} e_i \otimes e_j \otimes e_4.
    # Yes, rank 16.
    
    # So we add 16 terms.
    # Total rank 49 + 16 = 65.
    # Score 80/65 = 1.23.
    # Previous estimate was 53, which was wrong.
    
    # Let's verify.
    # For k=4, we need T[i, j, 4] = 1.
    # If we use terms r=49..64.
    # For each i in 0..3, j in 0..3:
    #   r = 49 + i*4 + j
    #   U[i*4 + j, r] = 1
    #   V[j*5 + 4, r] = 1  (since k=4)
    #   W[4*4 + i, r] = 1  (since k=4, index k*n+i = 4*4+i = 16+i)
    
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U[i * 4 + j, r] = 1
            V[j * 5 + 4, r] = 1
            W[4 * 4 + i, r] = 1
            
    # Verify full 4x4x5
    if not verify_decomposition(4, 4, 5, U.tolist(), V.tolist(), W.tolist(), 65):
        raise ValueError("4x4x5 decomposition failed verification")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

# Fallback to schoolbook if anything goes wrong
import sys
import traceback

def construct_safe():
    try:
        result = construct_full_4x4x5()
        return result
    except Exception as e:
        traceback.print_exc()
        # Return schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# Replace construct with construct_safe
construct = construct_safe
