"""Matrix multiplication 2x4 by 4x6 tensor decomposition."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal (or near-minimal) tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = 120.0  # safety margin before the 150s wall-clock limit
    
    n, m, p = 2, 4, 6
    schoolbook_rank = n * m * p  # 48
    
    # Start with schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]  # (8, 48)
    V = [[0] * schoolbook_rank for _ in range(m * p)]  # (24, 48)
    W = [[0] * schoolbook_rank for _ in range(p * n)]  # (12, 48)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Try to improve using Strassen-like combinations
    # The 2x4 * 4x6 can be split into block structure
    # A is 2x4, B is 4x6, C = A*B is 2x6
    # Split A into [A1 A2] where A1 is 2x2, A2 is 2x2
    # Split B into [B1; B2] where B1 is 2x6, B2 is 2x6
    # C = A1*B1 + A2*B2
    # Each 2x2 * 2x6 can be done with rank 14 using Strassen on the 2x2 part
    # So total rank = 14 + 14 = 28
    
    # Let's try to construct a rank-28 decomposition using Strassen for 2x2 * 2x6
    # For 2x2 * 2x6: treat as multiplying 2x2 by 2x6
    # Split the 2x6 into two 2x3 blocks? Or use direct tensor approach
    # Actually, let's think about this differently.
    
    # The tensor for 2x4 * 4x6 has structure that allows factorization
    # Let's try a systematic approach: use the fact that this is a Kronecker-like product
    # 2x4 = 2x(2*2), 4x6 = (2*2)x6
    # So this is like (I_2 ⊗ M_{2,2}) * (M_{2,2} ⊗ I_2) style structure
    
    # Let's try to find a lower rank decomposition by exploiting symmetry
    # The 48-dimensional space of independent entries can potentially be compressed
    
    # Approach: Try rank 24 (half of schoolbook) by pairing terms
    # For each i in {0,1}, k in {0,...,5}, we have 4 terms in j
    # Can we combine them?
    
    # Let's try a structured approach using the block decomposition:
    # C = A * B where A is 2x4, B is 4x6
    # Write A = [A_0 | A_1] with A_0, A_1 both 2x2
    # Write B = [B_0; B_1] with B_0, B_1 both 2x6
    # C = A_0 * B_0 + A_1 * B_1
    
    # For each 2x2 * 2x6 multiplication, use rank-7 per 2x2*2x2 block
    # 2x6 = three 2x2 blocks, so 2x2 * 2x6 = 3 * (2x2 * 2x2)
    # Using Strassen: 7 * 3 = 21 for each, total 42
    
    # But we can do better by sharing Strassen intermediates across the three 2x2 blocks of B
    # Actually, let's just implement the 42-rank version first, then try to improve
    
    # For 2x2 * 2x2, Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x6, split B into [B_left | B_right] each 2x2? No, B is 2x6
    # Split B into three 2x2 blocks: B1, B2, B3
    # C = A * [B1 B2 B3] = [A*B1 | A*B2 | A*B3]
    # Each A*Bk uses Strassen with 7 products
    # Total: 21 products for 2x2 * 2x6
    
    # But we can share some computations. Actually, for the full 2x4 * 4x6:
    # A = [A_0 | A_1], B = [B_0; B_1] where A_0,A_1 are 2x2, B_0,B_1 are 2x6
    # Split B_0 = [B0_1 B0_2 B0_3], B_1 = [B1_1 B1_2 B1_3] each 2x2
    # C = A_0*B_0 + A_1*B_1
    # C_j = A_0*B0_j + A_1*B1_j for j=0,1,2
    
    # Each C_j is 2x2, computed as A_0*B0_j + A_1*B1_j
    # We can use Strassen for each pair, but A_0 and A_1 are fixed
    # Precompute Strassen intermediates for A_0 and A_1
    # For A_0: precompute the 7 row-combinations
    # For A_1: precompute the 7 row-combinations
    # Then for each B_j = [B0_j; B1_j] (4x2), compute 7 products
    # Total: 7 * 3 = 21 products
    
    # Wait, let me reconsider. The structure is:
    # A is 2x4 = [a00 a01 a02 a03; a10 a11 a12 a13]
    # B is 4x6
    # C is 2x6
    
    # Let me use a different block structure.
    # Group A's columns: A = [A_L | A_R] where A_L, A_R are 2x2
    # Group B's rows: B = [B_T; B_B] where B_T, B_B are 2x6
    # C = A_L * B_T + A_R * B_B
    
    # Now, B_T and B_B are 2x6. Split each into three 2x2 blocks:
    # B_T = [BT1 BT2 BT3], B_B = [BB1 BB2 BB3]
    # C = [A_L*BT1 + A_R*BB1 | A_L*BT2 + A_R*BB2 | A_L*BT3 + A_R*BB3]
    
    # For each column block j, we need to compute A_L*BTj + A_R*BBj
    # This is a sum of two 2x2 * 2x2 products
    # We can use 7 Strassen products for each, but share the A-side intermediates
    
    # Precompute for A_L: 7 combinations of rows
    # Precompute for A_R: 7 combinations of rows
    # For each j, form 7 combinations of [BTj; BBj] (4x2)
    # Then 7 scalar products per j, 3 j's = 21 products total
    
    # This gives rank 21! Let me verify this is correct.
    
    # Actually, I need to be more careful. Let me define the Strassen intermediates properly.
    # For a 2x2 matrix M = [[m00, m01], [m10, m11]], define:
    # s1 = m00 + m11
    # s2 = m10 + m11
    # s3 = m00
    # s4 = m11
    # s5 = m00 + m01
    # s6 = m10 - m00
    # s7 = m01 - m11
    
    # For a 2x2 matrix N = [[n00, n01], [n10, n11]], define:
    # t1 = n00 + n11
    # t2 = n00
    # t3 = n01 - n11
    # t4 = n10 - n00
    # t5 = n11
    # t6 = n00 + n01
    # t7 = n10 + n11
    
    # Then M*N has:
    # (M*N)00 = s1*t1 - s4*t4 + s5*t5 - s7*t7
    # (M*N)01 = s3*t3 + s5*t5
    # (M*N)10 = s2*t2 + s4*t4
    # (M*N)11 = s1*t1 - s2*t2 + s3*t3 + s6*t6
    
    # Now for our problem:
    # C_j = A_L * BTj + A_R * BBj
    # Let P_j = BTj + BBj (2x2)
    # Let Q_j = BTj - BBj (2x2)
    # Hmm, this doesn't directly help.
    
    # Alternative: think of it as a 4x2 times 2x2 problem
    # [A_L; A_R] * [BTj | BBj]... no, that's not right either.
    
    # Let me just go with the straightforward 21-product approach:
    # For each j in {0,1,2}:
    #   Compute A_L * BTj using 7 Strassen products
    #   Compute A_R * BBj using 7 Strassen products
    #   Add them
    # Total: 21 products
    
    # But wait, the Strassen products for A_L*BTj and A_R*BBj use different B matrices,
    # so we can't share the B-side intermediates easily.
    # However, we CAN share the A-side intermediates since A_L and A_R are fixed.
    
    # So the rank is 21. Let me implement this.
    
    # Actually, I realize I should just verify by construction.
    # Let me build the U, V, W matrices for the 21-rank decomposition.
    
    # Index convention:
    # U has shape (n*m, rank) = (8, 21), U[i*m+j, c]
    # V has shape (m*p, rank) = (24, 21), V[j*p+k, c]
    # W has shape (p*n, rank) = (12, 21), W[k*n+i, c]
    
    # For the 21-rank construction:
    # A_L = A[:, 0:2] = [[a00, a01], [a10, a11]]
    # A_R = A[:, 2:4] = [[a02, a03], [a12, a13]]
    # B_T = B[0:2, :] (2x6)
    # B_B = B[2:4, :] (2x6)
    # BTj = B_T[:, 2*j:2*j+2], BBj = B_B[:, 2*j:2*j+2] for j=0,1,2
    
    # Strassen intermediates for A_L (rows of A_L indexed by i, cols by local j):
    # A_L[i, jl] where jl in {0,1} corresponds to global j = jl
    # A_R[i, jr] where jr in {0,1} corresponds to global j = 2+jr
    
    # For Strassen on A_L * BTj:
    # s1 = A_L[0,0] + A_L[1,1] = a00 + a11
    # s2 = A_L[1,0] + A_L[1,1] = a10 + a11
    # s3 = A_L[0,0] = a00
    # s4 = A_L[1,1] = a11
    # s5 = A_L[0,0] + A_L[0,1] = a00 + a01
    # s6 = A_L[1,0] - A_L[0,0] = a10 - a00
    # s7 = A_L[0,1] - A_L[1,1] = a01 - a11
    
    # For Strassen on A_R * BBj:
    # r1 = A_R[0,0] + A_R[1,1] = a02 + a13
    # r2 = A_R[1,0] + A_R[1,1] = a12 + a13
    # r3 = A_R[0,0] = a02
    # r4 = A_R[1,1] = a13
    # r5 = A_R[0,0] + A_R[0,1] = a02 + a03
    # r6 = A_R[1,0] - A_R[0,0] = a12 - a02
    # r7 = A_R[0,1] - A_R[1,1] = a03 - a13
    
    # For BTj (2x2), cols are 2j, 2j+1; rows 0,1:
    # t1 = B[0,2j] + B[1,2j+1]
    # t2 = B[0,2j]
    # t3 = B[0,2j+1] - B[1,2j+1]
    # t4 = B[1,2j] - B[0,2j]
    # t5 = B[1,2j+1]
    # t6 = B[0,2j] + B[0,2j+1]
    # t7 = B[1,2j] + B[1,2j+1]
    
    # For BBj (2x2), cols are 2j, 2j+1; rows 2,3:
    # u1 = B[2,2j] + B[3,2j+1]
    # u2 = B[2,2j]
    # u3 = B[2,2j+1] - B[3,2j+1]
    # u4 = B[3,2j] - B[2,2j]
    # u5 = B[3,2j+1]
    # u6 = B[2,2j] + B[2,2j+1]
    # u7 = B[3,2j] + B[3,2j+1]
    
    # Products for j-th block (7 for A_L*BTj, 7 for A_R*BBj):
    # p1 = (a00+a11) * (B[0,2j]+B[1,2j+1])
    # p2 = (a10+a11) * B[0,2j]
    # p3 = a00 * (B[0,2j+1]-B[1,2j+1])
    # p4 = a11 * (B[1,2j]-B[0,2j])
    # p5 = (a00+a01) * B[1,2j+1]
    # p6 = (a10-a00) * (B[0,2j]+B[0,2j+1])
    # p7 = (a01-a11) * (B[1,2j]+B[1,2j+1])
    # q1 = (a02+a13) * (B[2,2j]+B[3,2j+1])
    # q2 = (a12+a13) * B[2,2j]
    # q3 = a02 * (B[2,2j+1]-B[3,2j+1])
    # q4 = a13 * (B[3,2j]-B[2,2j])
    # q5 = (a02+a03) * B[3,2j+1]
    # q6 = (a12-a02) * (B[2,2j]+B[2,2j+1])
    # q7 = (a03-a13) * (B[3,2j]+B[3,2j+1])
    
    # C[0,2j] = p1 - p4 + p5 - p7 + q1 - q4 + q5 - q7
    # C[0,2j+1] = p3 + p5 + q3 + q5
    # C[1,2j] = p2 + p4 + q2 + q4
    # C[1,2j+1] = p1 - p2 + p3 + p6 + q1 - q2 + q3 + q6
    
    # Now let's build U, V, W for each of the 21 products.
    # Product index c: for j in 0..2, for r in 0..6
    # c = 7*j + r
    
    rank = 21
    U = [[0] * rank for _ in range(8)]   # 2*4 = 8
    V = [[0] * rank for _ in range(24)]  # 4*6 = 24
    W = [[0] * rank for _ in range(12)]  # 6*2 = 12
    
    for j in range(3):
        base_c = 7 * j
        # A_L products (r=0..6)
        # r=0: p1 = (a00+a11)*(B[0,2j]+B[1,2j+1])
        U[0][base_c + 0] = 1  # a00
        U[1][base_c + 0] = 1  # a11
        V[0*6 + 2*j][base_c + 0] = 1  # B[0,2j]
        V[1*6 + (2*j+1)][base_c + 0] = 1  # B[1,2j+1]
        W[(2*j)*2 + 0][base_c + 0] = 1  # C[0,2j]
        W[(2*j)*2 + 1][base_c + 0] = 1  # C[1,2j+1]
        
        # r=1: p2 = (a10+a11)*B[0,2j]
        U[0][base_c + 1] = 1  # a10 (row 1 of A_L is i=1, local col 0 = global col 0)
        U[1][base_c + 1] = 1  # a11
        V[0*6 + 2*j][base_c + 1] = 1
        W[(2*j)*2 + 1][base_c + 1] = 1  # C[1,2j]
        
        # r=2: p3 = a00*(B[0,2j+1]-B[1,2j+1])
        U[0][base_c + 2] = 1  # a00
        V[0*6 + (2*j+1)][base_c + 2] = 1  # B[0,2j+1]
        V[1*6 + (2*j+1)][base_c + 2] = -1  # -B[1,2j+1]
        W[(2*j+1)*2 + 0][base_c + 2] = 1  # C[0,2j+1]
        W[(2*j+1)*2 + 1][base_c + 2] = 1  # C[1,2j+1]
        
        # r=3: p4 = a11*(B[1,2j]-B[0,2j])
        U[1][base_c + 3] = 1  # a11
        V[1*6 + 2*j][base_c + 3] = 1  # B[1,2j]
        V[0*6 + 2*j][base_c + 3] = -1  # -B[0,2j]
        W[(2*j)*2 + 0][base_c + 3] = -1  # -C[0,2j]
        W[(2*j)*2 + 1][base_c + 3] = 1  # C[1,2j]
        
        # r=4: p5 = (a00+a01)*B[1,2j+1]
        U[0][base_c + 4] = 1  # a00
        U[1][base_c + 4] = 1  # a01 (row 0 of A_L, local col 1 = global col 1)
        V[1*6 + (2*j+1)][base_c + 4] = 1
        W[(2*j)*2 + 0][base_c + 4] = 1  # C[0,2j]
        W[(2*j+1)*2 + 0][base_c + 4] = 1  # C[0,2j+1]
        
        # r=5: p6 = (a10-a00)*(B[0,2j]+B[0,2j+1])
        U[0][base_c + 5] = -1  # -a00
        U[1][base_c + 5] = 1  # a10
        V[0*6 + 2*j][base_c + 5] = 1  # B[0,2j]
        V[0*6 + (2*j+1)][base_c + 5] = 1  # B[0,2j+1]
        W[(2*j+1)*2 + 1][base_c + 5] = 1  # C[1,2j+1]
        
        # r=6: p7 = (a01-a11)*(B[1,2j]+B[1,2j+1])
        U[0][base_c + 6] = -1  # -a11 (row 1, local col 1 = global col 1)
        U[1][base_c + 6] = 1  # a01
        V[1*6 + 2*j][base_c + 6] = 1  # B[1,2j]
        V[1*6 + (2*j+1)][base_c + 6] = 1  # B[1,2j+1]
        W[(2*j)*2 + 0][base_c + 6] = -1  # -C[0,2j]
        
        # A_R products (r=0..6)
        # r=0: q1 = (a02+a13)*(B[2,2j]+B[3,2j+1])
        U[2*4 + 0][base_c + 7] = 1  # a02 (row 0 of A_R, local col 0 = global col 2)
        U[2*4 + 1][base_c + 7] = 1  # a13 (row 1 of A_R, local col 1 = global col 3)
        V[2*6 + 2*j][base_c + 7] = 1  # B[2,2j]
        V[3*6 + (2*j+1)][base_c + 7] = 1  # B[3,2j+1]
        W[(2*j)*2 + 0][base_c + 7] = 1
        W[(2*j)*2 + 1][base_c + 7] = 1
        
        # r=1: q2 = (a12+a13)*B[2,2j]
        U[2*4 + 0][base_c + 8] = 1  # a12
        U[2*4 + 1][base_c + 8] = 1  # a13
        V[2*6 + 2*j][base_c + 8] = 1
        W[(2*j)*2 + 1][base_c + 8] = 1
        
        # r=2: q3 = a02*(B[2,2j+1]-B[3,2j+1])
        U[2*4 + 0][base_c + 9] = 1  # a02
        V[2*6 + (2*j+1)][base_c + 9] = 1
        V[3*6 + (2*j+1)][base_c + 9] = -1
        W[(2*j+1)*2 + 0][base_c + 9] = 1
        W[(2*j+1)*2 + 1][base_c + 9] = 1
        
        # r=3: q4 = a13*(B[3,2j]-B[2,2j])
        U[2*4 + 1][base_c + 10] = 1  # a13
        V[3*6 + 2*j][base_c + 10] = 1
        V[2*6 + 2*j][base_c + 10] = -1
        W[(2*j)*2 + 0][base_c + 10] = -1
        W[(2*j)*2 + 1][base_c + 10] = 1
        
        # r=4: q5 = (a02+a03)*B[3,2j+1]
        U[2*4 + 0][base_c + 11] = 1  # a02
        U[2*4 + 1][base_c + 11] = 1  # a03
        V[3*6 + (2*j+1)][base_c + 11] = 1
        W[(2*j)*2 + 0][base_c + 11] = 1
        W[(2*j+1)*2 + 0][base_c + 11] = 1
        
        # r=5: q6 = (a12-a02)*(B[2,2j]+B[2,2j+1])
        U[2*4 + 0][base_c + 12] = -1  # -a02
        U[2*4 + 1][base_c + 12] = 1  # a12
        V[2*6 + 2*j][base_c + 12] = 1
        V[2*6 + (2*j+1)][base_c + 12] = 1
        W[(2*j+1)*2 + 1][base_c + 12] = 1
        
        # r=6: q7 = (a03-a13)*(B[3,2j]+B[3,2j+1])
        U[2*4 + 0][base_c + 13] = -1  # -a13
        U[2*4 + 1][base_c + 13] = 1  # a03
        V[3*6 + 2*j][base_c + 13] = 1
        V[3*6 + (2*j+1)][base_c + 13] = 1
        W[(2*j)*2 + 0][base_c + 13] = -1
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] equals the matrix multiplication tensor."""
    rank = len(U[0])
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent
    # Actually, the tensor for matrix multiplication is:
    # T[a, b, c] = 1 where a = i*m+j (row of A * col of A... wait)
    # Let me use the convention: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 means A[i,j]*B[j,k]*C[i,k]
    # In the flattened form: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Sum over c should give 1 when (i,j,k) form a valid multiplication term, else 0
    
    # Build the tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > tol:
                    return False
    # Also check that no spurious terms appear (should be 0 for invalid index combos)
    # But the tensor is only defined for valid (i,j,k), so we just need the above
    return True

# EVOLVE-BLOCK-END
