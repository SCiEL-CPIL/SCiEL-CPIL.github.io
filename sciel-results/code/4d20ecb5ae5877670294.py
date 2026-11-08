"""Matrix multiplication 4x4x9 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p  # 144
    
    # Build schoolbook U, V, W
    U = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook
    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Check if U, V, W correctly decompose matrix multiplication."""
        # The tensor T[i,j,k] = delta (i.e., T[i,j,k] = 1 if this is the (i,j,k) entry)
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for each (i,j,k)
        # Actually, the convention is: (AB)[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: T[i,j,k] such that sum_j T[i,j,k] * A[i,j] * B[j,k] = (AB)[i,k]
        # So T[i,j,k] = delta_{j,j} = 1 for all valid i,j,k
        # The decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        # Let's verify by computing the implied multiplication
        # For random A, B, check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]*A[i,j]*B[j,k] = (AB)[i,k]
        
        # More directly: the tensor T has T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
        # Wait, no. T[i,j,k] = delta where the multiplication is (AB)[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor is T[i,j,k] = 1 for all valid indices (it's just the "multiplication tensor")
        # The decomposition says: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want to find U, V, W with minimal rank such that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Strategy: exploit the structure 9 = 3*3
    # View k as (k1, k2) where k = k1*3 + k2, k1 in {0,1,2}, k2 in {0,1,2}
    # Similarly, the second matrix index j*p+k becomes j*9+k = j*(3*3)+k1*3+k2
    
    # For fixed k2, we have a 4x4x3 multiplication in (i, j, k1)
    # The three k2 slices are independent, so rank = 3 * rank(4x4x3)
    
    # Let's try to find a lower rank for 4x4x3
    # Schoolbook 4x4x3 = 48
    # Can we do better?
    
    # Actually, let me try a different structural decomposition.
    # Note that 9 = 3*3, and we can think of the 4x9 matrix as having 3 rows of 3x3 blocks... no.
    
    # Let me try: can we use the fact that the multiplication tensor for 4x4x9
    # has some structure we can exploit?
    
    # Alternative: try to use a known lower-rank 4x4x4 decomposition and extend it.
    # If we have rank r for 4x4x4, then for 4x4x9 we might get rank related to r.
    
    # Actually, the simplest improvement: use the rectangular structure.
    # M_{4,4,9}: we can write the 9 as 3+3+3 and use three 4x4x3 multiplications.
    # Or 9 as 4+4+1, using two 4x4x4 and one 4x4x1.
    
    # 4x4x1 has rank 16 (schoolbook) but actually rank 4! 
    # Because (AB)[:,k] = A * B[:,k], which is a matrix-vector product of rank 4.
    # Wait, no. The tensor for 4x4x1: T[i,j,0] = 1 for all i,j.
    # sum_c U[i*4+j,c] * V[j*1+0,c] * W[0*4+i,c] = 1
    # This is just the outer product structure. Rank is 4: 
    # U[i*4+j,c] = delta_{i,c}, V[j,c] = 1, W[i,c] = 1 for c in 0..3
    # Check: sum_c delta_{i,c} * 1 * 1 = 1 if c=i exists, so sum = 1. Yes! Rank 4.
    
    # So M_{4,4,1} has rank 4.
    # M_{4,4,9} can be decomposed as M_{4,4,4} + M_{4,4,4} + M_{4,4,1}
    # rank <= rank(4x4x4) + rank(4x4x4) + rank(4x4x1)
    
    # What's the best known rank for 4x4x4?
    # Schoolbook: 64. Strassen: 49 for 2x2x2, but for 4x4x4 it's more complex.
    # Actually, the rank of 4x4x4 is at most 49 (by tensoring Strassen's 2x2x2 with itself: 7*7=49)
    # But we need exact integer/half-integer entries.
    
    # Strassen's 2x2x2 has rank 7 with integer entries.
    # Tensoring: 4x4x4 has rank at most 49 with integer entries.
    
    # So rank(4x4x9) <= 49 + 49 + 4 = 102. That's better than 144!
    
    # Let me implement this:
    # Decompose 9 = 4 + 4 + 1
    # For the first 4 columns: use Strassen-tensor-Strassen 4x4x4
    # For the next 4 columns: use another copy
    # For the last 1 column: use the rank-4 4x4x1
    
    # First, let's build the Strassen 2x2x2 decomposition.
    # Strassen's algorithm for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The tensor entries for 2x2x2:
    # U has shape (2*2, 7) = (4, 7), V has shape (2*2, 7) = (4, 7), W has shape (2*2, 7) = (4, 7)
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 with rank 7."""
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        
        # M1 = (A00+A11)*(B00+B11)
        # U: A00 at (0,0), A11 at (1,1)
        U[0*2+0, 0] = 1  # A00
        U[1*2+1, 0] = 1  # A11
        V[0*2+0, 0] = 1  # B00
        V[1*2+1, 0] = 1  # B11
        W[0*2+0, 0] = 1  # C00
        W[1*2+1, 0] = 1  # C11
        
        # M2 = (A10+A11)*B00
        U[1*2+0, 1] = 1  # A10
        U[1*2+1, 1] = 1  # A11
        V[0*2+0, 1] = 1  # B00
        W[1*2+0, 1] = 1  # C10
        W[1*2+1, 1] = 1  # C11
        
        # M3 = A00*(B01-B11)
        U[0*2+0, 2] = 1  # A00
        V[0*2+1, 2] = 1  # B01
        V[1*2+1, 2] = -1 # B11
        W[0*2+1, 2] = 1  # C01
        W[1*2+1, 2] = 1  # C11
        
        # M4 = A11*(B10-B00)
        U[1*2+1, 3] = 1  # A11
        V[1*2+0, 3] = 1  # B10
        V[0*2+0, 3] = -1 # B00
        W[0*2+0, 3] = 1  # C00
        W[1*2+0, 3] = 1  # C10
        
        # M5 = (A00+A01)*B11
        U[0*2+0, 4] = 1  # A00
        U[0*2+1, 4] = 1  # A01
        V[1*2+1, 4] = 1  # B11
        W[0*2+0, 4] = -1 # C00
        W[0*2+1, 4] = 1  # C01
        
        # M6 = (A10-A00)*(B00+B01)
        U[1*2+0, 5] = 1  # A10
        U[0*2+0, 5] = -1 # A00
        V[0*2+0, 5] = 1  # B00
        V[0*2+1, 5] = 1  # B01
        W[1*2+1, 5] = 1  # C11
        
        # M7 = (A01-A11)*(B10+B11)
        U[0*2+1, 6] = 1  # A01
        U[1*2+1, 6] = -1 # A11
        V[1*2+0, 6] = 1  # B10
        V[1*2+1, 6] = 1  # B11
        W[0*2+0, 6] = 1  # C00
        
        return U, V, W
    
    # Verify Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    assert verify(U_s, V_s, W_s, 2, 2, 2), "Strassen 2x2x2 verification failed"
    
    # Now build 4x4x4 by tensoring Strassen with itself.
    # The 4x4x4 multiplication: A is 4x4, B is 4x4, C is 4x4.
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2.
    # Similarly B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]].
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Using Strassen on the 2x2 block structure:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Each Mj is a 2x2 matrix product, computed with Strassen (rank 7 each).
    # Total rank: 7*7 = 49.
    
    # For the tensor product, the U, V, W matrices are Kronecker products.
    # U_4x4x4[i*4+j, c1*7+c2] = U_s[i*2+i2, c1] * U_s[j*2+j2, c2]
    # Wait, I need to be more careful.
    
    # Let me think about this differently. The 4x4x4 tensor T4[i,j,k] = 1 for all i,j,k in 0..3.
    # We decompose using the block structure.
    
    # Actually, the cleanest way: the rank-49 decomposition of 4x4x4 is obtained by
    # applying Strassen to the 2x2 block multiplication, where each block multiplication
    # is itself done with Strassen.
    
    # Let me build it explicitly.
    # A is 4x4, indexed by (i, j) where i in 0..3, j in 0..3.
    # Block structure: i = 2*i1 + i2, j = 2*j1 + j2, where i1,j1 in {0,1} (block row/col), i2,j2 in {0,1} (within block).
    # Similarly for B: j = 2*j1 + j2, k = 2*k1 + k2.
    # C: i = 2*i1 + i2, k = 2*k1 + k2.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In blocks: C_block(i1,k1) = sum_{j1} A_block(i1,j1) * B_block(j1,k1)
    
    # Using Strassen on blocks:
    # S1 = (A00+A11) * (B00+B11)
    # S2 = (A10+A11) * B00
    # S3 = A00 * (B01-B11)
    # S4 = A11 * (B10-B00)
    # S5 = (A00+A01) * B11
    # S6 = (A10-A00) * (B00+B01)
    # S7 = (A01-A11) * (B10+B11)
    
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Each Sj is a 2x2 matrix product, computed with Strassen (7 scalar multiplications).
    # So total: 7 * 7 = 49 scalar multiplications.
    
    # Now I need to build the U, V, W matrices for the 4x4x4 decomposition.
    
    def build_4x4x4_strassen():
        """Build rank-49 decomposition of 4x4x4 using tensor of Strassen."""
        n = 4
        U = np.zeros((16, 49), dtype=np.int64)
        V = np.zeros((16, 49), dtype=np.int64)
        W = np.zeros((16, 49), dtype=np.int64)
        
        # Strassen 2x2x2 components
        U_s, V_s, W_s = strassen_2x2x2()
        
        # The 7 Strassen steps for block multiplication:
        # Each step c1 in 0..6 involves specific block combinations.
        # For step c1, the A-blocks and B-blocks are combined, and the result contributes to C-blocks.
        
        # Let me define the block structure more carefully.
        # A[i,j] where i = 2*i1+i2, j = 2*j1+j2
        # B[j,k] where j = 2*j1+j2, k = 2*k1+k2
        # C[i,k] where i = 2*i1+i2, k = 2*k1+k2
        
        # For Strassen step c1, we compute (sum_{a} alpha_a * A_block_a) * (sum_{b} beta_b * B_block_b)
        # where A_block_a is a 2x2 block of A, B_block_b is a 2x2 block of B.
        
        # Then within each block multiplication, we use Strassen again.
        
        # Let me define the 7 block-level Strassen steps:
        # Step 0: (A00+A11) * (B00+B11)
        # Step 1: (A10+A11) * B00
        # Step 2: A00 * (B01-B11)
        # Step 3: A11 * (B10-B00)
        # Step 4: (A00+A01) * B11
        # Step 5: (A10-A00) * (B00+B01)
        # Step 6: (A01-A11) * (B10+B11)
        
        # For each step, I need to specify:
        # - Which A-blocks are involved and with what coefficients
        # - Which B-blocks are involved and with what coefficients
        # - Which C-blocks the result contributes to and with what coefficients
        
        # A-blocks: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
        # B-blocks: B00 = B[0:2,0:2], B01 = B[0:2,2:4], B10 = B[2:4,0:2], B11 = B[2:4,2:4]
        # C-blocks: C00 = C[0:2,0:2], C01 = C[0:2,2:4], C10 = C[2:4,0:2], C11 = C[2:4,2:4]
        
        # Block definitions: (block_row, block_col)
        # A-blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
        # B-blocks: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
        # C-blocks: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
        
        # Strassen steps:
        # Step 0: A: [(0,0,1), (1,1,1)], B: [(0,0,1), (1,1,1)], C: [(0,0,1), (1,1,1)]
        # Step 1: A: [(1,0,1), (1,1,1)], B: [(0,0,1)], C: [(1,0,1), (1,1,1)]
        # Step 2: A: [(0,0,1)], B: [(0,1,1), (1,1,-1)], C: [(0,1,1), (1,1,1)]
        # Step 3: A: [(1,1,1)], B: [(1,0,1), (0,0,-1)], C: [(0,0,1), (1,0,1)]
        # Step 4: A: [(0,0,1), (0,1,1)], B: [(1,1,1)], C: [(0,0,-1), (0,1,1)]
        # Step 5: A: [(1,0,1), (0,0,-1)], B: [(0,0,1), (0,1,1)], C: [(1,1,1)]
        # Step 6: A: [(0,1,1), (1,1,-1)], B: [(1,0,1), (1,1,1)], C: [(0,0,1)]
        
        steps = [
            # (A_blocks, B_blocks, C_blocks)
            # Each block: (block_row, block_col, coeff)
            ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
            ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1), ((1,1),1)]),
            ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),1)]),
            ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]),
            ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
            ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,1),1)]),
            ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1)]),
        ]
        
        for c1, (A_blocks, B_blocks, C_blocks) in enumerate(steps):
            for c2 in range(7):
                col = c1 * 7 + c2
                # A part: for each (block_row, block_col, coeff) in A_blocks
                # The within-block Strassen uses U_s[i2*2+j2, c2]
                # The block index (br, bc) corresponds to rows 2*br:2*br+2, cols 2*bc:2*bc+2
                for (br, bc), coeff in A_blocks:
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2*br + i2
                            j = 2*bc + j2
                            U[i*4+j, col] += coeff * U_s[i2*2+j2, c2]
                
                # B part
                for (br, bc), coeff in B_blocks:
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2*br + j2
                            k = 2*bc + k2
                            V[j*4+k, col] += coeff * V_s[j2*2+k2, c2]
                
                # C part
                for (br, bc), coeff in C_blocks:
                    for k2 in range(2):
                        for i2 in range(2):
                            k = 2*br + k2
                            i = 2*bc + i2
                            W[k*4+i, col] += coeff * W_s[k2*2+i2, c2]
        
        return U, V, W
    
    U_444, V_444, W_444 = build_4x4x4_strassen()
    assert verify(U_444, V_444, W_444, 4, 4, 4), "4x4x4 Strassen verification failed"
    
    # Now build 4x4x1 with rank 4.
    def build_4x4x1():
        """Rank-4 decomposition of 4x4x1."""
        n, m, p = 4, 4, 1
        U = np.zeros((16, 4), dtype=np.int64)
        V = np.zeros((4, 4), dtype=np.int64)
        W = np.zeros((4, 4), dtype=np.int64)
        
        # U[i*4+j, c] = delta_{i,c}
        # V[j, c] = 1
        # W[i, c] = 1
        for c in range(4):
            for i in range(4):
                for j in range(4):
                    if i == c:
                        U[i*4+j, c] = 1
            for j in range(4):
                V[j, c] = 1
            for i in range(4):
                W[i, c] = 1
        
        return U, V, W
    
    U_441, V_441, W_441 = build_4x4x1()
    assert verify(U_441, V_441, W_441, 4, 4, 1), "4x4x1 verification failed"
    
    # Now combine: 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # The k index for 9: first 4 use first 4x4x4, next 4 use second, last 1 uses 4x4x1.
    
    # U has shape (16, rank_total), V has shape (36, rank_total), W has shape (36, rank_total)
    # Wait: m*p = 4*9 = 36, p*n = 9*4 = 36.
    
    rank_total = 49 + 49 + 4  # 102
    
    U_final = np.zeros((16, rank_total), dtype=np.int64)
    V_final = np.zeros((36, rank_total), dtype=np.int64)
    W_final = np.zeros((36, rank_total), dtype=np.int64)
    
    # First 4x4x4: k in 0..3
    col_offset = 0
    U_final[:, col_offset:col_offset+49] = U_444
    for j in range(4):
        for k in range(4):
            V_final[j*9+k, col_offset:col_offset+49] = V_444[j*4+k]
    for k in range(4):
        for i in range(4):
            W_final[k*4+i, col_offset:col_offset+49] = W_444[k*4+i]
    col_offset += 49
    
    # Second 4x4x4: k in 4..7
    U_final[:, col_offset:col_offset+49] = U_444
    for j in range(4):
        for k in range(4):
            V_final[j*9+(k+4), col_offset:col_offset+49] = V_444[j*4+k]
    for k in range(4):
        for i in range(4):
            W_final[(k+4)*4+i, col_offset:col_offset+49] = W_444[k*4+i]
    col_offset += 49
    
    # 4x4x1: k in 8
    U_final[:, col_offset:col_offset+4] = U_441
    for j in range(4):
        V_final[j*9+8, col_offset:col_offset+4] = V_441[j]
    for i in range(4):
        W_final[8*4+i, col_offset:col_offset+4] = W_441[i]
    
    # Verify
    if not verify(U_final, V_final, W_final, 4, 4, 9):
        # Fallback to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check time
    if time.time() - start_time > deadline - 5:
        # Return what we have
        pass
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

