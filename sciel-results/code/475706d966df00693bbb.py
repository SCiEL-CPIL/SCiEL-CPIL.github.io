"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 120
    
    n, m, p = DIMENSIONS  # A: n×m, B: m×p, C: n×p
    
    # Partition: rows of A: 2+1, middle: 2+2, cols of B: 2+3
    # Sub-blocks (row_block_size, mid_block_size, col_block_size) -> rank
    # Using Strassen for (2,2,2) blocks, schoolbook otherwise
    
    # Strassen decomposition for (2,2)×(2,2)->(2,2)
    # U: 2x2 pattern, V: 2x2 pattern, W: 2x2 pattern for each of 7 terms
    # M1 = (A11 + A22)(B11 + B22) -> contributes to C11, C12, C21, C22
    # M2 = (A21 + A22)B11 -> C21, C22  
    # M3 = A11(B21 - B11) -> C11, C12
    # M4 = A22(B22 - B12) -> C21, C22
    # M5 = (A11 + A12)B22 -> C11, C12
    # M6 = (A21 - A11)(B11 + B12) -> C21, C22
    # M7 = (A12 - A22)(B21 + B22) -> C11, C12
    
    # Strassen coefficients: for term t, U[i,j] is coefficient of A_{i,j}, 
    # V[j,k] is coefficient of B_{j,k}, W[i,k] is coefficient added to C_{i,k}
    # Note: using 1-indexed in formulas, 0-indexed in code
    
    # Let me define Strassen terms explicitly (0-indexed):
    # Term 1: U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1)
    #   W: M1 contributes +1 to C(0,0), +1 to C(0,1)... wait let me re-derive.
    
    # Standard Strassen for 2x2 matrices (0-indexed):
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    #   C[0,0] += M1, C[1,1] += M1 (but need to subtract others)
    # Actually the standard form:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (U_pattern_2x2, V_pattern_2x2, W_pattern_2x2)
        # M1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], contributes to C: [[1,0],[0,1]]
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        # M2: U=[[0,0],[1,1]], V=[[1,0],[0,0]], contributes to C: [[0,0],[1,-1]]
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,-1]]),
        # M3: U=[[1,0],[0,0]], V=[[0,1],[0,0]], contributes to C: [[0,1],[0,0]]
        ([[1,0],[0,0]], [[0,1],[0,0]], [[0,1],[0,0]]),
        # M4: U=[[0,0],[0,1]], V=[[0,0],[0,1]], contributes to C: [[0,0],[1,0]]
        ([[0,0],[0,1]], [[0,0],[0,1]], [[0,0],[1,0]]),
        # M5: U=[[1,1],[0,0]], V=[[0,0],[0,1]], contributes to C: [[0,0],[-1,0]]
        ([[1,1],[0,0]], [[0,0],[0,1]], [[0,0],[-1,0]]),
        # M6: U=[[-1,0],[1,0]], V=[[1,1],[0,0]], contributes to C: [[0,0],[0,1]]
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        # M7: U=[[0,1],[-1,0]], V=[[0,0],[1,1]], contributes to C: [[1,0],[0,0]]
        ([[0,1],[-1,0]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    # Verify Strassen: let me double-check with a simple example
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # M2 = (c+d)e = ce + de
    # M3 = a(f) = af
    # M4 = d(h) = dh
    # M5 = (a+b)h = ah + bh
    # M6 = (c-a)(e+f) = ce + cf - ae - af
    # M7 = (b-d)(g+h) = bg + bh - dg - dh
    
    # C[0,0] should be ae+bg. 
    # M1 - M2 + M4 - M5 + M7 = (ae+ah+de+dh) - (ce+de) + dh - (ah+bh) + (bg+bh-dg-dh)
    # = ae + ah + de + dh - ce - de + dh - ah - bh + bg + bh - dg - dh
    # = ae + bg - ce + dh - dg + dh - dh... let me redo
    # = ae + (ah-ah) + (de-de) + (dh+dh-dh) + bg + (-bh+bh) - dg - ce
    # = ae + dh + bg - dg - ce... that's not right.
    
    # Let me use the standard Strassen formulas more carefully:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4  
    # C11 = M1 - M2 + M3 + M6
    
    # Recomputing:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # = (a+d)(e+h) + d(h-f) - (a+b)h + (b-d)(g+h)
    # = ae+ah+de+dh + dh-df - ah-bh + bg+bh-dg-dh
    # = ae + (ah-ah) + de + (dh+dh-dh) + bg + (-df-dg) + (-bh+bh)
    # = ae + de + dh + bg - df - dg
    # That's not ae+bg either. Something's off.
    
    # I think I'm confusing myself. Let me just implement it and verify numerically.
    # For now, let me use a well-known set of Strassen coefficients.
    
    # After careful re-derivation, here are correct Strassen terms:
    # For C = A*B where A, B, C are 2x2 (0-indexed):
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[0,0])
    # M4 = A[1,1]*(B[1,1]-B[0,1])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #
    # C[0,0] = M1 - M3 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M4 + M6
    
    # Let me verify C[0,0] = M1 - M3 - M5 + M7:
    # M1 - M3 - M5 + M7
    # = (a+d)(e+h) - a(g-e) - (a+b)h + (b-d)(g+h)
    # = ae+ah+de+dh - ag+ae - ah-bh + bg+bh-dg-dh
    # = 2ae + de + dh - ag + bg - dg
    # Still not right. I must be using wrong formulas.
    
    # Let me look up the standard:
    # p1 = (a + d)·(e + h)
    # p2 = (c + d)·e
    # p3 = a·(f - h)
    # p4 = d·(g - e)
    # p5 = (a + b)·h
    # p6 = (c - a)·(e + f)
    # p7 = (b - d)·(g + h)
    #
    # a' = p1 + p4 - p5 + p7
    # b' = p3 + p5
    # c' = p2 + p4
    # d' = p1 - p2 + p3 + p6
    
    # Verify a' = p1 + p4 - p5 + p7:
    # = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    # = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    # = ae + bg + (ah-ah) + (de-de) + (dh-dh) + (-bh+bh) + (dg-dg)
    # = ae + bg ✓
    
    # OK so the correct Strassen terms are:
    strassen_terms = [
        # M1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W for contribution to C
        # C[0,0]+=1, C[0,1]+=0, C[1,0]+=0, C[1,1]-=1 (from d' = p1 - ...)
        # Wait, p1 contributes +1 to C[0,0] and +1 to C[1,1]
        # But then p4, p5, p7 also contribute to C[0,0] with various signs.
        # So for each term, I need to specify which C entries it contributes to and with what sign.
        #
        # C[0,0] = p1 + p4 - p5 + p7
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p4
        # C[1,1] = p1 - p2 + p3 + p6
        #
        # So:
        # p1: contributes +1 to C[0,0], +1 to C[1,1]
        # p2: contributes +1 to C[1,0], -1 to C[1,1]
        # p3: contributes +1 to C[0,1], +1 to C[1,1]
        # p4: contributes +1 to C[0,0], +1 to C[1,0]
        # p5: contributes -1 to C[0,0], +1 to C[0,1]
        # p6: contributes +1 to C[1,1]
        # p7: contributes +1 to C[0,0]
        
        # p1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        # p2: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[0,0],[1,-1]]
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,-1]]),
        # p3: U=[[1,0],[0,0]], V=[[0,-1],[0,0]], W=[[0,1],[0,1]]
        ([[1,0],[0,0]], [[0,-1],[0,0]], [[0,1],[0,1]]),
        # p4: U=[[0,0],[0,1]], V=[[-1,0],[0,0]], W=[[1,0],[1,0]]
        ([[0,0],[0,1]], [[-1,0],[0,0]], [[1,0],[1,0]]),
        # p5: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[-1,0],[0,1]]
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]]),
        # p6: U=[[-1,0],[1,0]], V=[[1,1],[0,0]], W=[[0,0],[0,1]]
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        # p7: U=[[0,1],[-1,0]], V=[[0,0],[1,1]], W=[[1,0],[0,0]]
        ([[0,1],[-1,0]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    # Verify by constructing the full 2x2x2 tensor and checking rank 7
    # Let's do a numerical check
    def verify_decomposition(U_full, V_full, W_full, expected_U_shape, expected_V_shape, expected_W_shape):
        """Verify that sum of rank-1 terms equals the matrix multiplication tensor."""
        import numpy as np
        n, m, p = DIMENSIONS
        U_full = np.array(U_full)
        V_full = np.array(V_full)
        W_full = np.array(W_full)
        
        # Reconstruct tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The matrix multiplication tensor is T[i,j,k] = delta[i,k]... no
        # T[i,j,k] = 1 if this is a valid (i,j) from A and (j,k) from B producing (i,k) in C
        # Actually the tensor for matrix multiplication: T_{ijk} = 1 where A has indices (i,j) and B has (j,k)
        # So T[i,j,k] = 1 for all i,j,k (it's the identity on the index j)
        # The tensor is: sum_{i,j,k} A[i,j] * B[j,k] * C_{output}[i,k]
        # In the rank-1 decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where the sum over j is implicit? No, U has shape (n*m, rank), so U[i*m+j, r] for fixed i but j varies.
        # 
        # The decomposition means: for all A, B: (AB)[i,k] = sum_{r} (sum_j U[i*m+j, r] * A[i,j]) * ... 
        # No wait. The tensor decomposition is:
        # The matrix multiplication tensor M has entries M[i,j,k] = 1 if the j-index matches, i.e., it's the
        # tensor such that M : A : B = A*B where (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # 
        # M[i,j,k] = 1 (it's a 3-tensor with all entries 1? No)
        # M[i,j,k] = delta condition: the tensor is such that contracting with A and B gives AB
        # <M, A⊗B> = sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] = sum_{i,k} (AB)[i,k] * 1
        # So M[i,j,k] = 1 for all i,j,k? That would give sum_{i,j,k} A[i,j]B[j,k] = (sum A)(sum B) which is wrong.
        # 
        # Actually M[i,j,k] = 1, and the contraction is sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] 
        # But this sums over independent i,j,k, which doesn't enforce j matching.
        # 
        # I think the correct tensor is: M is defined on indices (i, j, k) where j is shared:
        # The tensor has entries indexed by (row_of_A, col_of_A_row_of_B, col_of_B_col_of_C)
        # M[i, j, k] = 1, and the product tensor = sum_{i,j,k} M[i,j,k] * A[i,j] * B[j,k]
        # This equals sum_{i,j,k} A[i,j] B[j,k] which is indeed (A*B) viewed as a scalar... no.
        # sum_{i,j,k} A[i,j]B[j,k] is a scalar, not a matrix.
        # 
        # I think I'm overcomplicating this. The rank-1 decomposition of the matrix multiplication tensor
        # means: A*B = sum_r (U_r · A) ⊙ (B · V_r^T) · W_r where ⊙ is outer product or something.
        # 
        # Actually, the standard formulation: the tensor for matrix multiplication C = AB has the property
        # that each entry C[i,k] = sum_j A[i,j] B[j,k]. In the tensor network, this is a single tensor
        # M of dimension n×m×p (or n×p×m depending on convention) where M[i,k,j] = 1, and
        # C[i,k] = sum_j A[i,j] B[j,k] means we're contracting the j-index.
        # 
        # The rank-r decomposition means: M = sum_{r=1}^r u_r ⊗ v_r ⊗ w_r where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, 
        # w_r ∈ R^{p*n} (following the W[k*n+i] convention).
        # 
        # So the reconstruction is: for each (i,j,k) in A×B, the contribution to C[i,k] from term r is
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]. But this gives C[i,k] as a function of i,k, and we
        # need it to equal sum_j A[i,j]B[j,k].
        # 
        # The way it works: the "input" tensors A and B are multiplied by U and V to get intermediate
        # vectors, then those are multiplied and projected by W.
        # 
        # More precisely: define for term r:
        # a_r = U_r^T * A_flat (but A is 2D...)
        # 
        # I think the correct check is: for random A (n×m) and B (m×p), verify that
        # A*B = sum_r (outer(A * U[:,r] ... ))
        # 
        # Let me just do the numerical check directly by constructing the tensor and checking.
        
        # Build the full tensor T[i, j, k] = 1 for the matrix multiplication
        # T is such that sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} (A*B)[i,k]
        # This requires T[i,j,k] to "pick out" the right j for each i,k pair.
        # Actually, the tensor M for matrix multiplication satisfies:
        # (M · (A⊗B)) = vec(A*B)^T where the dot is the appropriate contraction.
        # 
        # In index notation: (AB)_{ik} = sum_j A_{ij} B_{jk}
        # The tensor M has indices (i, j, k) and M_{ijk} = 1, and the equation is:
        # sum_j M_{ijk} A_{ij} B_{jk} = (AB)_{ik}
        # This works if M_{ijk} = 1 for all i,j,k, because then LHS = sum_j A_{ij} B_{jk} = (AB)_{ik} ✓
        # 
        # Wait, that's just the definition! So M_{ijk} = 1 for all i, j, k.
        # 
        # Now the rank-1 decomposition: M = sum_r u_r ⊗ v_r ⊗ w_r means:
        # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i, j, k
        # 
        # But that's saying the tensor of all 1s is decomposed, which doesn't make sense for matrix mult.
        # 
        # I think the issue is that the tensor for matrix multiplication isn't the all-1s tensor.
        # The correct tensor is: for the bilinear form to compute AB, we need a different structure.
        # 
        # After more thought: the rank of the matrix multiplication tensor refers to the minimum number
        # of rank-1 tensors whose sum equals the matrix multiplication tensor, where the matrix multiplication
        # tensor M is defined by M[i, j, k] = δ_{j, j'} (i.e., it enforces index matching). But in the 
        # n×m, m×p → n×p setting, the j-index is shared, so M is a 3-tensor of size n×m×p with M[i,j,k] = 1.
        # 
        # The rank-1 decomposition M = sum_r u_r v_r w_r with u_r ∈ R^n, v_r ∈ R^m, w_r ∈ R^p would be
        # standard, but here we have u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n} with the specific indexing.
        # 
        # The check should be: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # because M[i,j,k] = 1.
        
        T_reconstructed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U_full.shape[1]):
                        s += U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r]
                    T_reconstructed[i,j,k] = s
        
        # The tensor should be all 1s
        expected = np.ones((n, m, p))
        if not np.allclose(T_reconstructed, expected):
            return False
        return True
    
    # Now construct the decomposition with partitioning
    # Partition: A rows: [0,1] and [2], B cols: [0,1] and [2,3,4], middle: [0,1] and [2,3]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(row_offset, row_count, mid_offset, mid_count, col_offset, col_count, use_strassen=False):
        """Add terms for a (row_count × mid_count) × (mid_count × col_count) block."""
        nonlocal U_cols, V_cols, W_cols
        U_local = [[0]*row_count for _ in range(mid_count)]  # Actually U is (n*m, rank), indexed by i*m+j
        # U has shape (n*m, rank). For a sub-block with rows i0..i0+rc-1, cols j0..j0+mc-1:
        # U[i*m + j, r] for i in [i0, i0+rc), j in [j0, j0+mc)
        
        if use_strassen and row_count == 2 and mid_count == 2 and col_count == 2:
            # Use Strassen
            for u_pat, v_pat, w_pat in strassen_terms:
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                for di in range(2):
                    for dj in range(2):
                        u_col[(row_offset + di) * m + (mid_offset + dj)] = u_pat[di][dj]
                for dj in range(2):
                    for dk in range(2):
                        v_col[(mid_offset + dj) * p + (col_offset + dk)] = v_pat[dj][dk]
                for dk in range(2):
                    for di in range(2):
                        w_col[(col_offset + dk) * n + (row_offset + di)] = w_pat[di][dk]
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
        else:
            # Schoolbook: each (i,j,k) triple gives one term
            for di in range(row_count):
                for dj in range(mid_count):
                    for dk in range(col_count):
                        u_col = [0] * (n * m)
                        v_col = [0] * (m * p)
                        w_col = [0] * (p * n)
                        i = row_offset + di
                        j = mid_offset + dj
                        k = col_offset + dk
                        u_col[i * m + j] = 1
                        v_col[j * p + k] = 1
                        w_col[k * n + i] = 1
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
    
    # Sub-blocks:
    # (rows 0-1, mid 0-1, cols 0-1): 2x2x2 -> Strassen, rank 7
    # (rows 0-1, mid 0-1, cols 2-4): 2x2x3 -> schoolbook, rank 12
    # (rows 0-1, mid 2-3, cols 0-1): 2x2x2 -> Strassen, rank 7
    # (rows 0-1, mid 2-3, cols 2-4): 2x2x3 -> schoolbook, rank 12
    # (row 2, mid 0-1, cols 0-1): 1x2x2 -> schoolbook, rank 4
    # (row 2, mid 0-1, cols 2-4): 1x2x3 -> schoolbook, rank 6
    # (row 2, mid 2-3, cols 0-1): 1x2x2 -> schoolbook, rank 4
    # (row 2, mid 2-3, cols 2-4): 1x2x3 -> schoolbook, rank 6
    
    add_block(0, 2, 0, 2, 0, 2, use_strassen=True)
    add_block(0, 2, 0, 2, 2, 3, use_strassen=False)
    add_block(0, 2, 2, 2, 0, 2, use_strassen=True)
    add_block(0, 2, 2, 2, 2, 3, use_strassen=False)
    add_block(2, 1, 0, 2, 0, 2, use_strassen=False)
    add_block(2, 1, 0, 2, 2, 3, use_strassen=False)
    add_block(2, 1, 2, 2, 0, 2, use_strassen=False)
    add_block(2, 1, 2, 2, 2, 3, use_strassen=False)
    
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U[i][r] = U_cols[r][i]
        for i in range(m*p):
            V[i][r] = V_cols[r][i]
        for i in range(p*n):
            W[i][r] = W_cols[r][i]
    
    # Verify
    import numpy as np
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                max_err = max(max_err, abs(s - 1))
    
    if max_err > 1e-10:
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
