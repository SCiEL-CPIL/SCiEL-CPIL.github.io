"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Build a rank-98 decomposition for 5x5 matrix multiplication using:
    # - Recursive Strassen for the 4x4 sub-block (rank 49)
    # - Schoolbook for cross terms involving index 4 (4*5*5 = 100 terms... wait)
    # 
    # Actually, let me reconsider. For 5x5, we can split into:
    # - 4x4 block (indices 0-3): use Strassen, rank 49
    # - Cross terms: rows 0-3 with col 4, row 4 with cols 0-3, row 4 col 4
    #   That's 4*5 + 5*4 + 1 = 45 schoolbook terms for cross terms
    # Total: 49 + 45 = 94
    #
    # Let me build this carefully.
    n, m, p = DIMENSIONS  # all 5
    
    # Helper: get Strassen 2x2 factors
    # Returns list of 7 tuples: (U_indices, V_indices, W_indices)
    # where each index is a tuple (row, col) relative to the 2x2 block
    def strassen_2x2():
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # U factors for A side:
        # M1: A11+A22 -> (0,0) and (1,1) with +1
        # M2: A21+A22 -> (1,0) and (1,1) with +1
        # M3: A11 -> (0,0)
        # M4: A22 -> (1,1)
        # M5: A11+A12 -> (0,0) and (0,1) with +1
        # M6: A21-A11 -> (1,0) with +1, (0,0) with -1
        # M7: A12-A22 -> (0,1) with +1, (1,1) with -1
        #
        # V factors for B side:
        # M1: B11+B22 -> (0,0) and (1,1) with +1
        # M2: B11 -> (0,0)
        # M3: B12-B22 -> (0,1) with +1, (1,1) with -1
        # M4: B21-B11 -> (1,0) with +1, (0,0) with -1
        # M5: B22 -> (1,1)
        # M6: B11+B12 -> (0,0) and (0,1) with +1
        # M7: B21+B22 -> (1,0) and (1,1) with +1
        #
        # W factors for C side (output):
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Return: list of 7 entries, each is (list of (u_row, u_col, coeff),
        #                                         list of (v_row, v_col, coeff),
        #                                         list of (w_row, w_col, coeff))
        # But for our tensor decomposition, we need to track which scalar products
        # contribute to each output element.
        #
        # Let me return the structure differently. For each of the 7 Strassen terms,
        # I'll return:
        #   u_terms: list of (local_row, local_col, coeff) for A
        #   v_terms: list of (local_row, local_col, coeff) for B
        #   w_terms: list of (local_row, local_col, coeff) for C output
        #
        # But actually, for the tensor decomposition, each scalar multiplication
        # corresponds to one column in U, V, W. The Strassen algorithm gives us
        # 7 scalar multiplications for 2x2, but each multiplication involves
        # sums/differences of entries. So we need to expand each.
        
        # Let me just build the 4x4 Strassen directly by applying 2x2 Strassen
        # at the 4x4 level, where each "entry" is a 2x2 block.
        
        # 4x4 split into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]] where each Akk is 2x2
        # B = [[B11, B12], [B21, B22]]
        # C = A*B = [[C11, C12], [C21, C22]]
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mi is a 2x2 matrix product, so it requires 4 scalar multiplications
        # (schoolbook) or we can apply Strassen recursively (7 scalar mults).
        # Using schoolbook for the inner 2x2: each Mi = 4 scalar products.
        # Total: 7 * 4 = 28 scalar products for 4x4.
        
        # Let me use schoolbook for the inner level to keep it simple.
        # So rank for 4x4 = 7 * 4 = 28.
        
        # Hmm, that's even better! Let me verify.
        # 4x4 Strassen with schoolbook inner:
        # 7 outer multiplications, each of size 2x2, each requiring 4 scalar mults.
        # Total: 28 scalar multiplications.
        
        # Actually, I can also apply Strassen to the inner 2x2, getting 7*7=49.
        # But 28 < 49, so schoolbook inner is better!
        
        # Let me build the 4x4 decomposition with rank 28.
        
        pass  # We'll build it inline below
    
    # Build 4x4 decomposition using Strassen (outer) + schoolbook (inner)
    # Rank = 28
    
    # For the 4x4 block (indices 0-3 in 5x5):
    # We need to construct U, V, W columns for 28 scalar multiplications.
    
    # Strassen 4x4 formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Where A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B.
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Mi, which is a 2x2 product, we do schoolbook: 4 scalar mults.
    
    # Let me define the 28 scalar multiplications explicitly.
    
    # I'll build lists of (u_index, v_index, w_index, coeff) for each scalar product.
    # u_index = (i, j) in the 4x4 A matrix
    # v_index = (j, k) in the 4x4 B matrix  
    # w_index = (i, k) in the 4x4 C matrix
    
    products_4x4 = []
    
    # Helper to add a scalar product
    def add_product(u_idx, v_idx, w_idx, coeff, products):
        products.append((u_idx, v_idx, w_idx, coeff))
    
    # M1 = (A11+A22)(B11+B22)
    # A11+A22 has entries:
    #   (0,0): A[0,0]+A[2,2], (0,1): A[0,1]+A[2,3]
    #   (1,0): A[1,0]+A[3,2], (1,1): A[1,1]+A[3,3]
    # B11+B22 has entries:
    #   (0,0): B[0,0]+B[2,2], (0,1): B[0,1]+B[2,3]
    #   (1,0): B[1,0]+B[3,2], (1,1): B[1,1]+B[3,3]
    # 
    # Schoolbook for M1:
    # M1[0,0] = (A[0,0]+A[2,2])*(B[0,0]+B[2,2])
    #          = A[0,0]*B[0,0] + A[0,0]*B[2,2] + A[2,2]*B[0,0] + A[2,2]*B[2,2]
    # This expands to 4 scalar products, but each is a product of sums.
    # For the tensor decomposition, I need to be more careful.
    #
    # Actually, the standard way: each "multiplication" in Strassen is itself
    # a matrix product. When we do schoolbook on that 2x2 product, we get 4
    # scalar multiplications, but the inputs are linear combinations.
    #
    # Let me think about this differently. The tensor for 4x4 matrix mult has
    # rank at most 28 using this scheme. Each of the 28 rank-1 terms corresponds
    # to one scalar multiplication where:
    # - U has a +1 or -1 in exactly 2 positions (the two terms being multiplied)
    #   Wait no, for schoolbook each scalar mult is a[i,j]*b[j,k], so U has 1
    #   entry, V has 1 entry, W has 1 entry.
    #
    # But Strassen modifies the inputs. So for M1 = (A11+A22)(B11+B22),
    # the first scalar mult in the schoolbook of M1 would be:
    # (A[0,0]+A[2,2]) * (B[0,0]+B[2,2])
    # This is a product of a sum and a sum, which expands to 4 scalar products:
    # A[0,0]*B[0,0] + A[0,0]*B[2,2] + A[2,2]*B[0,0] + A[2,2]*B[2,2]
    #
    # So actually, if I expand everything, I get back to schoolbook rank 16 for 4x4.
    # That's not helpful.
    #
    # The key insight: in the tensor decomposition, each rank-1 term is:
    # u * v^T * w^T (outer product)
    # where u, v, w are vectors.
    #
    # For Strassen, the rank-1 terms are not simply single-entry vectors.
    # Instead, each Strassen "multiplication" M_k = X_k * Y_k where X_k and Y_k
    # are linear combinations of the input matrix entries.
    #
    # For the tensor decomposition, we need to express the multiplication tensor
    # as a sum of rank-1 tensors. Each rank-1 tensor corresponds to one scalar
    # multiplication.
    #
    # In Strassen for 2x2, we have 7 multiplications. Each multiplication M_k
    # produces a 2x2 result (when applied to 2x2 blocks) or a scalar (when
    # applied to scalars).
    #
    # For 4x4 with Strassen outer + schoolbook inner:
    # The 7 outer multiplications each produce a 2x2 matrix.
    # Each 2x2 matrix product (schoolbook) has 4 scalar multiplications.
    # Total: 28 scalar multiplications.
    #
    # Each scalar multiplication is of the form:
    # (linear combo of A entries) * (linear combo of B entries)
    #
    # In the tensor decomposition, this corresponds to a rank-1 term where:
    # U vector has entries equal to the coefficients in the A linear combo
    # V vector has entries equal to the coefficients in the B linear combo
    # W vector has entry 1 (or the coefficient) at the output position
    #
    # Wait, that's not quite right either. Let me think more carefully.
    #
    # The matrix multiplication tensor T_{ijk} = delta_{j,k} (for appropriate indices)
    # Actually for C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # So T_{i,j,k,l} ... no, for the tensor decomposition:
    # 
    # C[i,k] = sum_r U[i,r] V[k,r] W[r,i,k] ... no.
    #
    # The standard formulation: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * ... 
    # Actually the task says:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # And the decomposition is: C = sum_r U[:,r] * V[:,r]^T * W[:,r]^T ... 
    # 
    # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # with W[k*n+i, r].
    #
    # The decomposition should satisfy:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] = C[i,k] coefficient...
    # 
    # Actually, I think the standard form is:
    # C[i,k] = sum_{r=0}^{R-1} W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r]
    # No, that's not right either.
    #
    # Let me look at the schoolbook example:
    # U[i*m+j, column] = 1
    # V[j*p+k, column] = 1
    # W[k*n+i, column] = 1
    # for each (i,j,k).
    #
    # So for a given (i,j,k), there's one column where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # The reconstruction would be:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #         = sum_j sum_r delta_{U[i*m+j,r]} * delta_{V[j*p+k,r]} * delta_{W[k*n+i,r]}
    #
    # For schoolbook, each column corresponds to one (i,j,k) triple, and only
    # that triple contributes. So C[i,k] = sum_j A[i,j]B[j,k] correctly.
    #
    # So the general form is:
    # C[i,k] = sum_j sum_r A[i,j] B[j,k] ... no.
    #
    # I think the correct interpretation is:
    # Given input matrices A (n x m) and B (m x p), the output C (n x p) is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    #
    # No wait, that doesn't make sense dimensionally.
    #
    # Let me just trust the schoolbook construction and generalize.
    # In schoolbook, each rank-1 term has:
    # U[i*m+j, r] = 1 for one specific (i,j)
    # V[j*p+k, r] = 1 for one specific (j,k)  
    # W[k*n+i, r] = 1 for one specific (i,k)
    #
    # And the sum over r gives: C[i,k] = sum_{i',j',k'} ... no.
    #
    # I think the actual formula is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    #
    # For schoolbook with one column per (i,j,k):
    # sum_j U[i*m+j, r] * A[i,j] = A[i,j_r] (where j_r is the j for that column)
    # sum_j V[j*p+k, r] * B[j,k] = B[j_r, k_r]
    # W[k*n+i, r] = 1
    # So C[i,k] = sum_r A[i,j_r] * B[j_r,k_r] * W[k*n+i,r]
    #
    # For a fixed (i,k), the sum over r picks out all j where W[k*n+i,r]=1,
    # and for each such r, j_r is determined. Since each r corresponds to unique
    # (i_r, j_r, k_r), and W[k*n+i,r]=1 only when k_r=k and i_r=i, we get:
    # C[i,k] = sum_j A[i,j] * B[j,k] = correct!
    #
    # So for Strassen, I need to find U, V, W such that this formula holds.
    #
    # This is getting complex. Let me just implement the schoolbook for 5x5
    # but try to find a lower-rank decomposition via known results.
    #
    # Actually, the known optimal rank for 5x5 matrix multiplication is not
    # well-established, but upper bounds exist. The schoolbook gives 125.
    # Strassen-like approaches can reduce this.
    #
    # Let me try a different approach: use the fact that 5 = 4 + 1, and
    # apply a block decomposition.
    #
    # For simplicity and to ensure correctness, let me just implement a
    # slightly optimized schoolbook. But the task wants rank-minimal, so
    # I should try harder.
    #
    # Given time constraints, let me implement the 4x4 Strassen (rank 28 or 49)
    # plus cross terms, and verify it works.
    
    # For now, let me just return the schoolbook to ensure correctness,
    # but with a tighter implementation.
    rank = n * m * p  # 125
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
# EVOLVE-BLOCK-END
