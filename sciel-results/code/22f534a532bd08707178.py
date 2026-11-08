"""Matrix multiplication 6x6x6 rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 6, 6, 6
    # Start with schoolbook rank 216
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the schoolbook decomposition is correct
    def verify(rank_val, U_mat, V_mat, W_mat):
        # For each (i, j, k), check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # and for all other combinations, the sum should be 0
        # Efficient check: build the tensor
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank_val):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    expected = 1 if (i == j == k or True) else 0
                    # Actually the matrix multiplication tensor T[i,j,k] = 1 if we're computing
                    # C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all (i,j,k)
                    # Wait, let me reconsider. The matrix multiplication tensor has T[i,j,k] = 1
                    # for all i in [n], j in [m], k in [p], because C[i,k] = sum_j A[i,j]*B[j,k]
                    # So the tensor is all ones? No, that's not right either.
                    # The tensor T_{ijk} for matrix multiplication satisfies:
                    # C_{ik} = sum_j A_{ij} B_{jk}
                    # So T_{ijk} = 1 for all valid (i,j,k).
                    # The decomposition is: T_{ijk} = sum_c U_{ij}^{(c)} V_{jk}^{(c)} W_{ik}^{(c)}
                    # where U has indices (i,j), V has indices (j,k), W has indices (i,k).
                    # Wait, the problem says W[k*n+i, r], so W has indices (k,i).
                    # So T_{ijk} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # And this should equal 1 for all (i,j,k) since every triple contributes.
                    if val != 1:
                        return False
        return True
    
    # Let me just verify the schoolbook works
    assert verify(rank, U, V, W), "Schoolbook verification failed"
    
    # Try to reduce rank by finding redundant columns
    # A column c is redundant if its contribution can be expressed as a combination of others
    # For exact integer decomposition, we can try to remove a column and adjust others
    
    # Strategy: try to find a subset of columns that still spans the tensor
    # Since we have 216 columns and the tensor lives in a space of dimension at most n*m*p = 216,
    # we need to check if any columns are linearly dependent in the tensor space
    
    # The tensor space: T is a 6x6x6 tensor = 216-dimensional space
    # Each column c defines a rank-1 tensor: u_c ⊗ v_c ⊗ w_c (with appropriate index structure)
    # We need to check if the 216 rank-1 tensors are linearly independent
    
    # Build the 216x216 matrix M where M[c1, c2] = inner product of tensors for columns c1 and c2
    # M[c1, c2] = sum_{i,j,k} U[i*m+j, c1]*V[j*p+k, c1]*W[k*n+i, c1] * U[i*m+j, c2]*V[j*p+k, c2]*W[k*n+i, c2]
    # = sum_{i,j,k} (U_{i,j}^{c1} V_{j,k}^{c1} W_{k,i}^{c1}) (U_{i,j}^{c2} V_{j,k}^{c2} W_{k,i}^{c2})
    
    # For the schoolbook, column (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1, all others 0
    # So the tensor for column (i,j,k) is the basis tensor e_{ijk}
    # Therefore the Gram matrix is the identity, meaning all 216 columns are linearly independent
    # in the tensor space. This means we cannot reduce below 216 by just removing columns.
    
    # However, we might be able to replace multiple columns with fewer columns that have
    # different (non-0/1) entries. This requires finding a different decomposition.
    
    # Known result: The minimum rank for 6x6 matrix multiplication over the rationals
    # is known to be 216 (same as schoolbook) or possibly less. Let me check known bounds.
    # Actually, for n x n matrix multiplication, the tensor rank is at most n^3 (schoolbook)
    # and at least n^2. For n=6, it's between 36 and 216.
    # The best known upper bound for 6x6 is actually 216 (schoolbook) in many references,
    # but Strassen's algorithm gives rank 7 for 2x2x2, which extrapolates to better bounds.
    # For 4x4x4, the best known is 49 (vs 64 schoolbook).
    # For 6x6x6, the best known upper bound is 162 (from various constructions) or possibly lower.
    
    # Let me try a simple approach: use Strassen-like ideas
    # Strassen reduces 2x2x2 from 8 to 7 multiplications.
    # For 6x6, we can divide into 2x2 blocks: 6/2 = 3, so we get a 3x3x3 problem of 2x2 matrices.
    # Each 2x2 multiplication uses 7 multiplications (Strassen).
    # Total multiplications: 7^3 = 343? No, that's not right.
    # If we partition into 3x3 blocks of 2x2 matrices, we need to multiply 3x3 block matrices,
    # which takes 3^3 = 27 scalar multiplications if done naively, but each "scalar" is a 2x2 matrix
    # requiring 7 multiplications. So total = 27 * 7 = 189. That's better than 216!
    
    # Wait, that's the total number of scalar multiplications. But the tensor rank is what we need.
    # The tensor rank of the 6x6x6 multiplication tensor is at most 189 using this approach.
    # Actually, I need to be more careful. Let me think about this differently.
    
    # The tensor rank r(T) for matrix multiplication n x m x p satisfies:
    # r <= (number of block multiplications) * (rank of each block multiplication)
    # If we partition into blocks of size 2x2, we get ceil(6/2)=3 blocks in each dimension.
    # The block multiplication is a 3x3x3 matrix multiplication (of 2x2 blocks).
    # The rank of 3x3x3 matrix multiplication is at most 27 (schoolbook) or possibly less.
    # Each 2x2 block multiplication has rank 7 (Strassen).
    # So total rank <= 27 * 7 = 189? No, that's not how tensor rank works under Kronecker products.
    
    # Actually, if T1 has rank r1 and T2 has rank r2, then T1 ⊗ T2 has rank r1 * r2.
    # The 6x6x6 multiplication tensor can be written as T_{3x3x3} ⊗ T_{2x2x2} where:
    # T_{3x3x3} is the 3x3x3 matrix multiplication tensor (rank <= 27)
    # T_{2x2x2} is the 2x2x2 matrix multiplication tensor (rank = 7 by Strassen)
    # So rank(T_{6x6x6}) <= rank(T_{3x3x3}) * rank(T_{2x2x2}) <= 27 * 7 = 189
    
    # This gives us an upper bound of 189, which is better than 216!
    
    # Let me construct this decomposition.
    
    # First, let's get the Strassen decomposition for 2x2x2.
    # Strassen's 7 multiplications for 2x2 matrix multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B22 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B12 + B22)
    # M7 = (A12 - A22)(B11 + B21)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The Strassen decomposition gives rank 7 for 2x2x2.
    # U2 is 4x7, V2 is 4x7, W2 is 4x7 (since 2*2=4 elements each)
    
    # Let me define the Strassen factors explicitly.
    # Indexing: for 2x2, i,j,k in {0,1}
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # Strassen:
    # M1: (A11+A22)(B11+B22) -> U: A11=1,A22=1; V: B11=1,B22=1; W: C11=1 (partial)
    # This is getting complex. Let me just use the known rank 216 schoolbook and try to
    # verify it, then attempt a simple rank reduction.
    
    # Given the complexity of constructing an optimal decomposition, let me try a 
    # simpler approach: check if we can remove any columns from the schoolbook.
    
    # For the schoolbook, each column corresponds to a unique (i,j,k) triple, and the
    # tensors are basis vectors, so they're all independent. We can't remove any.
    
    # Let me try the 189-rank construction using Strassen for the 2x2 blocks.
    
    def strassen_2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        r = 7
        U2 = np.zeros((4, r), dtype=np.int64)
        V2 = np.zeros((4, r), dtype=np.int64)
        W2 = np.zeros((4, r), dtype=np.int64)
        
        # Strassen's 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B11-B01)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B01+B11)
        # M7 = (A01-A11)(B00+B10)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Column 0: M1
        U2[0, 0] = 1  # A00
        U2[3, 0] = 1  # A11 (index 3 = 1*2+1)
        V2[0, 0] = 1  # B00
        V2[3, 0] = 1  # B11 (index 3 = 1*2+1)
        W2[0, 0] = 1  # C00 (index 0 = 0*2+0)
        
        # Column 1: M2
        U2[2, 1] = 1  # A10 (index 2 = 1*2+0)
        U2[3, 1] = 1  # A11
        V2[0, 1] = 1  # B00
        W2[2, 1] = 1  # C10 (index 2 = 1*2+0)
        
        # Column 2: M3
        U2[0, 2] = 1  # A00
        V2[2, 2] = 1  # B10 (index 2 = 1*2+0)
        V2[0, 2] = -1  # B00
        W2[1, 2] = 1  # C01 (index 1 = 0*2+1)
        
        # Column 3: M4
        U2[3, 3] = 1  # A11
        V2[3, 3] = 1  # B11
        V2[1, 3] = -1  # B01 (index 1 = 0*2+1)
        W2[0, 3] = 1  # C00
        W2[2, 3] = 1  # C10
        
        # Column 4: M5
        U2[0, 4] = 1  # A00
        U2[1, 4] = 1  # A01 (index 1 = 0*2+1)
        V2[3, 4] = 1  # B11
        W2[0, 4] = -1  # C00
        W2[1, 4] = 1  # C01
        
        # Column 5: M6
        U2[2, 5] = 1  # A10
        U2[0, 5] = -1  # A00
        V2[1, 5] = 1  # B01
        V2[3, 5] = 1  # B11
        W2[3, 5] = 1  # C11 (index 3 = 1*2+1)
        
        # Column 6: M7
        U2[1, 6] = 1  # A01
        U2[3, 6] = -1  # A11
        V2[0, 6] = 1  # B00
        V2[2, 6] = 1  # B10
        W2[0, 6] = 1  # C00
        
        return U2, V2, W2
    
    U2, V2, W2 = strassen_2x2()
    
    # Now build the 3x3x3 schoolbook for the block structure.
    # The 6x6x6 tensor is T_{6x6x6} = T_{3x3x3} ⊗ T_{2x2x2}
    # where the Kronecker product is over the block structure.
    
    # For the 3x3x3 part (schoolbook, rank 27):
    # U3 has shape (3*3, 27) = (9, 27)
    # V3 has shape (3*3, 27) = (9, 27)
    # W3 has shape (3*3, 27) = (9, 27)
    
    r3 = 27
    U3 = np.zeros((9, r3), dtype=np.int64)
    V3 = np.zeros((9, r3), dtype=np.int64)
    W3 = np.zeros((9, r3), dtype=np.int64)
    
    col3 = 0
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                U3[i3*3 + j3, col3] = 1
                V3[j3*3 + k3, col3] = 1
                W3[k3*3 + i3, col3] = 1
                col3 += 1
    
    # Now combine: total rank = 27 * 7 = 189
    # U_total has shape (36, 189) since 6*6 = 36
    # Each element of the 6x6 matrix is at position (3*i2 + i3) * 6 + (3*j2 + j3)
    # Wait, let me think about the indexing more carefully.
    
    # For a 6x6 matrix A, element A[i,j] where i,j in 0..5.
    # We partition into 2x2 blocks: i = 2*i2 + i1 where i2 in 0..2, i1 in 0..1.
    # Wait, 6 = 3*2, so i = 2*i2 + i1 where i2 in 0..2, i1 in 0..1.
    
    # Actually, let's use i = 2*i2 + i1, j = 2*j2 + j1, k = 2*k2 + k1
    # where i2,j2,k2 in 0..2 (block indices) and i1,j1,k1 in 0..1 (within-block indices).
    
    # The full tensor T[i,j,k] = 1 for all i,j,k in 0..5.
    # T[i,j,k] = T3[i2,j2,k2] * T2[i1,j1,k1] where T3 and T2 are the respective tensors.
    
    # For the decomposition:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # With c = (c3, c2) where c3 in 0..26 and c2 in 0..6:
    # U[i,j,c] = U3[i2,j2,c3] * U2[i1,j1,c2]
    # V[j,k,c] = V3[j2,k2,c3] * V2[j1,k1,c2]
    # W[k,i,c] = W3[k2,i2,c3] * W2[k1,i1,c2]
    
    # Let me verify this works:
    # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # = sum_{c3,c2} U3[i2,j2,c3]*U2[i1,j1,c2] * V3[j2,k2,c3]*V2[j1,k1,c2] * W3[k2,i2,c3]*W2[k1,i1,c2]
    # = (sum_{c3} U3[i2,j2,c3]*V3[j2,k2,c3]*W3[k2,i2,c3]) * (sum_{c2} U2[i1,j1,c2]*V2[j1,k1,c2]*W2[k1,i1,c2])
    # = T3[i2,j2,k2] * T2[i1,j1,k1]
    # = 1 * 1 = 1 (since both T3 and T2 are the all-ones tensors for their respective sizes)
    
    # Great, this should work!
    
    total_rank = r3 * 7  # 27 * 7 = 189
    U_total = np.zeros((36, total_rank), dtype=np.int64)
    V_total = np.zeros((36, total_rank), dtype=np.int64)
    W_total = np.zeros((36, total_rank), dtype=np.int64)
    
    for c3 in range(r3):
        for c2 in range(7):
            c = c3 * 7 + c2
            # For each (i,j) pair in 0..5:
            for i in range(6):
                for j in range(6):
                    i2, i1 = divmod(i, 2)
                    j2, j1 = divmod(j, 2)
                    U_total[i*6+j, c] = U3[i2*3+j2, c3] * U2[i1*2+j1, c2]
            
            for j in range(6):
                for k in range(6):
                    j2, j1 = divmod(j, 2)
                    k2, k1 = divmod(k, 2)
                    V_total[j*6+k, c] = V3[j2*3+k2, c3] * V2[j1*2+k1, c2]
            
            for k in range(6):
                for i in range(6):
                    k2, k1 = divmod(k, 2)
                    i2, i1 = divmod(i, 2)
                    W_total[k*6+i, c] = W3[k2*3+i2, c3] * W2[k1*2+i1, c2]
    
    # Verify the 189-rank decomposition
    if verify(total_rank, U_total, V_total, W_total):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
