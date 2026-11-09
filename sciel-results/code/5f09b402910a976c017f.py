import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # A is n x m, B is m x p, C is n x p
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # C[i, j] = sum_k A[i, k] B[k, j]
    # Decomposition: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Note the W index: W[k*n+i, r] in the task description says "W[k*n+i,r]" but standard is W[i, j] -> W[j*n+i].
    # Let's check the task description: "W[k*n+i,r]".
    # In the seed:
    # w_idx = k * n + i
    # So the row index of W is k*n + i, where k is the column of A (which is row of B) and i is row of A (col of C).
    # Wait, C[i, j] = sum_k A[i, k] B[k, j].
    # The seed uses:
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # Here i is row of A/C, j is col of A/row of B, k is col of B/C.
    # So W index is k*n + i. This matches C[i, k] if we swap i and k?
    # Let's trace:
    # Term = A[i, j] * B[j, k].
    # U[i*m+j] is 1. V[j*p+k] is 1. W[k*n+i] is 1.
    # Sum over r: U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # This reconstructs C[i, k].
    # So W row index is col_of_C * n + row_of_C.
    # This is consistent.

    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal U, V, W, rank
        U[u_idx][rank] = val
        V[v_idx][rank] = val
        W[w_idx][rank] = val
        rank += 1

    def add_strassen_2x2x2(a_row_offset, b_col_offset):
        # a_row_offset: 0 or 1 (which 2x2 block of A)
        # b_col_offset: 0, 1, or 2 (which 2x1 block of B)
        # A block is A[a_row_offset*2 : a_row_offset*2+2, 0:2] (since m=3, we use cols 0,1)
        # B block is B[0:2, b_col_offset*2 : b_col_offset*2+2] (since p=5, we use cols 0..3)
        
        # Indices in A: i in {a_row_offset*2, a_row_offset*2+1}, j in {0, 1}
        # Indices in B: j in {0, 1}, k in {b_col_offset*2, b_col_offset*2+1}
        
        # Strassen variables:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Let's define helper to get A[i, j] index in U
        # U index: i * m + j
        # V index: j * p + k
        # W index: k * n + i
        
        # A11 is at i=a_row_offset*2, j=0
        # A12 is at i=a_row_offset*2, j=1
        # A21 is at i=a_row_offset*2+1, j=0
        # A22 is at i=a_row_offset*2+1, j=1
        
        # B11 is at j=0, k=b_col_offset*2
        # B12 is at j=0, k=b_col_offset*2+1
        # B21 is at j=1, k=b_col_offset*2
        # B22 is at j=1, k=b_col_offset*2+1
        
        # We will add terms for each M.
        # Each M is a product of a linear combo of A entries and a linear combo of B entries.
        # We can decompose each M into rank-1 terms.
        # e.g. M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # This would take 4 terms.
        # Strassen reduces the number of multiplications, but in tensor decomposition,
        # each "multiplication" M_r corresponds to a set of rank-1 terms if the factors are sums.
        # Wait. The rank of the tensor is the number of rank-1 terms in the sum.
        # Strassen algorithm computes 7 products. Each product is of two linear combinations.
        # (sum a_i A_i) * (sum b_j B_j) = sum_{i,j} a_i b_j (A_i B_j).
        # So one Strassen "product" expands into multiple rank-1 terms.
        # The rank of 2x2x2 matrix multiplication is 7.
        # This means there exists a decomposition into 7 rank-1 tensors.
        # The coefficients in U, V, W will be such that the sum works out.
        
        # Let's use the known rank-7 decomposition coefficients.
        # The 7 tensors (u, v, w) for 2x2x2:
        # 1: u=(1,1,1,1), v=(1,0,1,0), w=(1,0,0,1) -> (A11+A22)(B11+B22) ... wait, indices
        # Let's map A to 4 entries: A11, A12, A21, A22.
        # B to 4 entries: B11, B12, B21, B22.
        # C to 4 entries: C11, C12, C21, C22.
        
        # T1: u=[1,0,0,1], v=[1,0,1,0], w=[1,0,0,1]  => (A11+A22)(B11+B22) -> C11, C22
        # T2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]  => (A21+A22)B11 -> C21
        # T3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0] => A11(B12-B22) -> C12
        # T4: u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,1,0] => A22(B21-B11) -> C21, C22?
        # T5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,-1] => (A11+A12)B22 -> C12, C22?
        # T6: u=[0,-1,1,0], v=[1,1,0,0], w=[0,0,0,1] => (A21-A11)(B11+B12) -> C22?
        # T7: u=[1,0,-1,0], v=[0,1,1,1], w=[1,0,0,0] => (A12-A22)(B21+B22) -> C11?
        
        # Let's verify T1:
        # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # C11 = A11B11 + A12B21. C22 = A21B12 + A22B22.
        # T1 contributes to C11 (A11B11) and C22 (A22B22).
        # w=[1,0,0,1] means C11 += 1, C22 += 1.
        # u=[1,0,0,1] means A11, A22.
        # v=[1,0,1,0] means B11, B22.
        # Product: (A11+A22)(B11+B22).
        # Contribution to C11: A11B11. Correct.
        # Contribution to C22: A22B22. Correct.
        # What about A11B22 and A22B11? They are not in C11 or C22.
        # So T1 is not a rank-1 tensor for the multiplication tensor?
        # A rank-1 tensor is u_i v_j w_k.
        # If u has two non-zeros, v has two non-zeros, w has two non-zeres, it's not rank 1?
        # No, a rank-1 tensor is an outer product.
        # If u = e1 + e4, v = e1 + e3, w = e1 + e4.
        # Then the tensor is (e1+e4) \otimes (e1+e3) \otimes (e1+e4).
        # This is NOT a rank-1 tensor in the standard sense of "u_i v_j w_k" where u, v, w are vectors.
        # Wait. The decomposition is T = sum_r u^{(r)} \otimes v^{(r)} \otimes w^{(r)}.
        # Each term is a rank-1 tensor.
        # A rank-1 tensor has exactly one non-zero entry in each slice? No.
        # A rank-1 tensor T_{ijk} = u_i v_j w_k.
        # If u has two non-zeros, v has two non-zeros, w has two non-zeros, then T has 2*2*2=8 non-zeros.
        # This is a valid rank-1 tensor.
        # So we can use these.
        
        # Let's define the 7 vectors for 2x2x2.
        # Indices: 0:A11, 1:A12, 2:A21, 3:A22
        # 0:B11, 1:B12, 2:B21, 3:B22
        # 0:C11, 1:C12, 2:C21, 3:C22
        
        # 1: u=[1,0,0,1], v=[1,0,1,0], w=[1,0,0,1]
        # 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # 3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
        # 4: u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,1,0]
        # 5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,-1]
        # 6: u=[0,-1,1,0], v=[1,1,0,0], w=[0,0,0,1]
        # 7: u=[1,0,-1,0], v=[0,1,1,1], w=[1,0,0,0]
        
        # Let's check if this sums to the multiplication tensor.
        # C11 = A11B11 + A12B21.
        # From 1: A11B11.
        # From 7: A12B21. (u1=1, v3=1, w0=1 -> A12 B22? No. v=[0,1,1,1] -> B12, B21, B22.
        # T7: u=[1,0,-1,0] -> A11 - A21. v=[0,1,1,1] -> B12+B21+B22. w=[1,0,0,0] -> C11.
        # Contribution: (A11-A21)(B12+B21+B22) to C11.
        # This doesn't look right.
        
        # Let's look up the standard rank-7 decomposition.
        # It is easier to just use the schoolbook for 2x2x2 if I can't get the coefficients right quickly.
        # Rank 4 for 2x2x2.
        # Total rank would be:
        # 3 blocks of 2x2x2? No.
        # A is 3x3. B is 3x5.
        # We can split A into 3x1, 3x1, 3x1? No.
        # Split A into 2x2, 2x1, 1x2, 1x1?
        # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
        # B = [B1; B2] where B1 is 2x5, B2 is 1x5.
        # C = A11 B1 + A12 B2 + A21 B1 + A22 B2.
        
        # A11 B1: 2x2 * 2x5.
        # We can split B1 into two 2x2 and one 2x1.
        # B1 = [B11 B12 B13] where B11, B12 are 2x2, B13 is 2x1.
        # A11 B11: 2x2*2x2. Rank 4 (schoolbook) or 7 (Strassen).
        # A11 B12: 2x2*2x2. Rank 4 or 7.
        # A11 B13: 2x2*2x1. Rank 4.
        
        # A12 B2: 2x1 * 1x5. Rank 2*1*5 = 10? No.
        # A12 is 2x1. B2 is 1x5.
        # C_part = A12 B2.
        # C_part[i, j] = A12[i, 0] B2[0, j].
        # This is outer product of A12 and B2.
        # Rank 1? No, because A12 has 2 entries, B2 has 5.
        # The tensor is sum_{i,j} A12[i] B2[j] C[i,j].
        # This is a sum of 2*5 = 10 rank-1 terms?
        # No, it's one term if we can factor it?
        # u_i v_j w_k.
        # u has 2 non-zeros. v has 5 non-zeros. w has 10 non-zeros.
        # This is not a single rank-1 term in the U,V,W representation?
        # In U,V,W representation, each column is a rank-1 term.
        # For A12 B2:
        # U[i*m+1, r] = A12[i]. V[1*p+j, r] = B2[j]. W[j*n+i, r] = 1.
        # We need to represent this.
        # Since A12 and B2 are vectors, we can use rank 1 if we allow U and V to have multiple non-zeros?
        # Yes! U is a matrix. A column of U can have multiple non-zeros.
        # So A12 B2 can be done in rank 1?
        # Let's check.
        # Term: u \otimes v \otimes w.
        # u_i = A12[i] for i=0,1. 0 else.
        # v_j = B2[j] for j=0..4. 0 else.
        # w_k = 1 for k corresponding to (i,j).
        # But w is fixed for the column.
        # The product is sum_r u_i v_j w_k.
        # If we use 1 column:
        # u_i = A12[i]. v_j = B2[j].
        # Then sum_r u_i v_j w_k = A12[i] B2[j] w_k.
        # We want this to be C[i, j] = A12[i] B2[j].
        # So we need w_k = 1 for k = j*n+i.
        # But w is a vector. It can have 10 non-zeros.
        # So yes, rank 1 is sufficient for 2x1 * 1x5.
        
        # Similarly for A21 B1: 1x2 * 2x5.
        # Rank 1?
        # u_j = A21[j]. v_j = B1[j, k].
        # C[i, k] = sum_j A21[j] B1[j, k].
        # Term: u_j v_j w_k.
        # u has 2 non-zeros. v has 10 non-zeros. w has 5 non-zeros.
        # Yes, rank 1.
        
        # A22 B2: 1x1 * 1x5.
        # Rank 1.
        
        # So the only "hard" part is A11 B1.
        # A11 is 2x2. B1 is 2x5.
        # Split B1 into B11 (2x2), B12 (2x2), B13 (2x1).
        # A11 B11: 2x2*2x2. Rank 7 (Strassen) or 4 (Schoolbook).
        # A11 B12: 2x2*2x2. Rank 7 or 4.
        # A11 B13: 2x2*2x1. Rank 4.
        
        # Total rank:
        # Using Schoolbook for 2x2x2:
        # 4 + 4 + 4 + 1 + 1 + 1 = 15.
        # Using Strassen for 2x2x2:
        # 7 + 7 + 4 + 1 + 1 + 1 = 21.
        # So Schoolbook is better here!
        
        # Can we do better for 2x2*2x5?
        # Rank of 2x2*2x5 is 4*5 = 20? No.
        # It's the rank of the tensor.
        # 2x2*2x2 is 4.
        # 2x2*2x1 is 4.
        # So 2x2*2x5 is at most 4+4+4 = 12.
        # Is it 12?
        # Probably yes.
        
        # So total rank = 12 (A11 B1) + 1 (A12 B2) + 1 (A21 B1) + 1 (A22 B2) = 15.
        
        # Wait, A21 B1 is 1x2 * 2x5.
        # C[i, k] = sum_j A21[j] B1[j, k].
        # i is 0..2. j is 0..1. k is 0..4.
        # A21 is row 2 of A. So i=2.
        # B1 is rows 0,1 of B.
        # So this contributes to C[2, k].
        # Rank 1?
        # u_j = A21[j]. v_j = B1[j, k].
        # w_k = 1 for k=0..4.
        # Yes, rank 1.
        
        # A12 B2:
        # A12 is col 1 of A. i=0,1.
        # B2 is row 2 of B. k=0..4.
        # Contributes to C[i, k] for i=0,1.
        # Rank 1.
        
        # A22 B2:
        # A22 is A[2,1].
        # B2 is row 2 of B.
        # Contributes to C[2, k].
        # Rank 1.
        
        # So total rank 15.
        
        # Let's implement this.
        
        # A11 B1:
        # A11 is A[0:2, 0:2].
        # B1 is B[0:2, 0:5].
        # Split B1 into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5].
        
        # Term 1: A11 * B[0:2, 0:2]
        # Schoolbook 2x2x2.
        # i in 0,1. j in 0,1. k in 0,1.
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_term(u_idx, v_idx, w_idx, 1.0)
                    
        # Term 2: A11 * B[0:2, 2:4]
        # i in 0,1. j in 0,1. k in 2,3.
        for i in range(2):
            for j in range(2):
                for k in range(2, 4):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_term(u_idx, v_idx, w_idx, 1.0)
                    
        # Term 3: A11 * B[0:2, 4:5]
        # i in 0,1. j in 0,1. k in 4.
        for i in range(2):
            for j in range(2):
                for k in range(4, 5):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_term(u_idx, v_idx, w_idx, 1.0)
                    
        # Term 4: A12 * B2
        # A12 is A[0:2, 1]. (i=0,1; j=1)
        # B2 is B[2, 0:5]. (j=2; k=0..4)
        # C[i, k] += A[i, 1] B[2, k]
        # Rank 1.
        u_idx = 0 * m + 1 # A[0, 1]
        # Wait, rank 1 means ONE column.
        # But A12 has 2 entries. B2 has 5 entries.
        # We need to represent sum_{i,k} A[i,1] B[2,k] C[i,k].
        # Using 1 column:
        # U[i*m+1, r] = A[i,1] ? No, U entries are coefficients.
        # We don't know A and B values. We are constructing the decomposition.
        # The decomposition must work for ALL A and B.
        # So U, V, W are fixed matrices.
        # For A12 B2:
        # We need sum_r U[i*m+1, r] V[2*p+k, r] W[k*n+i, r] = A[i,1] B[2,k].
        # This must hold for all i, k.
        # If we use 1 column:
        # U[i*m+1, 0] * V[2*p+k, 0] * W[k*n+i, 0] = A[i,1] B[2,k].
        # This is impossible with constant coefficients unless A and B are specific.
        # Ah, I see. The decomposition is T = sum u \otimes v \otimes w.
        # The entries of U, V, W are the coefficients.
        # For the term A[i,1] B[2,k], we need a rank-1 tensor that picks out this product.
        # But A[i,1] and B[2,k] are variables.
        # The tensor for A[i,1] B[2,k] is e_i \otimes e_2 \otimes e_k (in the A, B, C space).
        # In the U, V, W space:
        # U has index (i, j). V has index (j, k). W has index (k, i).
        # The term A[i,1] B[2,k] corresponds to:
        # U[i*m+1, r] = 1
        # V[1*p+k, r] = 1  (Wait, B index is j*p+k. Here j=1? No, B is m x p. j is 0..2.
        # A is n x m. i is 0..2, j is 0..2.
        # A12 is A[0:2, 1]. So j=1.
        # B2 is B[2, 0:5]. So j=2.
        # The product is A[i, 1] * B[2, k].
        # This involves A at (i, 1) and B at (2, k).
        # In the tensor, the index for A is (i, 1). Index for B is (2, k).
        # So U[i*m+1, r] = 1.
        # V[2*p+k, r] = 1.
        # W[k*n+i, r] = 1.
        # This is a rank-1 term for each (i, k) pair.
        # So we need 2 * 5 = 10 terms for A12 B2.
        # My previous "Rank 1" was wrong. That was for the matrix product value, not the tensor rank.
        
        # So A12 B2 rank is 10.
        # A21 B1 rank is 1 * 2 * 5 = 10?
        # A21 is A[2, 0:2]. (i=2, j=0,1).
        # B1 is B[0:2, 0:5]. (j=0,1; k=0..4).
        # Product A[2, j] B[j, k].
        # Terms: i=2, j=0..1, k=0..4.
        # 1 * 2 * 5 = 10 terms.
        
        # A22 B2 rank is 1 * 1 * 5 = 5 terms.
        
        # A11 B1 rank is 2 * 2 * 5 = 20 terms.
        
        # Total rank = 20 + 10 + 10 + 5 = 45.
        # This is just the schoolbook rank.
        
        # So my attempt to reduce rank failed because I misunderstood tensor rank vs matrix product complexity.
        # The tensor rank of A(n,m) * B(m,p) is n*m*p for the schoolbook.
        # To reduce rank, we need to combine terms.
        # Strassen does this for 2x2x2.
        # For 2x2x2, rank is 7 < 8.
        # So we should use Strassen for the 2x2x2 blocks.
        
        # Let's go back to Strassen for 2x2x2.
        # We need the correct coefficients.
        
        # I will use a known valid set of coefficients for 2x2x2 rank 7.
        # From literature:
        # 1: u=[1,0,0,1], v=[1,0,1,0], w=[1,0,0,1]
        # 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # 3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
        # 4: u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,1,0]
        # 5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,-1]
        # 6: u=[0,-1,1,0], v=[1,1,0,0], w=[0,0,0,1]
        # 7: u=[1,0,-1,0], v=[0,1,1,1], w=[1,0,0,0]
        
        # Let's verify this set sums to the multiplication tensor.
        # I will write a small check in my head.
        # C11 = A11B11 + A12B21.
        # T1: A11B11. (u1=1, v1=1, w1=1).
        # T7: (A11-A21)(B12+B21+B22) -> C11.
        # Contribution: A11B12 + A11B21 + A11B22 - A21B12 - A21B21 - A21B22.
        # This is messy.
        
        # Let's use a simpler approach.
        # I will use the schoolbook for the whole thing.
        # It is robust and correct.
        # Rank 45.
        
        # Is there any simple improvement?
        # Maybe for 3x3x5, there is a known rank < 45?
        # Probably yes, but I don't know it.
        
        # I will stick with schoolbook.
        
        pass

    # Re-initialize for schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
