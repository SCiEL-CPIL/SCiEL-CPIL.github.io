"""Matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns: {'U', 'V', 'W'} with shapes (20, rank), (25, rank), (20, rank)
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    
    start_time = time.time()
    deadline = start_time + 115

    # Build schoolbook decomposition (rank = 100)
    schoolbook_rank = n * m * p  # 100
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = schoolbook_rank

    # Precompute the target tensor T flattened
    # The tensor for matrix mult C = A * B is defined such that C[i, k] = sum_j A[i, j] B[j, k].
    # Indices:
    # U row index: i*m + j  (A index)
    # V row index: j*p + k  (B index)
    # W row index: k*n + i  (C index)
    # The tensor element T[A_idx, B_idx, C_idx] is 1 if A_idx=(i,j), B_idx=(j,k), C_idx=(k,i) for some i,j,k.
    T_flat = np.zeros(nm * mp * pn)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_flat[a * mp * pn + b * pn + c] = 1.0

    def verify_fast(U, V, W, tol=1e-9):
        """
        Fast verification using NumPy.
        U: (nm, r), V: (mp, r), W: (pn, r)
        Computes T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        Flattens and compares to T_flat.
        """
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        T_approx_flat = T_approx.flatten()
        return np.allclose(T_approx_flat, T_flat, atol=tol)

    # Attempt to find a rank reduction using a block decomposition approach.
    # We know R(4,4,4) = 49 (Lathauwer et al. 2003). 
    # If we can hardcode the 49 terms for 4x4x4, we can build a decomposition for 4x5x5.
    # The 4x5x5 tensor can be viewed as:
    # C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4]  (4x4x4 block, rank 49)
    # C[0:4, 4]   = A[0:4, 0:5] * B[0:5, 4]    (4x5x1 block)
    # C[4, 0:5]   is not defined, C is 4x5. 
    # Wait, A is 4x5, B is 5x5, C is 4x5.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # We can split B into B_left (5x4) and B_right (5x1).
    # C_left (4x4) = A (4x5) * B_left (5x4).
    # C_right (4x1) = A (4x5) * B_right (5x1).
    
    # This doesn't directly use R(4,4,4).
    # Let's try to use the 49-term 4x4x4 decomposition for the top-left 4x4x4 block?
    # No, the dimensions don't match directly for a simple block split.
    # A is 4x5, B is 5x5.
    # If we split A into A1 (4x4) and A2 (4x1), and B into B1 (4x5) and B2 (1x5).
    # C = A1*B1 + A2*B2.
    # A1 is 4x4, B1 is 4x5. A2 is 4x1, B2 is 1x5.
    # A2*B2 is outer product of A2 (4x1) and B2 (1x5), which is rank 1? No, it's a sum of 1 term?
    # A2[i, 4] * B2[4, k]. This is a single term if we treat the indices correctly?
    # No, it's sum over j=4. So it's one term in the sum for C[i,k].
    # So C = (A1 * B1) + (A2 * B2).
    # A1 * B1 is 4x4 times 4x5.
    # A2 * B2 is 4x1 times 1x5.
    
    # The 4x1 * 1x5 part is just the 4th column of A times the 4th row of B.
    # This can be done with 1 term? No, it's a matrix product of size 4x5.
    # C2[i, k] = A[i, 4] * B[4, k].
    # This is a single rank-1 tensor? No, it's 20 elements.
    # It can be represented as sum of 1 term in the tensor decomposition?
    # T2[a, b, c] = 1 if a=(i,4), b=(4,k), c=(k,i).
    # This is a set of 20 non-zero entries.
    # Can this be rank 1? No, it's not a single product of three vectors.
    # It's the outer product of A[:,4] and B[4,:].
    # In tensor decomposition terms, this is a specific pattern.
    # Rank of A[:,4] * B[4,:] is not 1 in the matrix multiplication tensor sense.
    # It requires 1 term? No.
    # C2[i, k] = A[i, 4] * B[4, k].
    # This is just one multiplication in the schoolbook.
    # So the rank is 1 for this part?
    # Let's check: U[a, r] V[b, r] W[c, r].
    # If we set r=1, U[i*5+4, 0] = A[i,4] (but U must be constant, not depending on A).
    # Ah, the decomposition is for the *algorithm*, not the specific matrices.
    # So U, V, W are fixed coefficients.
    # For C2[i, k] = A[i, 4] * B[4, k], we need a term that picks out A[i, 4] and B[4, k] and multiplies them.
    # This is exactly 1 term in the schoolbook decomposition!
    # Term: U[i*5+4, r] = 1, V[4*5+k, r] = 1, W[k*4+i, r] = 1.
    # Wait, V index is j*p + k. Here j=4, p=5. So V[4*5+k, r] = V[20+k, r] = 1.
    # U index is i*m + j. Here i is row of A, j=4. So U[i*5+4, r] = 1.
    # W index is k*n + i. Here k is col of C, i is row of C. So W[k*4+i, r] = 1.
    # So yes, the "cross term" A2*B2 is just 1 term? No, it's 4*5 = 20 terms?
    # No, the term in the tensor decomposition corresponds to a specific (i, j, k) in the sum C[i, k] = sum_j A[i, j] B[j, k].
    # The schoolbook has n*m*p = 4*5*5 = 100 terms.
    # Each term corresponds to a unique (i, j, k).
    # The "A2*B2" part corresponds to j=4.
    # There are 4 (values of i) * 5 (values of k) = 20 such terms (i, 4, k).
    # So this part has rank 20 in the schoolbook.
    # Can it be reduced?
    # C2[i, k] = A[i, 4] * B[4, k].
    # This is an outer product of two vectors.
    # Can we compute an outer product with rank < 20?
    # No, the tensor T2 is the sum of 20 rank-1 tensors (one for each i, k).
    # And these 20 tensors are linearly independent?
    # T2[i*5+4, 20+k, k*4+i] = 1.
    # These are 20 distinct entries.
    # The rank of T2 is at least the number of non-zero entries in any slice?
    # It is known that the rank of the outer product tensor (vector * vector) is 1?
    # No, the outer product of two vectors is a matrix, not a 3-way tensor.
    # Here, the "tensor" is defined by the multiplication algorithm.
    # The algorithm for C2[i, k] = A[i, 4] * B[4, k] requires 1 multiplication.
    # So it is rank 1 in the sense of "number of multiplications"?
    # No, the rank is the number of terms in the sum C[i, k] = sum_r ...
    # If we can compute C2 with 1 term, then the rank is 1.
    # Let's check:
    # We want T2[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
    # Non-zeros of T2 are at (i*5+4, 20+k, k*4+i) for i in 0..3, k in 0..4.
    # Can we find U, V, W with r=1 such that U[i*5+4, 0] V[20+k, 0] W[k*4+i, 0] = 1?
    # Let U[i*5+4, 0] = u_i, V[20+k, 0] = v_k, W[k*4+i, 0] = w_{k,i}.
    # We need u_i v_k w_{k,i} = 1 for all i, k.
    # This is possible!
    # Let u_i = 1, v_k = 1, w_{k,i} = 1.
    # Then the product is 1.
    # But we also need the product to be 0 for all other (a, b, c).
    # For T2, the only non-zeros are at those 20 spots.
    # If we set U, V, W to be zero elsewhere, does it work?
    # U is non-zero only at indices i*5+4.
    # V is non-zero only at indices 20+k.
    # W is non-zero only at indices k*4+i.
    # For any other (a, b, c), at least one of U[a, 0], V[b, 0], W[c, 0] is 0.
    # So the product is 0.
    # Thus, the "cross term" part (j=4) can be computed with rank 1!
    
    # This is a huge reduction.
    # The schoolbook has 100 terms.
    # The j=4 part has 20 terms, but can be done with rank 1.
    # The j=0..3 part has 4*4*5 = 80 terms.
    # This is a 4x4x5 multiplication (A is 4x4, B is 4x5, C is 4x5).
    # Can we reduce this?
    # R(4,4,5)?
    # We know R(4,4,4) = 49.
    # R(4,4,5) <= R(4,4,4) + R(4,4,1)?
    # R(4,4,1) is the rank of 4x4 times 4x1.
    # C[i, 0] = sum_j A[i, j] B[j, 0].
    # This is a vector-matrix product.
    # Can this be done with rank < 16?
    # Similar to the j=4 case, if it was a single column?
    # No, here B is 4x1, A is 4x4.
    # C[i, 0] = sum_j A[i, j] B[j, 0].
    # This is 16 terms in schoolbook.
    # Can it be rank 1?
    # T[a, b, c] = 1 if a=(i,j), b=(j,0), c=(0,i).
    # Non-zeros: (i*4+j, j*5+0, 0*4+i) = (i*4+j, 5j, i).
    # For i in 0..3, j in 0..3.
    # Can we find U, V, W with r=1?
    # U[i*4+j, 0] V[5j, 0] W[i, 0] = 1?
    # Let U[i*4+j, 0] = u_{i,j}, V[5j, 0] = v_j, W[i, 0] = w_i.
    # u_{i,j} v_j w_i = 1.
    # This is possible (set all to 1).
    # But we need it to be 0 elsewhere.
    # U is non-zero at i*4+j.
    # V is non-zero at 5j.
    # W is non-zero at i.
    # For any other (a, b, c), is the product 0?
    # If a is not of form i*4+j, U is 0.
    # If b is not of form 5j, V is 0.
    # If c is not of form i, W is 0.
    # But wait, the indices a, b, c are coupled in the tensor definition.
    # The tensor T is defined for the whole 4x4x5 multiplication.
    # We are looking for a decomposition of the whole tensor.
    # We can't just decompose the j=4 part separately and the j=0..3 part separately and add the ranks,
    # because the U, V, W matrices must be the same for all terms.
    # However, we can use the fact that the j=4 part uses disjoint indices in U and V?
    # U indices for j=4: i*5+4.
    # U indices for j=0..3: i*5+j (j<4).
    # These are disjoint.
    # V indices for j=4: 4*5+k = 20+k.
    # V indices for j=0..3: j*5+k.
    # These are disjoint.
    # W indices for j=4: k*4+i.
    # W indices for j=0..3: k*4+i.
    # These are NOT disjoint.
    # So we can't simply add the ranks.
    
    # However, we can construct a decomposition by combining them.
    # Let r1 be the rank for j=4 part (which is 1).
    # Let r2 be the rank for j=0..3 part.
    # The total rank is r1 + r2.
    # We just need to make sure the W indices don't conflict.
    # They don't conflict in the sense that we can have separate terms.
    # Term r1 uses W[k*4+i, r1].
    # Term r2 uses W[k*4+i, r2].
    # Since r1 != r2, they are different columns in W.
    # So the total rank is 1 + r2.
    
    # Now, what is r2?
    # r2 is the rank of the 4x4x5 multiplication tensor (A 4x4, B 4x5, C 4x5).
    # We can split B into B1 (4x4) and B2 (4x1).
    # C = A*B1 + A*B2.
    # A*B1 is 4x4x4, rank 49.
    # A*B2 is 4x4x1, rank ?
    # Let's check the rank of 4x4x1.
    # C[i, 0] = sum_j A[i, j] B[j, 0].
    # As shown above, this can be done with rank 1?
    # Let's re-verify.
    # T2[a, b, c] = 1 if a=(i,j), b=(j,0), c=(0,i).
    # Non-zeros: (i*4+j, 5j, i).
    # We need U[i*4+j, r] V[5j, r] W[i, r] = 1.
    # And 0 elsewhere.
    # If we set U, V, W to be 1 at these positions and 0 elsewhere, does it work?
    # Yes, as argued before.
    # So the 4x4x1 part has rank 1.
    
    # So R(4,4,5) <= R(4,4,4) + R(4,4,1) = 49 + 1 = 50.
    # And R(4,5,5) <= R(4,4,5) + R(4,1,5) = 50 + 1 = 51.
    
    # This is a massive improvement over 100!
    # But I need the actual 49-term decomposition for 4x4x4.
    # If I don't have it, I can't use it.
    # The schoolbook for 4x4x4 has rank 64.
    # If I use schoolbook for 4x4x4, the total rank is 1 + (64 + 1) + 1 = 67.
    # This is still an improvement over 100.
    
    # Let's implement the 67-rank solution.
    # Terms:
    # 1. j=4 part: 1 term.
    # 2. j=0..3, k=0..3 part (4x4x4 schoolbook): 64 terms.
    # 3. j=0..3, k=4 part (4x4x1 schoolbook): 1 term.
    
    # Total rank: 1 + 64 + 1 = 66?
    # Wait, the 4x4x4 part is A(4x4) * B(4x4) -> C(4x4).
    # The 4x4x1 part is A(4x4) * B(4x1) -> C(4x1).
    # The j=4 part is A(4x1) * B(1x5) -> C(4x5).
    
    # Let's check the indices.
    # A is 4x5, B is 5x5, C is 4x5.
    # C[i, k] = sum_{j=0..4} A[i, j] B[j, k].
    # Split sum into j=0..3 and j=4.
    # C[i, k] = sum_{j=0..3} A[i, j] B[j, k] + A[i, 4] B[4, k].
    
    # Part 1: j=4.
    # C1[i, k] = A[i, 4] B[4, k].
    # Rank 1 term:
    # U[i*5+4, 0] = 1
    # V[4*5+k, 0] = 1  (i.e., V[20+k, 0] = 1)
    # W[k*4+i, 0] = 1
    
    # Part 2: j=0..3.
    # C2[i, k] = sum_{j=0..3} A[i, j] B[j, k].
    # Split k into k=0..3 and k=4.
    # C2[i, k] = sum_{j=0..3} A[i, j] B[j, k] (for k<4) + sum_{j=0..3} A[i, j] B[j, 4] (for k=4).
    
    # Part 2a: k=0..3.
    # C2a[i, k] = sum_{j=0..3} A[i, j] B[j, k] for i<4, k<4.
    # This is 4x4x4. Schoolbook rank 64.
    # Terms for i in 0..3, j in 0..3, k in 0..3.
    # U[i*5+j, r] = 1
    # V[j*5+k, r] = 1
    # W[k*4+i, r] = 1
    
    # Part 2b: k=4.
    # C2b[i, 4] = sum_{j=0..3} A[i, j] B[j, 4].
    # This is 4x4x1. Rank 1 term?
    # C2b[i, 4] = sum_{j=0..3} A[i, j] B[j, 4].
    # This is a dot product of A[i, 0:4] and B[0:4, 4].
    # Can this be rank 1?
    # T[a, b, c] = 1 if a=(i,j), b=(j,4), c=(4,i).
    # Non-zeros: (i*5+j, j*5+4, 4*4+i) = (i*5+j, 5j+4, 16+i).
    # We need U[i*5+j, r] V[5j+4, r] W[16+i, r] = 1.
    # Set U, V, W to 1 at these positions.
    # Is it 0 elsewhere?
    # U is non-zero at i*5+j (j<4).
    # V is non-zero at 5j+4.
    # W is non-zero at 16+i.
    # For any other (a, b, c), is the product 0?
    # Yes, if the indices don't match the pattern.
    # So this part is rank 1.
    
    # Total rank: 1 (Part 1) + 64 (Part 2a) + 1 (Part 2b) = 66.
    
    # Let's construct this 66-rank solution.
    U_66 = np.zeros((nm, 66))
    V_66 = np.zeros((mp, 66))
    W_66 = np.zeros((pn, 66))
    
    # Part 1: j=4
    r = 0
    for i in range(4):
        for k in range(5):
            U_66[i*5+4, r] = 1
            V_66[20+k, r] = 1
            W_66[k*4+i, r] = 1
    r += 1
    
    # Part 2a: j=0..3, k=0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_66[i*5+j, r] = 1
                V_66[j*5+k, r] = 1
                W_66[k*4+i, r] = 1
                r += 1
    
    # Part 2b: j=0..3, k=4
    for i in range(4):
        for j in range(4):
            U_66[i*5+j, r] = 1
            V_66[j*5+4, r] = 1
            W_66[16+i, r] = 1
            r += 1  # Wait, this is 16 terms, not 1.
    
    # My rank 1 argument for Part 2b was flawed.
    # C2b[i, 4] = sum_{j=0..3} A[i, j] B[j, 4].
    # This is 4 values of i, each requiring a sum of 4 terms.
    # The tensor T has non-zeros at (i*5+j, 5j+4, 16+i).
    # There are 16 such non-zeros.
    # Can they be covered by 1 term?
    # We need U[i*5+j, 0] V[5j+4, 0] W[16+i, 0] = 1 for all i, j.
    # Let U[i*5+j, 0] = u_{i,j}, V[5j+4, 0] = v_j, W[16+i, 0] = w_i.
    # u_{i,j} v_j w_i = 1.
    # This is possible (set all to 1).
    # But we need it to be 0 for all other (a, b, c).
    # The only non-zeros in T for Part 2b are these 16.
    # If we set U, V, W to 0 elsewhere, does it work?
    # Yes.
    # So Part 2b IS rank 1.
    
    # Let's re-verify Part 2b construction.
    # I added 16 terms in the loop above. That's wrong.
    # It should be 1 term.
    
    # Let's restart the construction of the 66-rank solution.
    U_66 = np.zeros((nm, 66))
    V_66 = np.zeros((mp, 66))
    W_66 = np.zeros((pn, 66))
    
    # Part 1: j=4 (1 term)
    r = 0
    for i in range(4):
        for k in range(5):
            U_66[i*5+4, r] = 1
            V_66[20+k, r] = 1
            W_66[k*4+i, r] = 1
    r += 1
    
    # Part 2a: j=0..3, k=0..3 (64 terms)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_66[i*5+j, r] = 1
                V_66[j*5+k, r] = 1
                W_66[k*4+i, r] = 1
                r += 1
    
    # Part 2b: j=0..3, k=4 (1 term)
    for i in range(4):
        for j in range(4):
            U_66[i*5+j, r] = 1
            V_66[j*5+4, r] = 1
            W_66[16+i, r] = 1
    r += 1
    
    # Verify the 66-rank solution
    if verify_fast(U_66, V_66, W_66):
        best_result = {"U": U_66.tolist(), "V": V_66.tolist(), "W": W_66.tolist()}
        best_rank = 66

    return best_result

# EVOLVE-BLOCK-END
