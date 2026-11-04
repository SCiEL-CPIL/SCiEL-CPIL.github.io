"""Rewritten program for the 2x3x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # We use a rank-9 decomposition based on the Kronecker product structure:
    #   T[i,j,k] = delta_{i(a),i(c)} * delta_{j(a),j(b)} * delta_{k(b),k(c)}
    # Split the k dimension (size 6) into k = 2*k1 + k2 with k1 in {0,1,2}, k2 in {0,1}.
    # Split the i dimension (size 2) as i in {0,1}.
    # 
    # The tensor factors as:
    #   T = (delta_i) ⊗ (delta_j) ⊗ (delta_{k1}) ⊗ (delta_{k2})
    # where delta_i is 2x2 (rank 2), delta_j is 3x3 (rank 3), delta_{k1} is 3x3 (rank 3),
    # delta_{k2} is 2x2 (rank 2).
    #
    # The Kronecker product of tensors has rank equal to the product of ranks,
    # but we can do better by grouping:
    #   T = [delta_i ⊗ delta_{k2}] ⊗ [delta_j ⊗ delta_{k1}]
    # where delta_i ⊗ delta_{k2} is a (2*2)x(2*2) = 4x4 tensor of rank 4 (product of ranks 2*2),
    # and delta_j ⊗ delta_{k1} is a (3*3)x(3*3) = 9x9 tensor of rank 9 (product of ranks 3*3).
    # Total rank would be 4*9 = 36, which is schoolbook.
    #
    # Instead, we use the fact that the matrix multiplication tensor for 2x3 * 3x6 can be
    # decomposed using the identity:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # with k = 2*k1 + k2, so B[j, 2*k1+k2] is a 3x6 matrix.
    #
    # We construct 9 rank-1 terms by combining:
    #   - 3 terms for the j dimension (j = 0, 1, 2)
    #   - 3 terms for the (i, k1) dimension, exploiting that i has 2 values and k1 has 3 values,
    #     but the delta structure allows us to use 3 combined indices.
    #
    # Specifically, for each j in {0,1,2} and each r in {0,1,2}, we define:
    #   U[i*m + j, (j,r)] = 1 if i == r % 2 else 0
    #   V[j*p + k, (j,r)] = 1 if k//2 == r//1 and k%2 == r%2 ... 
    #
    # Let me think more carefully. The key insight is:
    # delta_{i(a),i(c)} * delta_{k2(b),k2(c)} can be written as sum over 2 terms (for i=0,1)
    # delta_{j(a),j(b)} * delta_{k1(b),k1(c)} can be written as sum over 3 terms (for k1=0,1,2)
    # So total rank = 2 * 3 = 6? No, because the i and j are on different sides.
    #
    # Let me just explicitly construct rank-9 factors and verify.
    # 
    # Index convention:
    #   U: (n*m, rank) = (6, rank), row index = i*m + j = 3*i + j
    #   V: (m*p, rank) = (18, rank), row index = j*p + k = 6*j + k
    #   W: (p*n, rank) = (12, rank), row index = k*n + i = 2*k + i
    #
    # For rank-9, we index columns by (j, s) where j in {0,1,2} and s in {0,1,2}.
    # 
    # The condition is:
    #   sum_{j,s} U[3*i+j, (j,s)] * V[6*j+k, (j,s)] * W[2*k+i, (j,s)] = delta_{j,a=j} * delta_{k,b=k} * delta_{i,c=i}
    #   which should equal 1 if the indices are consistent, i.e., this is the matrix mult tensor.
    #
    # Actually, the tensor T[a,b,c] where a = i*m+j (input A index), b = j*p+k (input B index),
    # c = k*n+i (output C index) should be 1 iff the three indices are consistent, i.e.,
    # a = 3*i+j, b = 6*j+k, c = 2*k+i for some i,j,k.
    #
    # T[3*i+j, 6*j+k, 2*k+i] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4,5}.
    #
    # For rank-9 with columns (j, s), j in {0,1,2}, s in {0,1,2}:
    # We need:
    #   U[3*i+j, (j,s)] = u_{i,s}
    #   V[6*j+k, (j,s)] = v_{k,s}  (independent of j for the V part since j is fixed by column)
    #   W[2*k+i, (j,s)] = w_{k,i,s}
    #
    # And we need: sum_s u_{i,s} * v_{k,s} * w_{k,i,s} = delta_{k, k} = 1 for all k, i.
    # Wait, that's not right. The sum is over all columns (j,s), but for a given (a,b,c) = (3*i+j, 6*j+k, 2*k+i),
    # only columns with the same j contribute (since U has j in the row, V has j in the row).
    # So for fixed i, j, k:
    #   sum_s U[3*i+j, (j,s)] * V[6*j+k, (j,s)] * W[2*k+i, (j,s)] should equal 1.
    #
    # Let U[3*i+j, (j,s)] = A[i,s]
    # Let V[6*j+k, (j,s)] = B[k,s]  (same for all j)
    # Let W[2*k+i, (j,s)] = C[k,i,s]
    #
    # We need: sum_s A[i,s] * B[k,s] * C[k,i,s] = 1 for all i in {0,1}, k in {0,1,2,3,4,5}.
    #
    # This is 12 equations with 3*2 + 6*3 + 6*2*3 = 6 + 18 + 36 = 60 unknowns (many are zero by structure).
    #
    # Simple solution: set A[i,s] = 1, B[k,s] = 1, C[k,i,s] = 1/s ... no, that gives sum_s 1 = 3.
    #
    # Let's try: A[i,s] = 1, B[k,s] = 1, C[k,i,s] = delta_{k,s'} where s' depends on k.
    # If we set C[k,i,s] = 1 if k//2 == s and k%2 == i, else 0, then:
    #   sum_s A[i,s]*B[k,s]*C[k,i,s] = sum_s 1*1*delta_{k//2,s}*delta_{k%2,i} = 1 if k%2==i, else 0.
    # That only works for half the k values.
    #
    # Alternative: use A[i,s] = delta_{i, s%2}, B[k,s] = delta_{k//2, s//1}... 
    #
    # Let me try a cleaner approach. Since we need 12 equations and have freedom, let's use:
    # For s = 0,1,2, define:
    #   A[i,s] = 1 for all i, s
    #   B[k,s] = 1 for all k, s  
    #   C[k,i,s] = (1/3) for all k, i, s
    # Then sum_s = 3 * (1/3) = 1. But C depends on i and k trivially.
    # However, C[k,i,s] must be consistent: W[2*k+i, (j,s)] = C[k,i,s] for all j.
    # This works! W[2*k+i, (j,s)] = 1/3 for all j.
    #
    # Let's verify: U[3*i+j, (j,s)] = 1, V[6*j+k, (j,s)] = 1, W[2*k+i, (j,s)] = 1/3.
    # Sum over s (3 terms): 3 * 1 * 1 * 1/3 = 1. Correct!
    #
    # But wait, do these entries need to be integers or half-integers? 1/3 is neither.
    # The task says "integer/half-integer entries". So 1/3 is not allowed.
    #
    # Let me use rank-18 instead to get integer entries, or find a rank-9 with integer entries.
    #
    # For rank-9 with integer entries, I need sum_s A[i,s]*B[k,s]*C[k,i,s] = 1 with integers.
    # Try: A[i,s] = 1, B[k,s] = 1, C[k,i,s] = delta_{s, f(k,i)} for some function f.
    # Then sum_s = 1 if there's exactly one s with f(k,i)=s, which is always true if f is defined.
    # So set C[k,i,s] = 1 if s == f(k,i), else 0.
    # We need f: {0,1} x {0..5} -> {0,1,2} to be a well-defined function.
    # For example, f(k,i) = (k + i) % 3.
    #
    # Let me verify this works:
    # U[3*i+j, (j,s)] = 1
    # V[6*j+k, (j,s)] = 1
    # W[2*k+i, (j,s)] = 1 if s == (k+i)%3 else 0
    #
    # For fixed i,j,k: sum_s 1 * 1 * delta_{s,(k+i)%3} = 1. Correct!
    #
    # All entries are 0 or 1, which are integers. 
    rank = 9
    n, m, p = DIMENSIONS
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for s in range(3):
                col = j * 3 + s
                U[i * m + j][col] = 1
    
    for j in range(m):
        for k in range(p):
            for s in range(3):
                col = j * 3 + s
                V[j * p + k][col] = 1
    
    for k in range(p):
        for i in range(n):
            for j in range(m):
                s = (k + i) % 3
                col = j * 3 + s
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
