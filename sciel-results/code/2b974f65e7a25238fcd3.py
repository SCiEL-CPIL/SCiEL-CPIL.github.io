"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns a rank-42 tensor decomposition for 2x3*3x7 matrix multiplication.
    
    Uses Strassen's algorithm for the 2x2 sub-block of the 2x3 matrix
    and schoolbook for the remaining parts.
    """
    start_time = time.time()
    DEADLINE = 150.0  # Safety margin for the 120s evaluation limit + overhead
    
    n, m, p = 2, 3, 7
    
    # We split the 3 columns of A into 2+1 and the 7 columns of B into 2+2+2+1
    # A = [A_2x2 | A_2x1], B = [B_2x2; B_2x2; B_2x2; B_1x1] (splitting rows of B)
    # Wait, B is 3x7. So B has 3 rows. A has 3 columns.
    # A = [A_1 (2x2) | A_2 (2x1)]
    # B = [B_1 (2x7) ; B_2 (1x7)]
    # C = A_1 B_1 + A_2 B_2
    
    # A_1 is 2x2, B_1 is 2x7.
    # We can use Strassen for A_1 B_1 if we view B_1 as having 2 blocks of columns?
    # Strassen computes 2x2 * 2x2.
    # Here we have 2x2 * 2x7.
    # We can split B_1 into B_1a (2x3) and B_1b (2x4).
    # Or just use schoolbook for the whole thing? No, that's rank 42.
    # Let's try to use Strassen on the 2x2 part.
    
    # Actually, the simplest valid improvement is to ensure the code runs fast and is correct.
    # The schoolbook rank is 42.
    # Can we do better?
    # Rank(2x2*2x2) = 7.
    # Rank(2x1*1x7) = 2.
    # If we split B into 2x7 as 2x2 + 2x2 + 2x2 + 2x1.
    # C = A_1 B_1 + A_2 B_2
    # A_1 is 2x2. B_1 is 2x7.
    # Let B_1 = [B_1_1, B_1_2, B_1_3, B_1_4] where B_1_1 is 2x2, B_1_2 is 2x2, B_1_3 is 2x2, B_1_4 is 2x1.
    # A_1 B_1 = A_1 B_1_1 + A_1 B_1_2 + A_1 B_1_3 + A_1 B_1_4.
    # We can use Strassen for A_1 B_1_1 (rank 7).
    # Schoolbook for A_1 B_1_2 (rank 4).
    # Schoolbook for A_1 B_1_3 (rank 4).
    # Schoolbook for A_1 B_1_4 (rank 2).
    # Schoolbook for A_2 B_2 (A_2 is 2x1, B_2 is 1x7, rank 2).
    # Total rank = 7 + 4 + 4 + 2 + 2 = 19?
    # Wait, the decomposition must be a single sum of outer products of vectors of length 42 (U), 21 (V), 14 (W).
    # The rank is the number of terms.
    
    # Let's build a decomposition of rank 19.
    # A = [A_1 | A_2] where A_1 is 2x2, A_2 is 2x1.
    # B = [B_1; B_2] where B_1 is 2x7, B_2 is 1x7.
    # C = A_1 B_1 + A_2 B_2.
    
    # Split B_1 into B_1a (2x2), B_1b (2x2), B_1c (2x2), B_1d (2x1).
    # C = A_1 B_1a + A_1 B_1b + A_1 B_1c + A_1 B_1d + A_2 B_2.
    
    # For A_1 B_1a (2x2 * 2x2), use Strassen (7 terms).
    # For A_1 B_1b (2x2 * 2x2), use Strassen (7 terms).
    # For A_1 B_1c (2x2 * 2x2), use Strassen (7 terms).
    # For A_1 B_1d (2x2 * 2x1), use schoolbook (2 terms).
    # For A_2 B_2 (2x1 * 1x7), use schoolbook (2 terms).
    # Total rank = 7+7+7+2+2 = 25.
    
    # Can we do better for 2x2 * 2x1?
    # A_1 B_1d: A_1 = [[a,b],[c,d]], B_1d = [[e],[f]].
    # Result = [[ae+bf],[ce+df]].
    # Rank is 2? Yes, it's just two independent linear forms.
    # Actually, we can combine A_2 B_2 with something?
    # A_2 is 2x1, B_2 is 1x7.
    # A_2 = [g; h], B_2 = [i,j,k,l,m,n,o].
    # Result = [g i, g j, ..., h o].
    # This is rank 2 (g * row1 + h * row2).
    
    # So rank 25 is possible.
    # Can we do better for 2x2 * 2x2?
    # Strassen is 7.
    
    # Is there a better decomposition for 2x3 * 3x7?
    # Maybe split B_1 into 2x3 + 2x4?
    # A_1 B_1 = A_1 (2x3) + A_1 (2x4).
    # 2x2 * 2x3: Rank?
    # 2x2 * 2x1 is rank 2. 2x2 * 2x2 is rank 7.
    # 2x2 * 2x3 = 2x2*2x2 + 2x2*2x1 = 7 + 2 = 9.
    # 2x2 * 2x4 = 2x2*2x2 + 2x2*2x2 = 7 + 7 = 14.
    # Total 9 + 14 = 23. Plus A_2 B_2 (2) = 25.
    
    # What if we split A_1 differently?
    # No, Strassen is optimal for 2x2.
    
    # Let's implement the rank 25 decomposition.
    
    # Indices:
    # A is 2x3. A[i,j] for i in 0..1, j in 0..2.
    # B is 3x7. B[j,k] for j in 0..2, k in 0..6.
    # C is 2x7. C[i,k] for i in 0..1, k in 0..6.
    
    # A_1 = A[:, 0:2] (2x2)
    # A_2 = A[:, 2] (2x1)
    # B_1 = B[0:2, :] (2x7)
    # B_2 = B[2, :] (1x7)
    
    # B_1a = B[0:2, 0:2] (2x2)
    # B_1b = B[0:2, 2:4] (2x2)
    # B_1c = B[0:2, 4:6] (2x2)
    # B_1d = B[0:2, 6:7] (2x1)
    
    # C = A_1 B_1a + A_1 B_1b + A_1 B_1c + A_1 B_1d + A_2 B_2
    
    # U has shape (2*3, rank) = (6, rank)
    # V has shape (3*7, rank) = (21, rank)
    # W has shape (7*2, rank) = (14, rank)
    # W index: k*n + i = k*2 + i
    
    rank = 25
    U = np.zeros((6, rank))
    V = np.zeros((21, rank))
    W = np.zeros((14, rank))
    
    col = 0
    
    # Helper to add a Strassen term for 2x2 * 2x2
    # A_sub = A[i1:i1+2, j1:j1+2]
    # B_sub = B[i1:i1+2, k1:k1+2]
    # C_sub = C[i1:i1+2, k1:k1+2]
    def add_strassen(i1, j1, k1):
        nonlocal col
        # A_00, A_01, A_10, A_11
        # B_00, B_01, B_10, B_11
        # C_00 = (A_00+A_11)(B_00+B_11) - M5 - M6
        # C_01 = A_11(B_01+B_10)
        # C_10 = (A_00+A_01)B_10
        # C_11 = (A_10+A_11)B_00 - M2 - M3
        
        # M1 = (A_00+A_11)(B_00+B_11)
        # M2 = (A_00+A_01)B_10
        # M3 = A_11(B_01+B_10)
        # M4 = (A_10+A_11)B_00
        # M5 = (A_00+A_11)B_11
        # M6 = A_00(B_10-B_11)
        # M7 = (A_01-A_10)(B_11+B_00)
        
        # We need to map these to U, V, W.
        # U indexes: A row, A col. A is 2x3.
        # V indexes: B row, B col. B is 3x7.
        # W indexes: C row, C col. C is 2x7.
        
        # For a term t = u * v * w:
        # U[idx_a, col] += u
        # V[idx_b, col] += v
        # W[idx_c, col] += w
        
        # A_00 is A[i1, j1]
        # A_01 is A[i1, j1+1]
        # A_10 is A[i1+1, j1]
        # A_11 is A[i1+1, j1+1]
        
        # B_00 is B[j1, k1]
        # B_01 is B[j1, k1+1]
        # B_10 is B[j1+1, k1]
        # B_11 is B[j1+1, k1+1]
        
        # C_00 is C[i1, k1]
        # C_01 is C[i1, k1+1]
        # C_10 is C[i1+1, k1]
        # C_11 is C[i1+1, k1+1]
        
        # M1
        # U: A[i1, j1] += 1, A[i1+1, j1+1] += 1
        # V: B[j1, k1] += 1, B[j1+1, k1+1] += 1
        # W: C[i1, k1] += 1, C[i1+1, k1+1] -= 1 (wait, C_11 has -M5-M6, not -M1. M1 contributes to C_00 and C_11? No.)
        # Strassen:
        # C_00 = M1 - M5 - M6
        # C_01 = M3
        # C_10 = M2
        # C_11 = M4 - M2 - M3 + M7 (wait, standard Strassen is different)
        
        # Standard Strassen:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B21 - B22)
        # M4 = A22 * (B12 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 - M5 + M7
        # C12 = M4 + M5
        # C21 = M2 + M3
        # C22 = M1 - M2 + M4
        
        # Let's stick to this standard version.
        # Indices 0,1 for A and B.
        
        # M1
        # U: A[i1, j1] += 1, A[i1+1, j1+1] += 1
        # V: B[j1, k1] += 1, B[j1+1, k1+1] += 1
        # W: C[i1, k1] += 1, C[i1+1, k1+1] += 1
        # Wait, C11 is top-left. C11 corresponds to i1, k1.
        # U index for A[i,j] is i*3+j.
        # V index for B[j,k] is j*7+k.
        # W index for C[i,k] is k*2+i.
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs:
                U[idx, col] += val
            for idx, val in v_coeffs:
                V[idx, col] += val
            for idx, val in w_coeffs:
                W[idx, col] += val
            col += 1
        
        a00 = i1*3+j1
        a01 = i1*3+j1+1
        a10 = (i1+1)*3+j1
        a11 = (i1+1)*3+j1+1
        
        b00 = j1*7+k1
        b01 = j1*7+k1+1
        b10 = (j1+1)*7+k1
        b11 = (j1+1)*7+k1+1
        
        c00 = k1*2+i1
        c01 = (k1+1)*2+i1
        c10 = k1*2+(i1+1)
        c11 = (k1+1)*2+(i1+1)
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes +1 to C00, +1 to C11
        add_term([(a00,1),(a11,1)], [(b00,1),(b11,1)], [(c00,1),(c11,1)])
        
        # M2 = (A10+A11)B00
        # Contributes +1 to C10
        add_term([(a10,1),(a11,1)], [(b00,1)], [(c10,1)])
        
        # M3 = A00(B10-B11)
        # Contributes +1 to C10
        add_term([(a00,1)], [(b10,1),(b11,-1)], [(c10,1)])
        
        # M4 = A11(B01-B00)
        # Contributes +1 to C01, +1 to C11
        add_term([(a11,1)], [(b01,1),(b00,-1)], [(c01,1),(c11,1)])
        
        # M5 = (A00+A01)B11
        # Contributes -1 to C00, +1 to C01
        add_term([(a00,1),(a01,1)], [(b11,1)], [(c00,-1),(c01,1)])
        
        # M6 = (A10-A00)(B00+B01)
        # Contributes +1 to C00
        add_term([(a10,1),(a00,-1)], [(b00,1),(b01,1)], [(c00,1)])
        
        # M7 = (A01-A11)(B10+B11)
        # Contributes +1 to C00, +1 to C11
        add_term([(a01,1),(a11,-1)], [(b10,1),(b11,1)], [(c00,1),(c11,1)])
        
    # A_1 B_1a: i1=0, j1=0, k1=0
    add_strassen(0, 0, 0)
    
    # A_1 B_1b: i1=0, j1=0, k1=2
    add_strassen(0, 0, 2)
    
    # A_1 B_1c: i1=0, j1=0, k1=4
    add_strassen(0, 0, 4)
    
    # A_1 B_1d: i1=0, j1=0, k1=6 (2x2 * 2x1)
    # A_1 = [[a,b],[c,d]], B_1d = [[e],[f]]
    # C_1d = [[ae+bf],[ce+df]]
    # Term 1: a e -> C[0,6]
    # Term 2: b f -> C[0,6]
    # Term 3: c e -> C[1,6]
    # Term 4: d f -> C[1,6]
    
    # U for A_1: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d.
    # V for B_1d: B[0,6]=e, B[1,6]=f.
    # W for C_1d: C[0,6], C[1,6].
    
    # a e
    add_term([(0*3+0, 1)], [(0*7+6, 1)], [(6*2+0, 1)])
    # b f
    add_term([(0*3+1, 1)], [(1*7+6, 1)], [(6*2+0, 1)])
    # c e
    add_term([(1*3+0, 1)], [(0*7+6, 1)], [(6*2+1, 1)])
    # d f
    add_term([(1*3+1, 1)], [(1*7+6, 1)], [(6*2+1, 1)])
    
    # A_2 B_2: A_2 is 2x1 (A[:,2]), B_2 is 1x7 (B[2,:])
    # A_2 = [g; h], B_2 = [i, j, k, l, m, n, o]
    # C_2 = [gi, gj, gk, gl, gm, gn, go; hi, hj, hk, hl, hm, hn, ho]
    
    # g * i
    add_term([(0*3+2, 1)], [(2*7+0, 1)], [(0*2+0, 1)])
    # h * i
    add_term([(1*3+2, 1)], [(2*7+0, 1)], [(0*2+1, 1)])
    # ... this is rank 14 if done schoolbook.
    # But A_2 is 2x1, B_2 is 1x7.
    # C_2 = A_2 B_2.
    # We can write this as 2 terms?
    # A_2 = [g; h] = g * [1; 0] + h * [0; 1]
    # B_2 = [i, ..., o] = i * [1, 0, ...] + ...
    # No, the tensor is A_2 B_2.
    # U vector for A_2: [0, 0, g; 0, 0, h] -> just the 3rd column of A.
    # V vector for B_2: [i, j, ...; 0...; 0...] -> just the 3rd row of B.
    # The product is the outer product of A_2 and B_2.
    # This is a single term in the tensor decomposition?
    # T(A,B) = A_2 B_2.
    # U has A_2, V has B_2, W has identity on C?
    # No, W must map the scalar to C.
    # The term is U_k * V_k * W_k.
    # If U_k has g,h and V_k has i,j,.., then the product is (g,h) * (i,j,..) which is a 2x7 matrix.
    # W_k must be 1 for all entries?
    # If W_k is a vector of 1s, then U_k V_k W_k is not defined.
    # The decomposition is sum_k U_k V_k W_k where U_k is 6x1, V_k is 21x1, W_k is 14x1.
    # The term is U_k (V_k (W_k^T?)) No.
    # The tensor is T_{i,j,k} = sum_c U_{ij, c} V_{jk, c} W_{ki, c}.
    # For A_2 B_2:
    # A_2 is A[:, 2]. B_2 is B[2, :].
    # The contribution to C[i, k] is A[i, 2] B[2, k].
    # This is a rank 1 tensor in the subspace spanned by A[:,2] and B[2,:].
    # Can we represent it with one term?
    # U_{i,2, c} = delta_{i, ?} No, U is 6x1.
    # U_{i, 2} is the coefficient of A[i, 2].
    # If we set U_{i, 2, c} = 1 for i=0,1? No, U is a vector.
    # If we set U to have 1 at indices corresponding to A[0,2] and A[1,2]?
    # Then V must have 1 at B[2,0], B[2,1]...
    # And W must have 1 at C[0,0], C[0,1]...
    # But a single term is a product of 3 vectors.
    # u * v * w.
    # If u has non-zeros at A[0,2] and A[1,2], v has non-zeros at B[2,0]...B[2,6], w has non-zeros at C[0,0]...C[1,6].
    # Then u v w will have non-zeros at all combinations.
    # Specifically, T_{i, 2, k} += U_{i,2} V_{2,k} W_{k,i}.
    # We want T_{i, 2, k} = A_{i,2} B_{2,k}.
    # So we need U_{i,2} V_{2,k} W_{k,i} = A_{i,2} B_{2,k}.
    # This looks like it might not be possible with 1 term unless the vectors are sparse in a specific way.
    # But A_{i,2} B_{2,k} is separable?
    # It is the outer product of vector A_2 (length 2) and vector B_2 (length 7).
    # In the tensor T_{i,j,k}, j is fixed to 2.
    # So it is a matrix M_{i,k} = A_{i,2} B_{2,k}.
    # This matrix has rank 1.
    # Can we represent a rank 1 matrix M_{i,k} as sum_c U_{i,2,c} V_{2,k,c} W_{k,i,c}?
    # If we set c=1: U_{i,2,1} = A_{i,2}, V_{2,k,1} = B_{2,k}, W_{k,i,1} = 1.
    # Then sum = A_{i,2} B_{2,k} * 1 = A_{i,2} B_{2,k}.
    # Yes! One term is enough for A_2 B_2.
    
    # So we need 1 term for A_2 B_2.
    # U: indices 0*3+2=2, 1*3+2=5.
    # V: indices 2*7+0=14, ..., 2*7+6=20.
    # W: indices 0*2+0=0, ..., 6*2+1=13.
    
    # But wait, the values of A and B are variables.
    # U, V, W must be constant matrices.
    # The decomposition is T = sum_c U_c V_c W_c.
    # The entries of T are linear in A and B.
    # U_c is a vector of coefficients for A.
    # If U_c has 1 at A[0,2] and 0 elsewhere, then it picks out A[0,2].
    # We cannot have U_c depend on A.
    # So the term U_c V_c W_c must be such that the sum reconstructs the bilinear form.
    # For A_2 B_2, the bilinear form is A[0,2] B[2,0] for C[0,0], etc.
    # This is a sum of 14 monomials: A[0,2]B[2,0], A[0,2]B[2,1], ..., A[1,2]B[2,6].
    # Each monomial is a rank 1 term in the tensor T_{i,j,k}.
    # T_{0,2,0} = A[0,2] B[2,0].
    # This requires U to have 1 at (0,2), V at (2,0), W at (0,0).
    # T_{0,2,1} = A[0,2] B[2,1].
    # This requires U at (0,2), V at (2,1), W at (1,0).
    # ...
    # So we need 14 terms for A_2 B_2?
    # Yes, because the support of the tensor for A_2 B_2 is 14 disjoint points (i, 2, k).
    # And they cannot be combined because the A and B indices vary independently.
    # So rank 14 for A_2 B_2.
    
    # Total rank = 7 (Strassen 1) + 7 (Strassen 2) + 7 (Strassen 3) + 4 (School 2x2x1) + 14 (School 2x1x7) = 39.
    # This is better than 42.
    
    # Let's add the 14 terms for A_2 B_2.
    for i in range(2):
        for k in range(7):
            add_term([(i*3+2, 1)], [(2*7+k, 1)], [(k*2+i, 1)])
            
    # And the 4 terms for A_1 B_1d were already added?
    # Yes, I added 4 terms above for A_1 B_1d.
    # Wait, A_1 B_1d is 2x2 * 2x1.
    # C = [[ae+bf], [ce+df]].
    # T_{0, 0, 6} = A[0,0]B[0,6] + A[0,1]B[1,6].
    # This is 2 monomials.
    # T_{1, 0, 6} = A[1,0]B[0,6] + A[1,1]B[1,6].
    # This is 2 monomials.
    # So 4 terms total. Correct.
    
    # So the total rank is 7+7+7+4+14 = 39.
    
    # Can we do better?
    # Maybe A_1 B_1d can be done with 2 terms?
    # No, the support is 4 points.
    
    # So 39 is the rank.
    
    # Verify time
    if time.time() > DEADLINE:
        # Fallback to schoolbook if we ran out of time (shouldn't happen)
        rank = 42
        U = np.zeros((6, rank))
        V = np.zeros((21, rank))
        W = np.zeros((14, rank))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(7):
                    U[i*3+j, col] = 1
                    V[j*7+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
