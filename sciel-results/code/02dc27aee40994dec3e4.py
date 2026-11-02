import time
import sys

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    
    # We split B (2x10) into two blocks B1 (2x5) and B2 (2x5).
    # We compute C1 = A B1 and C2 = A B2 using Strassen's algorithm for 2x2 matrices.
    # Strassen for 2x2x2 has rank 7.
    # Here, the "matrices" in Strassen are 2x2 for A, and 2x5 for B.
    # The products in Strassen are of the form (scalar) * (1x5 vector).
    # Each such product has rank 5.
    # There are 7 such products.
    # Total rank = 7 * 5 = 35.
    
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)] # 4x35
    V = [[0] * rank for _ in range(m * p)] # 20x35
    W = [[0] * rank for _ in range(p * n)] # 20x35
    
    # Strassen coefficients for A (2x2)
    # M1 = (A11 + A22) * B1
    # M2 = (A21 + A22) * B2
    # M3 = A11 * B3
    # M4 = A22 * B4
    # M5 = (A11 + A12) * B5
    # M6 = (A21 - A11) * B6
    # M7 = (A12 - A22) * B7
    
    # B is 2x5. Let B = [B_top; B_bottom] where B_top is 1x5, B_bottom is 1x5.
    # The "B" matrices in Strassen terms are 2x5 matrices.
    # Let's define the 7 B-matrices for one block (2x5).
    # B1 = B_top + B_bottom
    # B2 = B_top
    # B3 = B_bottom - B_top  (Note: standard Strassen uses B12-B22, here indices are rows)
    # Let's stick to standard Strassen notation for 2x2 block multiplication.
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]] where bij are 1x5 vectors.
    
    # M1 = (a11+a22) * (b11+b22)
    # M2 = (a21+a22) * b11
    # M3 = a11 * (b12-b22)
    # M4 = a22 * (b21-b11)
    # M5 = (a11+a12) * b22
    # M6 = (a21-a11) * (b11+b12)
    # M7 = (a12-a22) * (b21+b22)
    
    # Result C:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # We have two blocks, Block 1 (cols 1-5) and Block 2 (cols 6-10).
    # We will create 7 terms for Block 1 and 7 terms for Block 2.
    # Total 14 terms? No, 7 terms per block, each term is a 1x5 vector product.
    # So 7 * 5 = 35 scalar multiplications per block?
    # No, the rank is the number of scalar multiplications.
    # Each "M_i" is a 1x5 vector. Computing it requires 5 scalar multiplications.
    # So 7 * 5 = 35 multiplications for one block.
    # Two blocks: 70 multiplications.
    # Wait, this is worse than 40.
    
    # My previous analysis was:
    # "Treat the 2x10 multiplication as a 2x2 matrix multiplication over the ring of 1x5 row vectors"
    # This means we do ONE Strassen decomposition, but the entries of B are 1x5 vectors.
    # So we compute 7 intermediate products M1...M7, each of which is a 1x5 vector.
    # To compute M1 = s1 * v1, we need 5 scalar multiplications.
    # Total 7 * 5 = 35 scalar multiplications.
    # This gives the full 2x10 result C.
    # Yes, this is correct.
    
    # So we just need to set up the 35 terms.
    
    col = 0
    for block in range(2):
        # This loop is not needed if we treat B as 2x10 directly with 1x5 vector entries?
        # No, Strassen splits B into 2x2 blocks of entries.
        # If entries are 1x5 vectors, then B is a 2x2 matrix of 1x5 vectors.
        # This corresponds to B being 2x10.
        # So we only do the Strassen decomposition ONCE.
        
        pass

    # Let's define the 7 Strassen terms for the 2x2 block structure where entries are 1x5 vectors.
    # A is 2x2 scalars.
    # B is 2x2 matrix of 1x5 vectors.
    # B11 = cols 1-5 of row 1
    # B12 = cols 6-10 of row 1
    # B21 = cols 1-5 of row 2
    # B22 = cols 6-10 of row 2
    
    # Wait, standard Strassen assumes B is 2x2.
    # If B is 2x10, we can view it as 2x2 where each entry is 1x5.
    # This works.
    
    # Term 1: s1 * v1
    # s1 = a11 + a22
    # v1 = B11 + B22 (vector sum)
    # U: a11, a22 get +1
    # V: B11, B22 get +1 (all 5 components)
    # W: C11, C22 get +1 (all 5 components)
    
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for ui, uv in zip(u_indices, u_vals):
            U[ui][col] += uv
        for vi, vv in zip(v_indices, v_vals):
            V[vi][col] += vv
        for wi, wv in zip(w_indices, w_vals):
            W[wi][col] += wv
        col += 1
        
    # A indices: 0:11, 1:12, 2:21, 3:22
    # B indices: 
    # Row 1, Col 1-5: 0-4
    # Row 1, Col 6-10: 5-9
    # Row 2, Col 1-5: 10-14
    # Row 2, Col 6-10: 15-19
    # C indices:
    # Row 1, Col 1-5: 0-4
    # Row 1, Col 6-10: 5-9
    # Row 2, Col 1-5: 10-14
    # Row 2, Col 6-10: 15-19
    
    # Term 1: (A11+A22)(B11+B22) -> C11, C22
    add_term([0, 3], [1, 1], 
             [0,1,2,3,4, 10,11,12,13,14], [1]*10, 
             [0,1,2,3,4, 10,11,12,13,14], [1]*10)
             
    # Term 2: (A21+A22)B11 -> C21, C22 (neg)
    add_term([2, 3], [1, 1], 
             [0,1,2,3,4], [1]*5, 
             [10,11,12,13,14], [1]*5) # C21
    # C22 gets -1 from M2
    # We need to add this to W for C22 in the same column?
    # No, W is per column.
    # M2 contributes to C21 (+1) and C22 (-1).
    # So we add to W[10..14] +1 and W[15..19] -1.
    # But I already added W for Term 1.
    # I must combine W contributions for the same term (column).
    
    # Let's restart the construction more carefully.
    
    U = [[0] * 35 for _ in range(4)]
    V = [[0] * 35 for _ in range(20)]
    W = [[0] * 35 for _ in range(20)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        
    # We need to add multiple contributions to the same column.
    # So I will define the 7 terms explicitly.
    
    # B11: 0-4, B12: 5-9, B21: 10-14, B22: 15-19
    # C11: 0-4, C12: 5-9, C21: 10-14, C22: 15-19
    
    # Term 1: (A11+A22)(B11+B22)
    # U: 0:+1, 3:+1
    # V: 0-4:+1, 15-19:+1
    # W: C11(0-4):+1, C22(15-19):+1
    U[0][col] += 1; U[3][col] += 1
    for i in range(5):
        V[i][col] += 1
        V[15+i][col] += 1
        W[i][col] += 1
        W[15+i][col] += 1
    col += 1
    
    # Term 2: (A21+A22)B11
    # U: 2:+1, 3:+1
    # V: 0-4:+1
    # W: C21(10-14):+1, C22(15-19):-1
    U[2][col] += 1; U[3][col] += 1
    for i in range(5):
        V[i][col] += 1
        W[10+i][col] += 1
        W[15+i][col] -= 1
    col += 1
    
    # Term 3: A11(B12-B22)
    # U: 0:+1
    # V: 5-9:+1, 15-19:-1
    # W: C12(5-9):+1, C22(15-19):+1
    U[0][col] += 1
    for i in range(5):
        V[5+i][col] += 1
        V[15+i][col] -= 1
        W[5+i][col] += 1
        W[15+i][col] += 1
    col += 1
    
    # Term 4: A22(B21-B11)
    # U: 3:+1
    # V: 10-14:+1, 0-4:-1
    # W: C11(0-4):+1, C21(10-14):+1
    U[3][col] += 1
    for i in range(5):
        V[10+i][col] += 1
        V[i][col] -= 1
        W[i][col] += 1
        W[10+i][col] += 1
    col += 1
    
    # Term 5: (A11+A12)B22
    # U: 0:+1, 1:+1
    # V: 15-19:+1
    # W: C11(0-4):-1, C12(5-9):+1
    U[0][col] += 1; U[1][col] += 1
    for i in range(5):
        V[15+i][col] += 1
        W[i][col] -= 1
        W[5+i][col] += 1
    col += 1
    
    # Term 6: (A21-A11)(B11+B12)
    # U: 2:+1, 0:-1
    # V: 0-4:+1, 5-9:+1
    # W: C21(10-14):+1
    U[2][col] += 1; U[0][col] -= 1
    for i in range(5):
        V[i][col] += 1
        V[5+i][col] += 1
        W[10+i][col] += 1
    col += 1
    
    # Term 7: (A12-A22)(B21+B22)
    # U: 1:+1, 3:-1
    # V: 10-14:+1, 15-19:+1
    # W: C11(0-4):-1, C22(15-19):+1
    U[1][col] += 1; U[3][col] -= 1
    for i in range(5):
        V[10+i][col] += 1
        V[15+i][col] += 1
        W[i][col] -= 1
        W[15+i][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

