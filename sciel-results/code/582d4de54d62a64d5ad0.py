"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    # Matrix multiplication 2x2 @ 2x5.
    # Dimensions: n=2, m=2, p=5.
    # Shapes: U (n*m, rank) = (4, rank), V (m*p, rank) = (10, rank), W (p*n, rank) = (10, rank).
    # Indices:
    # U row: i*m + j (i in 0..1, j in 0..1)
    # V row: j*p + k (j in 0..1, k in 0..4)
    # W row: k*n + i (k in 0..4, i in 0..1)
    
    # To find a lower rank decomposition, we can exploit the linearity.
    # The tensor for matrix multiplication is T_{i,j,k} = A_{i,j} B_{j,k} -> C_{i,k}.
    # Rank of 2x2x2 multiplication is 7 (Strassen).
    # Rank of 2x2x1 multiplication is 4 (schoolbook, as 2x2 @ 2x1).
    # For 2x2 @ 2x5, we can view it as two separate 2x2 @ 2x1 multiplications? No, B is 2x5.
    # We can split B into B[:, 0:2] and B[:, 2:5].
    # C[:, 0:2] = A @ B[:, 0:2] (2x2 @ 2x2) -> Rank 7.
    # C[:, 2:5] = A @ B[:, 2:5] (2x2 @ 2x3).
    # Rank of 2x2 @ 2x3?
    # It is equivalent to multiplying a 2x2 matrix by two 2x1 matrices.
    # Each 2x2 @ 2x1 has rank 4.
    # Since the inputs A are shared, can we combine them?
    # The tensor is T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
    # We can use the "block diagonal" argument? No.
    # Let's look at the rank of 2x2x5.
    # Lower bound: max(rank(2x2x2), rank(2x2x3), ...)?
    # Actually, for m x n @ n x p, the rank is at least max(mn, mp, np, n^2 log_2 n ...)?
    # For 2x2x5, schoolbook is 2*2*5 = 20.
    # Strassen on the 2x2x2 part gives 7 for the first 2 columns of output.
    # The remaining 3 columns of output (2x2 @ 2x3) can be decomposed.
    # A 2x2 @ 2x3 multiplication can be seen as two 2x2 @ 2x1 multiplications.
    # If we did them independently, it would be 4 + 4 = 8. Total 7 + 8 = 15.
    # Can we do better?
    # The tensor for 2x2 @ 2x3 is T_{i,j,k}.
    # This is equivalent to the tensor for 2x2 @ 2x2 plus 2x2 @ 2x1? No, it's a single tensor.
    # However, the rank of a direct sum is the sum of ranks.
    # The tensor space is decomposable by the output index k if the input A is the same?
    # T = sum_k T_k where T_k is the tensor for column k.
    # T_k maps (A, B[:,k]) to C[:,k].
    # Since A is shared, we cannot simply sum the ranks of independent problems unless we are careful about the shared input.
    # BUT, the decomposition U, V, W allows sharing.
    # If we use Strassen for the first 2 columns, we use 7 products.
    # For the next 2 columns (cols 2,3), we can use Strassen again? No, B[:, 2:4] is different from B[:, 0:2].
    # Strassen products are linear combinations of B entries.
    # P1 = (A00+A11)(B00+B11) ... depends on B.
    # If we run Strassen for col 0,1 and Strassen for col 2,3, we are using different linear combinations of B.
    # So we need 7 products for cols 0,1 and 7 products for cols 2,3?
    # That would be 14 products.
    # Then for the last column (col 4), we need a 2x2 @ 2x1 multiplication.
    # Rank of 2x2 @ 2x1 is 4.
    # Total 14 + 4 = 18.
    
    # Is 18 optimal?
    # Let's check if we can share.
    # The Strassen products for the first block are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # These are specific linear combinations of B[:, 0:2].
    # For the second block (cols 2,3), we would have:
    # M8 = (A00+A11)(B22+B33)  # Note: B indices are now B[:,2], B[:,3]
    # ...
    # Since the linear combinations of A are the same (A00, A11, etc), but B is different, the "U" vectors (coefficients for A) can be shared if we structure it correctly?
    # In the decomposition C = sum_r W_r U_r V_r, U_r is a vector of length n*m=4.
    # If we have two Strassen blocks, we have 14 distinct U vectors?
    # U for M1 is [1, 0, 0, 1] (coeffs for A00, A01, A10, A11).
    # U for M8 is [1, 0, 0, 1].
    # They are the same U vector!
    # But V is different. V for M1 involves B00, B11 (indices 0, 3 in V matrix? No, V row index is j*p+k).
    # B00 is at V[0*5+0, r] = V[0, r]. B11 is at V[1*5+1, r] = V[6, r].
    # For M8, B22 is at V[1*5+2, r] = V[7, r]. B33 is at V[1*5+3, r] = V[8, r].
    # So V vectors are different.
    # However, the rank is the number of terms in the sum.
    # If U1 == U8, can we merge them?
    # Term 1: U1 V1 W1
    # Term 8: U8 V8 W8
    # If U1 == U8, we have U1 (V1 W1 + V8 W8). This doesn't reduce the rank immediately unless V1 W1 + V8 W8 has lower rank structure?
    # No, rank of sum is sum of ranks if supports are disjoint or generic.
    # So 7 + 7 + 4 = 18 seems to be the construction.
    
    # Can we do better?
    # What if we use a single decomposition for 2x2x5?
    # The rank of 2x2x5 is known to be 18?
    # Let's verify the rank of 2x2 @ 2x1.
    # A (2x2) @ B (2x1) = C (2x1).
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Products:
    # P1 = A00 B0 -> C0
    # P2 = A01 B1 -> C0
    # P3 = A10 B0 -> C1
    # P4 = A11 B1 -> C1
    # Rank 4.
    
    # Is there a rank 3 decomposition for 2x2 @ 2x1?
    # This is equivalent to multiplying a 2x2 matrix by a vector.
    # This is a linear map from (A,B) to C.
    # The tensor is T_{i,j,k}. Here n=2, m=2, p=1.
    # T_{i,j,k} = delta_{j, k?} No.
    # C_i = sum_j A_{i,j} B_j.
    # So T_{i,j,b} = A_{i,j} B_b C_i?
    # The tensor components are T_{i,j,b} = 1 if the term A_{i,j} B_b contributes to C_i.
    # It contributes if index matches.
    # The rank of this tensor is 4. (It is not decomposable further over reals/complexes for small sizes).
    
    # So, if we split 2x2x5 into 2x2x2, 2x2x2, 2x2x1:
    # Rank <= 7 + 7 + 4 = 18.
    
    # Is it possible to get 17?
    # Some sources suggest rank of 2x2x5 is 18.
    # Let's construct the 18-rank decomposition.
    
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to set entries
    def set_entry(mat, row, col, val):
        mat[row][col] = val
        
    # --- Block 1: Columns 0, 1 (2x2 @ 2x2) ---
    # Strassen Products for A (2x2) @ B (2x2) -> C (2x2)
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will map these to rank indices 0..6.
    
    # M1
    r = 0
    # U: A00 (1), A01 (0), A10 (0), A11 (1)
    set_entry(U, 0, r, 1); set_entry(U, 3, r, 1)
    # V: B00 (1, idx 0*5+0=0), B01 (0), B10 (0), B11 (1, idx 1*5+1=6)
    set_entry(V, 0, r, 1); set_entry(V, 6, r, 1)
    # W: C00 (1, idx 0*2+0=0), C01 (0), C10 (0), C11 (1, idx 1*2+1=3)
    set_entry(W, 0, r, 1); set_entry(W, 3, r, 1)
    
    # M2
    r = 1
    # U: A00 (0), A01 (0), A10 (1), A11 (1)
    set_entry(U, 2, r, 1); set_entry(U, 3, r, 1)
    # V: B00 (1, idx 0)
    set_entry(V, 0, r, 1)
    # W: C10 (1, idx 1*2+0=2), C11 (1, idx 3) -> Wait, C10 = M2+M4, C11 = M1-M2+M3+M6.
    # So W for M2 contributes to C10 and C11?
    # C10 = M2 + M4. So W_{C10} += 1.
    # C11 = M1 - M2 + M3 + M6. So W_{C11} += -1.
    set_entry(W, 2, r, 1); set_entry(W, 3, r, -1)
    
    # M3
    r = 2
    # U: A00 (1)
    set_entry(U, 0, r, 1)
    # V: B10 (1, idx 1*5+0=5), B11 (-1, idx 6)
    set_entry(V, 5, r, 1); set_entry(V, 6, r, -1)
    # W: C01 (1, idx 0*2+1=1), C11 (1, idx 3)
    # C01 = M3 + M5.
    # C11 = ... + M3 ...
    set_entry(W, 1, r, 1); set_entry(W, 3, r, 1)
    
    # M4
    r = 3
    # U: A11 (1)
    set_entry(U, 3, r, 1)
    # V: B01 (1, idx 0*5+1=1), B00 (-1, idx 0)
    set_entry(V, 1, r, 1); set_entry(V, 0, r, -1)
    # W: C00 (1, idx 0), C10 (1, idx 2)
    # C00 = M1 + M4 ...
    # C10 = M2 + M4
    set_entry(W, 0, r, 1); set_entry(W, 2, r, 1)
    
    # M5
    r = 4
    # U: A00 (1), A01 (1)
    set_entry(U, 0, r, 1); set_entry(U, 1, r, 1)
    # V: B11 (1, idx 6)
    set_entry(V, 6, r, 1)
    # W: C00 (-1, idx 0), C01 (1, idx 1)
    # C00 = M1 + M4 - M5 ...
    # C01 = M3 + M5
    set_entry(W, 0, r, -1); set_entry(W, 1, r, 1)
    
    # M6
    r = 5
    # U: A10 (1), A00 (-1)
    set_entry(U, 2, r, 1); set_entry(U, 0, r, -1)
    # V: B00 (1, idx 0), B10 (1, idx 5)
    set_entry(V, 0, r, 1); set_entry(V, 5, r, 1)
    # W: C11 (1, idx 3)
    set_entry(W, 3, r, 1)
    
    # M7
    r = 6
    # U: A01 (1), A11 (-1)
    set_entry(U, 1, r, 1); set_entry(U, 3, r, -1)
    # V: B01 (1, idx 1), B11 (1, idx 6)
    set_entry(V, 1, r, 1); set_entry(V, 6, r, 1)
    # W: C00 (-1, idx 0)
    # C00 = M1 + M4 - M5 - M7
    set_entry(W, 0, r, -1)
    
    # --- Block 2: Columns 2, 3 (2x2 @ 2x2) ---
    # Same A, B columns 2,3.
    # B indices: B22 (idx 1*5+2=7), B23 (idx 1*5+3=8), B32 (idx 0*5+2=2), B33 (idx 0*5+3=3).
    # Wait, B is 2x5. Row j, Col k.
    # V index is j*p + k.
    # j=0 (Row 0 of B, which corresponds to A's col 0):
    #   k=2 -> idx 2
    #   k=3 -> idx 3
    # j=1 (Row 1 of B, which corresponds to A's col 1):
    #   k=2 -> idx 7
    #   k=3 -> idx 8
    
    # C indices for block 2:
    # C02 (idx 2*2+0=4), C03 (idx 3*2+0=6), C12 (idx 2*2+1=5), C13 (idx 3*2+1=7).
    # Wait, W index is k*n + i.
    # k=2, i=0 -> 4
    # k=3, i=0 -> 6
    # k=2, i=1 -> 5
    # k=3, i=1 -> 7
    
    # Map Strassen products to rank 7..13.
    
    # M8 (M1 equivalent)
    r = 7
    # U: A00, A11
    set_entry(U, 0, r, 1); set_entry(U, 3, r, 1)
    # V: B32 (idx 2), B23 (idx 8)? No.
    # M1 = (A00+A11)(B00+B11). Here B00 is B[0,2], B11 is B[1,3].
    # B[0,2] is idx 2. B[1,3] is idx 8.
    set_entry(V, 2, r, 1); set_entry(V, 8, r, 1)
    # W: C02 (idx 4), C13 (idx 7)
    set_entry(W, 4, r, 1); set_entry(W, 7, r, 1)
    
    # M9 (M2 equivalent)
    r = 8
    # U: A10, A11
    set_entry(U, 2, r, 1); set_entry(U, 3, r, 1)
    # V: B32 (idx 2)
    set_entry(V, 2, r, 1)
    # W: C12 (idx 5), C13 (idx 7) -> C12 = M9+M11, C13 = M8-M9+M10+M13
    set_entry(W, 5, r, 1); set_entry(W, 7, r, -1)
    
    # M10 (M3 equivalent)
    r = 9
    # U: A00
    set_entry(U, 0, r, 1)
    # V: B22 (idx 7), B23 (-1, idx 8)
    # M3 = A00(B10-B11). Here B10 is B[1,2] (idx 7), B11 is B[1,3] (idx 8).
    set_entry(V, 7, r, 1); set_entry(V, 8, r, -1)
    # W: C03 (idx 6), C13 (idx 7)
    # C03 = M10+M12, C13 = ... + M10 ...
    set_entry(W, 6, r, 1); set_entry(W, 7, r, 1)
    
    # M11 (M4 equivalent)
    r = 10
    # U: A11
    set_entry(U, 3, r, 1)
    # V: B33 (idx 3), B32 (-1, idx 2)
    # M4 = A11(B01-B00). Here B01 is B[0,3] (idx 3), B00 is B[0,2] (idx 2).
    set_entry(V, 3, r, 1); set_entry(V, 2, r, -1)
    # W: C02 (idx 4), C12 (idx 5)
    # C02 = M8+M11-M12-M14
    # C12 = M9+M11
    set_entry(W, 4, r, 1); set_entry(W, 5, r, 1)
    
    # M12 (M5 equivalent)
    r = 11
    # U: A00, A01
    set_entry(U, 0, r, 1); set_entry(U, 1, r, 1)
    # V: B23 (idx 8)
    # M5 = (A00+A01)B11. Here B11 is B[1,3] (idx 8).
    set_entry(V, 8, r, 1)
    # W: C02 (-1, idx 4), C03 (1, idx 6)
    set_entry(W, 4, r, -1); set_entry(W, 6, r, 1)
    
    # M13 (M6 equivalent)
    r = 12
    # U: A10, A00 (-1)
    set_entry(U, 2, r, 1); set_entry(U, 0, r, -1)
    # V: B32 (idx 2), B22 (idx 7)
    # M6 = (A10-A00)(B00+B10). Here B00 is B[0,2] (idx 2), B10 is B[1,2] (idx 7).
    set_entry(V, 2, r, 1); set_entry(V, 7, r, 1)
    # W: C13 (1, idx 7)
    set_entry(W, 7, r, 1)
    
    # M14 (M7 equivalent)
    r = 13
    # U: A01, A11 (-1)
    set_entry(U, 1, r, 1); set_entry(U, 3, r, -1)
    # V: B33 (idx 3), B23 (idx 8)
    # M7 = (A01-A11)(B01+B11). Here B01 is B[0,3] (idx 3), B11 is B[1,3] (idx 8).
    set_entry(V, 3, r, 1); set_entry(V, 8, r, 1)
    # W: C02 (-1, idx 4)
    set_entry(W, 4, r, -1)
    
    # --- Block 3: Column 4 (2x2 @ 2x1) ---
    # Schoolbook rank 4.
    # Q1 = A00 B34 -> C04
    # Q2 = A01 B24 -> C04
    # Q3 = A10 B34 -> C14
    # Q4 = A11 B24 -> C14
    
    # B indices: B34 is B[0,4] (idx 4), B24 is B[1,4] (idx 9).
    # C indices: C04 is idx 4*2+0=8, C14 is idx 4*2+1=9.
    
    # Q1
    r = 14
    # U: A00
    set_entry(U, 0, r, 1)
    # V: B34 (idx 4)
    set_entry(V, 4, r, 1)
    # W: C04 (idx 8)
    set_entry(W, 8, r, 1)
    
    # Q2
    r = 15
    # U: A01
    set_entry(U, 1, r, 1)
    # V: B24 (idx 9)
    set_entry(V, 9, r, 1)
    # W: C04 (idx 8)
    set_entry(W, 8, r, 1)
    
    # Q3
    r = 16
    # U: A10
    set_entry(U, 2, r, 1)
    # V: B34 (idx 4)
    set_entry(V, 4, r, 1)
    # W: C14 (idx 9)
    set_entry(W, 9, r, 1)
    
    # Q4
    r = 17
    # U: A11
    set_entry(U, 3, r, 1)
    # V: B24 (idx 9)
    set_entry(V, 9, r, 1)
    # W: C14 (idx 9)
    set_entry(W, 9, r, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
