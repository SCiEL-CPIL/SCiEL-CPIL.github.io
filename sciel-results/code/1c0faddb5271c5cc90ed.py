def _build_schoolbook_2x2_2x2():
    """
    Build schoolbook decomposition for 2x2 * 2x2 -> 2x2.
    Rank 4.
    U, V, W shapes (4, 4).
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = a*e -> C[0,0]
    # M2 = b*f -> C[0,1]
    # M3 = c*g -> C[1,0]
    # M4 = d*h -> C[1,1]
    # Wait, this is not quite right.
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    
    # We need 4 multiplications?
    # Standard schoolbook for 2x2 is 4 multiplications? No, it's 4 scalar multiplications for each output?
    # No, 2x2 matrix multiplication requires 8 scalar multiplications in schoolbook.
    # Rank of 2x2x2 tensor is 7 (Strassen).
    # Schoolbook rank is 8.
    # My previous code used Strassen (7).
    # If I use schoolbook, rank is 8.
    # Then total rank:
    # C11: 8+8=16
    # C12: 6+6=12
    # C21: 8+8=16
    # C22: 6+6=12
    # Total: 56.
    # This is worse than Strassen (76? No, Strassen was 14+12+14+12=52 if rank was 52, but actual cols were 76).
    # Wait, Strassen rank is 7.
    # My previous calculation:
    # C11: 7+7=14
    # C12: 6+6=12
    # C21: 7+7=14
    # C22: 6+6=12
    # Total 52.
    # But the code allocated 52 and used 76 cols?
    # Let's re-verify `add_schoolbook_2x2_2x3`.
    # It adds 12 cols.
    # `add_strassen_block` adds 7 cols.
    # So 14 + 12 + 14 + 12 = 52.
    # Why did it fail?
    # Ah, `add_strassen_block` increments `col` by 7.
    # `add_schoolbook_2x2_2x3` increments `col` by 12.
    # Sum is 52.
    # So why IndexError?
    # Let's check `add_strassen_block` loop.
    # `for c_idx in range(7):`
    # Inside, it iterates `u_i` in 4, `v_i` in 4, `w_i` in 4.
    # It writes to `U[...][col]`.
    # If `col` reaches 52, and list size is 52, indices 0..51 are valid.
    # So it should fit.
    # Maybe the index calculation inside `add_strassen_block` for W is wrong and writes out of bounds of the *row*?
    # `W[global_k * n + global_i][col]`
    # `global_k` is up to 3 (for 4x4). `n=4`.
    # `global_k * 4 + global_i` max is $3*4+3 = 15$.
    # `W` has 20 rows. So row index is safe.
    # `col` is the column index.
    # If `col` is 51, `W[row][51]` is valid.
    # If `col` becomes 52, it crashes.
    # Does `col` become 52?
    # 14 (C11) + 12 (C12) + 14 (C21) + 12 (C22) = 52.
    # The last write happens at `col=51`.
    # Then `col` becomes 52.
    # So it should be fine.
    
    # Let's look at the error again: `IndexError: list assignment index out of range`.
    # Maybe `global_k * n + global_i` is out of bounds for W?
    # W has shape (PN, rank) = (20, rank).
    # `global_k` is from `c_col_off + local_k`.
    # In `add_strassen_block`, `c_col_off` is passed.
    # For C11 (0,0), `c_col_off=0`. `local_k` is 0 or 1. `global_k` is 0 or 1.
    # For C12 (0,2), `c_col_off=2`. `local_k` is 0 or 1?
    # Wait, C12 is 2x3. The Strassen block is for 2x2*2x2.
    # I used Strassen for C11 (2x2) and C21 (2x2).
    # I used Schoolbook for C12 (2x3) and C22 (2x3).
    # So `add_strassen_block` is only called for 2x2 output blocks.
    # So `global_k` and `global_i` are within 0..1 or 2..3.
    # So `W` index is safe.
    
    # What about `U` and `V`?
    # `U` has 16 rows.
    # `V` has 20 rows.
    # Indices seem safe.
    
    # Is it possible that `rank` is not 52?
    # `rank = 52`.
    # Maybe one of the blocks adds more columns?
    # `add_strassen_block`: 7 cols.
    # `add_schoolbook_2x2_2x3`: 12 cols.
    # Total 52.
    
    # Let's check `add_schoolbook_2x2_2x3` again.
    # `for a_i in range(2):`
    # `  for a_j in range(2):`
    # `    for b_k in range(3):`
    # `      col += 1`
    # 12 increments.
    
    # I suspect the error might be in `_verify` or `_verify_fast`?
    # No, "list assignment index out of range" usually happens during construction.
    
    # Let's assume the rank calculation was wrong in a subtle way or there's an off-by-one in the loop.
    # To be safe, I will increase `rank` to 80 (schoolbook) or just use the full schoolbook decomposition which is guaranteed to work.
    # Or, I will implement the 4x4x4 (19) + 4x4x1 (16) = 35 strategy.
    # Since I can't easily code BCL 19, I will use the schoolbook 4x4x4 (rank 16) + schoolbook 4x4x1 (rank 16) = 32?
    # No, schoolbook 4x4x4 is rank 16.
    # Schoolbook 4x4x1 is rank 16.
    # Total 32.
    # This is better than 40 or 52.
    # And it's easy to code.
    
    # Plan:
    # 1. Decompose B into B1 (4x4) and B2 (4x1).
    # 2. Use schoolbook for A*B1 (4x4x4). Rank 16.
    # 3. Use schoolbook for A*B2 (4x4x1). Rank 16.
    # 4. Total rank 32.
    
    # This is much simpler and safer than block Strassen.
    
    # I will replace `_build_block_decomposition` with `_build_split_decomposition`.
    
    def _build_split_decomposition():
        rank = 32
        U = [[0]*rank for _ in range(NM)]
        V = [[0]*rank for _ in range(MP)]
        W = [[0]*rank for _ in range(PN)]
        
        col = 0
        # Part 1: A * B1 (4x4 * 4x4)
        # B1 is B[:, 0:4]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        
        # Part 2: A * B2 (4x4 * 4x1)
        # B2 is B[:, 4:5]
        # k index for B is 4.
        for i in range(4):
            for j in range(4):
                # k is fixed at 4 for the input column, but output k is 0..3?
                # Wait, B2 is 4x1.
                # A is 4x4.
                # C2 = A * B2 is 4x1.
                # C2[i] = sum_j A[i,j] B2[j]
                # B2[j] is B[j, 4].
                # So for each i, j:
                # U[i, j] * V[j, 4] * W[?, i]
                # The output is a vector, so k (output column) is 0?
                # The tensor is T[i, j, k].
                # For the 4x4x1 part, the "k" dimension of the tensor corresponds to the output column.
                # Since the output is 4x1, there is only 1 output column.
                # But the tensor T is defined for the whole 4x4x5 multiplication.
                # The "k" index in T[i,j,k] refers to the column of B and C.
                # So for the B2 part, k=4 (the 5th column of B).
                # But C only has 5 columns? No, C is 4x5.
                # Wait, A is 4x4, B is 4x5. C is 4x5.
                # So k ranges 0..4.
                # My previous assumption "4x4x1 block" meant the part of the tensor where k=4.
                # So T[i,j,4] = sum_l A[i,l] B[l,4].
                # This is a rank-16 sub-tensor.
                # U[i, l] * V[l, 4] * W[4, i]?
                # W index is k*n+i.
                # Here k=4 (the column index of B/C).
                # So W[4*4 + i] = W[16+i].
                # W has 20 rows (5 cols * 4 rows).
                # So W[16..19] are used.
                
                for j in range(4):
                    U[i*m + j][col] = 1
                    V[j*p + 4][col] = 1
                    W[4*n + i][col] = 1
                    col += 1
        
        return U, V, W

    # This gives rank 32.
    # I will use this.
