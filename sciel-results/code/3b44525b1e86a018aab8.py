    # A_blocks
    A11 = [get_A_vec(0,0), get_A_vec(0,1), get_A_vec(1,0), get_A_vec(1,1)]
    A12 = [get_A_vec(0,2), get_A_vec(0,3), get_A_vec(1,2), get_A_vec(1,3)]
    A21 = [get_A_vec(2,0), get_A_vec(2,1), get_A_vec(3,0), get_A_vec(3,1)]
    A22 = [get_A_vec(2,2), get_A_vec(2,3), get_A_vec(3,2), get_A_vec(3,3)]
    
    # B_blocks (offset 0)
    B11 = [get_B_vec(0,0), get_B_vec(0,1), get_B_vec(1,0), get_B_vec(1,1)]
    B12 = [get_B_vec(0,2), get_B_vec(0,3), get_B_vec(1,2), get_B_vec(1,3)]
    B21 = [get_B_vec(2,0), get_B_vec(2,1), get_B_vec(3,0), get_B_vec(3,1)]
    B22 = [get_B_vec(2,2), get_B_vec(2,3), get_B_vec(3,2), get_B_vec(3,3)]
    
    # Helper to combine blocks
    def combine_blocks(b1, b2, op):
        if op == 'add':
            return [add_vecs(x, y) for x, y in zip(b1, b2)]
        elif op == 'sub':
            return [sub_vecs(x, y) for x, y in zip(b1, b2)]
        else:
            return b1

    # M1: (A11+A22)(B11+B22)
    A_M1 = combine_blocks(A11, A22, 'add')
    B_M1 = combine_blocks(B11, B22, 'add')
    
    # M2: (A21+A22)B11
    A_M2 = combine_blocks(A21, A22, 'add')
    B_M2 = B11
    
    # M3: A11(B12-B22)
    A_M3 = A11
    B_M3 = combine_blocks(B12, B22, 'sub')
    
    # M4: A22(B21-B11)
    A_M4 = A22
    B_M4 = combine_blocks(B21, B11, 'sub')
    
    # M5: (A11+A12)B22
    A_M5 = combine_blocks(A11, A12, 'add')
    B_M5 = B22
    
    # M6: (A21-A11)(B11+B12)
    A_M6 = combine_blocks(A21, A11, 'sub')
    B_M6 = combine_blocks(B11, B12, 'add')
    
    # M7: (A12-A22)(B21+B22)
    A_M7 = combine_blocks(A12, A22, 'sub')
    B_M7 = combine_blocks(B21, B22, 'add')
    
    A_parts = [A_M1, A_M2, A_M3, A_M4, A_M5, A_M6, A_M7]
    B_parts = [B_M1, B_M2, B_M3, B_M4, B_M5, B_M6, B_M7]
