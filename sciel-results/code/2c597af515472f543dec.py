"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Rank 7 decomposition (Strassen)
    # U: (4, 7)
    # V: (4, 7)
    # W: (4, 7)
    
    # Indices:
    # U[i, r] corresponds to A[i] where i is 0..3 (flattened 2x2)
    # V[j, r] corresponds to B[j]
    # W[k, r] corresponds to C[k]
    
    # Strassen's 7 products:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b01 - b11)
    # p4 = a11(b10 - b00)
    # p5 = (a00 + a01)b11
    # p6 = (a10 - a00)(b00 + b01)
    # p7 = (a01 - a11)(b10 + b11)
    
    # Flattened indices:
    # A: 00=0, 01=1, 10=2, 11=3
    # B: 00=0, 01=1, 10=2, 11=3
    # C: 00=0, 01=1, 10=2, 11=3
    
    # Define columns for U, V, W
    # Each column r corresponds to one product term.
    
    U = []
    V = []
    W = []
    
    # p1
    u_col = [1, 0, 0, 1]
    v_col = [1, 0, 0, 1]
    w_col = [1, 0, 0, 1]
    U.append(u_col)
    V.append(v_col)
    W.append(w_col)
    
    # p2
    u_col = [0, 0, 1, 1]
    v_col = [1, 0, 0, 0]
    w_col = [0, 1, 0, -1] # C10 += p2, C11 -= p2? 
    # Let's check Strassen formula:
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # My previous w_col for p2 was [0, 1, 0, -1] which means C01 += 0, C10 += 1, C11 -= 1.
    # Formula: C10 = p2 + p4. So C10 gets +1.
    # Formula: C11 = ... - p2 ... So C11 gets -1.
    # Correct.
    
    U.append(u_col)
    V.append(v_col)
    W.append(w_col)
    
    # p3
    u_col = [1, 0, 0, 0]
    v_col = [0, 1, 0, -1]
    w_col = [0, 1, 0, 1] # C01 += 1, C11 += 1
    # Formula: C01 = p3 + p5. So C01 gets +1.
    # Formula: C11 = ... + p3 ... So C11 gets +1.
    # Correct.
    
    U.append(u_col)
    V.append(v_col)
    W.append(w_col)
    
    # p4
    u_col = [0, 0, 0, 1]
    v_col = [-1, 0, 1, 0] # b10 - b00. B indices: 00=0, 01=1, 10=2, 11=3.
    # b10 is index 2. b00 is index 0.
    # v_col should be [ -1, 0, 1, 0 ].
    # My previous v3 was [-1, 0, 1, 0]. Correct.
    w_col = [1, 0, 1, 0] # C00 += 1, C10 += 1
    # Formula: C00 = ... + p4 ...
    # Formula: C10 = ... + p4 ...
    # Correct.
    
    U.append(u_col)
    V.append(v_col)
    W.append(w_col)
    
    # p5
    u_col = [1, 1, 0, 0] # a00 + a01
    v_col = [0, 0, 0, 1] # b11
    w_col = [-1, 1, 0, 0] # C00 -= 1, C01 += 1
    # Formula: C00 = ... - p5 ...
    # Formula: C01 = ... + p5 ...
    # Correct.
    
    U.append(u_col)
    V.append(v_col)
    W.append(w_col)
    
    # p6
    u_col = [-1, 0, 1, 0] # a10 - a00
    v_col = [1, 1, 0, 0] # b00 + b01
    w_col = [0, 0, 0, 1] # C11 += 1
    # Formula: C11 = ... + p6 ...
    # Correct.
    
    U.append(u_col)
    V.append(v_col)
    W.append(w_col)
    
    # p7
    u_col = [0, 1, 0, -1] # a01 - a11
    v_col = [0, 0, 1, 1] # b10 + b11
    w_col = [1, 0, 0, 0] # C00 += 1
    # Formula: C00 = ... + p7 ...
    # Correct.
    
    U.append(u_col)
    V.append(v_col)
    W.append(w_col)
    
    # The output expects U, V, W as matrices of shape (4, 7).
    # I have constructed them as lists of columns.
    # U is currently [[col0], [col1], ...] which is a list of 7 lists of length 4.
    # This is shape (7, 4).
    # The task requires shape (4, 7).
    # So I need to transpose.
    
    U_final = [list(row) for row in zip(*U)]
    V_final = [list(row) for row in zip(*V)]
    W_final = [list(row) for row in zip(*W)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
