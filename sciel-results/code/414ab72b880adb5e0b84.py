"""
Strategy: Deterministic placement of a single 2x2x2 Strassen block to reduce rank from 105 to 104.
The schoolbook rank is 3*5*7 = 105.
A 2x2x2 block has schoolbook rank 8 but Strassen rank 7.
We can fit exactly one disjoint 2x2x2 block in 3x5x7 (using 2 of 3 rows, 2 of 5 cols, 2 of 7 idxs).
The remaining tensor elements are handled by schoolbook.
Total rank = 7 + (105 - 8) = 104.
"""

def construct():
    n, m, p = 3, 5, 7
    # We will construct U, V, W matrices.
    # Rank = 104
    
    # Indices for the Strassen block
    # Rows: 0, 1
    # Cols: 0, 1
    # Idxs: 0, 1
    
    # Strassen's 7 multiplications for 2x2x2:
    # Let A = [[a00, a01], [a10, a11]]
    # Let B = [[b00, b01], [b10, b11]]
    # M1 = a00 * (b00 + b11)
    # M2 = (a00 + a01) * b11
    # M3 = (a10 + a11) * b00
    # M4 = a11 * (b10 + b00)
    # M5 = (a00 + a11) * (b01 + b10)
    # M6 = (a01 - a11) * (b11 + b10)
    # M7 = (a10 - a00) * (b00 + b01)
    # C00 = M1 + M4 - M2 + M6
    # C01 = M5 + M2 - M4 - M7
    # C10 = M3 + M5 + M7 - M1
    # C11 = M1 + M2 - M3 + M4
    
    # We need to map these to the global U, V, W matrices.
    # U has shape (n*m, rank) = (15, 104)
    # V has shape (m*p, rank) = (35, 104)
    # W has shape (p*n, rank) = (21, 104)
    
    # Let's assign the first 7 columns to the Strassen block.
    # The remaining 97 columns to the schoolbook terms for the rest.
    
    # Strassen Block Details:
    # A elements: A[0,0] (U[0,0]), A[0,1] (U[1,0]), A[1,0] (U[5,0]), A[1,1] (U[6,0])
    # Note: U index is i*m + j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 5
    # i=1, j=1 -> 6
    
    # B elements: B[0,0] (V[0,0]), B[0,1] (V[1,0]), B[1,0] (V[7,0]), B[1,1] (V[8,0])
    # Note: V index is j*p + k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 7
    # j=1, k=1 -> 8
    
    # C elements: C[0,0] (W[0,0]), C[0,1] (W[3,0]), C[1,0] (W[1,0]), C[1,1] (W[4,0])
    # Note: W index is k*n + i.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 3
    # k=0, i=1 -> 1
    # k=1, i=1 -> 4
    
    # Strassen Formulas in terms of U, V, W coefficients for each multiplication term t in 0..6:
    # We define the coefficient vectors u_t, v_t, w_t for each of the 7 Strassen multiplications.
    
    # t=0: M1 = a00 * (b00 + b11)
    # U: {0: 1}
    # V: {0: 1, 8: 1}
    # W: {0: 1, 4: 1}  # C00 += M1, C11 += M1
    
    # t=1: M2 = (a00 + a01) * b11
    # U: {0: 1, 1: 1}
    # V: {8: 1}
    # W: {0: -1, 4: 1} # C00 -= M2, C11 += M2
    
    # t=2: M3 = (a10 + a11) * b00
    # U: {5: 1, 6: 1}
    # V: {0: 1}
    # W: {1: 1, 4: -1} # C10 += M3, C11 -= M3
    
    # t=3: M4 = a11 * (b10 + b00)
    # U: {6: 1}
    # V: {7: 1, 0: 1}
    # W: {0: 1, 4: 1} # C00 += M4, C11 += M4
    
    # t=4: M5 = (a00 + a11) * (b01 + b10)
    # U: {0: 1, 6: 1}
    # V: {1: 1, 7: 1}
    # W: {3: 1, 1: 1} # C01 += M5, C10 += M5
    
    # t=5: M6 = (a01 - a11) * (b11 + b10)
    # U: {1: 1, 6: -1}
    # V: {8: 1, 7: 1}
    # W: {0: 1, 3: -1} # C00 += M6, C01 -= M6
    
    # t=6: M7 = (a10 - a00) * (b00 + b01)
    # U: {5: 1, 0: -1}
    # V: {0: 1, 1: 1}
    # W: {3: -1, 1: 1} # C01 -= M7, C10 += M7
    
    # Let's verify the W signs for Strassen:
    # C00 = M1 + M4 - M2 + M6
    # C01 = M5 + M2 - M4 - M7
    # C10 = M3 + M5 + M7 - M1
    # C11 = M1 + M2 - M3 + M4
    
    # My W assignments above:
    # t=0 (M1): W[0]=1, W[4]=1. Correct (C00, C11).
    # t=1 (M2): W[0]=-1, W[4]=1. Correct (C00, C11).
    # t=2 (M3): W[1]=1, W[4]=-1. Correct (C10, C11).
    # t=3 (M4): W[0]=1, W[4]=1. Correct (C00, C11).
    # t=4 (M5): W[3]=1, W[1]=1. Correct (C01, C10).
    # t=5 (M6): W[0]=1, W[3]=-1. Correct (C00, C01).
    # t=6 (M7): W[3]=-1, W[1]=1. Correct (C01, C10).
    
    # The set of indices involved in the Strassen block:
    # U indices: 0, 1, 5, 6
    # V indices: 0, 1, 7, 8
    # W indices: 0, 1, 3, 4
    
    # These correspond to:
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1) in (k,i) space?
    # W index k*n+i.
    # W[0] -> k=0, i=0. C[0,0].
    # W[1] -> k=0, i=1. C[1,0].
    # W[3] -> k=1, i=0. C[0,1].
    # W[4] -> k=1, i=1. C[1,1].
    
    # So the Strassen block covers C[0,0], C[1,0], C[0,1], C[1,1].
    # This is correct for a 2x2x2 block at A[0:2, 0:2] * B[0:2, 0:2] = C[0:2, 0:2].
    
    # Now, we need to handle the rest of the tensor with schoolbook terms.
    # The total number of schoolbook terms is 105.
    # We have "used" 8 terms for the 2x2x2 block in the schoolbook count.
    # So we need 105 - 8 = 97 additional terms.
    # Total rank = 7 (Strassen) + 97 (Schoolbook) = 104.
    
    # We need to identify which (i,j,k) tuples are NOT in the 2x2x2 block.
    # Block tuples: i in {0,1}, j in {0,1}, k in {0,1}.
    # There are 2*2*2 = 8 such tuples.
    # All other 97 tuples will be handled by schoolbook.
    
    U = [[0] * 104 for _ in range(15)]
    V = [[0] * 104 for _ in range(35)]
    W = [[0] * 104 for _ in range(21)]
    
    # Fill Strassen block (columns 0-6)
    strassen_data = [
        # (u_dict, v_dict, w_dict)
        ({0: 1}, {0: 1, 8: 1}, {0: 1, 4: 1}),
        ({0: 1, 1: 1}, {8: 1}, {0: -1, 4: 1}),
        ({5: 1, 6: 1}, {0: 1}, {1: 1, 4: -1}),
        ({6: 1}, {7: 1, 0: 1}, {0: 1, 4: 1}),
        ({0: 1, 6: 1}, {1: 1, 7: 1}, {3: 1, 1: 1}),
        ({1: 1, 6: -1}, {8: 1, 7: 1}, {0: 1, 3: -1}),
        ({5: 1, 0: -1}, {0: 1, 1: 1}, {3: -1, 1: 1})
    ]
    
    for t in range(7):
        u_d, v_d, w_d = strassen_data[t]
        for idx, val in u_d.items():
            U[idx][t] = val
        for idx, val in v_d.items():
            V[idx][t] = val
        for idx, val in w_d.items():
            W[idx][t] = val
            
    # Fill Schoolbook terms (columns 7-103)
    col = 7
    for i in range(3):
        for j in range(5):
            for k in range(7):
                # Skip if in Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verify col is 104
    # if col != 104: raise Exception(f"Col mismatch: {col}")
    
    return {"U": U, "V": V, "W": W}
