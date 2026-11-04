import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # Define 2x2 Strassen factors (rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indexing: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # Term 2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # Term 3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # Term 4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00, C10
    
    # Term 5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01
    
    # Term 6: (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # Term 7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00

    # Construct 4x4 factors (rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r in range(7):
        for s in range(7):
            idx = r * 7 + s
            # U4
            for br in range(2):
                for bc in range(2):
                    coeff = U2[br * 2 + bc, r]
                    if coeff != 0:
                        # Block (br, bc) in 4x4 matrix
                        # Rows: br*2, br*2+1
                        # Cols: bc*2, bc*2+1
                        for ir in range(2):
                            for ic in range(2):
                                # Inner index
                                inner_idx = ir * 2 + ic
                                val = coeff * U2[inner_idx, s]
                                # Global index
                                global_row = br * 2 + ir
                                global_col = bc * 2 + ic
                                U4[global_row * 4 + global_col, idx] = val
                                
            # V4
            for br in range(2):
                for bc in range(2):
                    coeff = V2[br * 2 + bc, r]
                    if coeff != 0:
                        for ir in range(2):
                            for ic in range(2):
                                inner_idx = ir * 2 + ic
                                val = coeff * V2[inner_idx, s]
                                global_row = br * 2 + ir
                                global_col = bc * 2 + ic
                                V4[global_row * 4 + global_col, idx] = val
                                
            # W4
            for br in range(2):
                for bc in range(2):
                    coeff = W2[br * 2 + bc, r]
                    if coeff != 0:
                        for ir in range(2):
                            for ic in range(2):
                                inner_idx = ir * 2 + ic
                                val = coeff * W2[inner_idx, s]
                                global_row = br * 2 + ir
                                global_col = bc * 2 + ic
                                W4[global_row * 4 + global_col, idx] = val

    # Convert to list of lists as required
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
