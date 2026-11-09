import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    # Base Strassen 2x2 by 2x2
    # U, V, W are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Now build the full decomposition
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Outer Strassen: 7 products
    # Each product is (2x2) by (2x4)
    # We need to define the linear combinations for the outer Strassen
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We iterate through the 7 outer terms
    for outer_r in range(7):
        # For each outer term, we have 2 inner terms (for the 2x4 split)
        for inner_r in range(2):
            # For each inner term, we have 7 base Strassen terms
            for base_r in range(7):
                # Determine which A block and B block are involved
                # This depends on outer_r and inner_r
                
                # We need to map outer_r, inner_r, base_r to specific A and B entries
                # and add to U, V, W
                
                # This logic is complex to hardcode.
                # It's better to generate the vectors programmatically.
                
                pass
                
    # The above loop is a placeholder.
    # We need to actually fill U, V, W.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
