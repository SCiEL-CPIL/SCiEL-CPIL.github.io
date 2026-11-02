import time

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x10 matrix.
    
    The rank is 35, which is the optimal rank for this problem (5 * rank(2x2x2) = 5 * 7).
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen decomposition for 2x2 matrix multiplication.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The 7 terms of Strassen's algorithm:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Vectors for A (4 elements: A00, A01, A10, A11)
    a_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Vectors for B (4 elements: B00, B01, B10, B11)
    b_vecs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Vectors for C (4 elements: C00, C01, C10, C11)
    # Note: The task specifies W[k*n+i, r].
    # For a 2x2 block, the indices in W are:
    # C00 -> 0
    # C01 -> 1
    # C10 -> 2
    # C11 -> 3
    c_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Verify Strassen coefficients
    # C00: M1(1) + M4(1) - M5(1) + M7(1) -> 1+1-1+1 = 2? 
    # Wait.
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # M1 = (A00+A11)(B00+B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M7 = (A01-A11)(B10+B11)
    
    # Let's check C00 coefficient for M1: 1
    # M4: 1
    # M5: -1
    # M7: 1
    # My c_vecs[0] for C00 is 1. Correct.
    # c_vecs[3] for C00 is 1. Correct.
    # c_vecs[4] for C00 is -1. Correct.
    # c_vecs[6] for C00 is 1. Correct.
    
    # C01 = M3 + M5
    # M3: 1
    # M5: 1
    # c_vecs[2] for C01 (index 1) is 1. Correct.
    # c_vecs[4] for C01 (index 1) is 1. Correct.
    
    # C10 = M2 + M4
    # M2: 1
    # M4: 1
    # c_vecs[1] for C10 (index 2) is 1. Correct.
    # c_vecs[3] for C10 (index 2) is 1. Correct.
    
    # C11 = M1 - M2 + M3 + M6
    # M1: 1
    # M2: -1
    # M3: 1
    # M6: 1
    # c_vecs[0] for C11 (index 3) is 1. Correct.
    # c_vecs[1] for C11 (index 3) is 0. Wait.
    # M2 is involved in C11 with coeff -1.
    # My c_vecs[1] is [0, 0, 1, 0]. So C11 coeff is 0.
    # Error in c_vecs[1].
    # M2 = (A10+A11)B00.
    # C11 = M1 - M2 + M3 + M6.
    # So c_vecs[1] should have -1 at index 3.
    
    # Let's re-verify all c_vecs.
    
    # M1: C00(1), C01(0), C10(0), C11(1) -> [1, 0, 0, 1]
    # M2: C00(0), C01(0), C10(1), C11(-1) -> [0, 0, 1, -1]
    # M3: C00(0), C01(1), C10(0), C11(1) -> [0, 1, 0, 1]
    # M4: C00(1), C01(0), C10(1), C11(0) -> [1, 0, 1, 0]
    # M5: C00(-1), C01(1), C10(0), C11(0) -> [-1, 1, 0, 0]
    # M6: C00(0), C01(0), C10(0), C11(1) -> [0, 0, 0, 1]
    # M7: C00(1), C01(0), C10(0), C11(0) -> [1, 0, 0, 0]
    
    # Let's re-verify M6 and M7 in standard Strassen.
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Check C11:
    # M1: 1
    # M2: -1
    # M3: 1
    # M6: 1
    # M4: 0
    # M5: 0
    # M7: 0
    
    # So c_vecs should be:
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 1]
    # M4: [1, 0, 1, 0]
    # M5: [-1, 1, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [1, 0, 0, 0]
    
    # The previous c_vecs in the thought process were slightly off or I misremembered.
    # I will use the corrected c_vecs.

    corrected_c_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col = 0
    for k in range(5):
        for idx in range(7):
            a_vec = a_vecs[idx]
            b_vec = b_vecs[idx]
            c_vec = corrected_c_vecs[idx]
            
            # Fill U
            for i in range(4):
                U[i][col] = a_vec[i]
                
            # Fill V
            # B indices for block k:
            # B00 -> 2k
            # B01 -> 2k+1
            # B10 -> 10 + 2k
            # B11 -> 10 + 2k + 1
            V[2*k][col] = b_vec[0]
            V[2*k+1][col] = b_vec[1]
            V[10+2*k][col] = b_vec[2]
            V[10+2*k+1][col] = b_vec[3]
            
            # Fill W
            # C indices for block k:
            # C00 -> 4k
            # C01 -> 4k+1
            # C10 -> 4k+2
            # C11 -> 4k+3
            W[4*k][col] = c_vec[0]
            W[4*k+1][col] = c_vec[1]
            W[4*k+2][col] = c_vec[2]
            W[4*k+3][col] = c_vec[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

