import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 4
    
    # We use a block-Strassen decomposition.
    # The matrix A is 2x4, B is 4x4.
    # We split A into blocks A1 (2x2), A2 (2x2) along the column dimension (j).
    # We split B into blocks B1 (2x2), B2 (2x2) along the row dimension (j) and column dimension (k).
    # Specifically:
    # A = [A1 A2] where A1 uses j in {0,1}, A2 uses j in {2,3}
    # B = [B11 B12; B21 B22] where B11 uses j in {0,1}, k in {0,1}
    #                           B12 uses j in {0,1}, k in {2,3}
    #                           B21 uses j in {2,3}, k in {0,1}
    #                           B22 uses j in {2,3}, k in {2,3}
    #
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    #
    # We apply Strassen's algorithm to each of the 4 multiplications of 2x2 matrices.
    # Each Strassen multiplication has rank 7.
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    
    # Local Strassen factors for 2x2x2
    # Indices for local 2x2 matrices:
    # U_local[i*2+j], V_local[j*2+k], W_local[k*2+i]
    # where i,j,k in {0,1}
    
    # The 7 Strassen terms for 2x2 matrix multiplication:
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22) -> U: {0:1, 3:1}, V: {0:1, 3:1}, W: {0:1, 3:1}
    # M2 = a22(b21-b11)       -> U: {3:1}, V: {2:1, 0:-1}, W: {1:1, 3:-1}  (Note: V index 2 is b21, 0 is b11)
    #       Wait, let's re-verify the standard Strassen terms carefully.
    
    # Standard Strassen:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Output C:
    # C11 = M1 + M4 - M2 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M7
    
    # Let's map indices:
    # A: 00=0, 01=1, 10=2, 11=3
    # B: 00=0, 01=1, 10=2, 11=3
    # C: 00=0, 01=1, 10=2, 11=3
    
    # M1:
    # U: A00(0), A11(3) -> {0:1, 3:1}
    # V: B00(0), B11(3) -> {0:1, 3:1}
    # W: C00(0), C11(3) -> {0:1, 3:1}
    
    # M2:
    # U: A10(2), A11(3) -> {2:1, 3:1}
    # V: B00(0) -> {0:1}
    # W: C10(2), C00(0) -> {2:1, 0:1}
    # Wait, C10 is index 2, C00 is index 0.
    # Formula: C10 += M2, C00 -= M2. So W: {2:1, 0:-1}
    
    # M3:
    # U: A00(0) -> {0:1}
    # V: B01(1), B11(3) -> {1:1, 3:-1}
    # W: C01(1), C11(3) -> {1:1, 3:-1}
    # Formula: C01 += M3, C11 -= M3. So W: {1:1, 3:-1}
    
    # M4:
    # U: A11(3) -> {3:1}
    # V: B10(2), B00(0) -> {2:1, 0:-1}
    # W: C00(0), C10(2) -> {0:1, 2:1}
    # Formula: C00 += M4, C10 += M4. So W: {0:1, 2:1}
    
    # M5:
    # U: A00(0), A01(1) -> {0:1, 1:1}
    # V: B11(3) -> {3:1}
    # W: C01(1), C11(3) -> {1:1, 3:1}
    # Formula: C01 += M5, C11 += M5. So W: {1:1, 3:1}
    
    # M6:
    # U: A10(2), A00(0) -> {2:1, 0:-1}
    # V: B00(0), B01(1) -> {0:1, 1:1}
    # W: C00(0) -> {0:1}
    # Formula: C00 += M6. So W: {0:1}
    
    # M7:
    # U: A01(1), A11(3) -> {1:1, 3:-1}
    # V: B10(2), B11(3) -> {2:1, 3:1}
    # W: C11(3) -> {3:-1}
    # Formula: C11 -= M7. So W: {3:-1}
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),      # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 0: -1}),            # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: -1}),           # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 3: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {0: 1}),            # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {3: -1}),           # M7
    ]
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    col = 0
    
    # We have 4 blocks of 2x2 multiplications:
    # Block 0: A1 * B11 (j in {0,1}, k in {0,1})
    # Block 1: A1 * B12 (j in {0,1}, k in {2,3})
    # Block 2: A2 * B21 (j in {2,3}, k in {0,1})
    # Block 3: A2 * B22 (j in {2,3}, k in {2,3})
    
    for block in range(4):
        # Determine offsets for j and k
        # Block 0: j_offset=0, k_offset=0
        # Block 1: j_offset=0, k_offset=2
        # Block 2: j_offset=2, k_offset=0
        # Block 3: j_offset=2, k_offset=2
        
        j_offset = (block // 2) * 2
        k_offset = (block % 2) * 2
        
        for U_loc, V_loc, W_loc in strassen_terms:
            # Fill U
            for u_idx, val in U_loc.items():
                i = u_idx // 2
                j_loc = u_idx % 2
                j = j_offset + j_loc
                global_u_idx = i * m + j
                U[global_u_idx][col] += val
            
            # Fill V
            for v_idx, val in V_loc.items():
                j_loc = v_idx // 2
                k_loc = v_idx % 2
                j = j_offset + j_loc
                k = k_offset + k_loc
                global_v_idx = j * p + k
                V[global_v_idx][col] += val
            
            # Fill W
            for w_idx, val in W_loc.items():
                k_loc = w_idx // 2
                i = w_idx % 2
                k = k_offset + k_loc
                global_w_idx = k * n + i
                W[global_w_idx][col] += val
            
            col += 1
            
            if time.time() - start_time > 100:
                break
        if time.time() - start_time > 100:
            break
            
    return {"U": U, "V": V, "W": W}
