import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    timeout = 140.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    np_ = n * p  # 10
    
    R = 28
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((np_, R))
    
    col = 0
    
    # Helper to add a Strassen term for 2x2x2
    # a_indices: [a00, a01, a10, a11]
    # b_indices: [b00, b01, b10, b11]
    # c_indices: [c00, c01, c10, c11]
    def add_strassen_222(a_idx, b_idx, c_idx):
        nonlocal col
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        
        # M1 = (A00+A11)*(B00+B11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M2 = (A10+A11)*B00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # M3 = A00*(B01-B11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        col += 1
        
        # M4 = A11*(B10-B00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1

    # Block 1: j in {0,1}, k in {0,1}
    # A indices:
    # A00: i=0, j=0 -> 0
    # A01: i=0, j=1 -> 1
    # A10: i=1, j=0 -> 3
    # A11: i=1, j=1 -> 4
    a_idx_1 = [0, 1, 3, 4]
    
    # B indices:
    # B00: j=0, k=0 -> 0
    # B01: j=0, k=1 -> 1
    # B10: j=1, k=0 -> 5
    # B11: j=1, k=1 -> 6
    b_idx_1 = [0, 1, 5, 6]
    
    # C indices:
    # C00: i=0, k=0 -> 0
    # C01: i=0, k=1 -> 1
    # C10: i=1, k=0 -> 2
    # C11: i=1, k=1 -> 3
    c_idx_1 = [0, 1, 2, 3]
    
    add_strassen_222(a_idx_1, b_idx_1, c_idx_1)
    
    # Block 2: j in {0,1}, k in {2,3}
    # A indices same
    a_idx_2 = [0, 1, 3, 4]
    
    # B indices:
    # B00': j=0, k=2 -> 2
    # B01': j=0, k=3 -> 3
    # B10': j=1, k=2 -> 7
    # B11': j=1, k=3 -> 8
    b_idx_2 = [2, 3, 7, 8]
    
    # C indices:
    # C00': i=0, k=2 -> 2
    # C01': i=0, k=3 -> 3
    # C10': i=1, k=2 -> 4
    # C11': i=1, k=3 -> 5
    c_idx_2 = [2, 3, 4, 5]
    
    add_strassen_222(a_idx_2, b_idx_2, c_idx_2)
    
    # Block 3: j in {0,1}, k=4
    # Schoolbook 2x2x1
    # Terms:
    # 1. A00 * B04 -> C04
    # 2. A01 * B14 -> C04
    # 3. A10 * B04 -> C14
    # 4. A11 * B14 -> C14
    
    # 1.
    U[0, col] = 1
    V[4, col] = 1 # B04: j=0, k=4 -> 4
    W[4, col] = 1 # C04: i=0, k=4 -> 4
    col += 1
    
    # 2.
    U[1, col] = 1
    V[9, col] = 1 # B14: j=1, k=4 -> 9
    W[4, col] = 1 # C04: i=0, k=4 -> 4
    col += 1
    
    # 3.
    U[3, col] = 1
    V[4, col] = 1 # B04
    W[6, col] = 1 # C14: i=1, k=4 -> 6
    col += 1
    
    # 4.
    U[4, col] = 1
    V[9, col] = 1 # B14
    W[6, col] = 1 # C14
    col += 1
    
    # Block 4: j=2
    # Schoolbook 2x1x5
    # A indices:
    # A02: i=0, j=2 -> 2
    # A12: i=1, j=2 -> 5
    # B indices: j=2, k=0..4 -> 10..14
    # C indices:
    # C0k: i=0, k=0..4 -> 0..4
    # C1k: i=1, k=0..4 -> 2..6
    
    for k in range(5):
        # i=0
        U[2, col] = 1
        V[10+k, col] = 1
        W[k, col] = 1
        col += 1
        
        # i=1
        U[5, col] = 1
        V[10+k, col] = 1
        W[2+k, col] = 1
        col += 1
        
    assert col == 28, f"Expected 28, got {col}"
    
    # Verify
    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_calc[a, b, c] - 1.0) > 1e-9:
                    print(f"Error at ({a},{b},{c}): {T_calc[a,b,c]}")
    
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
                
    for a in range(nm):
        for b in range(mp):
            for c in range(np_):
                if (a, b, c) not in target_set:
                    if abs(T_calc[a, b, c]) > 1e-9:
                        print(f"Non-zero at ({a},{b},{c}): {T_calc[a,b,c]}")
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
