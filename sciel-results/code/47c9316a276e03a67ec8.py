import time
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 construction for 2x2 by 2x11
    # 5 blocks of 2x2 using Strassen (5 * 7 = 35)
    # 1 block of 2x1 using schoolbook (4)
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2 blocks
    # A indices in U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices in V: j*p + k
    # C indices in W: k*n + i
    
    # Strassen terms for a block at column offset k_off
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    # M2: (A10+A11)B01 -> C10 + C11 (Wait, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B12
    # M3 = A11 B21
    # M4 = A22 B22
    # M5 = (A11+A12)(B22-B21)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12+A22)(B21-B11)
    # C11 = M1 + M4 - M7 + M5
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M6
    # )
    
    # Let's map 0-indexed:
    # A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    # B00, B01, B10, B11 in block k
    
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            U[idx][col] += val
        for idx, val in v_vals.items():
            V[idx][col] += val
        for idx, val in w_vals.items():
            W[idx][col] += val
        col += 1

    for block in range(5):
        k_off = block * 2
        # B indices
        b00 = 0 * p + k_off      # 11*0 + k
        b01 = 0 * p + k_off + 1  # 11*0 + k+1
        b10 = 1 * p + k_off      # 11*1 + k
        b11 = 1 * p + k_off + 1  # 11*1 + k+1
        
        # C indices
        c00 = k_off * n + 0      # 2*k + 0
        c01 = (k_off + 1) * n + 0 # 2*(k+1) + 0
        c10 = k_off * n + 1      # 2*k + 1
        c11 = (k_off + 1) * n + 1 # 2*(k+1) + 1
        
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        add_term({0: 1, 3: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        
        # M2: (A10+A11)B01 -> C10 + C11 ? No, C21 = M2 + M4. C22 has M2.
        # Standard: C21 = M2 + M4. C22 = M1 - M3 + M2 + M6.
        # So M2 contributes to C10 (index 1,0 in local block? No, C21 is row 1, col 0 of block?
        # Wait. C is 2x2.
        # C00, C01
        # C10, C11
        # My indices:
        # c00: row 0, col k
        # c01: row 0, col k+1
        # c10: row 1, col k
        # c11: row 1, col k+1
        
        # M2: (A10+A11) B01
        # A10 is U[2], A11 is U[3].
        # B01 is V[b01].
        # Contributes to C10 (row 1, col k) and C11 (row 1, col k+1)?
        # Formula: C21 (local) = M2 + M4. C22 (local) = ... + M2 + ...
        # Local C21 corresponds to global c10.
        # Local C22 corresponds to global c11.
        add_term({2: 1, 3: 1}, {b01: 1}, {c10: 1, c11: 1})
        
        # M3: A00 B10 -> C01 + C11?
        # Formula: C12 (local) = M3 + M7. C22 (local) = ... - M3 ...
        # Local C12 is global c01.
        # Local C22 is global c11.
        add_term({0: 1}, {b10: 1}, {c01: 1, c11: -1})
        
        # M4: A11 B11 -> C00 + C10
        # Formula: C11 (local) = ... + M4 ... (C00)
        # C21 (local) = M2 + M4 (C10)
        add_term({3: 1}, {b11: 1}, {c00: 1, c10: 1})
        
        # M5: (A00+A01)(B11-B10) -> C00 - C11?
        # Formula: C11 (local) = ... + M5 (C00)
        # C22 (local) has no M5?
        # Wait, C11 = M1 + M4 - M7 + M5.
        # C22 = M1 - M3 + M2 + M6.
        # So M5 only contributes to C00?
        # Let's re-verify Strassen.
        # C11 = M1 + M4 - M7 + M5
        # C12 = M3 + M7
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 + M6
        # Yes, M5 is only in C11.
        add_term({0: 1, 1: 1}, {b11: 1, b10: -1}, {c00: 1})
        
        # M6: (A10-A00)(B00+B01) -> C11
        # Formula: C22 = ... + M6
        add_term({2: 1, 0: -1}, {b00: 1, b01: 1}, {c11: 1})
        
        # M7: (A01+A11)(B10-B00) -> C00 - C12?
        # Formula: C11 = ... - M7
        # C12 = M3 + M7
        add_term({1: 1, 3: 1}, {b10: 1, b00: -1}, {c00: -1, c01: 1})
        
    # Last column (index 10)
    k = 10
    b0 = 0 * p + k
    b1 = 1 * p + k
    c0 = k * n + 0
    c1 = k * n + 1
    
    # C0 = A00 B0 + A01 B1
    add_term({0: 1}, {b0: 1}, {c0: 1})
    add_term({1: 1}, {b1: 1}, {c0: 1})
    
    # C1 = A10 B0 + A11 B1
    add_term({2: 1}, {b0: 1}, {c1: 1})
    add_term({3: 1}, {b1: 1}, {c1: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
