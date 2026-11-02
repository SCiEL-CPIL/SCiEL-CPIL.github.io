import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    R = 39
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, col):
        # u_indices: dict {a_index: coeff}
        # v_indices: dict {b_index: coeff}
        # w_indices: dict {c_index: coeff}
        for a, c in u_indices.items():
            U[a, col] = c
        for b, c in v_indices.items():
            V[b, col] = c
        for c_idx, c in w_indices.items():
            W[c_idx, col] = c

    col = 0
    
    # Strassen block helper
    def add_strassen_block(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11, start_col):
        # M1 = (a00+a11)(b00+b11) -> c00(+), c11(+)
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1}, start_col); start_col += 1
        # M2 = (a10+a11)b00 -> c10(+)
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1}, start_col); start_col += 1
        # M3 = a00(b01-b11) -> c01(+)
        add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1}, start_col); start_col += 1
        # M4 = a11(b10-b00) -> c00(+), c10(+)
        add_term({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1}, start_col); start_col += 1
        # M5 = (a00+a01)b11 -> c00(-), c01(+)
        add_term({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1}, start_col); start_col += 1
        # M6 = (a10-a00)(b00+b01) -> c11(+)
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1}, start_col); start_col += 1
        # M7 = (a01-a11)(b10+b11) -> c00(+), c11(+)
        add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1, c11: 1}, start_col); start_col += 1
        return start_col

    # Block 1: A_L B_T1
    # A_L: A00(0), A01(1), A10(3), A11(4)
    # B_T1: B00(0), B01(1), B10(7), B11(8)
    # C: C00(0), C01(2), C10(1), C11(3)
    col = add_strassen_block(0, 1, 3, 4, 0, 1, 7, 8, 0, 2, 1, 3, col)
    
    # Block 2: A_L B_T2
    # B_T2: B02(2), B03(3), B12(9), B13(10)
    # C: C02(4), C03(6), C12(5), C13(7)
    col = add_strassen_block(0, 1, 3, 4, 2, 3, 9, 10, 4, 6, 5, 7, col)
    
    # Block 3: A_L B_T3
    # B_T3: B04(4), B05(5), B14(11), B15(12)
    # C: C04(8), C05(10), C14(9), C15(11)
    col = add_strassen_block(0, 1, 3, 4, 4, 5, 11, 12, 8, 10, 9, 11, col)
    
    # Block 4: A_L B_T4 (Schoolbook 2x2x1)
    # B_T4: B06(6), B16(13)
    # C: C06(12), C16(13)
    # Term 1: A00 * B06 -> C06
    add_term({0: 1}, {6: 1}, {12: 1}, col); col += 1
    # Term 2: A01 * B16 -> C06
    add_term({1: 1}, {13: 1}, {12: 1}, col); col += 1
    # Term 3: A10 * B06 -> C16
    add_term({3: 1}, {6: 1}, {13: 1}, col); col += 1
    # Term 4: A11 * B16 -> C16
    add_term({4: 1}, {13: 1}, {13: 1}, col); col += 1
    
    # Block 5: A_R B_B (Schoolbook 2x1x7)
    # A_R: A02(2), A12(5)
    # B_B: B20(14), B21(15), ..., B26(20)
    # C: C0k(2k), C1k(2k+1)
    for k in range(7):
        b_idx = 14 + k
        c0_idx = 2 * k
        c1_idx = 2 * k + 1
        # A02 * B2k -> C0k
        add_term({2: 1}, {b_idx: 1}, {c0_idx: 1}, col); col += 1
        # A12 * B2k -> C1k
        add_term({5: 1}, {b_idx: 1}, {c1_idx: 1}, col); col += 1
        
    if col != R:
        raise ValueError(f"Column mismatch: {col} vs {R}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
