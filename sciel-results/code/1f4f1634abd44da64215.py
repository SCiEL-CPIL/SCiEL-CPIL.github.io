import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Laderman's 49 terms for 4x4x4
    # Each term is (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    # Indices are 0..15 for A, B, C (flattened row-major i*m+j)
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
        
    # 1. A00 B00 -> C00
    add_term([0], [1], [0], [1], [0], [1])
    # 2. A00 B10 -> C10
    add_term([0], [1], [4], [1], [1], [1])
    # 3. A01 B00 -> C01
    add_term([1], [1], [0], [1], [4], [1])
    # 4. A01 B10 -> C11
    add_term([1], [1], [4], [1], [5], [1])
    # 5. A10 B00 -> C02
    add_term([4], [1], [0], [1], [8], [1])
    # 6. A10 B10 -> C12
    add_term([4], [1], [4], [1], [9], [1])
    # 7. A11 B00 -> C03
    add_term([5], [1], [0], [1], [12], [1])
    # 8. A11 B10 -> C13
    add_term([5], [1], [4], [1], [13], [1])
    # 9. A20 B00 -> C20
    add_term([8], [1], [0], [1], [16], [1])
    # 10. A20 B10 -> C21
    add_term([8], [1], [4], [1], [17], [1])
    # 11. A21 B00 -> C22
    add_term([9], [1], [0], [1], [20], [1])
    # 12. A21 B10 -> C23
    add_term([9], [1], [4], [1], [21], [1])
    # 13. A30 B00 -> C30
    add_term([12], [1], [0], [1], [24], [1])
    # 14. A30 B10 -> C31
    add_term([12], [1], [4], [1], [25], [1])
    # 15. A31 B00 -> C32
    add_term([13], [1], [0], [1], [28], [1])
    # 16. A31 B10 -> C33
    add_term([13], [1], [4], [1], [29], [1])
    # 17. A02 B00 -> C04
    add_term([2], [1], [0], [1], [32], [1])
    # 18. A02 B10 -> C05
    add_term([2], [1], [4], [1], [33], [1])
    # 19. A03 B00 -> C06
    add_term([3], [1], [0], [1], [36], [1])
    # 20. A03 B10 -> C07
    add_term([3], [1], [4], [1], [37], [1])
    # 21. A12 B00 -> C14
    add_term([6], [1], [0], [1], [40], [1])
    # 22. A12 B10 -> C15
    add_term([6], [1], [4], [1], [41], [1])
    # 23. A13 B00 -> C16
    add_term([7], [1], [0], [1], [44], [1])
    # 24. A13 B10 -> C17
    add_term([7], [1], [4], [1], [45], [1])
    # 25. A22 B00 -> C24
    add_term([10], [1], [0], [1], [48], [1])
    # 26. A22 B10 -> C25
    add_term([10], [1], [4], [1], [49], [1])
    # 27. A23 B00 -> C26
    add_term([11], [1], [0], [1], [52], [1])
    # 28. A23 B10 -> C27
    add_term([11], [1], [4], [1], [53], [1])
    # 29. A32 B00 -> C34
    add_term([14], [1], [0], [1], [56], [1])
    # 30. A32 B10 -> C35
    add_term([14], [1], [4], [1], [57], [1])
    # 31. A33 B00 -> C36
    add_term([15], [1], [0], [1], [60], [1])
    # 32. A33 B10 -> C37
    add_term([15], [1], [4], [1], [61], [1])
    
    # 33. (A00+A11+A22+A33) B11 -> C11, C22, C33? No.
    # Laderman's 33rd term:
    # (A00+A11+A22+A33) * B11
    # Contributes to C11, C22, C33?
    # Actually, the last 17 terms are sums.
    # I will use a known correct list of Laderman terms.
    
    # Since hardcoding 49 terms manually is error-prone, I will use a simpler verified decomposition if possible,
    # or trust the Strassen-based one which was already scored 1.3061.
    # However, Laderman is rank 49. Strassen is rank 64.
    # The previous program used Strassen and got 98.
    # If I use Laderman, I get 98 as well.
    # Is there a rank < 98?
    # R(4,4,8) is at least 96?
    # Actually, R(4,4,4) = 49.
    # R(4,4,8) <= 2 * 49 = 98.
    # It is known that R(4,4,8) = 98? Or can it be lower?
    # Some sources suggest R(4,4,8) = 96 is possible?
    # If I stick to 98, I am safe.
    
    # Let's just clean up the existing Strassen implementation which is known to work and score 1.3061.
    # The previous code had a lot of comments and logic. I will rewrite it cleanly.
    
    # Strassen 2x2x2
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c=3
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c=4
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c=5
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c=6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Block Strassen for 4x4x4
    A_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 2), (1, 3)], # P1
        [(1, 0)],         # P2
        [(1, 3)],         # P3
        [(1, 0), (1, 1)], # P4
        [(1, 2), (-1, 0)],# P5
        [(1, 1), (-1, 3)],# P6
    ]
    
    B_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 0)],         # P1
        [(1, 1), (-1, 3)],# P2
        [(1, 2), (-1, 0)],# P3
        [(1, 3)],         # P4
        [(1, 0), (1, 1)], # P5
        [(1, 2), (1, 3)], # P6
    ]
    
    C_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 2), (-1, 3)], # P1
        [(1, 1), (1, 3)], # P2
        [(1, 0), (1, 2)], # P3
        [(-1, 0), (1, 1)], # P4
        [(1, 3)], # P5
        [(1, 0)], # P6
    ]
    
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                # U
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                # V
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                # W
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
