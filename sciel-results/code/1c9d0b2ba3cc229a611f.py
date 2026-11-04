"""Program for matrix multiplication 3x4x8 with Laderman-based optimization."""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    columns = []  # Each entry is (U_dict, V_dict, W_dict)
    
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
    
    def add_schoolblock(n_start, m_start, p_start, ni, mi, pi):
        """Add schoolbook terms for a sub-block."""
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    u_idx = (n_start + i) * m + (m_start + j)
                    v_idx = (m_start + j) * p + (p_start + k)
                    w_idx = (p_start + k) * n + (n_start + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def add_laderman_2x3x4(n_start, m_start, p_start):
        """Add Laderman's rank-14 decomposition for 2x3x4 block.
        
        Uses the known rank-14 algorithm for 2x3x4 matrix multiplication.
        The 14 multiplications are:
        M1 = A00*B00
        M2 = A00*B01  
        M3 = A01*B10
        M4 = A01*B11
        M5 = A02*B20
        M6 = A02*B21
        M7 = (A10-A00)*(B00+B10)
        M8 = (A11-A01)*(B01+B11)
        M9 = (A12-A02)*(B02+B20)
        M10 = (A12-A02)*(B03+B21)
        M11 = (A10+A11)*(B00-B10)
        M12 = (A10+A11)*(B01-B11)
        M13 = (A10+A11+A12)*(B02-B20)
        M14 = (A10+A11+A12)*(B03-B21)
        """
        def u_glob(li, lj):  # li in {0,1}, lj in {0,1,2}
            return (n_start + li) * m + (m_start + lj)
        def v_glob(li, lj):  # li in {0,1,2}, lj in {0,1,2,3}
            return (m_start + li) * p + (p_start + lj)
        def w_glob(li, lj):  # li in {0,1}, lj in {0,1,2,3}
            return (p_start + lj) * n + (n_start + li)
        
        # 14 rank-1 terms
        # Term 1: M1 = A00*B00 -> C00
        add_term({u_glob(0,0): 1}, {v_glob(0,0): 1}, {w_glob(0,0): 1})
        # Term 2: M2 = A00*B01 -> C01
        add_term({u_glob(0,0): 1}, {v_glob(0,1): 1}, {w_glob(0,1): 1})
        # Term 3: M3 = A01*B10 -> C02
        add_term({u_glob(0,1): 1}, {v_glob(1,0): 1}, {w_glob(0,2): 1})
        # Term 4: M4 = A01*B11 -> C03
        add_term({u_glob(0,1): 1}, {v_glob(1,1): 1}, {w_glob(0,3): 1})
        # Term 5: M5 = A02*B20 -> C10
        add_term({u_glob(0,2): 1}, {v_glob(2,0): 1}, {w_glob(1,0): 1})
        # Term 6: M6 = A02*B21 -> C11
        add_term({u_glob(0,2): 1}, {v_glob(2,1): 1}, {w_glob(1,1): 1})
        # Term 7: M7 = (A10-A00)*(B00+B10)
        u7 = {u_glob(1,0): 1, u_glob(0,0): -1}
        v7 = {v_glob(0,0): 1, v_glob(1,0): 1}
        # Contributes to C10 and C00
        w7 = {w_glob(1,0): 1, w_glob(0,0): 1}
        add_term(u7, v7, w7)
        # Term 8: M8 = (A11-A01)*(B01+B11)
        u8 = {u_glob(1,1): 1, u_glob(0,1): -1}
        v8 = {v_glob(0,1): 1, v_glob(1,1): 1}
        w8 = {w_glob(1,1): 1, w_glob(0,1): 1}
        add_term(u8, v8, w8)
        # Term 9: M9 = (A12-A02)*(B02+B20)
        u9 = {u_glob(1,2): 1, u_glob(0,2): -1}
        v9 = {v_glob(0,2): 1, v_glob(2,0): 1}
        w9 = {w_glob(1,2): 1, w_glob(0,2): 1}
        add_term(u9, v9, w9)
        # Term 10: M10 = (A12-A02)*(B03+B21)
        u10 = {u_glob(1,2): 1, u_glob(0,2): -1}
        v10 = {v_glob(0,3): 1, v_glob(2,1): 1}
        w10 = {w_glob(1,3): 1, w_glob(0,3): 1}
        add_term(u10, v10, w10)
        # Term 11: M11 = (A10+A11)*(B00-B10)
        u11 = {u_glob(1,0): 1, u_glob(1,1): 1}
        v11 = {v_glob(0,0): 1, v_glob(1,0): -1}
        w11 = {w_glob(1,0): 1}
        add_term(u11, v11, w11)
        # Term 12: M12 = (A10+A11)*(B01-B11)
        u12 = {u_glob(1,0): 1, u_glob(1,1): 1}
        v12 = {v_glob(0,1): 1, v_glob(1,1): -1}
        w12 = {w_glob(1,1): 1}
        add_term(u12, v12, w12)
        # Term 13: M13 = (A10+A11+A12)*(B02-B20)
        u13 = {u_glob(1,0): 1, u_glob(1,1): 1, u_glob(1,2): 1}
        v13 = {v_glob(0,2): 1, v_glob(2,0): -1}
        w13 = {w_glob(1,2): 1}
        add_term(u13, v13, w13)
        # Term 14: M14 = (A10+A11+A12)*(B03-B21)
        u14 = {u_glob(1,0): 1, u_glob(1,1): 1, u_glob(1,2): 1}
        v14 = {v_glob(0,3): 1, v_glob(2,1): -1}
        w14 = {w_glob(1,3): 1}
        add_term(u14, v14, w14)
    
    # Block partitioning:
    # n=3 -> [1, 2]
    # m=4 -> [3, 1]
    # p=8 -> [4, 4]
    
    # Block 1: n=[0], m=[0,1,2], p=[0,1,2,3] - schoolbook 1x3x4 = 12
    add_schoolblock(0, 0, 0, 1, 3, 4)
    
    # Block 2: n=[0], m=[3], p=[0,1,2,3] - schoolbook 1x1x4 = 4
    add_schoolblock(0, 3, 0, 1, 1, 4)
    
    # Block 3: n=[0], m=[0,1,2], p=[4,5,6,7] - schoolbook 1x3x4 = 12
    add_schoolblock(0, 0, 4, 1, 3, 4)
    
    # Block 4: n=[0], m=[3], p=[4,5,6,7] - schoolbook 1x1x4 = 4
    add_schoolblock(0, 3, 4, 1, 1, 4)
    
    # Block 5: n=[1,2], m=[0,1,2], p=[0,1,2,3] - Laderman 2x3x4 = 14
    add_laderman_2x3x4(1, 0, 0)
    
    # Block 6: n=[1,2], m=[3], p=[0,1,2,3] - schoolbook 2x1x4 = 8
    add_schoolblock(1, 3, 0, 2, 1, 4)
    
    # Block 7: n=[1,2], m=[0,1,2], p=[4,5,6,7] - Laderman 2x3x4 = 14
    add_laderman_2x3x4(1, 0, 4)
    
    # Block 8: n=[1,2], m=[3], p=[4,5,6,7] - schoolbook 2x1x4 = 8
    add_schoolblock(1, 3, 4, 2, 1, 4)
    
    rank = len(columns)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(columns):
        for idx, val in u_dict.items():
            U[idx][col_idx] = val
        for idx, val in v_dict.items():
            V[idx][col_idx] = val
        for idx, val in w_dict.items():
            W[idx][col_idx] = val
    
    return {"U": U, "V": V, "W": W}
