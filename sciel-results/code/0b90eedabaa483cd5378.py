"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_schoolbook}
    
    # Structured decomposition using Strassen on 2x2 x 2x2 subblocks
    # Partition: A (2x3) = [A_2x2 | A_2x1], B (3x9) = [B_2x9; B_1x9]
    # C = A_2x2 * B_2x9 + A_2x1 * B_1x9
    # For A_2x2 * B_2x9: split 9 columns into groups of 2 (four groups) + 1 remaining
    #   Each 2x2 x 2x2 block: rank 7 (Strassen)
    #   One 2x2 x 2x1 block: rank 4
    # For A_2x1 * B_1x9: rank 18 (outer product)
    # Total: 4*7 + 4 + 18 = 50
    
    U = []
    V = []
    W = []
    
    def add_term(u_entry, v_entry, w_entry):
        """Add a rank-1 term. u_entry: dict mapping A-index to coeff, etc."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, coeff in u_entry.items():
            u_vec[idx] = coeff
        for idx, coeff in v_entry.items():
            v_vec[idx] = coeff
        for idx, coeff in w_entry.items():
            w_vec[idx] = coeff
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Part 1: A_2x1 * B_1x9 (rank 18)
    # A column 2 (index 2), B row 2 (index 2)
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            add_term(
                {i * m + 2: 1},
                {2 * p + k: 1},
                {k * n + i: 1}
            )
    
    # Part 2: A_2x2 * B_2x9 using Strassen on column groups
    # A_2x2: columns 0,1 of A (indices 0,1 in A)
    # B_2x9: rows 0,1 of B (indices 0,1 in B)
    # Split 9 columns into groups: [0,1], [2,3], [4,5], [6,7], [8]
    # For each 2-column group, use Strassen 2x2 x 2x2 decomposition
    
    # Strassen for 2x2 x 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    
    # For a 2-column group starting at column offset ko:
    # A_2x2 has entries at A[0,0], A[0,1], A[1,0], A[1,1] (indices 0,1,3,4 in flat A)
    # B_2x2 has entries at B[0,ko], B[0,ko+1], B[1,ko], B[1,ko+1]
    #   (indices 0*p+ko, 0*p+ko+1, 1*p+ko, 1*p+ko+1 in flat B)
    # C entries: C[0,ko], C[0,ko+1], C[1,ko], C[1,ko+1]
    #   (indices ko*n+0, ko*n+1, (ko+1)*n+0, (ko+1)*n+1 in flat C)
    
    def strassen_2x2_2x2(ko):
        """Add 7 Strassen terms for 2x2 x 2x2 block at column offset ko."""
        # A indices: a00=0, a01=1, a10=3, a11=4 (flat: i*m+j)
        a00 = 0 * m + 0  # 0
        a01 = 0 * m + 1  # 1
        a10 = 1 * m + 0  # 3
        a11 = 1 * m + 1  # 4
        
        # B indices: b00=ko, b01=ko+1, b10=p+ko, b11=p+ko+1 (flat: j*p+k)
        b00 = 0 * p + ko
        b01 = 0 * p + (ko + 1)
        b10 = 1 * p + ko
        b11 = 1 * p + (ko + 1)
        
        # C indices: c00=ko, c01=ko+1, c10=n+ko, c11=n+ko+1 (flat: k*n+i)
        c00 = ko * n + 0
        c01 = ko * n + 1
        c10 = ko * n + 1  # wait, k*n+i: for k=ko, i=1: ko*n+1
        # Actually: C index = k*n + i
        # c00: k=ko, i=0 -> ko*n
        # c01: k=ko+1, i=0 -> (ko+1)*n
        # c10: k=ko, i=1 -> ko*n+1
        # c11: k=ko+1, i=1 -> (ko+1)*n+1
        c00 = ko * n + 0
        c01 = (ko + 1) * n + 0
        c10 = ko * n + 1
        c11 = (ko + 1) * n + 1
        
        # M1 = (a00+a11)(b00+b11)
        add_term(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c01: 0, c10: 0, c11: 1}  # c00 + c11
        )
        # Wait, the W vector needs to capture the linear combination for C.
        # M1 contributes +1 to c00 and +1 to c11
        # But we already added M1's U and V. Now W should have c00:1, c11:1
        # Let me redo this properly.
        
    # Let me restart the Strassen part more carefully
    U.clear()
    V.clear()
    W.clear()
    
    # Part 1: A_2x1 * B_1x9 (rank 18)
    for i in range(n):
        for k in range(p):
            add_term(
                {i * m + 2: 1},
                {2 * p + k: 1},
                {k * n + i: 1}
            )
    
    # Part 2: Strassen on 2x2 x 2x2 blocks
    # Column groups: [0,1], [2,3], [4,5], [6,7]
    for ko in [0, 2, 4, 6]:
        a00 = 0 * m + 0  # A[0,0]
        a01 = 0 * m + 1  # A[0,1]
        a10 = 1 * m + 0  # A[1,0]
        a11 = 1 * m + 1  # A[1,1]
        
        b00 = 0 * p + ko
        b01 = 0 * p + (ko + 1)
        b10 = 1 * p + ko
        b11 = 1 * p + (ko + 1)
        
        c00 = ko * n + 0
        c01 = (ko + 1) * n + 0
        c10 = ko * n + 1
        c11 = (ko + 1) * n + 1
        
        # M1 = (a00+a11)(b00+b11) -> contributes to c00: +1, c11: +1
        add_term(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c11: 1}
        )
        # M2 = (a10+a11)(b00) -> contributes to c10: +1, c11: -1
        add_term(
            {a10: 1, a11: 1},
            {b00: 1},
            {c10: 1, c11: -1}
        )
        # M3 = (a00)(b10-b11) -> contributes to c01: +1, c11: +1
        add_term(
            {a00: 1},
            {b10: 1, b11: -1},
            {c01: 1, c11: 1}
        )
        # M4 = (a11)(b01-b00) -> contributes to c00: +1, c10: +1
        add_term(
            {a11: 1},
            {b01: 1, b00: -1},
            {c00: 1, c10: 1}
        )
        # M5 = (a00+a01)(b11) -> contributes to c01: +1, c00: -1
        add_term(
            {a00: 1, a01: 1},
            {b11: 1},
            {c01: 1, c00: -1}
        )
        # M6 = (a10-a00)(b00+b10) -> contributes to c11: +1
        add_term(
            {a10: 1, a00: -1},
            {b00: 1, b10: 1},
            {c11: 1}
        )
        # M7 = (a01-a11)(b01+b11) -> contributes to c00: +1, c01: +1
        add_term(
            {a01: 1, a11: -1},
            {b01: 1, b11: 1},
            {c00: 1, c01: 1}
        )
    
    # Part 3: Remaining column ko=8 (2x2 x 2x1, rank 4)
    ko = 8
    a00 = 0 * m + 0
    a01 = 0 * m + 1
    a10 = 1 * m + 0
    a11 = 1 * m + 1
    
    b00 = 0 * p + ko
    b10 = 1 * p + ko
    
    c00 = ko * n + 0
    c10 = ko * n + 1
    
    # Schoolbook for 2x2 x 2x1: 4 terms
    add_term({a00: 1}, {b00: 1}, {c00: 1})
    add_term({a01: 1}, {b10: 1}, {c00: 1})
    add_term({a00: 1}, {b10: 1}, {c10: 1})
    add_term({a01: 1}, {b00: 1}, {c10: 1})
    
    rank_structured = len(U)
    
    # Verify correctness
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Check: sum of outer products should equal T
    reconstructed = np.zeros((n * m, m * p, p * n))
    for r in range(rank_structured):
        for a_idx in range(n * m):
            if U[r][a_idx] != 0:
                for b_idx in range(m * p):
                    if V[r][b_idx] != 0:
                        for c_idx in range(p * n):
                            if W[r][c_idx] != 0:
                                reconstructed[a_idx, b_idx, c_idx] += U[r][a_idx] * V[r][b_idx] * W[r][c_idx]
    
    if np.allclose(reconstructed, T, atol=1e-10):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
