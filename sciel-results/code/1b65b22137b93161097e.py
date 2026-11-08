"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 100
    
    rank = 65
    # Initialize as lists of lists to match expected interface strictly
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # --- Definition of Strassen Coefficients for 2x2 ---
    # Super level (4x4 composed of 2x2 blocks)
    # P = A * X, Q = Y * B
    # Strassen M1..M7
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # Coefficients for P terms (U side)
    # Each term is a combination of the 4 A blocks
    # inner_p[t] gives coefficients for [A00, A01, A10, A11]
    super_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=int)
    
    super_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=int)
    
    # Inner level (2x2 composed of 2x1 blocks? No, 2x2 of scalars)
    # For a 2x2 block P, we decompose into 7 terms using Strassen on 1x1? No, 2x2 Strassen uses 7 multiplications of 1x1?
    # Wait, the current code uses "Nested Strassen".
    # Level 1: 4x4 -> 7 products of 2x2.
    # Level 2: Each 2x2 product -> 7 products of 1x1 (scalars).
    # Total 49 scalar multiplications.
    
    # The "inner" Strassen coefficients for a 2x2 matrix multiplication
    # P (2x2) * Q (2x2) -> R (2x2)
    # P blocks: P00, P01, P10, P11
    # Q blocks: Q00, Q01, Q10, Q11
    
    inner_P = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=int)
    
    inner_Q = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=int)
    
    # Contribution to Result R (2x2)
    # R00, R01, R10, R11
    inner_R = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=int)
    
    # Contribution to Super Result C (4x4)
    # C00, C01, C10, C11
    super_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=int)
    
    def get_indices_2x2(block_idx, r, c):
        # block 0: 0:0, 1:1
        # block 1: 0:2, 1:3 (Col 2,3)
        # block 2: 2:0, 3:1 (Row 2,3)
        # block 3: 2:2, 3:3
        if block_idx == 0:
            return r, c
        elif block_idx == 1:
            return r, 2 + c
        elif block_idx == 2:
            return 2 + r, c
        else:
            return 2 + r, 2 + c

    col_idx = 0
    
    # Process the 49 Strassen terms
    for s in range(7):
        for t in range(7):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
            # Determine coefficients for U (A side)
            # U term s, t depends on super_A[s] and inner_P[t]
            # super_A[s] is [c00, c01, c10, c11] for A blocks
            # inner_P[t] is [d00, d01, d10, d11] for P elements (which are 2x2 blocks of A)
            # The effective A is a linear combination.
            
            # Actually, the structure is:
            # Term (s,t) corresponds to:
            # P_eff = sum_{b} super_A[s][b] * A_b
            # Then we apply inner Strassen t to P_eff and Q_eff.
            # This means the U matrix for this term is the Kronecker-like combination.
            
            # Let's build the list of (i, j, coeff) for U
            u_entries = []
            for b_a in range(4):
                ca = super_A[s][b_a]
                if ca == 0: continue
                for e_p in range(4):
                    cp = inner_P[t][e_p]
                    if cp == 0: continue
                    total_c = ca * cp
                    if total_c == 0: continue
                    # e_p is index into 2x2 block: 0:0,0 1:0,1 2:1,0 3:1,1
                    r_p, c_p = e_p // 2, e_p % 2
                    # Map to global i, j
                    i_g, j_g = get_indices_2x2(b_a, r_p, c_p)
                    u_entries.append((i_g, j_g, total_c))
            
            # V entries
            v_entries = []
            for b_b in range(4):
                cb = super_B[s][b_b]
                if cb == 0: continue
                for e_q in range(4):
                    cq = inner_Q[t][e_q]
                    if cq == 0: continue
                    total_c = cb * cq
                    if total_c == 0: continue
                    r_q, c_q = e_q // 2, e_q % 2
                    j_g, k_g = get_indices_2x2(b_b, r_q, c_q)
                    v_entries.append((j_g, k_g, total_c))
            
            # W entries
            w_entries = []
            for e_r in range(4):
                cr = inner_R[t][e_r]
                if cr == 0: continue
                r_r, c_r = e_r // 2, e_r % 2
                for b_c in range(4):
                    cc = super_C[s][b_c]
                    if cc == 0: continue
                    total_c = cr * cc
                    if total_c == 0: continue
                    i_g, k_g = get_indices_2x2(b_c, r_r, c_r)
                    # W index is k*n + i
                    w_entries.append((k_g, i_g, total_c))
            
            # Fill U, V, W
            for i, j, c in u_entries:
                U[i*m + j][col_idx] += c
            for j, k, c in v_entries:
                V[j*p + k][col_idx] += c
            for k, i, c in w_entries:
                W[k*n + i][col_idx] += c
                
            col_idx += 1

    # Process the 16 Schoolbook terms for the 5th column
    for i in range(4):
        for j in range(4):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
            U[i*m + j][col_idx] += 1.0
            V[j*p + 4][col_idx] += 1.0  # Column 4 of B
            W[4*n + i][col_idx] += 1.0  # Row 4 of C (index 4 in k)
            
            col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
