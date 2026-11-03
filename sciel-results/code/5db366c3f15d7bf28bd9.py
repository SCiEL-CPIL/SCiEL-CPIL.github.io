"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Uses a known 49-term decomposition for 4x4 matrix multiplication,
which is the current best-known upper bound on the tensor rank of
M_{4,4,4}.

The decomposition is built from Strassen's 7-term 2x2 multiplication
combined with an optimal 7-term 2x2 combination scheme (the "7x7"
Strassen variant), giving 7*7 = 49 terms.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return a rank-49 tensor decomposition for 4x4 matrix multiplication.
    
    The 4x4 matrices are partitioned into four 2x2 blocks:
        A = [[A00, A01],
             [A10, A11]]
        B = [[B00, B01],
             [B10, B11]]
        C = [[C00, C01],
             [C10, C11]]
    
    Each C block is a 2x2 matrix computed via Strassen's algorithm.
    
    The block structure gives 16 block-level products (4x4 of 2x2 blocks),
    but we can do better. The key insight: we need to compute
        C_ij = sum_k A_ik * B_kj
    for i,j in {0,1} at the block level, where each entry is a 2x2 matrix.
    
    Actually, the standard approach for rank 49 is:
    - Use Strassen's 7 multiplications at the top level for the 2x2 block
      multiplication of 2x2 blocks, BUT this gives 7 products of 2x2 matrices,
      and each 2x2 product needs rank 7, giving 7*7 = 49.
    
    Let me define this properly.
    
    At the block level (2x2 blocks), we use Strassen:
        M1 = (A00 + A11) * (B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B01 - B11)
        M4 = A11 * (B10 - B00)
        M5 = (A00 + A01) * B11
        M6 = (A10 - A00) * (B00 + B01)
        M7 = (A01 - A11) * (B10 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
    
    Each M_i is a product of two 2x2 matrices, which itself can be done
    with Strassen's 7-term decomposition.
    
    So total rank = 7 * 7 = 49.
    
    Now I need to express this as U, V, W matrices.
    
    U has shape (n*m, rank) = (16, 49)
    V has shape (m*p, rank) = (16, 49)
    W has shape (p*n, rank) = (16, 49)
    
    The entries are indexed as:
        U[i*m + j, r] for A[i,j]
        V[j*p + k, r] for B[j,k]
        W[k*n + i, r] for C[i,k]  (note: W[k*n+i,r] per the task)
    
    Wait, let me re-read: "W[k*n+i,r]" - so W is indexed by output position.
    For C[i,k] (row i, col k), the W index is k*n + i = k*4 + i.
    
    Let me define the Strassen 2x2 decomposition first.
    
    For a 2x2 matrix multiplication:
        A = [[a00, a01], [a10, a11]]
        B = [[b00, b01], [b10, b11]]
        C = A*B = [[c00, c01], [c10, c11]]
    
    Strassen's 7 products:
        p1 = (a00 + a11) * (b00 + b11)
        p2 = (a10 + a11) * b00
        p3 = a00 * (b01 - b11)
        p4 = a11 * (b10 - b00)
        p5 = (a00 + a01) * b11
        p6 = (a10 - a00) * (b00 + b01)
        p7 = (a01 - a11) * (b10 + b11)
    
        c00 = p1 + p4 - p5 + p7
        c01 = p3 + p5
        c10 = p2 + p4
        c11 = p1 - p2 + p3 + p6
    
    The tensor decomposition for a single 2x2 multiplication has rank 7.
    The U, V, W for this (with local indices):
    
    For each of the 7 terms:
        Term 1: A part = a00 + a11, B part = b00 + b11
                 U_local: a00: 1, a11: 1
                 V_local: b00: 1, b11: 1
        Term 2: A part = a10 + a11, B part = b00
                 U_local: a10: 1, a11: 1
                 V_local: b00: 1
        Term 3: A part = a00, B part = b01 - b11
                 U_local: a00: 1
                 V_local: b01: 1, b11: -1
        Term 4: A part = a11, B part = b10 - b00
                 U_local: a11: 1
                 V_local: b10: 1, b00: -1
        Term 5: A part = a00 + a01, B part = b11
                 U_local: a00: 1, a01: 1
                 V_local: b11: 1
        Term 6: A part = a10 - a00, B part = b00 + b01
                 U_local: a10: 1, a00: -1
                 V_local: b00: 1, b01: 1
        Term 7: A part = a01 - a11, B part = b10 + b11
                 U_local: a01: 1, a11: -1
                 V_local: b10: 1, b11: 1
    
    And W_local for each term:
        c00 = p1 + p4 - p5 + p7  => W for term t on c00: [1, 0, 0, 1, -1, 0, 1]
        c01 = p3 + p5            => W for term t on c01: [0, 0, 1, 0, 1, 0, 0]
        c10 = p2 + p4            => W for term t on c10: [0, 1, 0, 1, 0, 0, 0]
        c11 = p1 - p2 + p3 + p6  => W for term t on c11: [1, -1, 1, 0, 0, 1, 0]
    
    Let me organize this as arrays.
    
    U_local[t][local_a_idx]:
        t=0: [1, 0, 0, 1]  (a00, a01, a10, a11)
        t=1: [0, 0, 1, 1]
        t=2: [1, 0, 0, 0]
        t=3: [0, 0, 0, 1]
        t=4: [1, 1, 0, 0]
        t=5: [-1, 0, 1, 0]
        t=6: [0, 1, 0, -1]
    
    V_local[t][local_b_idx]:
        t=0: [1, 0, 0, 1]  (b00, b01, b10, b11)
        t=1: [1, 0, 0, 0]
        t=2: [0, 1, 0, -1]
        t=3: [-1, 0, 1, 0]
        t=4: [0, 0, 0, 1]
        t=5: [1, 1, 0, 0]
        t=6: [0, 0, 1, 1]
    
    W_local[t][local_c_idx]:
        t=0: [1, 0, 0, 1]  (c00, c01, c10, c11)
        t=1: [0, 0, 1, -1]
        t=2: [0, 1, 0, 1]
        t=3: [1, 0, 1, 0]
        t=4: [-1, 1, 0, 0]
        t=5: [0, 0, 0, 1]
        t=6: [1, 0, 0, 0]
    
    Wait, I need to double-check W. Let me re-derive.
    
    c00 = p1 + p4 - p5 + p7
    c01 = p3 + p5
    c10 = p2 + p4
    c11 = p1 - p2 + p3 + p6
    
    So for each term t (0-indexed, t=0 is p1, t=1 is p2, ..., t=6 is p7):
    c00: [1, 0, 0, 1, -1, 0, 1]
    c01: [0, 0, 1, 0, 1, 0, 0]
    c10: [0, 1, 0, 1, 0, 0, 0]
    c11: [1, -1, 1, 0, 0, 1, 0]
    
    Now for the 4x4 case:
    
    The 4x4 matrices are partitioned into 2x2 blocks of 2x2 matrices.
    Block (i,j) of A is A_ij, a 2x2 matrix with entries A[2i+u, 2j+v].
    
    We compute 7 "block products" M_s for s in 0..6, where each M_s is a 2x2
    matrix (the result of a Strassen term at the block level).
    
    For each M_s, we need to compute it as a product of two 2x2 matrices,
    which uses 7 more Strassen terms.
    
    So the total rank is 7 * 7 = 49.
    
    The global indices:
    For term (s, t) where s is the outer Strassen index (0..6) and t is the
    inner Strassen index (0..6), the global rank index is s*7 + t.
    
    For U:
        The outer Strassen term s uses certain blocks of A.
        Specifically, for the outer term s, the A-part is a sum of certain
        2x2 blocks A_ij.
        
        Let me define the outer Strassen A-coefficients:
        For block-level Strassen (computing 2x2 block products):
            M1 = (A00 + A11) * (B00 + B11)
            M2 = (A10 + A11) * B00
            M3 = A00 * (B01 - B11)
            M4 = A11 * (B10 - B00)
            M5 = (A00 + A01) * B11
            M6 = (A10 - A00) * (B00 + B01)
            M7 = (A01 - A11) * (B10 + B11)
        
        So the A-coefficients for each outer term s (0-indexed):
        s=0: A00: 1, A11: 1
        s=1: A10: 1, A11: 1
        s=2: A00: 1
        s=3: A11: 1
        s=4: A00: 1, A01: 1
        s=5: A10: 1, A00: -1
        s=6: A01: 1, A11: -1
        
        Let me encode this as O_A[s][block_idx] where block_idx = i*2 + j for block (i,j).
        Blocks: 00=0, 01=1, 10=2, 11=3
        
        O_A[0] = [1, 0, 0, 1]
        O_A[1] = [0, 0, 1, 1]
        O_A[2] = [1, 0, 0, 0]
        O_A[3] = [0, 0, 0, 1]
        O_A[4] = [1, 1, 0, 0]
        O_A[5] = [-1, 0, 1, 0]
        O_A[6] = [0, 1, 0, -1]
        
    For V (B-coefficients):
        s=0: B00: 1, B11: 1
        s=1: B00: 1
        s=2: B01: 1, B11: -1
        s=3: B10: 1, B00: -1
        s=4: B11: 1
        s=5: B00: 1, B01: 1
        s=6: B10: 1, B11: 1
        
        O_B[0] = [1, 0, 0, 1]
        O_B[1] = [1, 0, 0, 0]
        O_B[2] = [0, 1, 0, -1]
        O_B[3] = [-1, 0, 1, 0]
        O_B[4] = [0, 0, 0, 1]
        O_B[5] = [1, 1, 0, 0]
        O_B[6] = [0, 0, 1, 1]
        
    For W (C-coefficients), the block-level reconstruction:
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        O_C[block_idx][s] where block_idx = i*2 + j:
        C00 (block 0): [1, 0, 0, 1, -1, 0, 1]
        C01 (block 1): [0, 0, 1, 0, 1, 0, 0]
        C10 (block 2): [0, 1, 0, 1, 0, 0, 0]
        C11 (block 3): [1, -1, 1, 0, 0, 1, 0]
        
    Now, for each outer term s and inner term t, the global rank index is s*7 + t.
    
    For U[global_a_idx, s*7+t]:
        global_a_idx = (2*i + u) * 4 + (2*j + v) where block (i,j) and local (u,v)
        
        The value is O_A[s][i*2+j] * U_local[t][u*2+v]
        
        where U_local[t] is the 2x2 Strassen A-coefficient for term t.
        
    Similarly for V and W.
    
    Let me define U_local, V_local, W_local as 7x4 arrays (flattened 2x2).
    
    U_local (A coefficients for inner Strassen):
        t=0: [1, 0, 0, 1]
        t=1: [0, 0, 1, 1]
        t=2: [1, 0, 0, 0]
        t=3: [0, 0, 0, 1]
        t=4: [1, 1, 0, 0]
        t=5: [-1, 0, 1, 0]
        t=6: [0, 1, 0, -1]
        
    V_local (B coefficients for inner Strassen):
        t=0: [1, 0, 0, 1]
        t=1: [1, 0, 0, 0]
        t=2: [0, 1, 0, -1]
        t=3: [-1, 0, 1, 0]
        t=4: [0, 0, 0, 1]
        t=5: [1, 1, 0, 0]
        t=6: [0, 0, 1, 1]
        
    W_local (C coefficients for inner Strassen):
        t=0: [1, 0, 0, 1]
        t=1: [0, 0, 1, -1]
        t=2: [0, 1, 0, 1]
        t=3: [1, 0, 1, 0]
        t=4: [-1, 1, 0, 0]
        t=5: [0, 0, 0, 1]
        t=6: [1, 0, 0, 0]
        
    Now let me code this up.
    """
    
    # Define local Strassen coefficients
    U_local = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_local = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_local = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Outer Strassen block coefficients
    # O_A[s][block] where block = i*2+j
    O_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    O_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # O_C[block][s] where block = i*2+j
    O_C = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U: for each block (i,j) and local (u,v)
            for i in range(2):
                for j in range(2):
                    block_idx = i * 2 + j
                    outer_coeff = O_A[s][block_idx]
                    if outer_coeff == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            global_a_idx = (2 * i + u) * m + (2 * j + v)
                            local_coeff = U_local[t][u * 2 + v]
                            if local_coeff != 0:
                                U[global_a_idx][col] += outer_coeff * local_coeff
            
            # V: for each block (k,l) of B and local
            for k in range(2):
                for l in range(2):
                    block_idx = k * 2 + l
                    outer_coeff = O_B[s][block_idx]
                    if outer_coeff == 0:
                        continue
                    for kk in range(2):
                        for ll in range(2):
                            global_b_idx = (2 * k + kk) * p + (2 * l + ll)
                            local_coeff = V_local[t][kk * 2 + ll]
                            if local_coeff != 0:
                                V[global_b_idx][col] += outer_coeff * local_coeff
            
            # W: for each block (c_row, c_col) of C and local
            for c_row in range(2):
                for c_col in range(2):
                    block_idx = c_row * 2 + c_col
                    outer_coeff = O_C[block_idx][s]
                    if outer_coeff == 0:
                        continue
                    for cu in range(2):
                        for cv in range(2):
                            # C[c_row*2+cu, c_col*2+cv]
                            # W index: k*n + i where k is col of C, i is row of C
                            # So W[(c_col*2+cv) * n + (c_row*2+cu), col]
                            global_c_idx = (c_col * 2 + cv) * n + (c_row * 2 + cu)
                            local_coeff = W_local[t][cu * 2 + cv]
                            if local_coeff != 0:
                                W[global_c_idx][col] += outer_coeff * local_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
