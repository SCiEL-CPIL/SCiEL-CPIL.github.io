import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen 2x2*2x2 factors
    # U_s, V_s, W_s are 4x7 matrices representing the 7 terms of Strassen
    # Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    
    # Term 0: (A00+A11)(B00+B11) -> C00, C11
    # Term 1: (A10+A11)B00 -> C10, C11(-)
    # Term 2: A00(B01-B11) -> C01, C11
    # Term 3: A11(B10-B00) -> C00, C10
    # Term 4: (A00+A01)B11 -> C00(-), C01
    # Term 5: (A10-A00)(B00+B01) -> C11
    # Term 6: (A01-A11)(B10+B11) -> C00
    
    # We construct the sparse representation of the factors
    # For each term t, we store the non-zero (index, value) pairs for U, V, W
    
    strassen_terms = []
    
    # Helper to add a term
    def add_term(u_entries, v_entries, w_entries):
        strassen_terms.append((u_entries, v_entries, w_entries))

    # t0
    add_term([(0,1.0), (3,1.0)], [(0,1.0), (3,1.0)], [(0,1.0), (3,1.0)])
    # t1
    add_term([(2,1.0), (3,1.0)], [(0,1.0)], [(2,1.0), (3,-1.0)])
    # t2
    add_term([(0,1.0)], [(1,1.0), (3,-1.0)], [(1,1.0), (3,1.0)])
    # t3
    add_term([(3,1.0)], [(2,1.0), (0,-1.0)], [(0,1.0), (2,1.0)])
    # t4
    add_term([(0,1.0), (1,1.0)], [(3,1.0)], [(0,-1.0), (1,1.0)])
    # t5
    add_term([(2,1.0), (0,-1.0)], [(0,1.0), (1,1.0)], [(3,1.0)])
    # t6
    add_term([(1,1.0), (3,-1.0)], [(2,1.0), (3,1.0)], [(0,1.0)])

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Part 1: 4x4*4x4 Strassen
    # We use the recursive structure.
    # The 4x4*4x4 Strassen is composed of 49 terms.
    # Each term (t,s) in 7x7 grid corresponds to a product of 2x2 Strassen terms.
    
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            # U factor for this term
            # U_4x4[i,j] for this term is U_s[2*I+J, t] * U_s[2*i0+j0, s]
            # where i = 2*I + i0, j = 2*J + j0
            
            u_t_entries = strassen_terms[t][0]
            u_s_entries = strassen_terms[s][0]
            
            for (i_t, c_t) in u_t_entries:
                I = i_t // 2
                i0 = i_t % 2
                for (i_s, c_s) in u_s_entries:
                    J = i_s // 2
                    j0 = i_s % 2
                    # Wait, the indices in strassen_terms are for 2x2 blocks.
                    # i_t is index in 2x2 U matrix.
                    # The global index in 4x4 U is i = 2*I + i0?
                    # No.
                    # In the recursive step, the "U" for the 4x4 case is built from the "U" of the 2x2 case.
                    # The 2x2 U has indices 0..3.
                    # The 4x4 U has indices 0..15.
                    # The mapping is:
                    # 2x2 block (I, J) corresponds to rows 2*I..2*I+1, cols 2*J..2*J+1 in 4x4?
                    # No, the Strassen decomposition of 4x4*4x4 splits A into four 2x2 blocks.
                    # A = [[A00, A01], [A10, A11]].
                    # The Strassen terms are linear combinations of these blocks.
                    # The code in the original program does:
                    # U[i * 4 + j, col] += U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    # where i = 2*I + i0, j = 2*J + j0.
                    # This implies that the 2x2 Strassen factors are applied to the "block indices" and the "element indices" within the block?
                    # Actually, the original code treats the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
                    # The Strassen factors U_s, V_s, W_s are 2x2 matrices (flattened to 4).
                    # The recursion multiplies these factors.
                    
                    # Let's stick to the original loop logic but optimized.
                    pass

    # Re-implementing the original loop logic for safety, as it is known to be correct.
    # The original logic:
    # U_s[i*2+j, t] is the coefficient for block (I,J) and element (i0,j0)?
    # No, U_s is 4x7. Index is i*2+j.
    # In the 2x2 Strassen, U is 2x2. Flattened: [U00, U01, U10, U11].
    # So U_s[k, t] where k=0..3.
    # k = i*2 + j.
    # In the 4x4 case, we have blocks I,J (0..1) and elements i0,j0 (0..1).
    # The coefficient for A[I,i0, J,j0] in term (t,s) is U_s[2*I+J, t] * U_s[2*i0+j0, s].
    
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = strassen_terms[t][0][2 * I + J] * strassen_terms[s][0][2 * i0 + j0]
                    # Note: strassen_terms entries are lists of (idx, val).
                    # We need to get the value at specific index.
                    # It's better to use dense arrays for U_s, V_s, W_s.
                    pass

    # Let's just use the dense arrays from the original code, it's fast enough.
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    # Part 1: 4x4*4x4
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
