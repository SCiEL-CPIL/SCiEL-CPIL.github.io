import time
import sys

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We are constructing a rank 34 decomposition.
    # Rank 34 is derived from:
    # 7 (Strassen for A11 B11)
    # 7 (Strassen for A11 B12)
    # 4 (Schoolbook for A21 B11)
    # 4 (Schoolbook for A21 B12)
    # 4 (Schoolbook for A12 B21)
    # 4 (Schoolbook for A12 B22)
    # 2 (Schoolbook for A22 B21)
    # 2 (Schoolbook for A22 B22)
    # Total: 34
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(i, l, k, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[i * m + l][col] = u_val
        V[l * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1

    def add_strassen(i_offset, l_offset, k_offset):
        """
        Adds 7 Strassen terms for a 2x2 block multiplication.
        A block is A[i_offset:i_offset+2, l_offset:l_offset+2]
        B block is B[l_offset:l_offset+2, k_offset:k_offset+2]
        Result C block is C[i_offset:i_offset+2, k_offset:k_offset+2]
        """
        nonlocal col
        # m1 = (x00+x11)(y00+y11)
        # We need to distribute the sums.
        # This function adds the 7 terms.
        
        # Term 1: (x00+x11)(y00+y11)
        # This expands to 4 products in schoolbook, but in tensor decomposition, 
        # we define the linear combinations in U, V, W.
        
        # U part: x00+x11
        # V part: y00+y11
        # W part: contributes to z00, z11
        
        # Let's map the Strassen factors to U, V, W entries.
        # For each factor m_j = (linear combo of X) * (linear combo of Y)
        
        # m1: X = x00 + x11, Y = y00 + y11
        # U: x00 at (i0, l0), x11 at (i1, l1)
        # V: y00 at (l0, k0), y11 at (l1, k1)
        # W: 
        # z00 gets +1
        # z11 gets +1
        
        # m2: X = x10 + x11, Y = y00
        # U: x10 at (i1, l0), x11 at (i1, l1)
        # V: y00 at (l0, k0)
        # W:
        # z10 gets +1
        # z11 gets -1
        
        # m3: X = x00, Y = y10 - y11
        # U: x00 at (i0, l0)
        # V: y10 at (l1, k0), y11 at (l1, k1) with -1
        # W:
        # z01 gets +1
        # z11 gets +1
        
        # m4: X = x11, Y = y00 - y10
        # U: x11 at (i1, l1)
        # V: y00 at (l0, k0), y10 at (l1, k0) with -1
        # W:
        # z00 gets +1
        # z10 gets +1
        
        # m5: X = x00 + x01, Y = y11
        # U: x00 at (i0, l0), x01 at (i0, l1)
        # V: y11 at (l1, k1)
        # W:
        # z00 gets -1
        # z01 gets +1
        
        # m6: X = x10 - x00, Y = y00 + y10
        # U: x10 at (i1, l0), x00 at (i0, l0) with -1
        # V: y00 at (l0, k0), y10 at (l1, k0)
        # W:
        # z11 gets +1
        
        # m7: X = x01 - x11, Y = y10 + y11
        # U: x01 at (i0, l1), x11 at (i1, l1) with -1
        # V: y10 at (l1, k0), y11 at (l1, k1)
        # W:
        # z00 gets +1
        # z11 gets +1

        # Helper to add a single scalar multiplication term
        # We need to add multiple (U, V) pairs for each Strassen factor?
        # No, each Strassen factor is ONE term in the decomposition.
        # The "linear combo" is encoded in the U and V vectors for that term.
        
        # So for m1:
        # U vector has 1 at (i0, l0) and 1 at (i1, l1).
        # V vector has 1 at (l0, k0) and 1 at (l1, k1).
        # W vector has 1 at (k0, i0) [z00] and 1 at (k1, i1) [z11].
        
        # Let's implement add_strassen_factor
        def add_factor(u_indices, v_indices, w_indices, w_signs):
            nonlocal col
            if col >= rank:
                return
            for (u_i, u_l), val in u_indices:
                U[u_i * m + u_l][col] = val
            for (v_l, v_k), val in v_indices:
                V[v_l * p + v_k][col] = val
            for (w_k, w_i), val in zip(w_indices, w_signs):
                W[w_k * n + w_i][col] = val
            col += 1

        i0, i1 = i_offset, i_offset + 1
        l0, l1 = l_offset, l_offset + 1
        k0, k1 = k_offset, k_offset + 1

        # m1
        add_factor(
            [((i0, l0), 1), ((i1, l1), 1)],
            [((l0, k0), 1), ((l1, k1), 1)],
            [(k0, i0), (k1, i1)],
            [1, 1]
        )
        
        # m2
        add_factor(
            [((i1, l0), 1), ((i1, l1), 1)],
            [((l0, k0), 1)],
            [(k0, i1), (k1, i1)],
            [1, -1]
        )
        
        # m3
        add_factor(
            [((i0, l0), 1)],
            [((l1, k0), 1), ((l1, k1), -1)],
            [(k1, i0), (k1, i1)],
            [1, 1]
        )
        
        # m4
        add_factor(
            [((i1, l1), 1)],
            [((l0, k0), 1), ((l1, k0), -1)],
            [(k0, i0), (k0, i1)],
            [1, 1]
        )
        
        # m5
        add_factor(
            [((i0, l0), 1), ((i0, l1), 1)],
            [((l1, k1), 1)],
            [(k0, i0), (k1, i0)],
            [-1, 1]
        )
        
        # m6
        add_factor(
            [((i1, l0), 1), ((i0, l0), -1)],
            [((l0, k0), 1), ((l1, k0), 1)],
            [(k1, i1)],
            [1]
        )
        
        # m7
        add_factor(
            [((i0, l1), 1), ((i1, l1), -1)],
            [((l1, k0), 1), ((l1, k1), 1)],
            [(k0, i0), (k1, i1)],
            [1, 1]
        )

    # 1. Strassen for A11 B11
    # A11 is i in {0,1}, l in {0,1}
    # B11 is l in {0,1}, k in {0,1}
    add_strassen(0, 0, 0)
    
    # 2. Strassen for A11 B12
    # A11 is i in {0,1}, l in {0,1}
    # B12 is l in {0,1}, k in {2,3}
    add_strassen(0, 0, 2)
    
    # 3. Schoolbook for A21 B11
    # A21 is i in {2}, l in {0,1}
    # B11 is l in {0,1}, k in {0,1}
    # Products: A20*B0k + A21*B1k
    for l in [0, 1]:
        for k in [0, 1]:
            add_term(2, l, k, 1, 1, 1)
            
    # 4. Schoolbook for A21 B12
    # A21 is i in {2}, l in {0,1}
    # B12 is l in {0,1}, k in {2,3}
    for l in [0, 1]:
        for k in [2, 3]:
            add_term(2, l, k, 1, 1, 1)
            
    # 5. Schoolbook for A12 B21
    # A12 is i in {0,1}, l in {2}
    # B21 is l in {2}, k in {0,1}
    for i in [0, 1]:
        for k in [0, 1]:
            add_term(i, 2, k, 1, 1, 1)
            
    # 6. Schoolbook for A12 B22
    # A12 is i in {0,1}, l in {2}
    # B22 is l in {2}, k in {2,3}
    for i in [0, 1]:
        for k in [2, 3]:
            add_term(i, 2, k, 1, 1, 1)
            
    # 7. Schoolbook for A22 B21
    # A22 is i in {2}, l in {2}
    # B21 is l in {2}, k in {0,1}
    for k in [0, 1]:
        add_term(2, 2, k, 1, 1, 1)
        
    # 8. Schoolbook for A22 B22
    # A22 is i in {2}, l in {2}
    # B22 is l in {2}, k in {2,3}
    for k in [2, 3]:
        add_term(2, 2, k, 1, 1, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
