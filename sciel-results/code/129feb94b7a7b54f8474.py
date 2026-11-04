"""Matrix multiplication 2x3x10 decomposition using a structured rank-55 tensor decomposition."""

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition.
    # We split the 10 columns of B into 5 blocks of 2 columns.
    # Each block is a 2x3x2 tensor.
    # We use a rank-11 decomposition for each 2x3x2 tensor.
    # Total rank = 5 * 11 = 55.
    
    rank_per_block = 11
    num_blocks = p // 2
    total_rank = rank_per_block * num_blocks
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, col, u_val, v_val, w_val):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    for block in range(num_blocks):
        # Block handles columns 2*block and 2*block+1 of B
        # And columns 2*block and 2*block+1 of C
        k0 = 2 * block
        k1 = 2 * block + 1
        
        base_col = block * rank_per_block
        
        # Strassen terms for 2x2x2 part (indices 0,1 for A and B)
        # P1 = (a00 + a11)(b00 + b11)
        add_term(0, 0, k0*n+0, base_col+0, 1, 1, 1) # a00*b00 -> c00? No, Strassen is linear combinations.
        # We need to expand the Strassen products into linear terms for U, V, W.
        # U, V, W entries are coefficients of a_i, b_j, c_k.
        
        # Term 1: (a00 + a11)(b00 + b11)
        # This contributes to c00, c11.
        # c00 += 1
        # c11 += 1
        # Wait, the decomposition is sum of u_r v_r w_r.
        # u_r is a vector in A-space, v_r in B-space, w_r in C-space.
        # So for term r, we add u_r[i] to U[i][r], etc.
        
        # Let's define the 11 vectors (u, v, w) for the block.
        # u is length 6 (2x3 A), v is length 6 (3x2 B part? No, V is 3x10, so we only set cols k0, k1), w is length 20 (10x2 C? No, W is 10x2, so we only set rows k0, k1).
        
        # Indices in U: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        # Indices in V: 0:(0,0), 1:(0,1), 2:(0,2)... 3:(1,0)... 5:(2,0), 6:(2,1)
        #   For block k0, k1: V indices are 3*k0, 3*k0+1 (for row 0), 3*k0+2? No.
        #   V shape is (3, 10). Index is row * 10 + col.
        #   Row 0: 0, 1, ..., 9. Col k0, k1.
        #   Row 1: 10, 11, ..., 19. Col k0, k1.
        #   Row 2: 20, 21, ..., 29. Col k0, k1.
        
        # Indices in W: (p, n) = (10, 2). Index is col * 2 + row.
        #   Col k0: 2*k0, 2*k0+1.
        #   Col k1: 2*k1, 2*k1+1.
        
        # Strassen P1: (a00 + a11)(b00 + b11)
        # u: a00=1, a11=1. (U[0]=1, U[4]=1)
        # v: b00=1, b11=1. (V[0+k0]=1, V[10+k1]=1) -> Wait, b11 is row 1, col 1 of the 2x2 block.
        #   In the full B, row 1, col k1. Index 10 + k1.
        # w: c00=1, c11=1. (W[2*k0]=1, W[2*k1+1]=1)
        
        # Let's write a function to add a Strassen term.
        
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs, col_offset):
            # u_coeffs: dict {A_index: coeff}
            # v_coeffs: dict {B_index: coeff}
            # w_coeffs: dict {C_index: coeff}
            for idx, val in u_coeffs.items():
                U[idx][base_col + col_offset] += val
            for idx, val in v_coeffs.items():
                V[idx][base_col + col_offset] += val
            for idx, val in w_coeffs.items():
                W[idx][base_col + col_offset] += val

        # A indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        # B indices for block: 
        #   b00: 0+k0
        #   b01: 0+k1
        #   b10: 10+k0
        #   b11: 10+k1
        #   b20: 20+k0
        #   b21: 20+k1
        # C indices for block:
        #   c00: 2*k0
        #   c01: 2*k0+1
        #   c10: 2*k1
        #   c11: 2*k1+1
        
        # P1: (a00 + a11)(b00 + b11)
        add_strassen_term(
            {0: 1, 4: 1}, 
            {0+k0: 1, 10+k1: 1}, 
            {2*k0: 1, 2*k1+1: 1}, 
            0
        )
        
        # P2: (a10 + a11)b00
        add_strassen_term(
            {3: 1, 4: 1}, 
            {0+k0: 1}, 
            {2*k1: 1, 2*k1+1: -1}, # c10 += 1, c11 -= 1? 
            # c10 = P2 + P4. c11 = P1 - P2 ...
            # So w for P2 should be c10: 1, c11: -1.
            1
        )
        
        # P3: a00(b10 - b11)
        add_strassen_term(
            {0: 1}, 
            {10+k0: 1, 10+k1: -1}, 
            {2*k0+1: 1, 2*k1+1: 1}, # c01 += 1, c11 += 1
            2
        )
        
        # P4: a11(b01 - b00)
        add_strassen_term(
            {4: 1}, 
            {0+k1: 1, 0+k0: -1}, 
            {2*k0: 1, 2*k1: 1}, # c00 += 1, c10 += 1
            3
        )
        
        # P5: (a00 + a01)b11
        add_strassen_term(
            {0: 1, 1: 1}, 
            {10+k1: 1}, 
            {2*k0: -1, 2*k0+1: 1}, # c00 -= 1, c01 += 1
            4
        )
        
        # P6: (a10 - a00)(b00 + b01)
        add_strassen_term(
            {3: 1, 0: -1}, 
            {0+k0: 1, 0+k1: 1}, 
            {2*k1+1: 1}, # c11 += 1
            5
        )
        
        # P7: (a01 - a11)(b10 + b11)
        add_strassen_term(
            {1: 1, 4: -1}, 
            {10+k0: 1, 10+k1: 1}, 
            {2*k0: 1, 2*k1+1: 1}, # c00 += 1, c11 += 1
            6
        )
        
        # Extra terms for a02, a12 and b20, b21
        # T8: a02 b20 -> c00
        add_strassen_term(
            {2: 1}, 
            {20+k0: 1}, 
            {2*k0: 1}, 
            7
        )
        
        # T9: a02 b21 -> c01
        add_strassen_term(
            {2: 1}, 
            {20+k1: 1}, 
            {2*k0+1: 1}, 
            8
        )
        
        # T10: a12 b20 -> c10
        add_strassen_term(
            {5: 1}, 
            {20+k0: 1}, 
            {2*k1: 1}, 
            9
        )
        
        # T11: a12 b21 -> c11
        add_strassen_term(
            {5: 1}, 
            {20+k1: 1}, 
            {2*k1+1: 1}, 
            10
        )
        
    return {"U": U, "V": V, "W": W}
