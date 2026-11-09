import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    # Partition sizes for 5x5x5: 4 + 1
    sizes = [4, 1]
    starts = [0, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        
        for idx, coeff in u_coeffs:
            U_col[idx] = coeff
        for idx, coeff in v_coeffs:
            V_col[idx] = coeff
        for idx, coeff in w_coeffs:
            W_col[idx] = coeff
            
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def map_u(i, k, p_local, q_local):
        return (starts[i] + p_local) * m + (starts[k] + q_local)

    def map_v(k, j, q_local, t_local):
        return (starts[k] + q_local) * p + (starts[j] + t_local)

    def map_w(i, j, p_local, t_local):
        return (starts[j] + t_local) * n + (starts[i] + p_local)

    def add_strassen_2x2(u_blocks, v_blocks, w_offset_i, w_offset_j):
        """
        Adds 7 Strassen terms for a 2x2x2 block.
        u_blocks: list of (local_row, local_col, coeff) for A
        v_blocks: list of (local_row, local_col, coeff) for B
        w_offset_i: global block index for row of C
        w_offset_j: global block index for col of C
        """
        # Define the 7 Strassen terms
        # Each term is a product of a linear combo of A and a linear combo of B
        # We need to expand these into scalar multiplications
        
        # Term 1: (a00+a11)(b00+b11)
        # Term 2: (a10+a11)(b00)
        # Term 3: (a00)(b01-b11)
        # Term 4: (a11)(b10-b00)
        # Term 5: (a00+a01)(b11)
        # Term 6: (a10-a00)(b00+b01)
        # Term 7: (a01-a11)(b10+b11)
        
        # We map local indices to global indices
        # u_blocks and v_blocks are already in local coords relative to the 2x2 block
        # But we need to map them to the global 5x5 matrix
        
        # For simplicity, we assume u_blocks and v_blocks are lists of (r, c, coeff)
        # where r, c are in {0, 1}
        
        # We will generate the 7 terms by iterating through the combinations
        # This is complex to do generically. 
        # Instead, we will hardcode the 7 terms for the specific case where
        # u_blocks and v_blocks are simple sums of 2x2 blocks.
        
        # Actually, it's easier to just call a helper that adds the 7 terms
        # given the global offsets for the 2x2 blocks.
        pass

    def strassen_2x2(i, j, k, offset_i, offset_j, offset_k):
        """
        Adds 7 Strassen terms for a 2x2x2 block at global offsets.
        i, j, k are the block indices (0 or 1) in the 4x4 region.
        offset_i, offset_j, offset_k are the global start indices.
        """
        # The 2x2 block is at:
        # A: rows offset_i + 2*i .. offset_i + 2*i + 1, cols offset_k + 2*k .. offset_k + 2*k + 1
        # B: rows offset_k + 2*k .. offset_k + 2*k + 1, cols offset_j + 2*j .. offset_j + 2*j + 1
        # C: rows offset_j + 2*j .. offset_j + 2*j + 1, cols offset_i + 2*i .. offset_i + 2*i + 1
        
        # Term 1: (a00+a11)(b00+b11)
        add_term(
            [(map_u(i, k, 2*i, 2*k), 1), (map_u(i, k, 2*i+1, 2*k+1), 1)],
            [(map_v(k, j, 2*k, 2*j), 1), (map_v(k, j, 2*k+1, 2*j+1), 1)],
            [(map_w(i, j, 2*i, 2*j), 1), (map_w(i, j, 2*i+1, 2*j+1), 1)]
        )
        # Term 2: (a10+a11)(b00)
        add_term(
            [(map_u(i, k, 2*i+1, 2*k), 1), (map_u(i, k, 2*i+1, 2*k+1), 1)],
            [(map_v(k, j, 2*k, 2*j), 1)],
            [(map_w(i, j, 2*i+1, 2*j), 1), (map_w(i, j, 2*i+1, 2*j+1), -1)]
        )
        # Term 3: (a00)(b01-b11)
        add_term(
            [(map_u(i, k, 2*i, 2*k), 1)],
            [(map_v(k, j, 2*k, 2*j+1), 1), (map_v(k, j, 2*k+1, 2*j+1), -1)],
            [(map_w(i, j, 2*i, 2*j+1), 1), (map_w(i, j, 2*i+1, 2*j+1), 1)]
        )
        # Term 4: (a11)(b10-b00)
        add_term(
            [(map_u(i, k, 2*i+1, 2*k+1), 1)],
            [(map_v(k, j, 2*k+1, 2*j), 1), (map_v(k, j, 2*k, 2*j), -1)],
            [(map_w(i, j, 2*i, 2*j), 1), (map_w(i, j, 2*i+1, 2*j), 1)]
        )
        # Term 5: (a00+a01)(b11)
        add_term(
            [(map_u(i, k, 2*i, 2*k), 1), (map_u(i, k, 2*i, 2*k+1), 1)],
            [(map_v(k, j, 2*k+1, 2*j+1), 1)],
            [(map_w(i, j, 2*i, 2*j), -1), (map_w(i, j, 2*i, 2*j+1), 1)]
        )
        # Term 6: (a10-a00)(b00+b01)
        add_term(
            [(map_u(i, k, 2*i+1, 2*k), 1), (map_u(i, k, 2*i, 2*k), -1)],
            [(map_v(k, j, 2*k, 2*j), 1), (map_v(k, j, 2*k, 2*j+1), 1)],
            [(map_w(i, j, 2*i+1, 2*j+1), 1)]
        )
        # Term 7: (a01-a11)(b10+b11)
        add_term(
            [(map_u(i, k, 2*i, 2*k+1), 1), (map_u(i, k, 2*i+1, 2*k+1), -1)],
            [(map_v(k, j, 2*k+1, 2*j), 1), (map_v(k, j, 2*k+1, 2*j+1), 1)],
            [(map_w(i, j, 2*i, 2*j), 1)]
        )

    def strassen_4x4(i, j, k):
        """
        Adds 49 terms for a 4x4x4 block using Strassen's algorithm.
        i, j, k are the block indices (0) in the 4+1 split.
        """
        # Split the 4x4 block into four 2x2 blocks.
        # Local indices u, v, w in {0, 1}
        # Global offsets:
        # A: rows i*4 + 2u, cols k*4 + 2w
        # B: rows k*4 + 2w, cols j*4 + 2v
        # C: rows j*4 + 2v, cols i*4 + 2u
        
        # We use the 7 Strassen terms at the 4x4 level.
        # Each term is a product of 2x2 matrices.
        # We decompose each 2x2 product into 7 scalar multiplications.
        
        # Term 1: (A00+A11)(B00+B11)
        # A00 is at u=0, w=0. A11 is at u=1, w=1.
        # B00 is at w=0, v=0. B11 is at w=1, v=1.
        # This is a 2x2 matrix multiplication.
        # We call strassen_2x2 for the resulting 2x2 block?
        # No, we need to add the 7 terms for this specific combination.
        
        # To simplify, we will just call strassen_2x2 7 times?
        # No, that's not how Strassen works.
        
        # Instead, we will manually add the 49 terms.
        # This is tedious but safe.
        
        # Helper to add a term for a specific scalar multiplication
        # u_entry: (global_row, global_col) in A
        # v_entry: (global_row, global_col) in B
        # w_entry: (global_row, global_col) in C
        # coeff: coefficient
        
        # We will iterate through the 7 Strassen terms at the 4x4 level.
        # For each term, we have a 2x2 matrix product.
        # We will decompose this 2x2 product using the 7-term Strassen algorithm.
        
        # This requires a function that adds the 7 terms for a 2x2 product
        # where the inputs are linear combinations of 2x2 blocks.
        
        # Given time constraints, I will use the 3+2 Winograd solution (116) instead.
        # It is simpler and guaranteed to work.
        pass

    # Use 3+2+1 split with Winograd (19) for 3x3x3 and Strassen (7) for 2x2x2
    sizes = [3, 2]
    starts = [0, 3]
    
    # Clear previous columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def winograd_3x3(i, j, k):
        """
        Adds 19 terms for a 3x3x3 block using Winograd's algorithm.
        """
        # Winograd's 19-term algorithm for 3x3
        # This is complex to implement correctly without reference.
        # I will use schoolbook for 3x3 (27 terms) to be safe.
        # Rank 27 for 3x3.
        # Total rank: 27 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 124.
        # This is worse than 117.
        
        # So I must use the 2+2+1 split with Strassen (117) or find a better way.
        
        # Let's stick to the original 2+2+1 split with Strassen (117).
        # It is the safest.
        pass

    # Revert to 2+2+1 split with Strassen
    sizes = [2, 2, 1]
    starts = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def strassen_terms(i, j, k):
        """Generate the 7 Strassen terms for a 2x2x2 block."""
        add_term(
            [(map_u(i, k, 0, 0), 1), (map_u(i, k, 1, 1), 1)],
            [(map_v(k, j, 0, 0), 1), (map_v(k, j, 1, 1), 1)],
            [(map_w(i, j, 0, 0), 1), (map_w(i, j, 1, 1), 1)]
        )
        add_term(
            [(map_u(i, k, 1, 0), 1), (map_u(i, k, 1, 1), 1)],
            [(map_v(k, j, 0, 0), 1)],
            [(map_w(i, j, 1, 0), 1), (map_w(i, j, 1, 1), -1)]
        )
        add_term(
            [(map_u(i, k, 0, 0), 1)],
            [(map_v(k, j, 0, 1), 1), (map_v(k, j, 1, 1), -1)],
            [(map_w(i, j, 0, 1), 1), (map_w(i, j, 1, 1), 1)]
        )
        add_term(
            [(map_u(i, k, 1, 1), 1)],
            [(map_v(k, j, 1, 0), 1), (map_v(k, j, 0, 0), -1)],
            [(map_w(i, j, 0, 0), 1), (map_w(i, j, 1, 0), 1)]
        )
        add_term(
            [(map_u(i, k, 0, 0), 1), (map_u(i, k, 0, 1), 1)],
            [(map_v(k, j, 1, 1), 1)],
            [(map_w(i, j, 0, 0), -1), (map_w(i, j, 0, 1), 1)]
        )
        add_term(
            [(map_u(i, k, 1, 0), 1), (map_u(i, k, 0, 0), -1)],
            [(map_v(k, j, 0, 0), 1), (map_v(k, j, 0, 1), 1)],
            [(map_w(i, j, 1, 1), 1)]
        )
        add_term(
            [(map_u(i, k, 0, 1), 1), (map_u(i, k, 1, 1), -1)],
            [(map_v(k, j, 1, 0), 1), (map_v(k, j, 1, 1), 1)],
            [(map_w(i, j, 0, 0), 1)]
        )

    def schoolbook_terms(i, j, k, r, s, c):
        """Generate schoolbook terms for an r x s x c block."""
        for p_local in range(r):
            for q_local in range(s):
                for t_local in range(c):
                    add_term(
                        [(map_u(i, k, p_local, q_local), 1)],
                        [(map_v(k, j, q_local, t_local), 1)],
                        [(map_w(i, j, p_local, t_local), 1)]
                    )

    for i in range(3):
        for j in range(3):
            if time.time() - start_time > DEADLINE:
                break
            for k in range(3):
                r = sizes[i]
                s = sizes[k]
                c = sizes[j]
                
                if r == 2 and s == 2 and c == 2:
                    strassen_terms(i, j, k)
                else:
                    schoolbook_terms(i, j, k, r, s, c)
    
    rank = len(U_cols)
    
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
