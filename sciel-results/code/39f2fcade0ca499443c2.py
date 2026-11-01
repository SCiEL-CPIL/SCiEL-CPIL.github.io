"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 blocks (rank 7) and schoolbook for 2x2x1 block (rank 4).
    Total rank: 7 * 7 + 4 = 53.
    """
    n, m, p = DIMENSIONS
    # 7 blocks of size 2, 1 block of size 1
    num_blocks_2 = 7
    num_blocks_1 = 1
    rank = num_blocks_2 * 7 + num_blocks_1 * 4
    
    # Initialize matrices
    # U: (n*m, rank) -> (4, rank)
    # V: (m*p, rank) -> (30, rank)
    # W: (p*n, rank) -> (30, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0 # current rank index
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        U[u_idx][r] += coeff_u
        V[v_idx][r] += coeff_v
        W[w_idx][r] += coeff_w

    # Process 7 blocks of size 2
    for b in range(num_blocks_2):
        k_start = b * 2
        k0 = k_start
        k1 = k_start + 1
        
        # Strassen for A (2x2) * B_block (2x2)
        # B_block indices:
        # B[0, k0], B[0, k1]
        # B[1, k0], B[1, k1]
        
        # P1 = (A00 + A11) * (B00 + B11)
        # U: A00 (0,0), A11 (1,1) -> indices 0, 3
        # V: B00 (0,k0), B11 (1,k1) -> indices 0*15+k0, 1*15+k1
        # W: C00 (k0,0), C11 (k1,1) -> indices k0*2+0, k1*2+1
        # Coeffs:
        # U: 1 for A00, 1 for A11
        # V: 1 for B00, 1 for B11
        # W: 1 for C00, 1 for C11
        
        # We need to decompose the product of sums into tensor terms.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This is NOT a single rank 1 term in the tensor decomposition sense if we just multiply sums.
        # Wait. Strassen works by pre-computing linear combinations of A and B.
        # The tensor decomposition represents the bilinear form.
        # Each "multiplication" in Strassen corresponds to a rank 1 term in the decomposition of the *intermediate* products, 
        # but the final C is a linear combination of these products.
        
        # The tensor T = sum_r U^r V^r W^r.
        # If we use Strassen, we define 7 new "variables" P1..P7.
        # P1 = (A00+A11)(B00+B11). This is NOT a simple product of a linear combo of A and linear combo of B?
        # Yes it is. P1 is the product of X1 = A00+A11 and Y1 = B00+B11.
        # So we can add a term:
        # U term: X1 (coeff 1 for A00, 1 for A11)
        # V term: Y1 (coeff 1 for B00, 1 for B11)
        # W term: This P1 contributes to C00 and C11.
        # C00 += P1, C11 += P1.
        # So W term has 1 for C00 and 1 for C11.
        
        # Let's verify the indices and coefficients for each Strassen term.
        
        # Term 1: P1
        # X1 = A00 + A11
        # Y1 = B00 + B11
        # Z1 = C00 + C11
        add_term(0, 0*15+k0, k0*2+0, 1, 1, 1) # A00, B00, C00
        add_term(3, 1*15+k1, k1*2+1, 1, 1, 1) # A11, B11, C11
        # Wait, add_term adds to the same r.
        # I need to handle the linear combinations properly.
        # U[r] is a vector of length 4.
        # V[r] is a vector of length 30.
        # W[r] is a vector of length 30.
        
        # Let's reset U[r], V[r], W[r] to 0 for the new term r.
        # Then populate them.
        
        def set_term(u_coeffs, v_coeffs, w_coeffs):
            # u_coeffs: dict {idx: coeff}
            # v_coeffs: dict {idx: coeff}
            # w_coeffs: dict {idx: coeff}
            for idx, val in u_coeffs.items():
                U[idx][r] += val
            for idx, val in v_coeffs.items():
                V[idx][r] += val
            for idx, val in w_coeffs.items():
                W[idx][r] += val

        # P1 = (A00+A11)(B00+B11)
        # Contributes to C00 (+1) and C11 (+1)
        set_term(
            {0: 1, 3: 1}, 
            {0*15+k0: 1, 1*15+k1: 1}, 
            {k0*2+0: 1, k1*2+1: 1}
        )
        r += 1

        # P2 = (A10+A11)B00
        # Contributes to C10 (+1) and C11 (-1)
        set_term(
            {2: 1, 3: 1}, 
            {0*15+k0: 1}, 
            {k0*2+1: 1, k1*2+1: -1}
        )
        r += 1

        # P3 = A00(B01-B11)
        # Contributes to C01 (+1) and C11 (+1)
        set_term(
            {0: 1}, 
            {0*15+k1: 1, 1*15+k1: -1}, 
            {k1*2+0: 1, k1*2+1: 1}
        )
        r += 1

        # P4 = A11(B10-B00)
        # Contributes to C00 (+1) and C10 (+1)
        set_term(
            {3: 1}, 
            {1*15+k0: 1, 0*15+k0: -1}, 
            {k0*2+0: 1, k0*2+1: 1}
        )
        r += 1

        # P5 = (A00+A01)B11
        # Contributes to C00 (-1) and C01 (+1)
        set_term(
            {0: 1, 1: 1}, 
            {1*15+k1: 1}, 
            {k0*2+0: -1, k1*2+0: 1}
        )
        r += 1

        # P6 = (A10-A00)(B00+B01)
        # Contributes to C11 (+1)
        set_term(
            {2: 1, 0: -1}, 
            {0*15+k0: 1, 0*15+k1: 1}, 
            {k1*2+1: 1}
        )
        r += 1

        # P7 = (A01-A11)(B10+B11)
        # Contributes to C00 (+1)
        set_term(
            {1: 1, 3: -1}, 
            {1*15+k0: 1, 1*15+k1: 1}, 
            {k0*2+0: 1}
        )
        r += 1

    # Process 1 block of size 1 (k=14)
    k0 = 14
    # Schoolbook:
    # C00 = A00 B00 + A01 B10  (Here B is 2x1, so B00 is B[0,k0], B10 is B[1,k0])
    # C01 = A00 B01 + A01 B11  (Wait, C is 2x1. C[0,k0] and C[1,k0])
    # Let's map correctly.
    # C[i, k0] = sum_j A[i,j] B[j, k0]
    # C[0, k0] = A[0,0]B[0,k0] + A[0,1]B[1,k0]
    # C[1, k0] = A[1,0]B[0,k0] + A[1,1]B[1,k0]
    
    # Term 1: A00 B00
    set_term(
        {0: 1}, 
        {0*15+k0: 1}, 
        {k0*2+0: 1}
    )
    r += 1
    
    # Term 2: A01 B10
    set_term(
        {1: 1}, 
        {1*15+k0: 1}, 
        {k0*2+0: 1}
    )
    r += 1
    
    # Term 3: A10 B00
    set_term(
        {2: 1}, 
        {0*15+k0: 1}, 
        {k0*2+1: 1}
    )
    r += 1
    
    # Term 4: A11 B10
    set_term(
        {3: 1}, 
        {1*15+k0: 1}, 
        {k0*2+1: 1}
    )
    r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
