"""Matrix multiplication 3,3,8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (24, rank)
    # W index convention: W[k*n + i, r]

    U = []
    V = []
    W = []

    def add_term(u_idx, v_idx, w_idx, coeff=1):
        """Add a rank-1 tensor term."""
        # U vector: 1 at u_idx
        u_vec = [0] * (n * m)
        u_vec[u_idx] = coeff
        U.append(u_vec)

        # V vector: 1 at v_idx
        v_vec = [0] * (m * p)
        v_vec[v_idx] = 1
        V.append(v_vec)

        # W vector: 1 at w_idx
        w_vec = [0] * (p * n)
        w_vec[w_idx] = 1
        W.append(w_vec)

    # --- Strassen for the 2x2 top-left block ---
    # A top-left 2x2 block: A[0,0], A[0,1], A[1,0], A[1,1]
    # B top-left 2x2 block for each column pair k, k+1
    # We process columns in pairs: (0,1), (2,3), (4,5), (6,7)
    
    # For each pair of columns (2b, 2b+1), b=0,1,2,3
    for b in range(p // 2):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen's 7 products for 2x2 x 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For column pair (k0, k1):
        # B00 = B[0,k0], B01 = B[0,k1], B10 = B[1,k0], B11 = B[1,k1]
        
        # We need to express each Strassen product as a sum of rank-1 terms.
        # Each product like (A00+A11)(B00+B11) expands to 4 terms, but we can
        # keep them as separate rank-1 contributions to the output.
        
        # Actually, for tensor decomposition, each "multiplication" is a bilinear
        # form. The Strassen algorithm uses 7 bilinear forms instead of 8.
        # Each bilinear form is a product of a linear combination of A entries
        # and a linear combination of B entries.
        
        # Let's define the 7 bilinear forms for this column pair:
        
        # M1 = (A00 + A11) * (B00 + B11)
        # Contributes to C00 += M1, C01 += M1, C10 += M1, C11 += M1
        # But wait, that's not right. Strassen computes:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mj is a product of a linear combo of A's and a linear combo of B's.
        # In the tensor decomposition, each Mj corresponds to a set of rank-1 terms
        # where the U vector has the coefficients of the A linear combo, the V vector
        # has the coefficients of the B linear combo, and the W vector distributes
        # the result to the appropriate output entries.
        
        # For M1 = (A00 + A11)(B00 + B11):
        # U: A00 coeff=1, A11 coeff=1 -> indices 0 and 3
        # V: B00 coeff=1, B11 coeff=1 -> indices 0*8+k0 and 1*8+k1
        # W: contributes to C00 (i=0,k=k0), C01 (i=0,k=k1), C10 (i=1,k=k0), C11 (i=1,k=k1)
        # But each output entry gets the full M1 value, so we need 4 separate rank-1 terms
        # OR we can use a single term where W has 1s at all four positions.
        
        # In tensor decomposition, a single rank-1 term has U, V, W as vectors.
        # The contribution to C[i,k] is U[u_idx] * V[v_idx] * W[w_idx] for each (u,v,w)
        # with non-zero entries. So if U has two non-zero entries and V has two non-zero
        # entries, that's actually 4 rank-1 terms combined.
        
        # To keep it simple and correct, let's expand each bilinear form into
        # individual rank-1 terms.
        
        # M1 = (A00 + A11)(B00 + B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Each contributes to C00, C01, C10, C11 with coefficient +1
        
        # Term A00*B00:
        # U: idx 0 (A[0,0]), V: idx 0*8+k0 (B[0,k0]), W: idx k0*3+0 (C[0,k0])
        add_term(0, 0 * 8 + k0, k0 * 3 + 0, 1)
        add_term(0, 0 * 8 + k0, k0 * 3 + 1, 1)
        add_term(0, 0 * 8 + k0, k1 * 3 + 0, 1)
        add_term(0, 0 * 8 + k0, k1 * 3 + 1, 1)
        
        # Term A00*B11:
        add_term(0, 1 * 8 + k1, k0 * 3 + 0, 1)
        add_term(0, 1 * 8 + k1, k0 * 3 + 1, 1)
        add_term(0, 1 * 8 + k1, k1 * 3 + 0, 1)
        add_term(0, 1 * 8 + k1, k1 * 3 + 1, 1)
        
        # Term A11*B00:
        add_term(3, 0 * 8 + k0, k0 * 3 + 0, 1)
        add_term(3, 0 * 8 + k0, k0 * 3 + 1, 1)
        add_term(3, 0 * 8 + k0, k1 * 3 + 0, 1)
        add_term(3, 0 * 8 + k0, k1 * 3 + 1, 1)
        
        # Term A11*B11:
        add_term(3, 1 * 8 + k1, k0 * 3 + 0, 1)
        add_term(3, 1 * 8 + k1, k0 * 3 + 1, 1)
        add_term(3, 1 * 8 + k1, k1 * 3 + 0, 1)
        add_term(3, 1 * 8 + k1, k1 * 3 + 1, 1)
        
        # M2 = (A10 + A11) * B00
        # = A10*B00 + A11*B00
        # Contributes to C10 += M2, C11 -= M2
        
        # Term A10*B00:
        add_term(1, 0 * 8 + k0, k0 * 3 + 1, 1)
        add_term(1, 0 * 8 + k0, k1 * 3 + 1, -1)
        
        # Term A11*B00:
        add_term(3, 0 * 8 + k0, k0 * 3 + 1, 1)
        add_term(3, 0 * 8 + k0, k1 * 3 + 1, -1)
        
        # M3 = A00 * (B01 - B11)
        # = A00*B01 - A00*B11
        # Contributes to C01 += M3, C11 += M3
        
        # Term A00*B01:
        add_term(0, 0 * 8 + k1, k0 * 3 + 1, 1)
        add_term(0, 0 * 8 + k1, k1 * 3 + 1, 1)
        
        # Term -A00*B11:
        add_term(0, 1 * 8 + k1, k0 * 3 + 1, -1)
        add_term(0, 1 * 8 + k1, k1 * 3 + 1, -1)
        
        # M4 = A11 * (B10 - B00)
        # = A11*B10 - A11*B00
        # Contributes to C00 += M4, C10 += M4
        
        # Term A11*B10:
        add_term(3, 1 * 8 + k0, k0 * 3 + 0, 1)
        add_term(3, 1 * 8 + k0, k1 * 3 + 0, 1)
        
        # Term -A11*B00:
        add_term(3, 0 * 8 + k0, k0 * 3 + 0, -1)
        add_term(3, 0 * 8 + k0, k1 * 3 + 0, -1)
        
        # M5 = (A00 + A01) * B11
        # = A00*B11 + A01*B11
        # Contributes to C00 -= M5, C01 += M5
        
        # Term A00*B11:
        add_term(0, 1 * 8 + k1, k0 * 3 + 0, -1)
        add_term(0, 1 * 8 + k1, k1 * 3 + 0, 1)
        
        # Term A01*B11:
        add_term(2, 1 * 8 + k1, k0 * 3 + 0, -1)
        add_term(2, 1 * 8 + k1, k1 * 3 + 0, 1)
        
        # M6 = (A10 - A00) * (B00 + B01)
        # = (A10*B00 + A10*B01 - A00*B00 - A00*B01)
        # Contributes to C11 += M6
        
        # Term A10*B00:
        add_term(1, 0 * 8 + k0, k1 * 3 + 1, 1)
        
        # Term A10*B01:
        add_term(1, 0 * 8 + k1, k1 * 3 + 1, 1)
        
        # Term -A00*B00:
        add_term(0, 0 * 8 + k0, k1 * 3 + 1, -1)
        
        # Term -A00*B01:
        add_term(0, 0 * 8 + k1, k1 * 3 + 1, -1)
        
        # M7 = (A01 - A11) * (B10 + B11)
        # = (A01*B10 + A01*B11 - A11*B10 - A11*B11)
        # Contributes to C00 += M7
        
        # Term A01*B10:
        add_term(2, 1 * 8 + k0, k0 * 3 + 0, 1)
        
        # Term A01*B11:
        add_term(2, 1 * 8 + k1, k0 * 3 + 0, 1)
        
        # Term -A11*B10:
        add_term(3, 1 * 8 + k0, k0 * 3 + 0, -1)
        
        # Term -A11*B11:
        add_term(3, 1 * 8 + k1, k0 * 3 + 0, -1)
    
    # --- Cross terms for the 3rd row/column ---
    # A[2,0], A[2,1], A[2,2] and B[2,k] for k=0..7
    # Also A[0,2], A[1,2] and B[0,k], B[1,k]
    
    # C[i,k] includes:
    # For i=0: A[0,2]*B[2,k]
    # For i=1: A[1,2]*B[2,k]
    # For i=2: A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    
    for k in range(p):
        # A[0,2]*B[2,k] -> C[0,k]
        add_term(2, 2 * 8 + k, k * 3 + 0, 1)
        
        # A[1,2]*B[2,k] -> C[1,k]
        add_term(5, 2 * 8 + k, k * 3 + 1, 1)
        
        # A[2,0]*B[0,k] -> C[2,k]
        add_term(6, 0 * 8 + k, k * 3 + 2, 1)
        
        # A[2,1]*B[1,k] -> C[2,k]
        add_term(7, 1 * 8 + k, k * 3 + 2, 1)
        
        # A[2,2]*B[2,k] -> C[2,k]
        add_term(8, 2 * 8 + k, k * 3 + 2, 1)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
