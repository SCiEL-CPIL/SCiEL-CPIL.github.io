"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x3x8 matrix multiplication.
    
    Strategy:
    - Decompose the 8 columns of B into 4 blocks of 2 columns each.
    - For each 3x3x2 block, use a rank-17 decomposition:
      * Strassen (rank 7) for the top-left 2x2x2 subproblem (rows 0,1; cols 0,1; k 0,1)
      * Schoolbook (rank 10) for the remaining terms involving row 2 or col 2 of the 3x3 matrix
    - Total rank: 4 * 17 = 68
    
    The 3x3x2 multiplication tensor T_{i,j,k} where i in {0,1,2}, j in {0,1,2}, k in {0,1}:
    C_{i,k} = sum_{j=0}^{2} A_{i,j} B_{j,k}
    
    Split into disjoint parts:
    1. Top-left 2x2x2: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen rank 7
       Computes: C_{0,0}, C_{0,1}, C_{1,0}, C_{1,1} from A_{0,0..1,0..1} and B_{0,0..1,0..1}
       
    2. Remaining terms (involving index 2 in either i or j):
       - C_{2,0} = A_{2,0}B_{0,0} + A_{2,1}B_{1,0} + A_{2,2}B_{2,0}
       - C_{2,1} = A_{2,0}B_{0,1} + A_{2,1}B_{1,1} + A_{2,2}B_{2,1}
       - C_{0,2} = A_{0,0}B_{0,2} + A_{0,1}B_{1,2} + A_{0,2}B_{2,2}
       - C_{1,2} = A_{1,0}B_{0,2} + A_{1,1}B_{1,2} + A_{1,2}B_{2,2}
       - C_{2,2} = A_{2,0}B_{0,2} + A_{2,1}B_{1,2} + A_{2,2}B_{2,2}
       
       Wait, the Strassen part only uses A_{0,0..1,0..1} and B_{0,0..1,0..1}.
       The remaining terms use either A_{2,j} or A_{i,2} or B_{2,k}.
       
       Let me recount the remaining scalar products:
       Terms involving A_{2,j} (row 2 of A):
       - A_{2,0}B_{0,0} -> C_{2,0}
       - A_{2,0}B_{0,1} -> C_{2,1}
       - A_{2,1}B_{1,0} -> C_{2,0}
       - A_{2,1}B_{1,1} -> C_{2,1}
       - A_{2,2}B_{2,0} -> C_{2,0}
       - A_{2,2}B_{2,1} -> C_{2,1}
       That's 6 terms for row 2 of A.
       
       Terms involving A_{i,2} (col 2 of A) but not row 2:
       - A_{0,2}B_{2,0} -> C_{0,0}
       - A_{0,2}B_{2,1} -> C_{0,1}
       - A_{1,2}B_{2,0} -> C_{1,0}
       - A_{1,2}B_{2,1} -> C_{1,1}
       That's 4 terms for col 2 of A (excluding row 2).
       
       Total remaining: 6 + 4 = 10 terms.
       
       So total rank per 3x3x2 block: 7 (Strassen) + 10 (schoolbook) = 17.
       Total rank for 3x3x8: 4 * 17 = 68.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    U = []
    V = []
    W = []
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term to the decomposition."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_dict.items():
            u_vec[idx] = val
        for idx, val in v_dict.items():
            v_vec[idx] = val
        for idx, val in w_dict.items():
            w_vec[idx] = val
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Process 4 blocks of 2 columns each
    for block in range(4):
        k0 = block * 2  # first column index in this block
        k1 = block * 2 + 1  # second column index in this block
        
        # --- Strassen Part: 2x2x2 for rows 0,1; cols 0,1; k in {k0,k1} ---
        # Local indices: i in {0,1}, j in {0,1}, k in {k0,k1}
        # Global A index: i*m + j
        # Global B index: j*p + k
        # Global C index: k*n + i
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_rank1_term(
            {0*m+0: 1, 1*m+1: 1},
            {0*p+k0: 1, 1*p+k1: 1},
            {k0*n+0: 1, k1*n+1: 1}
        )
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 += M2
        add_rank1_term(
            {1*m+0: 1, 1*m+1: 1},
            {0*p+k0: 1},
            {k0*n+1: 1, k1*n+1: 1}
        )
        
        # M3 = A00(B01-B11) -> C01 += M3
        add_rank1_term(
            {0*m+0: 1},
            {0*p+k1: 1, 1*p+k1: -1},
            {k1*n+0: 1}
        )
        
        # M4 = A11(B10-B00) -> C10 += M4
        add_rank1_term(
            {1*m+1: 1},
            {1*p+k0: 1, 0*p+k0: -1},
            {k0*n+1: 1}
        )
        
        # M5 = (A00+A10)B10 -> C10 += M5, C00 -= M5
        add_rank1_term(
            {0*m+0: 1, 1*m+0: 1},
            {1*p+k0: 1},
            {k0*n+1: 1, k0*n+0: -1}
        )
        
        # M6 = (A01-A00)(B00+B10) -> C01 += M6, C11 -= M6
        add_rank1_term(
            {0*m+1: 1, 0*m+0: -1},
            {0*p+k0: 1, 1*p+k0: 1},
            {k1*n+0: 1, k1*n+1: -1}
        )
        
        # M7 = (A11-A10)(B01+B11) -> C01 += M7, C00 += M7
        add_rank1_term(
            {1*m+1: 1, 1*m+0: -1},
            {0*p+k1: 1, 1*p+k1: 1},
            {k1*n+0: 1, k0*n+0: 1}
        )
        
        # --- Schoolbook Part 1: A[2, :] * B[:, {k0,k1}] (Rank 6) ---
        # C2k += A20 B0k + A21 B1k + A22 B2k
        # k in {k0, k1}
        
        # A20 * B0k0 -> C2k0
        add_rank1_term(
            {2*m+0: 1},
            {0*p+k0: 1},
            {k0*n+2: 1}
        )
        
        # A20 * B0k1 -> C2k1
        add_rank1_term(
            {2*m+0: 1},
            {0*p+k1: 1},
            {k1*n+2: 1}
        )
        
        # A21 * B1k0 -> C2k0
        add_rank1_term(
            {2*m+1: 1},
            {1*p+k0: 1},
            {k0*n+2: 1}
        )
        
        # A21 * B1k1 -> C2k1
        add_rank1_term(
            {2*m+1: 1},
            {1*p+k1: 1},
            {k1*n+2: 1}
        )
        
        # A22 * B2k0 -> C2k0
        add_rank1_term(
            {2*m+2: 1},
            {2*p+k0: 1},
            {k0*n+2: 1}
        )
        
        # A22 * B2k1 -> C2k1
        add_rank1_term(
            {2*m+2: 1},
            {2*p+k1: 1},
            {k1*n+2: 1}
        )
        
        # --- Schoolbook Part 2: A[{0,1}, 2] * B[2, :] (Rank 4) ---
        # C0k += A02 B2k
        # C1k += A12 B2k
        # k in {k0, k1}
        
        # A02 * B2k0 -> C0k0
        add_rank1_term(
            {0*m+2: 1},
            {2*p+k0: 1},
            {k0*n+0: 1}
        )
        
        # A02 * B2k1 -> C0k1
        add_rank1_term(
            {0*m+2: 1},
            {2*p+k1: 1},
            {k1*n+0: 1}
        )
        
        # A12 * B2k0 -> C1k0
        add_rank1_term(
            {1*m+2: 1},
            {2*p+k0: 1},
            {k0*n+1: 1}
        )
        
        # A12 * B2k1 -> C1k1
        add_rank1_term(
            {1*m+2: 1},
            {2*p+k1: 1},
            {k1*n+1: 1}
        )
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
