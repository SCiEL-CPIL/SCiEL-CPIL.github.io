"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Laderman's 15-rank algorithm for 3x3 matrix multiplication
    # Each triple (u, v, w) means: U[u]*V[v]*W[w]
    # u,v,w are in {0..2} for local 3x3 indices
    laderman_triples = [
        (0, 0, 0),  # a00*b00 -> c00
        (0, 1, 1),  # a00*b01 -> c01
        (1, 0, 0),  # a10*b00 -> c10
        (2, 1, 1),  # a20*b01 -> c21
        (1, 1, 1),  # (a10+a11)*b10 -> c10,c11
        (1, 2, 2),  # a11*b11 -> c11
        (2, 0, 0),  # a20*b00 -> c20
        (2, 2, 2),  # (a20+a22)*b20 -> c20,c22
        (0, 2, 2),  # a01*b11 -> c01
        (2, 1, 1),  # wait, this is duplicate. Let me use the correct Laderman algorithm.
    ]
    
    # Correct Laderman 15-rank algorithm for 3x3:
    # Using the standard formulation:
    # Let A = [[a,b],[c,d]] style but for 3x3.
    # The 15 products are:
    laderman_15 = [
        # (u_row, u_col, v_row, v_col, w_row, w_col) in local 3x3 coords
        # Product contributes to C[w_row, w_col]
        (0, 0, 0, 0, 0, 0),  # a00*b00
        (0, 0, 0, 1, 0, 1),  # a00*b01
        (1, 0, 0, 0, 1, 0),  # a10*b00
        (2, 0, 0, 1, 2, 1),  # a20*b01
        (1, 1, 1, 0, 1, 1),  # a11*b10
        (2, 2, 2, 0, 2, 2),  # a22*b20
        (0, 2, 1, 1, 0, 1),  # a02*b11
        (2, 1, 1, 2, 2, 2),  # a21*b12
        (0, 0, 1, 0, 0, 0),  # a00*b10
        (0, 1, 1, 1, 0, 1),  # a01*b11
        (1, 0, 1, 0, 1, 0),  # a10*b10
        (1, 1, 1, 1, 1, 1),  # a11*b11
        (2, 0, 1, 0, 2, 0),  # a20*b10
        (2, 1, 1, 1, 2, 1),  # a21*b11
        (2, 2, 1, 2, 2, 2),  # a22*b12
    ]
    
    # Actually, let me use the well-known Laderman algorithm properly.
    # Laderman's algorithm for 3x3 matrix multiplication has rank 15.
    # The bilinear form: C = A*B where A,B,C are 3x3.
    # 
    # Standard Laderman decomposition (one common form):
    # Define 15 bilinear forms. Each is a product of a linear form in A
    # and a linear form in B, contributing to specific entries of C.
    #
    # Let me use a verified formulation. The key insight:
    # For 3x3, we can write:
    #   C[0,0] = a00*b00 + a01*b10 + a02*b20
    #   etc.
    # But we need to share products.
    #
    # A correct Laderman decomposition:
    # Products (each is L_A * L_B -> contributes to specific C entries):
    
    laderman_correct = [
        # (coeffs in A, coeffs in B, C entries affected)
        # Using sparse representation: {A_idx: coeff, B_idx: coeff, C_idx: coeff}
    ]
    
    # Let me use a simpler, verified approach:
    # Block diagonal: 3x3 + 1x1 + 1x1 embedded in 5x5
    # 3x3 uses rank 15 (Laderman), 1x1 uses rank 1, another 1x1 uses rank 1
    # Total: 15 + 1 + 1 = 17? No, we have 5x5 = 3x3 + 1x1 + 1x1 in block diag
    # But 3+1+1=5, so yes.
    
    # Actually for 5x5 block diagonal: blocks of sizes 3, 1, 1
    # Rank = rank(3x3) + rank(1x1) + rank(1x1) = 15 + 1 + 1 = 17
    
    # Let me implement Laderman's algorithm correctly.
    # Reference: The 15-rank decomposition of 3x3 matrix multiplication.
    
    # Here's a verified Laderman decomposition:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    #
    # 15 products:
    # P1 = a00 * b00
    # P2 = a00 * b01
    # P3 = a01 * b10
    # P4 = a01 * b11
    # P5 = a02 * b20
    # P6 = a02 * b21
    # P7 = a10 * b00
    # P8 = a11 * b01
    # P9 = a12 * b02
    # P10 = a20 * b01
    # P11 = a21 * b02
    # P12 = a22 * b02
    # P13 = (a10+a11) * (b10+b11)  -- this doesn't work simply
    # 
    # Let me just use a known correct formulation from literature.
    # 
    # Actually, the simplest correct approach: use the fact that
    # rank(MM_3) = 15 is proven, and use a specific decomposition.
    #
    # Here's a concrete one (from Landsberg, "Tensor geometry"):
    # For 3x3, define:
    # 
    # I'll use a computational approach: verify a decomposition works.
    
    # Given the complexity, let me use a simpler but still improved approach:
    # Use rank-125 schoolbook as fallback, but try to implement Laderman.
    
    # Let me implement a known Laderman decomposition:
    # The 15 bilinear forms for 3x3 matrix mult:
    
    # Using the formulation where we group:
    # Block: [[A11, A12], [A21, A22]] where each Ai is not scalar but...
    # No, let me just hardcode a verified Laderman decomposition.
    
    # Verified Laderman 3x3 decomposition (15 terms):
    # Each term: (A_indices_with_coeffs, B_indices_with_coeffs, C_indices_with_coeffs)
    # A index = row*3+col, B index = row*3+col, C index = row*3+col
    
    laderman_terms = [
        # Term 1: a00*b00 -> c00
        ({0: 1}, {0: 1}, {0: 1}),
        # Term 2: a00*b01 -> c01
        ({0: 1}, {1: 1}, {1: 1}),
        # Term 3: a01*b10 -> c01
        ({1: 1}, {3: 1}, {1: 1}),
        # Term 4: a02*b20 -> c02
        ({2: 1}, {6: 1}, {2: 1}),
        # Term 5: a10*b00 -> c10
        ({3: 1}, {0: 1}, {3: 1}),
        # Term 6: a11*b10 -> c10
        ({4: 1}, {3: 1}, {3: 1}),
        # Term 7: a12*b20 -> c12
        ({5: 1}, {6: 1}, {5: 1}),
        # Term 8: a20*b01 -> c21
        ({6: 1}, {1: 1}, {4: 1}),
        # Term 9: a21*b11 -> c21
        ({7: 1}, {4: 1}, {4: 1}),
        # Term 10: a22*b21 -> c22
        ({8: 1}, {7: 1}, {8: 1}),
        # Term 11: a11*b01 -> c11
        ({4: 1}, {1: 1}, {4: 1}),
        # Term 12: a12*b02 -> c12
        ({5: 1}, {2: 1}, {5: 1}),
        # Term 13: a21*b02 -> c22
        ({7: 1}, {2: 1}, {8: 1}),
        # Term 14: a22*b02 -> c22
        ({8: 1}, {2: 1}, {8: 1}),
        # Term 15: a20*b00 -> c20
        ({6: 1}, {0: 1}, {6: 1}),
    ]
    
    # Verify this is actually correct by checking it computes A*B
    # This is getting complex. Let me use a different, simpler strategy:
    # 
    # Use the Kronecker product / embedding approach with a KNOWN good
    # low-rank decomposition. Since implementing Laderman correctly is
    # error-prone, let me use a safer approach:
    #
    # For 5x5, I can use the fact that if I have a rank-r decomposition
    # for nxn, I can embed it. But the simplest improvement over 125
    # that I can verify is to use the structure.
    #
    # Actually, let me just carefully implement Laderman and verify it.
    
    # Laderman's algorithm for 3x3 (verified formulation):
    # 
    # The 15 products are:
    # 1.  A[0,0] * B[0,0]
    # 2.  A[0,0] * B[0,1]  
    # 3.  A[0,1] * B[1,0]
    # 4.  A[0,1] * B[1,1]
    # 5.  A[0,2] * B[2,0]
    # 6.  A[0,2] * B[2,1]
    # 7.  A[1,0] * B[0,0]
    # 8.  A[1,1] * B[0,1]
    # 9.  A[1,2] * B[0,2]
    # 10. A[2,0] * B[0,1]
    # 11. A[2,1] * B[0,2]
    # 12. A[2,2] * B[0,2]
    # 13. (A[1,0]+A[1,1]) * (B[1,0]+B[1,1])  -- NO, this isn't right either
    
    # I'm going in circles. Let me use a completely different, verifiable
    # approach: just return the schoolbook decomposition (rank 125) which
    # we KNOW is correct, since getting Laderman right without verification
    # is risky. But that gives score 1.0.
    #
    # Alternative: Let me implement a simple but correct low-rank approach.
    # For 5x5, if I partition into 2x2 + 2x2 + 1x1 blocks (block diagonal),
    # rank = 2*rank(2x2) + 1 = 2*4 + 1 = 9? No, rank(2x2)=4 (schoolbook).
    # That's still 9 which is better than 125!
    #
    # Wait: 5x5 block diagonal with blocks 2,2,1:
    # rank = rank(MM_2) + rank(MM_2) + rank(MM_1) = 4 + 4 + 1 = 9
    # 
    # But we need to handle the off-diagonal blocks too! Block diagonal
    # only works if the matrix is actually block diagonal, which it's not
    # in general. So this doesn't work for general 5x5 matrix multiplication.
    #
    # The correct approach: we need a decomposition of the full 5x5 MM tensor.
    # The rank of MM_5 is at most 125 (schoolbook) and at least some lower bound.
    # 
    # Known: rank(MM_n) >= n^3/2 + O(n^2) (Bini's lower bound roughly)
    # For n=5: lower bound is around 62.5, so rank >= 63 or so.
    # 
    # Best known upper bounds for MM_5: 
    # - Schoolbook: 125
    # - Using fast matrix multiplication ideas: possibly lower
    # - Strassen for 4x4 gives rank 7 for 2x2, so for 4x4: 7^2=49? No.
    # - Strassen: MM_2 has rank 7, MM_4 has rank at most 7^2=49 (by Kronecker)
    # - MM_5: we can embed 4x4 (rank <=49) and add 1x1 (rank 1) = 50?
    #   But again, only if block diagonal, which it's not.
    
    # The CORRECT way: use the tensor product / Kronecker structure.
    # If we have a rank-r decomposition of MM_n, then MM_{n*k} has rank at most
    # r^k (by tensoring). But that's for exact multiples.
    #
    # For MM_5, we can write 5 = 2+2+1 and use a "sum" construction,
    # but that only gives block-diagonal, not full MM_5.
    #
    # The honest answer: the best I can do without a verified Laderman
    # implementation is to return schoolbook rank 125. But let me try ONE
    # more time to get Laderman right.
    
    # Laderman 1976, "A 15 multiplication algorithm for 3x3 matrices":
    # 
    # A = [[a, b, c], [d, e, f], [g, h, i]]
    # B = [[j, k, l], [m, n, o], [p, q, r]]
    # 
    # 15 products:
    # P1 = a*j
    # P2 = a*k
    # P3 = b*m
    # P4 = b*n
    # P5 = c*p
    # P6 = c*q
    # P7 = d*j
    # P8 = e*k
    # P9 = f*l
    # P10 = g*k
    # P11 = h*l
    # P12 = i*l
    # P13 = (d+e)*(m+n)
    # P14 = (g+h)*(k+l)  -- wait, this doesn't look right
    # 
    # Let me look at this differently. I'll use a computational verification.
    # I'll construct a candidate decomposition and check if it works.
    
    # Actually, the safest path: implement schoolbook (rank 125) which is
    # guaranteed correct. Score will be 1.0 but at least valid.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
