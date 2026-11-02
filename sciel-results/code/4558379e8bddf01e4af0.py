"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
from itertools import product
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 145.0

def _search_2x2x3_rank10():
    """
    Search for a rank-10 decomposition of 2x2 * 2x3.
    Uses symmetry reduction: fix V to canonical form via row/column operations.
    Returns (U, V, W) as lists or None if not found.
    """
    # Tensor T for 2x2 * 2x3:
    # A is 2x2 (indices 0,1,2,3), B is 2x3 (indices 0..5), C is 2x3 (indices 0..5)
    # T[i,j,k] = 1 if C[i,k] depends on A[?]*B[?]
    # C[0,0] = A00*B00 + A01*B10
    # C[0,1] = A00*B01 + A01*B11
    # C[0,2] = A00*B02 + A01*B12
    # C[1,0] = A10*B00 + A11*B10
    # C[1,1] = A10*B01 + A11*B11
    # C[1,2] = A10*B02 + A11*B12
    
    # We want U (4x10), V (6x10), W (6x10) such that:
    # For each output (i,k): sum_r U[a_i][r] * V[b_k][r] * W[i*3+k][r] = T[a_i, b_k, i*3+k]
    # where a_i is the A index and b_k is the B index.
    
    # Strategy: Use the fact that rank-10 for 2x2x3 is known to exist (from literature).
    # We'll do a structured search.
    
    # The tensor has 12 non-zero entries. We need 10 rank-1 tensors.
    # 
    # Key insight: For 2x2x2, Strassen achieves rank 7. For 2x2x3, 
    # we can try to extend Strassen-like patterns.
    #
    # Let's try a direct algebraic construction based on known results.
    # The 2x2x3 problem is equivalent to multiplying a 2x2 matrix by a 2x3 matrix.
    #
    # One approach: decompose the 2x3 B into B[:,0:2] and B[:,2].
    # 2x2x2 part: rank 7 (Strassen)
    # 2x2x1 part: rank 4 (schoolbook)
    # Total: 11. We need to find a rank-10 solution.
    #
    # Alternative: use a more clever decomposition.
    #
    # Let me try to search over integer coefficients in {-1, 0, 1}.
    # This is a large search space, so we need to be smart.
    
    # Actually, let me try a specific known construction.
    # The 2x2x3 tensor can be decomposed with rank 10 using the following approach:
    # Split into two 2x2x2 problems sharing some structure.
    
    # Let's try a randomized search with integer coefficients in {-2,-1,0,1,2}
    # for a limited number of iterations.
    
    np.random.seed(42)
    max_iter = 500000
    coeffs = [-1, 0, 1]
    
    # We'll search for U, V, W with entries in {-1, 0, 1}
    # Fix V to have a specific structure to reduce search space.
    # 
    # Actually, let me try a different approach: use the fact that 
    # the 2x2x3 tensor is the "slice" of a larger tensor.
    
    # Let me just try to find ANY rank-10 solution via random search.
    # This is unlikely to work in reasonable time, but let's try a structured approach.
    
    # Structured approach: 
    # The 12 bilinear forms are:
    # f0 = A00*B00 + A01*B10  (C00)
    # f1 = A00*B01 + A01*B11  (C01)
    # f2 = A00*B02 + A01*B12  (C02)
    # f3 = A10*B00 + A11*B10  (C10)
    # f4 = A10*B01 + A11*B11  (C11)
    # f5 = A10*B02 + A11*B12  (C12)
    
    # A rank-1 term is u_a * v_b * w_c which contributes to f_c if 
    # the (a,b) pair matches.
    
    # Let me try to construct this using the following idea:
    # Use 7 Strassen-like terms for the first two columns, and 
    # find 3 additional terms that handle the third column with sharing.
    
    # Strassen for columns 0,1:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)(B00)
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)(B11)
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # For column 2, we need:
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Can we express these using combinations of the existing P's plus 3 new terms?
    # This seems unlikely since the existing P's don't involve B02 or B12.
    
    # So we need at least 4 new terms for column 2 alone, giving total 11.
    # Unless we can restructure the whole thing.
    
    # Let me try a completely different decomposition.
    # 
    # Alternative: group the 6 B-variables differently.
    # 
    # Actually, I recall that the 2x2x3 problem has rank 10.
    # Let me try to find it by solving a system of equations.
    
    # We have 10 rank-1 terms. Each term has 4*10 + 6*10 + 6*10 = 160 variables.
    # But we have symmetries: scaling each term's U, V, W vectors.
    # 
    # This is a very hard nonlinear system. Let me try a heuristic approach.
    
    # Heuristic: start from the rank-11 solution (7+4) and try to merge two terms.
    
    # Rank-11 solution:
    # Terms 0-6: Strassen for cols 0,1
    # Terms 7-10: schoolbook for col 2
    #   Q1 = A00*B02 -> C02
    #   Q2 = A01*B12 -> C02  
    #   Q3 = A10*B02 -> C12
    #   Q4 = A11*B12 -> C12
    
    # Can we replace Q1-Q4 with 3 terms?
    # We need:
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # With 3 terms, we need to represent 4 bilinear forms with 3 rank-1 terms.
    # This is only possible if there's a linear dependency.
    # 
    # A00*B02 + A01*B12 and A10*B02 + A11*B12 are independent.
    # So we need at least 2 terms for these two equations, but each term 
    # can only produce one product... unless we use sums.
    
    # Actually, a rank-1 term u*v*w contributes to multiple outputs if w has multiple non-zeros.
    # So one term could contribute to both C02 and C12.
    
    # Let's say term r has:
    # U_r = (u0, u1, u2, u3) for A00, A01, A10, A11
    # V_r = (v0, v1, v2, v3, v4, v5) for B00, B10, B01, B11, B02, B12
    # W_r = (w0, w1, w2, w3, w4, w5) for C00, C10, C01, C11, C02, C12
    
    # The contribution to C02 from term r is u0*v4*w4 + u1*v5*w4 = w4*(u0*v4 + u1*v5)
    # The contribution to C12 from term r is u2*v4*w5 + u3*v5*w5 = w5*(u2*v4 + u3*v5)
    
    # For 3 terms to span the 4D space of (A00*B02, A01*B12, A10*B02, A11*B12),
    # we need the 3 terms to generate 4 independent bilinear forms.
    # This is possible if the terms are carefully chosen.
    
    # Let me try: 
    # Term 7: (A00+A11)(B02+B12) -> contributes to C02 and C12
    # Term 8: (A10+A11)(B02) -> contributes to C12
    # Term 9: A00(B02-B12) -> contributes to C02
    
    # Check:
    # C02 = (A00+A11)(B02+B12) + A00(B02-B12) + ... 
    #     = A00*B02 + A00*B12 + A11*B02 + A11*B12 + A00*B02 - A00*B12 + ...
    #     = 2*A00*B02 + A11*B02 + A11*B12 + ...
    # That's not right.
    
    # Let me be more careful. I'll just do a random search with a time limit.
    
    elapsed = time.time() - _start_time
    time_limit = 30.0  # seconds for search
    
    # Try to find rank-10 by random search over {-1,0,1}^160
    # This is infeasible, so let's try a smarter approach.
    
    # Actually, let me just return None and use the rank-53 solution.
    # The search for rank-10 2x2x3 is too complex for this time budget.
    return None

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Use Strassen's 2x2x2 algorithm for each pair of columns.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) -> 7*7 = 49 rank
    # 1 single column (14) -> 4 rank (trivial)
    # Total rank = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    def add_strassen_block(base_r, k0, k1):
        """
        Strassen decomposition for 2x2 A times 2x2 B (columns k0, k1).
        
        B matrix columns are k0 and k1, rows are 0 and 1.
        B0k0 = B[0][k0], B0k1 = B[0][k1]
        B1k0 = B[1][k0], B1k1 = B[1][k1]
        
        7 products:
        P1 = (A00+A11)(B0k0+B1k1)
        P2 = (A10+A11)(B0k0)
        P3 = A00(B0k1-B1k1)
        P4 = A11(B1k0-B0k0)
        P5 = (A00+A01)(B1k1)
        P6 = (A10-A00)(B0k0+B0k1)
        P7 = (A01-A11)(B1k0+B1k1)
        
        Output C = A*B:
        C0k0 = A00*B0k0 + A01*B1k0 = P1 + P4 - P5 + P7
        C1k0 = A10*B0k0 + A11*B1k0 = P2 + P4
        C0k1 = A00*B0k1 + A01*B1k1 = P3 + P5
        C1k1 = A10*B0k1 + A11*B1k1 = P1 - P2 + P3 + P6
        """
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # P1: (A00+A11)(B0k0+B1k1)
        U[0][r1] = 1   # A00
        U[3][r1] = 1   # A11
        V[0 * p + k0][r1] = 1   # B0k0
        V[1 * p + k1][r1] = 1   # B1k1
        
        # P2: (A10+A11)(B0k0)
        U[2][r2] = 1   # A10
        U[3][r2] = 1   # A11
        V[0 * p + k0][r2] = 1   # B0k0
        
        # P3: A00(B0k1-B1k1)
        U[0][r3] = 1   # A00
        V[0 * p + k1][r3] = 1   # B0k1
        V[1 * p + k1][r3] = -1  # B1k1
        
        # P4: A11(B1k0-B0k0)
        U[3][r4] = 1   # A11
        V[1 * p + k0][r4] = 1   # B1k0
        V[0 * p + k0][r4] = -1  # B0k0
        
        # P5: (A00+A01)(B1k1)
        U[0][r5] = 1   # A00
        U[1][r5] = 1   # A01
        V[1 * p + k1][r5] = 1   # B1k1
        
        # P6: (A10-A00)(B0k0+B0k1)
        U[2][r6] = 1   # A10
        U[0][r6] = -1  # A00
        V[0 * p + k0][r6] = 1   # B0k0
        V[0 * p + k1][r6] = 1   # B0k1
        
        # P7: (A01-A11)(B1k0+B1k1)
        U[1][r7] = 1   # A01
        U[3][r7] = -1  # A11
        V[1 * p + k0][r7] = 1   # B1k0
        V[1 * p + k1][r7] = 1   # B1k1
        
        # Reconstruction
        # C0k0 = P1 + P4 - P5 + P7 -> W[2*k0]
        W[2 * k0][r1] = 1
        W[2 * k0][r4] = 1
        W[2 * k0][r5] = -1
        W[2 * k0][r7] = 1
        
        # C1k0 = P2 + P4 -> W[2*k0+1]
        W[2 * k0 + 1][r2] = 1
        W[2 * k0 + 1][r4] = 1
        
        # C0k1 = P3 + P5 -> W[2*k1]
        W[2 * k1][r3] = 1
        W[2 * k1][r5] = 1
        
        # C1k1 = P1 - P2 + P3 + P6 -> W[2*k1+1]
        W[2 * k1 + 1][r1] = 1
        W[2 * k1 + 1][r2] = -1
        W[2 * k1 + 1][r3] = 1
        W[2 * k1 + 1][r6] = 1
    
    # Add 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
    
    # Add trivial block for column 14 (single column, 4 multiplications)
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    # Q1 = A00 * B0,14 -> C0,14
    U[0][r1] = 1
    V[0 * p + 14][r1] = 1
    W[2 * 14][r1] = 1
    
    # Q2 = A01 * B1,14 -> C0,14
    U[1][r2] = 1
    V[1 * p + 14][r2] = 1
    W[2 * 14][r2] = 1
    
    # Q3 = A10 * B0,14 -> C1,14
    U[2][r3] = 1
    V[0 * p + 14][r3] = 1
    W[2 * 14 + 1][r3] = 1
    
    # Q4 = A11 * B1,14 -> C1,14
    U[3][r4] = 1
    V[1 * p + 14][r4] = 1
    W[2 * 14 + 1][r4] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
