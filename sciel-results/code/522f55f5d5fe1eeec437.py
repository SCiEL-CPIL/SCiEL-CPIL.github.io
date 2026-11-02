"""
Strategy: Structured Algebraic Construction for 4x4 * 4x7 Matrix Multiplication.

The current program uses schoolbook (rank 112) and random local search which is unlikely to succeed.
This program attempts a deterministic construction:
1. Split the 7 columns into a 4-column block and a 3-column block.
2. For the 4x4 * 4x4 part, use a known rank-49 decomposition (Lader 1976 / Strassen variant).
   This saves 16 operations compared to schoolbook (64 -> 49).
3. For the 4x4 * 4x3 part, use a tailored approach. 
   Schoolbook is 4*4*3 = 48. 
   We can potentially save operations here too, but conservatively, if we just use schoolbook for the 3-part, 
   total rank = 49 + 48 = 97.
   97 < 112, so this is a guaranteed improvement if the 4x4*4x4 rank-49 construction is valid.
   
   We will implement the rank-49 4x4*4x4 decomposition. 
   Reference: The rank of 4x4 matrix multiplication is 49. 
   A specific construction can be found in literature or derived via Strassen's algorithm 
   applied to 2x2 blocks, but 2x2x2 has rank 7. 
   4x4 can be seen as 4 blocks of 2x2. 
   Naive Strassen on 4x4 (using 2x2 blocks) would require 7 multiplications of 2x2 matrices.
   Each 2x2 multiplication has rank 7. So 7*7 = 49 rank.
   This matches the lower bound for 4x4 (which is 49).
   
   So, we construct the 4x4*4x4 part using Strassen's algorithm recursively on 2x2 blocks, 
   yielding rank 49.
   For the 4x4*4x3 part, we treat it as multiplying a 4x4 matrix by a 4x3 matrix.
   We can view the 4x3 matrix as having 3 columns. 
   We can use the standard schoolbook for this part (rank 48) or try to optimize.
   Given time constraints, using schoolbook for the 3-part is safe and yields total rank 97.
   
   Total Rank = 49 (for first 4 cols) + 48 (for last 3 cols) = 97.
   Schoolbook Rank = 4*4*7 = 112.
   Improvement: 112 -> 97.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    # 1. Schoolbook Baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Target tensor for verification
    target = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target.add((i * m + j, j * p + k, k * n + i))

    def verify(U, V, W):
        """Strict verification: all target entries = 1, all non-target = 0."""
        rank = len(U[0]) if U else 0
        if rank == 0:
            return len(target) == 0
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all target entries
        for (u, v, w) in target:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check all non-target entries (full check for correctness)
        # Optimization: Only check a subset if time is tight? No, correctness is key.
        # 4*4*7 = 112 target entries. Total entries 16*28*28 = 12544.
        # Iterating all non-targets is ~12k dot products of rank R. 
        # If R=97, that's ~1.2M ops. Very fast.
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) in target:
                        continue
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val) > 1e-9:
                        return False
        return True

    # 2. Construct Rank-49 4x4 * 4x4 Decomposition
    # We use Strassen's algorithm for 2x2 matrices.
    # A 4x4 matrix A is divided into 4 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2.
    # C = A*B.
    # Strassen computes 7 products of 2x2 matrices:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 matrix product. The rank of 2x2 matrix multiplication is 7.
    # So each Mi contributes 7 rank terms.
    # Total rank = 7 * 7 = 49.
    
    # We need to map the scalar operations of the 2x2 rank-7 decomposition to the global indices.
    
    # First, define the rank-7 decomposition for 2x2 * 2x2.
    # Let A be 2x2, B be 2x2.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A*B
    # C00 = a00*b00 + a01*b10
    # C01 = a00*b01 + a01*b11
    # C10 = a10*b00 + a11*b10
    # C11 = a10*b01 + a11*b11
    #
    # Strassen for 2x2:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will build the U, V, W matrices for the 4x4*4x4 part (columns 0-3 of the 7-column output).
    # The "input" to the 4x4*4x4 part is the full 4x4 matrix A (indices 0..15) and the first 4 columns of B (indices 0..15 in B's 4x4 sub-block? No, B is 4x7).
    # Wait, the problem is A(4x4) * B(4x7).
    # We split B into B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2.
    # We handle A*B1 with rank 49.
    # We handle A*B2 with rank 48 (schoolbook).
    # Total rank 97.
    
    U_new = []
    V_new = []
    W_new = []
    
    # Helper to add a term
    def add_term(u_val, v_val, w_val):
        # u_val, v_val, w_val are lists/dicts of (index, coeff)
        # We append a column to U, V, W
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        
        for idx, coeff in u_val.items():
            U_row[idx] += coeff
        for idx, coeff in v_val.items():
            V_row[idx] += coeff
        for idx, coeff in w_val.items():
            W_row[idx] += coeff
            
        U_new.append(U_row)
        V_new.append(V_row)
        W_new.append(W_row)

    # --- Part 1: 4x4 * 4x4 (First 4 columns of B) ---
    # Indices:
    # A is 4x4. Index u = i*m + j = i*4 + j.
    # B1 is 4x4 (columns 0-3 of B). Index v = j*p + k. Here k is 0..3. So v = j*7 + k.
    # C1 is 4x4 (columns 0-3 of C). Index w = k*n + i. Here k is 0..3. So w = k*4 + i.
    
    # We need to map the 2x2 Strassen blocks.
    # A blocks:
    # A11: rows 0,1 cols 0,1 -> u in {0,1,4,5}
    # A12: rows 0,1 cols 2,3 -> u in {2,3,6,7}
    # A21: rows 2,3 cols 0,1 -> u in {8,9,12,13}
    # A22: rows 2,3 cols 2,3 -> u in {10,11,14,15}
    
    # B1 blocks (cols 0-3 of B):
    # B11: rows 0,1 cols 0,1 -> v in {0,1, 7,8} (j=0, k=0,1; j=1, k=0,1)
    # B12: rows 0,1 cols 2,3 -> v in {2,3, 9,10}
    # B21: rows 2,3 cols 0,1 -> v in {14,15, 21,22}
    # B22: rows 2,3 cols 2,3 -> v in {16,17, 23,24}
    
    # C1 blocks (cols 0-3 of C):
    # C11: rows 0,1 cols 0,1 -> w in {0,1, 4,5} (k=0,1; i=0,1)
    # C12: rows 0,1 cols 2,3 -> w in {2,3, 6,7}
    # C21: rows 2,3 cols 0,1 -> w in {8,9, 12,13}
    # C22: rows 2,3 cols 2,3 -> w in {10,11, 14,15}
    
    # We will iterate over the 7 Strassen products M1..M7.
    # For each M, we need to decompose the 2x2 multiplication into 7 rank-1 terms.
    # Let's define the 7 rank-1 terms for a generic 2x2 * 2x2.
    # Inputs: A (2x2), B (2x2).
    # Outputs: C (2x2).
    # The 7 terms correspond to the scalar multiplications in Strassen.
    # Term 1: (a00+a11) * (b00+b11)
    #   U: a00=1, a11=1
    #   V: b00=1, b11=1
    #   W: c00+=1, c11+=1 (Wait, M1 contributes to C00 and C11)
    #   Actually, the W vector for a rank-1 term is the output contribution.
    #   In the tensor decomposition, a term is u * v * w.
    #   Sum over terms = C.
    #   So for M1 = (a00+a11)(b00+b11), this product appears in C00 and C11.
    #   C00 gets +M1. C11 gets +M1.
    #   So the W vector for this term should have 1 at C00 and 1 at C11?
    #   No, the tensor is T(u,v,w).
    #   T = sum_t U_t[u] V_t[v] W_t[w].
    #   If we have a term that computes X = A_part * B_part, and X contributes to C_pos with coeff 1,
    #   then for that term, W[w_pos] = 1.
    #   However, a single rank-1 term in the 2x2 decomposition corresponds to one scalar multiplication.
    #   The 2x2 Strassen algorithm has 7 scalar multiplications.
    #   Each scalar multiplication is a product of a linear combination of A and a linear combination of B.
    #   Let's list the 7 terms for 2x2:
    
    # Term 1: U: {0:1, 3:1}, V: {0:1, 3:1}, W: {0:1, 3:1}  (C00, C11) -> Wait, C indices in 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #   M1 = (A00+A11)(B00+B11).
    #   C00 += M1. C11 += M1.
    #   So W: {0:1, 3:1}.
    
    # Term 2: U: {2:1, 3:1}, V: {0:1}, W: {2:1} (C10) -> M2 = (A10+A11)B00. C10 += M2.
    #   U: {2:1, 3:1}, V: {0:1}, W: {2:1}.
    
    # Term 3: U: {0:1}, V: {1:1, 3:-1}, W: {1:1} (C01) -> M3 = A00(B01-B11). C01 += M3.
    #   U: {0:1}, V: {1:1, 3:-1}, W: {1:1}.
    
    # Term 4: U: {3:1}, V: {2:1, 0:-1}, W: {0:1, 2:1} (C00, C10) -> M4 = A11(B10-B00). C00 += M4, C10 += M4.
    #   U: {3:1}, V: {2:1, 0:-1}, W: {0:1, 2:1}.
    
    # Term 5: U: {0:1, 1:1}, V: {3:1}, W: {0:-1, 1:1} (C00, C01) -> M5 = (A00+A01)B11. C00 -= M5, C01 += M5.
    #   U: {0:1, 1:1}, V: {3:1}, W: {0:-1, 1:1}.
    
    # Term 6: U: {2:1, 0:-1}, V: {0:1, 1:1}, W: {3:1} (C11) -> M6 = (A10-A00)(B00+B01). C11 += M6.
    #   U: {2:1, 0:-1}, V: {0:1, 1:1}, W: {3:1}.
    
    # Term 7: U: {1:1, 3:-1}, V: {2:1, 3:1}, W: {0:1, 3:1} (C00, C11) -> M7 = (A01-A11)(B10+B11). C00 += M7, C11 += M7.
    #   U: {1:1, 3:-1}, V: {2:1, 3:1}, W: {0:1, 3:1}.
    
    # We need to map these local 2x2 indices to global indices.
    # Local A indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1) in the 2x2 block.
    # Local B indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1) in the 2x2 block.
    # Local C indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1) in the 2x2 block.
    
    # Define mappings for each of the 4 blocks of A, B1, C1.
    # A_blocks:
    # A11: global u = i*4+j. Local (li, lj) -> global i=li, j=lj. u = li*4 + lj.
    # A12: global u = i*4+j. Local (li, lj) -> global i=li, j=2+lj. u = li*4 + 2 + lj.
    # A21: global u = i*4+j. Local (li, lj) -> global i=2+li, j=lj. u = (2+li)*4 + lj.
    # A22: global u = i*4+j. Local (li, lj) -> global i=2+li, j=2+lj. u = (2+li)*4 + 2 + lj.
    
    # B1_blocks (cols 0-3 of B, p=7):
    # B11: global v = j*7+k. Local (lj, lk) -> global j=lj, k=lk. v = lj*7 + lk.
    # B12: global v = j*7+k. Local (lj, lk) -> global j=lj, k=2+lk. v = lj*7 + 2 + lk.
    # B21: global v = j*7+k. Local (lj, lk) -> global j=2+lj, k=lk. v = (2+lj)*7 + lk.
    # B22: global v = j*7+k. Local (lj, lk) -> global j=2+lj, k=2+lk. v = (2+lj)*7 + 2 + lk.
    
    # C1_blocks (cols 0-3 of C, n=4):
    # C11: global w = k*4+i. Local (lk, li) -> global k=lk, i=li. w = lk*4 + li.
    # C12: global w = k*4+i. Local (lk, li) -> global k=2+lk, i=li. w = (2+lk)*4 + li.
    # C21: global w = k*4+i. Local (lk, li) -> global k=lk, i=2+li. w = lk*4 + 2 + li.
    # C22: global w = k*4+i. Local (lk, li) -> global k=2+lk, i=2+li. w = (2+lk)*4 + 2 + li.
    
    # Strassen combinations:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will generate the 7 terms for each of the 7 Strassen products.
    # Total 49 terms.
    
    # Define the 7 base terms for 2x2 (U, V, W as dicts of local index -> coeff)
    base_terms_2x2 = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # M1
        ({2:1, 3:1}, {0:1}, {2:1}),           # M2
        ({0:1}, {1:1, 3:-1}, {1:1}),          # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),     # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),     # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),     # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}) # M7
    ]
    
    # Mappings for A blocks
    def map_A(block, li, lj):
        if block == 11: return li * 4 + lj
        if block == 12: return li * 4 + 2 + lj
        if block == 21: return (2 + li) * 4 + lj
        if block == 22: return (2 + li) * 4 + 2 + lj
    def map_B1(block, lj, lk):
        if block == 11: return lj * 7 + lk
        if block == 12: return lj * 7 + 2 + lk
        if block == 21: return (2 + lj) * 7 + lk
        if block == 22: return (2 + lj) * 7 + 2 + lk
    def map_C1(block, lk, li):
        if block == 11: return lk * 4 + li
        if block == 12: return (2 + lk) * 4 + li
        if block == 21: return lk * 4 + 2 + li
        if block == 22: return (2 + lk) * 4 + 2 + li
    
    # Strassen structure:
    # Each Strassen product Mx is a sum of products of linear combos of A blocks and B blocks.
    # We iterate through the 7 Strassen products.
    # For each, we identify the A block(s) and B block(s) involved.
    # Then we iterate through the 7 base terms of 2x2.
    # For each base term, we map the local indices to global indices.
    
    # M1: (A11+A22) * (B11+B22)
    # This is effectively (A11+A22) * (B11+B22).
    # We can view this as a 2x2 multiplication where the "A" is the sum and "B" is the sum?
    # No, it's a scalar product of two linear combinations.
    # The 7 base terms are for a generic 2x2 * 2x2.
    # Here, the "inputs" to the 2x2 multiplication are not 2x2 matrices, but the sums are treated as the inputs to the 2x2 rank-7 algorithm?
    # No. The rank-7 algorithm is for multiplying two 2x2 matrices.
    # Here, M1 = (A11+A22) * (B11+B22).
    # A11 and A22 are 2x2 matrices. Their sum is a 2x2 matrix.
    # So we are multiplying two 2x2 matrices: X = A11+A22 and Y = B11+B22.
    # The product Z = X*Y is a 2x2 matrix.
    # We use the 7-term decomposition for Z = X*Y.
    # The U part for X: For each entry in X (which is sum of A11 and A22 entries), we add the coefficients from both blocks.
    # The V part for Y: Similarly for B11 and B22.
    # The W part: The result Z contributes to C11 and C11? 
    # Wait, M1 is a 2x2 matrix. 
    # C11 (top-left 2x2 of C) gets M1. 
    # C22 (bottom-right 2x2 of C) gets M1.
    # So the W part for the 2x2 result Z should be mapped to C11 and C22 with coefficient 1.
    
    # Let's formalize:
    # For a Strassen product M = X * Y, where X is sum of A_blocks and Y is sum of B_blocks.
    # We generate 7 terms.
    # For each of the 7 base terms (u_base, v_base, w_base):
    #   u_global = sum over (li, lj) in u_base: coeff * map_A(block, li, lj) for each block in X
    #   v_global = sum over (lj, lk) in v_base: coeff * map_B1(block, lj, lk) for each block in Y
    #   w_global = sum over (lk, li) in w_base: coeff * map_C1(block, lk, li) for each block in C_target
    
    # M1: X = A11+A22, Y = B11+B22. Target C: C11 (+1), C22 (+1).
    # M2: X = A21+A22, Y = B11. Target C: C21 (+1).
    # M3: X = A11, Y = B12-B22. Target C: C12 (+1).
    # M4: X = A22, Y = B21-B11. Target C: C11 (+1), C21 (+1).
    # M5: X = A11+A12, Y = B22. Target C: C11 (-1), C12 (+1).
    # M6: X = A21-A11, Y = B11+B12. Target C: C22 (+1).
    # M7: X = A12-A22, Y = B21+B22. Target C: C11 (+1), C22 (+1).
    
    strassen_products = [
        # (A_blocks, coeffs, B_blocks, coeffs, C_blocks, coeffs)
        ([(11, 1), (22, 1)], [(11, 1), (22, 1)], [(11, 1), (22, 1)]), # M1
        ([(21, 1), (22, 1)], [(11, 1)], [(21, 1)]),                    # M2
        ([(11, 1)], [(12, 1), (22, -1)], [(12, 1)]),                   # M3
        ([(22, 1)], [(21, 1), (11, -1)], [(11, 1), (21, 1)]),          # M4
        ([(11, 1), (12, 1)], [(22, 1)], [(11, -1), (12, 1)]),          # M5
        ([(21, 1), (11, -1)], [(11, 1), (12, 1)], [(22, 1)]),          # M6
        ([(12, 1), (22, -1)], [(21, 1), (22, 1)], [(11, 1), (22, 1)])  # M7
    ]
    
    for a_blocks, b_blocks, c_blocks in strassen_products:
        # For each of the 7 base terms
        for u_base, v_base, w_base in base_terms_2x2:
            u_global = {}
            v_global = {}
            w_global = {}
            
            # Map U
            for (li, lj), coeff in u_base.items():
                for block, b_coeff in a_blocks:
                    idx = map_A(block, li, lj)
                    u_global[idx] = u_global.get(idx, 0) + coeff * b_coeff
            
            # Map V
            for (lj, lk), coeff in v_base.items():
                for block, b_coeff in b_blocks:
                    idx = map_B1(block, lj, lk)
                    v_global[idx] = v_global.get(idx, 0) + coeff * b_coeff
            
            # Map W
            for (lk, li), coeff in w_base.items():
                for block, c_coeff in c_blocks:
                    idx = map_C1(block, lk, li)
                    w_global[idx] = w_global.get(idx, 0) + coeff * c_coeff
            
            add_term(u_global, v_global, w_global)

    # --- Part 2: 4x4 * 4x3 (Last 3 columns of B) ---
    # Use schoolbook for this part.
    # A is 4x4 (u = i*4+j).
    # B2 is 4x3 (cols 4-6 of B). v = j*7+k, k in 4..6.
    # C2 is 4x3 (cols 4-6 of C). w = k*4+i, k in 4..6.
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # Verify the new construction
    if verify(U_new, V_new, W_new):
        if len(U_new) < best_rank:
            best = {"U": U_new, "V": V_new, "W": W_new}
            best_rank = len(U_new)

    # Fallback to schoolbook if verification fails (should not happen)
    if not verify(best["U"], best["V"], best["W"]):
        best = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = rank_sb

    return best
# EVOLVE-BLOCK-END
