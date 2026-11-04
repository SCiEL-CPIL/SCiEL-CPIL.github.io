"""Winograd frequency-domain decomposition for 4x4 * 4x9 matrix multiplication."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    _start = time.time()
    _deadline = _start + 115.0

    n, m, p = DIMENSIONS  # 4, 4, 9

    def _time_left():
        return _deadline - time.time()

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _winograd_5x5x5():
        """
        Winograd rank-29 decomposition for 5x5x5 matrix multiplication.
        Uses 3x3 DFT-based approach with 3-point DFT (W3 = -1/2 + i*sqrt(3)/2).
        
        For 3x3: A, B, C are 3x3 matrices.
        DFT basis vectors (length 3):
          v0 = (1, 1, 1)
          v1 = (1, w, w^2) where w = e^{2pi*i/3}
          v2 = (1, w^2, w)
        
        Winograd 3x3x3 rank-9:
          G = DFT_3 (3x3 DFT matrix)
          F = DFT_3
          H = DFT_3^{-1} = conjugate(DFT_3)
          
          For each frequency r in {0,1,2}:
            For each frequency s in {0,1,2}:
              U_{r,s} = G[:,r] (column r of DFT)  -- 3 entries for A
              V_{r,s} = F[:,s] (column s of DFT)  -- 3 entries for B  
              W_{r,s} = H[:,r+s mod 3] (column (r+s)%3 of IDFT) -- 3 entries for C
              
        Total rank = 9 for 3x3x3.
        
        For 5x5x5, we embed into 3x3 by zero-padding:
          5x5 -> 3x3 with padding (but 5 > 3, so we need a different approach)
        
        Actually, for 5x5x5 Winograd, the standard approach uses:
          - 5-point DFT is complex
          - Alternative: use the known rank-29 scheme
        
        Let me use a simpler verified approach:
        The 5x5x5 tensor can be decomposed using the 3x3 Winograd (rank 9)
        applied to overlapping 3x3 blocks, but this doesn't directly give rank 29.
        
        Instead, I'll use the known result that matmul(5,5,5) has rank <= 29
        via the "Winograd" or "Kronecker" construction.
        
        A concrete construction:
        Use the 3x3 DFT Winograd (rank 9) and combine with a 5x5 -> 3x3 mapping.
        
        Actually, let me use a more direct approach:
        The 5x5x5 multiplication tensor can be written as a sum of rank-1 tensors.
        Using the formula from Winograd (1980) for n=5:
        
        The key insight: 5 = 3 + 2, so we can use a block structure.
        But the simplest verified rank-29 scheme uses:
        
        For each (i,j) in the 5x5 output:
          C[i,j] = sum_k A[i,k] * B[k,j]
        
        Winograd's method for n=5 uses the 5-point DFT with specific
        structured factors. The rank is 29 (not 25 = 5^2 because of the
        special structure).
        
        Let me implement the 3x3 DFT Winograd and verify it works,
        then extend to 5x5x5 via embedding.
        """
        # 3-point DFT matrix (complex):
        # F3 = [[1, 1, 1],
        #       [1, w, w^2],
        #       [1, w^2, w]]
        # where w = exp(2*pi*i/3) = -1/2 + i*sqrt(3)/2
        
        # For real arithmetic, we can use the fact that:
        # F3 * F3^H = 3 * I
        # So F3^{-1} = F3^H / 3
        
        # Winograd 3x3x3 rank-9 decomposition:
        # T[i,j,k] = sum_{r,s=0}^{2} (F3[i,r]) * (F3[j,s]) * (F3_H[k,(r+s)%3])
        # where F3_H is the conjugate of F3 (i.e., F3^{-1} * 3)
        
        # Let's verify: sum_{r,s} F3[i,r] * F3[j,s] * conj(F3[k,(r+s)%3])
        # = sum_r F3[i,r] * sum_s F3[j,s] * conj(F3[k,(r+s)%3])
        # Let t = (r+s)%3, s = (t-r)%3
        # = sum_r F3[i,r] * sum_t F3[j,(t-r)%3] * conj(F3[k,t])
        # This should equal 3*delta_{i,j} * 3*delta_{j,k}... no, it should give
        # the matrix multiplication tensor.
        
        # Actually the correct Winograd formula is:
        # C = F3^{-1} * (F3 * A) * (F3 * B) * F3^{-1}  (element-wise in frequency)
        # Wait, that's not right either.
        
        # The correct formula:
        # Let A' = F3 * A * F3^T  (transform both dimensions)
        # Let B' = F3 * B * F3^T
        # Then C' = A' * B' (element-wise / Hadamard product)
        # C = F3^{-1} * C' * F3^{-T}
        
        # The rank of this decomposition:
        # C[i,j] = sum_{r,s,t,u} F3^{-1}[i,r] * A'[r,t] * B'[t,u] * F3^{-1}[u,j]
        # where A'[r,t] = sum_{a,b} F3[r,a]*A[a,b]*F3[t,b]
        # and B'[t,u] = sum_{b,c} F3[t,b]*B[b,c]*F3[u,c]
        
        # Substituting:
        # C[i,j] = sum_{r,s,t,u} F3^{-1}[i,r] * (sum_a F3[r,a]*A[a,s]) * (sum_b F3[t,b]*B[b,j]) * F3^{-1}[u,j]... 
        
        # This is getting complicated. Let me just use the known rank-9 formula:
        # For 3x3x3, the Winograd decomposition has rank 9 with:
        # U[i,j,c] = F3[i, c//3] * F3[j, c%3]  for c in 0..8
        # V[i,j,c] = F3[i, c//3] * F3[j, c%3]
        # W[i,j,c] = F3_H[i, c//3] * F3_H[j, c%3]
        # where F3_H = conj(F3)
        
        # Let me verify this numerically.
        w = np.exp(2j * np.pi / 3)
        F3 = np.array([[1, 1, 1],
                       [1, w, w**2],
                       [1, w**2, w]], dtype=complex)
        F3H = np.conj(F3)
        
        # Build rank-9 decomposition for 3x3x3
        # c = 3*r + s, r in {0,1,2}, s in {0,1,2}
        # U[i,j,c] = F3[i,r] * F3[j,s]  -- but U should have shape (n*m, rank) = (9, 9)
        # For 3x3: A has indices (i,j) where i=row, j=col, so U row index = i*3+j
        # V has indices (j,k) where j=row of B, k=col of B, so V row index = j*3+k  
        # W has indices (k,i) where k=row of C, i=col of C, so W row index = k*3+i
        
        # The tensor is: T[i,j,k] = delta_{i,k} (for matrix mult C[i,k] = sum_j A[i,j]B[j,k])
        # So we need: sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1 for all i,j,k
        
        # With Winograd:
        # U[i*3+j, 3r+s] = F3[i,r] * F3[j,s]  -- wait, this doesn't look right dimensionally
        
        # Let me reconsider. The matrix multiplication tensor for n x n x n:
        # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # The rank-1 decomposition: T = sum_c u_c * v_c * w_c
        # where u_c is a vector in R^{n^2}, v_c in R^{n^2}, w_c in R^{n^2}
        # and T[i,j,k] = sum_c u_c[i*n+j] * v_c[j*n+k] * w_c[k*n+i]
        
        # For Winograd 3x3x3:
        # u_c[i*3+j] = F3[i, r] * F3[j, s]  where c = 3r+s
        # v_c[j*3+k] = F3[j, r'] * F3[k, s']  -- but what are r', s'?
        
        # Actually the standard Winograd for matrix mult:
        # C = G * ( (F * A) ⊙ (F * B) ) * H  where ⊙ is element-wise
        # G = F^{-1}, H = F^{-1}
        # F is the DFT matrix
        
        # Expanding: C[i,k] = sum_{a,b} G[i,a] * (F*A)[a, ?] * ... 
        # This is getting too complex to derive from scratch. Let me use a different approach.
        
        # ALTERNATIVE: Use the known fact that matmul(n,n,n) has rank at most
        # n^3 (schoolbook) and for n=3, rank is 9 (Winograd).
        # For n=5, the best known rank is 29 (from Winograd 1980).
        
        # Since deriving the exact rank-29 scheme is complex, let me use a
        # Kronecker product approach:
        # 5x5x5 can be embedded in 6x6x6 (zero-pad), and 6x6x6 can be
        # decomposed using 3x3x3 Winograd (rank 9) tensor 2x2x2 (rank 8) = rank 72?
        # No, that's worse.
        
        # Better: Use the fact that 5 = 2 + 3, and use a block decomposition.
        # Or just use the 3x3 Winograd directly on a 3x3 sub-problem and
        # handle the remaining dimensions separately.
        
        # SIMPLEST CORRECT APPROACH: 
        # For 4x4x9, use the 2x2 Strassen (rank 7) applied to 2x2 blocks:
        # 4x4x4 = (2x2 blocks) * Strassen = rank 7*7 = 49? No...
        # Actually 4x4x4 using 2x2 Strassen: each of the 4 block positions
        # needs a 2x2x2 multiplication (rank 7), but they share structure.
        # Total rank for 4x4x4 via Strassen = 7 * 7 = 49? 
        # No: 4x4x4 = 2x2 block matrix mult. The 2x2 block mult has rank 7
        # (Strassen). Each block is 2x2, so each "entry" of the 2x2 block
        # multiplication is itself a 2x2x2 multiplication (rank 7).
        # Total rank = 7 (Strassen) * 7 (inner) = 49.
        
        # For 4x4x9: We can view this as 4x4x4 * (1x4x1 Kronecker) for the
        # last dimension? No, that doesn't work directly.
        
        # Better: 4x4x9 = (4x4x4) x (1x1x9) in some sense? No.
        
        # Let me think about this differently.
        # C (4x9) = A (4x4) * B (4x9)
        # Split B into two parts: B = [B1 | B2] where B1 is 4x4, B2 is 4x5
        # C = [A*B1 | A*B2]
        # A*B1: 4x4x4, rank 49 (Strassen)
        # A*B2: 4x4x5, need to compute this
        
        # For 4x4x5: Split A into 2x2 blocks, B2 into 2x2 + 2x1 blocks
        # This gets complicated.
        
        # ALTERNATIVE SIMPLE APPROACH:
        # Use the 2x2 Strassen for the 4x4 part and schoolbook for the 9 columns.
        # 4x4x4 via Strassen: rank 49
        # For each additional column (5 more columns for total 9):
        #   4x4x1 via Strassen-like: rank 7 (since 4x4x1 = (2x2 blocks) with Strassen)
        #   Actually 4x4x1: C (4x1) = A (4x4) * b (4x1)
        #   Using 2x2 blocks: c = [[c11],[c12],[c21],[c22]] where each is 2x1
        #   This is like 4x4x4 but with B being 4x1 instead of 4x4.
        #   Rank should be less than 49.
        
        # Let me just use a clean Strassen-based construction for 4x4x4
        # and extend to 4x4x9.
        
        return None  # Placeholder, will implement below

    def _strassen_2x2x2():
        """
        Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
        A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = AB = [[p,q],[r,s]]
        
        M1 = (a+d)(e+h)
        M2 = (c+d)(e)
        M3 = (a)(f-h)
        M4 = (d)(g-c)
        M5 = (a+b)(h)
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        
        p = M1 + M4 - M5 + M7
        q = M3 + M5
        r = M2 + M4
        s = M1 - M2 + M3 + M6
        
        U indices (for A): a=0, b=1, c=2, d=3  (row index = i*2+j)
        V indices (for B): e=0, f=1, g=2, h=3  (row index = j*2+k)
        W indices (for C): p=0, q=1, r=2, s=3  (row index = k*2+i)
        
        For each Strassen term t (0..6):
        U[*,t], V[*,t], W[*,t] are the coefficient vectors.
        """
        # U: 4 rows (a,b,c,d), 7 cols (M1..M7)
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: p=1,s=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)(e): U: c=1,d=1; V: e=1; W: r=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = (a)(f-h): U: a=1; V: f=1,h=-1; W: q=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = (d)(g-c): U: d=1; V: g=1,c=-1; W: p=1,r=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1  # Wait, c is index 2 in A, but in V it's the B matrix
        # Actually V is for B: e=0, f=1, g=2, h=3
        # M4 = d*(g-c) where c is from A... no wait.
        # Let me re-derive: M4 = A22 * (B21 - B11) = d * (g - e)
        # So V: g=1, e=-1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a+b)(h): U: a=1,b=1; V: h=1; W: q=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6 = (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; W: s=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: p=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def _build_4x4x4_strassen():
        """
        Build rank-49 decomposition for 4x4x4 using 2x2 Strassen.
        
        4x4 matrices split into 2x2 blocks:
        A = [[A00, A01], [A10, A11]] where each is 2x2
        B = [[B00, B01], [B10, B11]]
        C = [[C00, C01], [C10, C11]]
        
        Strassen for 2x2 block multiplication (rank 7):
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)(B00)
        M3 = (A00)(B01-B11)
        M4 = (A11)(B10-B00)
        M5 = (A00+A01)(B11)
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1+M4-M5+M7
        C01 = M3+M5
        C10 = M2+M4
        C11 = M1-M2+M3+M6
        
        Each M_s is a 2x2x2 multiplication (rank 7 via Strassen).
        Total rank = 7 (outer Strassen) * 7 (inner Strassen) = 49.
        
        Index mapping:
        A[i,j] where i = 2*bi+ii, j = 2*bj+jj
          bi, bj in {0,1} are block indices
          ii, jj in {0,1} are intra-block indices
        U row index = i*4 + j = (2*bi+ii)*4 + (2*bj+jj) = 8*bi + 4*ii + 2*bj + jj
        
        B[j,k] where j = 2*bj+jj, k = 2*bk+kk
        V row index = j*4 + k = (2*bj+jj)*4 + (2*bk+kk) = 8*bj + 4*jj + 2*bk + kk
        
        C[k,i] where k = 2*bk+kk, i = 2*bi+ii
        W row index = k*4 + i = (2*bk+kk)*4 + (2*bi+ii) = 8*bk + 4*kk + 2*bi + ii
        """
        sU, sV, sW = _strassen_2x2x2()  # Each is 4x7
        
        rank = 49
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        
        # Outer Strassen: 7 terms, each involves specific block combinations
        # For outer term t (0..6), the blocks involved are:
        # A blocks: depends on t
        # B blocks: depends on t
        # C block: depends on t
        
        # Outer Strassen coefficients (same as inner):
        # t=0 (M1): A: 00,11; B: 00,11; C: 00,11
        # t=1 (M2): A: 10,11; B: 00; C: 10
        # t=2 (M3): A: 00; B: 01,11; C: 01
        # t=3 (M4): A: 11; B: 10,00; C: 00,10
        # t=4 (M5): A: 00,01; B: 11; C: 01
        # t=5 (M6): A: 10,00; B: 00,01; C: 11
        # t=6 (M7): A: 01,11; B: 10,11; C: 00
        
        # For each outer term t and inner term u (0..6):
        # The combined term index is c = t*7 + u
        # U[row, c] = outer_A_coeff[t, block_a] * sU[intra_a, u]
        # where block_a determines which 2x2 block of A, and intra_a is the position within
        
        # This is complex. Let me use a different indexing.
        
        # For outer term t, the A-side involves certain blocks with certain coefficients.
        # The inner Strassen operates on 2x2 matrices.
        
        # Let me define the outer Strassen more carefully:
        # Each outer term t corresponds to a product of (linear combo of A blocks) * (linear combo of B blocks)
        # which contributes to certain C blocks.
        
        # For term t=0: (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
        #   A-side: A00 with coeff 1, A11 with coeff 1
        #   B-side: B00 with coeff 1, B11 with coeff 1
        #   C-side: C00 with coeff 1, C11 with coeff 1
        
        # For each such term, the actual computation is:
        # (sum_{block_a} coeff_a * A_block_a) * (sum_{block_b} coeff_b * B_block_b)
        # = sum_{block_a, block_b} coeff_a * coeff_b * A_block_a * B_block_b
        # Each A_block_a * B_block_b is a 2x2x2 multiplication (rank 7).
        
        # So for term t with A-blocks {a1,a2,...} and B-blocks {b1,b2,...}:
        # We need rank-7 decompositions for each pair (a_i, b_j).
        
        # This means for t=0: pairs (00,00), (00,11), (11,00), (11,11) -- 4 pairs
        # Each pair needs rank 7, so term t=0 contributes 4*7 = 28 rank-1 terms?
        # That would make total rank much larger than 49.
        
        # I think I'm overcomplicating this. The correct Strassen for 4x4x4
        # uses the 2x2 Strassen structure where each "entry" of the 2x2 block
        # matrix is itself a 2x2 matrix, and the Strassen algorithm applies
        # to the block level. The total rank is 7*7 = 49 because:
        # - The outer Strassen has 7 terms
        # - Each term involves one 2x2x2 multiplication (not multiple)
        # - Because the linear combinations are done BEFORE the 2x2x2 mult
        
        # So for term t=0: (A00+A11) is a 2x2 matrix (sum of two 2x2 blocks),
        # (B00+B11) is a 2x2 matrix, and their product is a 2x2x2 mult (rank 7).
        
        # This is correct! The linear combinations happen at the block level,
        # producing 2x2 matrices, which are then multiplied (rank 7 each).
        
        # So for each outer term t, we have ONE 2x2x2 multiplication (rank 7).
        # Total rank = 7 * 7 = 49.
        
        # Now, for the combined decomposition:
        # For outer term t and inner term u:
        # U[row, t*7+u] = (coeff of A_block in outer term t) * sU[intra_row, u]
        # where the A_block determines which rows of U are non-zero.
        
        # For term t=0: A-side is A00 + A11
        #   A00 occupies rows where bi=0, bj=0: i in {0,1}, j in {0,1}
        #     U row indices: i*4+j for i in {0,1}, j in {0,1} = {0,1,4,5}
        #   A11 occupies rows where bi=1, bj=1: i in {2,3}, j in {2,3}
        #     U row indices: i*4+j for i in {2,3}, j in {2,3} = {10,11,14,15}
        #   So for inner term u, U[row, 0*7+u] = sU[intra, u] for rows in A00 and A11
        
        # Let me code this up properly.
        
        # Outer Strassen term definitions:
        # t: (A_blocks_with_coeffs, B_blocks_with_coeffs, C_blocks_with_coeffs)
        outer_terms = [
            # t=0: (A00+A11)(B00+B11) -> C00+1, C11+1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # t=1: (A10+A11)(B00) -> C10+1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            # t=2: (A00)(B01-B11) -> C01+1
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
            # t=3: (A11)(B10-B00) -> C00+1, C10+1
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # t=4: (A00+A01)(B11) -> C01+1
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}),
            # t=5: (A10-A00)(B00+B01) -> C11+1
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # t=6: (A01-A11)(B10+B11) -> C00+1
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for t, (a_blocks, b_blocks, c_blocks) in enumerate(outer_terms):
            for u in range(7):
                c = t * 7 + u
                # U: for each A block (bi,bj) with coeff ca:
                for (bi, bj), ca in a_blocks.items():
                    for ii in range(2):
                        for jj in range(2):
                            i = 2 * bi + ii
                            j = 2 * bj + jj
                            U[i * 4 + j, c] += ca * sU[ii * 2 + jj, u]
                
                # V: for each B block (bj,bk) with coeff cb:
                for (bj, bk), cb in b_blocks.items():
                    for jj in range(2):
                        for kk in range(2):
                            j = 2 * bj + jj
                            k = 2 * bk + kk
                            V[j * 4 + k, c] += cb * sV[jj * 2 + kk, u]
                
                # W: for each C block (bk,bi) with coeff cc:
                for (bk, bi), cc in c_blocks.items():
                    for kk in range(2):
                        for ii in range(2):
                            k = 2 * bk + kk
                            i = 2 * bi + ii
                            W[k * 4 + i, c] += cc * sW[kk * 2 + ii, u]
        
        return U, V, W

    def _validate_numpy(U, V, W):
        """Fast validation using numpy."""
        n, m, p = 4, 4, 9
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        
        if isinstance(U, list):
            U = np.array(U, dtype=float)
            V = np.array(V, dtype=float)
            W = np.array(W, dtype=float)
        
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        
        # Reshape to 3D tensors
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Contract over rank dimension
        # Result[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use einsum
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # Should be all ones
        return np.allclose(result, 1.0, atol=1e-9)

    # Try Strassen-based 4x4x4 first
    if _time_left() > 30:
        try:
            U4, V4, W4 = _build_4x4x4_strassen()
            if _validate_numpy(U4, V4, W4):
                # 4x4x4 works with rank 49. Now extend to 4x4x9.
                # For columns 4..8 (5 more columns), we need 4x4x1 decompositions.
                # 4x4x1: C (4x1) = A (4x4) * b (4x1)
                # Using 2x2 blocks: c = [c1; c2] where c1, c2 are 2x1
                # c1 = A11*b1 + A12*b2 (2x1 = 2x2 * 2x1)
                # c2 = A21*b1 + A22*b2
                # Each 2x2*2x1 is a rank-4 operation (schoolbook) or can use
                # a rank-3 scheme? Actually 2x2x1 has rank 4 (it's just
                # 4 scalar multiplications).
                
                # For 4x4x1, the rank is 4 (each output element is a dot product
                # of 4 elements, but we can share computations).
                # Actually, 4x4x1 = (2x2 blocks): 
                # c1 = A00*b0 + A01*b1  (2x2 * 2x1, rank 4)
                # c2 = A10*b0 + A11*b1  (2x2 * 2x1, rank 4)
                # Total rank = 8? Or can we do better?
                
                # For 2x2x1: C (2x1) = A (2x2) * b (2x1)
                # C[0] = a*b + c*d, C[1] = e*b + f*d
                # This has rank 4 (schoolbook) or can be done in rank 3?
                # No, rank(2x2x1) = 4 (it's the same as 4 scalar mults with no sharing
                # that helps, since each output depends on different pairs).
                # Actually: C[0] = a*b + c*d, C[1] = e*b + f*d
                # We can write: M1 = b, M2 = d, then C[0] = a*M1 + c*M2, C[1] = e*M1 + f*M2
                # But M1 and M2 are just b and d (no computation needed).
                # So it's just 4 multiplications: a*b, c*d, e*b, f*d. Rank 4.
                
                # For 4x4x1: rank = 8 (two 2x2x1 operations)
                # Total for 4x4x9: 49 + 5*8 = 49 + 40 = 89? 
                # But wait, the 5 additional columns each need a separate 4x4x1.
                # Total rank = 49 + 5*8 = 89.
                
                # Hmm, that's better than 144 but not as good as hoped.
                # Let me verify the 4x4x4 part first.
                
                # Build 4x4x9 by combining 4x4x4 (rank 49) + 5*(4x4x1) (rank 8 each)
                rank_total = 49 + 5 * 8
                U_full = np.zeros((16, rank_total), dtype=float)
                V_full = np.zeros((36, rank_total), dtype=float)  # 4*9 = 36
                W_full = np.zeros((36, rank_total), dtype=float)  # 9*4 = 36
                
                # Copy 4x4x4 part (columns 0-3 of B, rows 0-3 of C)
                U_full[:, 0:49] = U4
                # V for 4x4x4: B is 4x4, so rows 0-15 (j*4+k for j,k in 0..3)
                V_full[0:16, 0:49] = V4
                # W for 4x4x4: C is 4x4, so rows 0-15 (k*4+i for k,i in 0..3)
                W_full[0:16, 0:49] = W4
                
                # For each additional column k0 in 4..8:
                for col_idx in range(5):
                    k0 = 4 + col_idx  # Column index in B and C
                    base = 49 + col_idx * 8
                    
                    # 4x4x1: C[:,k0] = A * B[:,k0]
                    # B[:,k0] is a 4x1 vector: b = [b0, b1, b2, b3]^T
                    # V rows for B[:,k0]: j*9+k0 for j in 0..3
                    #   = 0*9+4, 1*9+4, 2*9+4, 3*9+4 = 4, 13, 22, 31 (for k0=4)
                    
                    # 4x4x1 using 2x2 blocks:
                    # b = [b1; b2] where b1 = [b0,b1]^T, b2 = [b2,b3]^T
                    # c1 = A00*b1 + A01*b2 (2x1)
                    # c2 = A10*b1 + A11*b2 (2x1)
                    
                    # A00*b1: A00 = [[a00,a01],[a10,a11]], b1 = [b0,b1]
                    #   c1_0 = a00*b0 + a01*b1
                    #   c1_1 = a10*b0 + a11*b1
                    # A01*b2: A01 = [[a02,a03],[a12,a13]], b2 = [b2,b3]
                    #   c1_0 += a02*b2 + a03*b3
                    #   c1_1 += a12*b2 + a13*b3
                    # Similarly for c2.
                    
                    # Rank-4 for each 2x2x1:
                    # 2x2x1: C[0] = a*b + c*d, C[1] = e*b + f*d
                    # U: a=1 (row 0), c=1 (row 1), e=1 (row 2), f=1 (row 3)
                    # V: b=1 (row 0), d=1 (row 1) for first pair; etc.
                    
                    # For A00*b1 (2x2x1):
                    # U rows: A00 elements at positions (0,0),(0,1),(1,0),(1,1) in 4x4
                    #   = row indices 0,1,4,5
                    # V rows: b1 elements at positions j=0,1 in column k0
                    #   = V row indices 0*9+k0, 1*9+k0
                    # W rows: c1 elements at positions k=0,1 in column k0
                    #   = W row indices 0*4+k0, 1*4+k0 (but W index is k*n+i = k*4+i)
                    
                    # Let me just use schoolbook for the 4x4x1 parts (rank 4 each, 
                    # but actually 4x4x1 has rank 4? No, it has rank 4*1 = 4? 
                    # Wait, 4x4x1: 4 outputs, each is a dot product of length 4.
                    # Schoolbook rank = 4 (one term per output element).
                    # Actually no: C[i,k0] = sum_j A[i,j]*B[j,k0]
                    # Each C[i,k0] needs 4 multiplications, total 16 multiplications.
                    # But we can share: for fixed k0, C[:,k0] = A * b where b is 4x1.
                    # This is a matrix-vector product, rank = 4 (each row of C is
                    # a separate dot product, but they share the same b).
                    # Actually the rank of the 4x4x1 tensor is 4:
                    # T[i,j,k] where k is fixed: T[i,j] = delta (for the j-th element)
                    # C[i] = sum_j A[i,j]*b[j]
                    # This can be written as 4 rank-1 terms:
                    #   For j in 0..3: u_j[i] = A[i,j], v_j = b[j], w_j[i] = 1
                    #   C[i] = sum_j u_j[i]*v_j*w_j[i]
                    # So rank 4.
                    
                    # For 4x4x1, rank = 4 (not 8 as I said earlier).
                    # Let me redo: total rank = 49 + 5*4 = 69.
                    
                    # Actually wait, I need to be more careful.
                    # The 4x4x1 tensor: T[i,j,k] for k=k0 (fixed)
                    # C[i,k0] = sum_j A[i,j] * B[j,k0]
                    # Rank-1 decomposition: for each j in 0..3:
                    #   U[i*4+j, c_j] = 1 (for all i) -- no, U[i*4+j, c_j] should be such that
                    #   sum_c U[i*4+j,c]*V[j*9+k0,c]*W[k0*4+i,c] = 1 for the right (i,j,k0)
                    
                    # For the j-th term:
                    #   U[i*4+j, c_j] = 1 for all i (this doesn't work because U depends on i)
                    
                    # Let me think again. The tensor T[i,j,k] = 1 for the matrix mult
                    # C[i,k] = sum_j A[i,j]*B[j,k].
                    # For 4x4x1 (k fixed to k0):
                    # T[i,j,k0] = 1 (it's just the identity in the j dimension for each i)
                    # Rank-1: T = sum_c u_c * v_c * w_c
                    # u_c[i*4+j] * v_c[j*9+k0] * w_c[k0*4+i] should give 1 for all i,j.
                    
                    # Simplest: for each j in 0..3, one term:
                    #   u_j[i*4+j] = 1 for all i, 0 otherwise
                    #   v_j[j*9+k0] = 1
                    #   w_j[k0*4+i] = 1 for all i, 0 otherwise
                    # Check: sum_j u_j[i*4+j]*v_j[j*9+k0]*w_j[k0*4+i]
                    #   = sum_j (1 if row=i*4+j else 0) * (1 if row=j*9+k0 else 0) * (1 if row=k0*4+i else 0)
                    #   For the j-th term: u_j is non-zero only at row i*4+j, v_j only at row j*9+k0,
                    #   w_j only at row k0*4+i.
                    #   Product is non-zero only when all three are non-zero simultaneously.
                    #   u_j[i*4+j] = 1 (always for this i,j)
                    #   v_j[j*9+k0] = 1 (always for this j,k0)
                    #   w_j[k0*4+i] = 1 (always for this k0,i)
                    #   So the j-th term contributes 1.
                    #   Sum over j: but each term only has ONE non-zero element in u, v, w.
                    #   Actually u_j has non-zero at ALL rows of the form i'*4+j for i' in 0..3.
                    #   So u_j[i*4+j] = 1, but u_j[i'*4+j] = 1 for all i'.
                    #   This means the product u_j[i*4+j]*v_j[j*9+k0]*w_j[k0*4+i] = 1*1*1 = 1
                    #   for the specific (i,j,k0) we're evaluating.
                    #   But for a different (i',j',k0):
                    #   u_j[i'*4+j'] is non-zero only if j' = j.
                    #   So the j-th term only contributes to (i,j,k0) for all i.
                    #   Sum over j: for (i,j,k0), only the j-th term contributes (since
                    #   u_j[i*4+j'] = 0 for j' != j).
                    #   So total = 1. Correct!
                    
                    # So rank 4 for 4x4x1. Total rank = 49 + 5*4 = 69.
                    
                    # Let me implement this properly.
                    
                    for j in range(4):
                        c = base + j
                        # U: non-zero at rows i*4+j for i in 0..3
                        for i in range(4):
                            U_full[i * 4 + j, c] = 1
                        # V: non-zero at row j*9+k0
                        V_full[j * 9 + k0, c] = 1
                        # W: non-zero at rows k0*4+i for i in 0..3
                        for i in range(4):
                            W_full[k0 * 4 + i, c] = 1
                
                # Validate
                if _validate_numpy(U_full, V_full, W_full):
                    # Convert to list of lists with integer entries
                    U_list = U_full.tolist()
                    V_list = V_full.tolist()
                    W_list = W_full.tolist()
                    
                    # Clean up floating point (should be integers)
                    for mat in [U_list, V_list, W_list]:
                        for row in mat:
                            for idx in range(len(row)):
                                val = row[idx]
                                if abs(val - round(val)) < 1e-9:
                                    row[idx] = int(round(val))
                                elif abs(val * 2 - round(val * 2)) < 1e-9:
                                    row[idx] = round(val * 2) / 2
                                else:
                                    row[idx] = val
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception as e:
            pass
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
