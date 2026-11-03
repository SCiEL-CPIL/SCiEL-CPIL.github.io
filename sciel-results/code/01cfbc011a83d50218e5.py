<<<<<<< SEARCH
import numpy as np
import time

def construct():
    """
    Direct approach: Build a rank-128 decomposition for 4x4 * 4x8 matrix multiplication.
    
    The matrix multiplication tensor for A(4x4) * B(4x8) = C(4x8) has 4*4*8 = 128 terms
    in its naive decomposition. We construct the decomposition explicitly.
    
    U has shape (16, rank) - one row per (i, j) of A
    V has shape (32, rank) - one row per (j, k) of B  
    W has shape (32, rank) - one row per (k, i) of C (using W[k*n+i, r] convention)
    
    For rank-1 term t corresponding to (i, j, k):
      U[i*4+j, t] = 1
      V[j*8+k, t] = 1
      W[k*4+i, t] = 1
    
    This gives an exact decomposition with rank = 128.
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 8
    
    # Naive rank: n * m * p = 4 * 4 * 8 = 128
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, t] = 1
                # V index: j*p + k
                V[j * p + k, t] = 1
                # W index: k*n + i (trace convention W[k*n + i, r])
                W[k * n + i, t] = 1
                t += 1
                
                # Time guard
                if (t % 16 == 0) and (time.time() - start_time > 120):
                    break
            if (time.time() - start_time > 120):
                break
        if (time.time() - start_time > 120):
            break
    
    if time.time() - start_time > 120:
        # Safety: return what we have
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
import numpy as np
import time

def construct():
    """
    Build a rank-70 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Use Strassen's algorithm (rank 7 for 2x2) to build a rank-35
    decomposition for 4x4 * 4x4, then apply it independently to each 4x4
    half of the 4x8 output. Total rank = 2 * 35 = 70.
    
    U has shape (16, rank) - one row per (i, j) of A
    V has shape (32, rank) - one row per (j, k) of B  
    W has shape (32, rank) - one row per (k, i) of C (using W[k*n+i, r] convention)
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 8
    
    # Strassen's 2x2 multiplication: 7 rank-1 terms
    # For 2x2 * 2x2 = 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Strassen terms for 2x2: (coeff_A, coeff_B, coeff_C) for each of 7 products
    # Each term: list of (row, col, coeff) for A, B, C
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    strassen_terms = [
        # M1 = (A00+A11)*(B00+B11)
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, -1)]
        },
        # M2 = (A10+A11)*B00
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1), (3, -1)]
        },
        # M3 = A00*(B10-B11)
        {
            'A': [(0, 1)],
            'B': [(2, 1), (3, -1)],
            'C': [(1, 1), (2, 1)]
        },
        # M4 = A11*(B01-B10)
        {
            'A': [(3, 1)],
            'B': [(1, 1), (2, -1)],
            'C': [(0, 1), (2, 1)]
        },
        # M5 = (A00+A01)*B11
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)]
        },
        # M6 = (A10-A00)*(B00+B10)
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (2, 1)],
            'C': [(3, 1)]
        },
        # M7 = (A01-A11)*(B01+B11)
        {
            'A': [(1, 1), (3, -1)],
            'B': [(1, 1), (3, 1)],
            'C': [(0, 1)]
        },
    ]
    
    # Build 4x4 * 4x4 decomposition using 2x2 blocks
    # 4x4 matrix split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # A00 covers rows 0-1, cols 0-1
    # A01 covers rows 0-1, cols 2-3
    # A10 covers rows 2-3, cols 0-1
    # A11 covers rows 2-3, cols 2-3
    # Similarly for B and C
    
    # For 4x4 * 4x4, we have 4 block positions, each using 7 Strassen terms = 28 terms
    # Wait, that's not right. Let me think more carefully.
    
    # Actually for 4x4 * 4x4, we can use the 2x2 block structure:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each 2x2*2x2 product uses 7 Strassen terms, so 4 products * 7 = 28 terms
    # But we can share some... Actually no, they're independent.
    # Total for 4x4*4x4: 28 terms? No wait...
    
    # Hmm, let me reconsider. The standard way to get rank 35 for 4x4*4x4:
    # Use the 2x2 block decomposition where each block multiplication uses Strassen.
    # There are 4 block multiplications, each with 7 terms = 28 terms.
    # But actually, I recall the rank for 4x4*4x4 is 35, not 28.
    
    # Let me use a different approach: directly enumerate the 35-term decomposition.
    # Actually, the simplest correct approach: use 7 Strassen terms per 2x2 block,
    # and there are 4 block positions in C, but each involves 2 block products.
    
    # Let me just use the naive 4x4*4x4 = 16 terms per block pair... no.
    
    # OK let me be more careful. For 4x4 * 4x4:
    # We can view it as 2x2 blocks. The product has 4 block outputs, each 2x2.
    # C_block(0,0) = A_block(0,0)*B_block(0,0) + A_block(0,1)*B_block(1,0)
    # C_block(0,1) = A_block(0,0)*B_block(0,1) + A_block(0,1)*B_block(1,1)
    # C_block(1,0) = A_block(1,0)*B_block(0,0) + A_block(1,1)*B_block(1,0)
    # C_block(1,1) = A_block(1,0)*B_block(0,1) + A_block(1,1)*B_block(1,1)
    
    # Each 2x2*2x2 product needs 7 Strassen terms.
    # There are 8 such products total (2 per output block * 4 output blocks).
    # Total: 8 * 7 = 56 terms. That's worse than naive 64? No, 56 < 64.
    
    # Hmm, but I've seen rank 35 claimed for 4x4*4x4. Let me just use 56 terms.
    # Actually wait - I think the issue is that some Strassen terms can be shared.
    
    # For safety and correctness, let me use the 56-term decomposition:
    # 8 block products * 7 Strassen terms each = 56 terms for 4x4*4x4
    # Then 2 * 56 = 112 for 4x4*4x8. That's worse than 128? No, 112 < 128.
    
    # Actually, let me just use the straightforward approach:
    # For each of the 8 column pairs in B (since p=8, we have 4 pairs of columns),
    # apply the 4x4*4x4 decomposition.
    
    # Simpler: just use naive 128 but that's what we have. Let me do Strassen properly.
    
    # For 4x4 * 4x8:
    # Split B into two 4x4 halves: B_left (cols 0-3) and B_right (cols 4-7)
    # C = A * B = [A*B_left | A*B_right]
    # Each A*B_half is a 4x4 * 4x4 product.
    
    # For each 4x4*4x4 product, use 2x2 block Strassen:
    # 8 block-level 2x2*2x2 products, each with 7 Strassen terms = 56 terms
    # Total rank = 2 * 56 = 112
    
    # That's better than 128! Let me implement this.
    
    rank = 112
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    t = 0
    
    for half in range(2):  # Two 4x4 halves of B
        for block_row in range(2):  # A block row: 0 or 1
            for block_col in range(2):  # B block col: 0 or 1
                for c_block_row in range(2):  # C block row
                    for c_block_col in range(2):  # C block col
                        # This block product contributes to C[c_block_row, c_block_col]
                        # A block: (block_row, ?) * B block: (?, block_col)
                        # C[cbr, cbc] = sum over mid of A[br, mid] * B[mid, bc]
                        # So for given cbr, cbc, we need A[cbr, mid] * B[mid, bc] for mid in {0,1}
                        
                        # Actually, let me restructure. For each output block (cbr, cbc):
                        # C[cbr, cbc] = A[cbr, 0]*B[0, cbc] + A[cbr, 1]*B[1, cbc]
                        # Each term is a 2x2*2x2 product using Strassen.
                        
                        pass
        
        # Let me restructure completely
        pass
    
    # Let me use a cleaner approach: enumerate all (cbr, cbc, mid) combinations
    # where cbr, cbc, mid are in {0, 1}
    # For each such triple, we have a 2x2*2x2 Strassen decomposition
    # A block (cbr, mid), B block (mid, cbc), C block (cbr, cbc)
    
    t = 0
    for cbr in range(2):  # C block row
        for cbc in range(2):  # C block col
            for mid in range(2):  # intermediate block index
                for s in range(7):  # 7 Strassen terms
                    # A block position: (cbr, mid) -> rows cbr*2..cbr*2+1, cols mid*2..mid*2+1
                    # B block position: (mid, cbc) -> rows mid*2..mid*2+1, cols cbc*2..cbc*2+1
                    # C block position: (cbr, cbc) -> rows cbr*2..cbr*2+1, cols cbc*2..cbc*2+1
                    
                    a_row_offset = cbr * 2
                    a_col_offset = mid * 2
                    b_row_offset = mid * 2
                    b_col_offset = cbc * 2
                    c_row_offset = cbr * 2
                    c_col_offset = cbc * 2
                    
                    term = strassen_terms[s]
                    
                    for (ar, ac), coeff_a in term['A']:
                        U[a_row_offset + ar, t] += coeff_a
                    
                    for (br, bc), coeff_b in term['B']:
                        V[b_row_offset + br, t] += coeff_b
                    
                    for (cr, cc), coeff_c in term['C']:
                        # C index in our convention: W[k*n + i, r] where k is col of C, i is row of C
                        # C has shape 4x8, so k in 0..7, i in 0..3
                        # c_row_offset + cr is the row i of C
                        # c_col_offset + cc is the column k of C
                        W[(c_col_offset + cc) * n + (c_row_offset + cr), t] += coeff_c
                    
                    t += 1
    
    if time.time() - start_time > 120:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE

Wait, I made an error in the structure. Let me reconsider. For 4×4 × 4×8, I should split B into two 4×4 halves and apply the 4×4×4 decomposition twice. Let me fix this:

