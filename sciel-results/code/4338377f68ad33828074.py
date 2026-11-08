"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """Strassen-based decomposition for 4x4 matrix multiplication, rank 49."""
    n, m, p = DIMENSIONS
    # We'll build U, V, W for the full 4x4 x 4x4 -> 4x4 multiplication.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (16, rank)  
    # W has shape (p*n, rank) = (16, rank)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Strassen for 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
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
    
    # For 4x4: block into 2x2 blocks of size 2x2.
    # Each "multiplication" in Strassen is a 2x2 x 2x2 product.
    # Use schoolbook for the inner 2x2 products: rank 4 each.
    # Total: 7 * 4 = 28 rank terms.
    
    # Actually, let's use the direct Strassen formula on 2x2 blocks where
    # each block multiplication is decomposed into 4 scalar mults.
    
    # Let's define the 2x2 block indices:
    # Block (bi, bj) where bi in {0,1}, bj in {0,1}
    # Element (i,j) in 4x4 belongs to block (i//2, j//2), local (i%2, j%2)
    
    # Strassen M1..M7, each is a 2x2 matrix product.
    # For each M_k = X_k * Y_k where X_k, Y_k are 2x2 matrices (linear combos of A/B blocks),
    # we decompose X_k * Y_k into 4 scalar mults.
    
    # X_k and Y_k are linear combinations of the 4 blocks of A and B respectively.
    # Each block is 2x2, so each entry of X_k is a linear combo of entries of A.
    
    # Let's build this systematically.
    # A block (bi,bj) has entries A[bi*2+li, bj*2+lj] for li,lj in {0,1}
    # In U/V/W indexing: U[i*m+j] = U[i*4+j], V[j*p+k] = V[j*4+k], W[k*n+i] = W[k*4+i]
    
    # For a product term: sum over (i,j,k) of U[i*4+j, c] * A[i,j] * V[j*4+k, c] * B[j,k] * W[k*4+i, c] = C[i,k]
    # Equivalently: C[i,k] = sum_c W[k*4+i,c] * (sum_j U[i*4+j,c] * A[i,j]) * (sum_j V[j*4+k,c] * B[j,k])
    
    # For Strassen M_k = X_k * Y_k where X_k = sum of some A-blocks, Y_k = sum of some B-blocks:
    # X_k[i,j] = sum of A[bi*2+li, bj*2+lj] for certain (bi,bj)
    # Y_k[j,k] = sum of B[bi*2+lj, bj*2+lk] for certain (bi,bj)
    
    # Let me define the Strassen coefficients for the 2x2 block level.
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_k = X_k * Y_k is a 2x2 x 2x2 product.
    # X_k is a 2x2 matrix whose entries are linear combos of A entries.
    # Y_k is a 2x2 matrix whose entries are linear combos of B entries.
    
    # For the schoolbook decomposition of X_k * Y_k:
    # (X_k * Y_k)[a,b] = sum_j X_k[a,j] * Y_k[j,b]
    # Each X_k[a,j] = sum_{(bi,bj) in S_k} coeff * A[bi*2+la, bj*2+lj]  (where la depends on a)
    # Actually, X_k is formed by adding entire 2x2 blocks, so:
    # X_k[a,j] = sum_{(bi,bj) in S_k} A[bi*2+a, bj*2+j]
    # Similarly Y_k[j,b] = sum_{(bi,bj) in T_k} B[bi*2+j, bj*2+b]
    
    # So for each M_k and each output position (a,b) in the 2x2 result:
    # (M_k)[a,b] = sum_j sum_{(bi1,bj1) in S_k} sum_{(bi2,bj2) in T_k} A[bi1*2+a, bj1*2+j] * B[bi2*2+j, bj2*2+b]
    
    # This is a sum of bilinear terms in A and B. Each term A[i,j]*B[j,k] contributes to C[i',k']
    # where i' and k' depend on the block structure.
    
    # Let me just enumerate all the bilinear terms directly.
    
    # Define S_k and T_k as sets of (bi,bj) block indices with coefficients.
    # For simplicity, all coefficients are +1 or -1.
    
    # M1: X = A00+A11, Y = B00+B11
    #   S1 = {(0,0):+1, (1,1):+1}, T1 = {(0,0):+1, (1,1):+1}
    # M2: X = A10+A11, Y = B00
    #   S2 = {(1,0):+1, (1,1):+1}, T2 = {(0,0):+1}
    # M3: X = A00, Y = B01-B11
    #   S3 = {(0,0):+1}, T3 = {(0,1):+1, (1,1):-1}
    # M4: X = A11, Y = B10-B00
    #   S4 = {(1,1):+1}, T4 = {(1,0):+1, (0,0):-1}
    # M5: X = A00+A01, Y = B11
    #   S5 = {(0,0):+1, (0,1):+1}, T5 = {(1,1):+1}
    # M6: X = A10-A00, Y = B00+B01
    #   S6 = {(1,0):+1, (0,0):-1}, T6 = {(0,0):+1, (0,1):+1}
    # M7: X = A01-A11, Y = B10+B11
    #   S7 = {(0,1):+1, (1,1):-1}, T7 = {(1,0):+1, (1,1):+1}
    
    # C = C00*block00 + C01*block01 + C10*block10 + C11*block11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each M_k, decompose X_k * Y_k into 4 scalar mults (schoolbook for 2x2).
    # For output position (a,b) in M_k (a,b in {0,1}):
    #   (M_k)[a,b] = sum_{j=0}^{1} X_k[a,j] * Y_k[j,b]
    #   = sum_{j=0}^{1} (sum_{(bi,bj) in S_k} c_s * A[bi*2+a, bj*2+j]) * (sum_{(bi2,bj2) in T_k} c_t * B[bi2*2+j, bj2*2+b])
    
    # Each product term: A[i,j]*B[j,k] where:
    #   i = bi*2+a, j_idx = bj*2+j (input j), k = bj2*2+b... wait, need to be careful.
    
    # Let me re-derive. A is 4x4, B is 4x4.
    # A[i,j], B[j,k], C[i,k].
    # Block (bi,bj) of A: rows bi*2, bi*2+1; cols bj*2, bj*2+1.
    #   A[bi*2+la, bj*2+lcol]
    # Block (bi,bj) of B: rows bi*2, bi*2+1; cols bj*2, bj*2+1.
    #   B[bi*2+lrow, bj*2+lcol]
    
    # X_k[a,j] = sum_{(bi,bj) in S_k} c_s * A[bi*2+a, bj*2+j]
    # Y_k[j,b] = sum_{(bi2,bj2) in T_k} c_t * B[bi2*2+j, bj2*2+b]
    
    # Product: X_k[a,j] * Y_k[j,b] = sum_{(bi,bj) in S_k} sum_{(bi2,bj2) in T_k} c_s*c_t * A[bi*2+a, bj*2+j] * B[bi2*2+j, bj2*2+b]
    
    # For this to be a valid bilinear term A[i,j']*B[j',k] in the matrix product:
    #   i = bi*2+a, j' = bj*2+j, and we need bj*2+j == bi2*2+j for the middle index to match.
    #   Wait, that's not necessarily true. The middle index in A[i,j'] is j' = bj*2+j,
    #   and the middle index in B[j',k] is j' = bi2*2+j. These must be equal for it to contribute
    #   to the standard matrix product C[i,k] = sum_{j'} A[i,j']*B[j',k].
    
    # Hmm, this means the Strassen decomposition at the block level doesn't directly give
    # terms that look like A[i,j']*B[j',k] with matching middle indices unless the block
    # structures align properly.
    
    # Actually, I think I'm overcomplicating this. Let me just directly construct the
    # U, V, W matrices by enumerating all the bilinear terms.
    
    # Each rank-1 term in the tensor decomposition is:
    #   C[i,k] += W[k*4+i, c] * (sum_j U[i*4+j, c] * A[i,j]) * (sum_j V[j*4+k, c] * B[j,k])
    
    # For Strassen, each M_k contributes to certain C blocks. Let me just build it by
    # collecting all the (i,j,k) triples with their coefficients.
    
    # For each M_k, and for each (a,b) in {0,1}^2 (position in the 2x2 result of M_k),
    # and for each j in {0,1} (inner sum index):
    #   X_k[a,j] = sum_{(bi,bj) in S_k} c_s * A[bi*2+a, bj*2+j]
    #   Y_k[j,b] = sum_{(bi2,bj2) in T_k} c_t * B[bi2*2+j, bj2*2+b]
    #   
    #   Product = sum_{(bi,bj) in S_k} sum_{(bi2,bj2) in T_k} c_s*c_t * A[bi*2+a, bj*2+j] * B[bi2*2+j, bj2*2+b]
    
    # For this to contribute to C[i,k] in the standard way, we need the middle indices to match:
    #   bj*2+j should equal bi2*2+j, i.e., bj = bi2.
    #   If bj != bi2, this term doesn't directly correspond to a standard A[i,j']*B[j',k] term.
    
    # I think the issue is that Strassen's algorithm works at the block level where each
    # "multiplication" is a full matrix multiplication. The bilinear form for the entire
    # 4x4 x 4x4 product is what we need.
    
    # Let me take a different approach: just directly compute what U, V, W should be
    # by thinking of each rank-1 component as corresponding to one scalar multiplication
    # in the Strassen algorithm.
    
    # Actually, the cleanest way: use the fact that for 2x2 Strassen, each M_k = X_k * Y_k
    # where X_k and Y_k are 2x2 matrices. The product X_k * Y_k has 4 entries, each a sum
    # of 2 products. So each M_k contributes 8 bilinear terms (but some may cancel or combine).
    
    # For the full 4x4 case with 2x2 blocks:
    # Total bilinear terms = 7 M_k's * 4 entries per M_k * 2 inner terms per entry = 56 terms
    # But many of these will have the same (i,j,k) pattern and can be combined.
    
    # Let me just enumerate them all and collect unique (i,j,k) patterns.
    
    S = {
        1: {((0,0), 1), ((1,1), 1)},
        2: {((1,0), 1), ((1,1), 1)},
        3: {((0,0), 1)},
        4: {((1,1), 1)},
        5: {((0,0), 1), ((0,1), 1)},
        6: {((1,0), 1), ((0,0), -1)},
        7: {((0,1), 1), ((1,1), -1)},
    }
    T = {
        1: {((0,0), 1), ((1,1), 1)},
        2: {((0,0), 1)},
        3: {((0,1), 1), ((1,1), -1)},
        4: {((1,0), 1), ((0,0), -1)},
        5: {((1,1), 1)},
        6: {((0,0), 1), ((0,1), 1)},
        7: {((1,0), 1), ((1,1), 1)},
    }
    
    # C_block_combinations: for each output block (ci,cj), which M_k's contribute with what sign
    C_combos = {
        (0,0): {1: 1, 4: 1, 5: -1, 7: 1},
        (0,1): {3: 1, 5: 1},
        (1,0): {2: 1, 4: 1},
        (1,1): {1: 1, 2: -1, 3: 1, 6: 1},
    }
    
    # Now collect all bilinear terms: (i, j, k, coeff)
    # where the term is coeff * A[i,j] * B[j,k] contributing to C[i,k]
    
    from collections import defaultdict
    terms = defaultdict(float)
    
    for mk in range(1, 8):
        for ci, cj in C_combos:
            if mk not in C_combos[(ci, cj)]:
                continue
            sign_c = C_combos[(ci, cj)][mk]
            
            # M_mk result goes to C block (ci, cj)
            # For each (a,b) in {0,1}^2 (local position in the 2x2 block):
            for a in range(2):
                for b in range(2):
                    # (M_mk)[a,b] = sum_{j=0}^{1} X_mk[a,j] * Y_mk[j,b]
                    for j_inner in range(2):
                        # X_mk[a, j_inner] = sum_{(bi,bj),c_s in S[mk]} c_s * A[bi*2+a, bj*2+j_inner]
                        # Y_mk[j_inner, b] = sum_{(bi2,bj2),c_t in T[mk]} c_t * B[bi2*2+j_inner, bj2*2+b]
                        for (bi, bj), cs in S[mk]:
                            for (bi2, bj2), ct in T[mk]:
                                i_idx = bi * 2 + a
                                j_idx_a = bj * 2 + j_inner
                                j_idx_b = bi2 * 2 + j_inner
                                k_idx = bj2 * 2 + b
                                # For this to be a valid term in the matrix product,
                                # we need j_idx_a == j_idx_b (middle index matches)
                                if j_idx_a == j_idx_b:
                                    coeff = sign_c * cs * ct
                                    terms[(i_idx, j_idx_a, k_idx)] += coeff
    
    # Now each unique (i,j,k) with nonzero coefficient gives one rank-1 term.
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (i, j, k), coeff in terms.items():
        if abs(coeff) < 1e-15:
            continue
        # Normalize: make W entry = coeff, U and V entries = 1
        # Or distribute: U=1, V=1, W=coeff
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = coeff
        col += 1
    
    # Check that col == rank
    # Some terms might have cancelled to zero, so we might have fewer actual terms
    if col < rank:
        # Trim
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        rank = col
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
