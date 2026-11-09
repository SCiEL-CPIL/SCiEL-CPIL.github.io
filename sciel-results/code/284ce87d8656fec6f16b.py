"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll build the decomposition using a block structure:
    # i: 4, j: 4+1, k: 4+1
    # Blocks: (4,4,4), (4,4,1), (4,1,4), (4,1,1)
    
    # Start with zero matrices
    total_rank = 0
    U_rows = n * m  # 20
    V_rows = m * p  # 25
    W_rows = p * n  # 20
    
    # We'll accumulate rank-1 terms
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(U_coeff, V_coeff, W_coeff):
        """Add a rank-1 term. U_coeff: dict (i*m+j -> val), V_coeff: dict (j*p+k -> val), W_coeff: dict (k*n+i -> val)"""
        U_terms.append(U_coeff)
        V_terms.append(V_coeff)
        W_terms.append(W_coeff)
    
    def strassen_2x2x2(oi, oj, ok):
        """Add Strassen decomposition for 2x2x2 block at offset (oi, oj, ok)."""
        # M1 = (a00+a11)(b00+b11)
        U1 = {oj*m + oj: 1, (oj+1)*m + (oj+1): 1}  # a00 + a11, where a_ij is A[oi+i, oj+j]
        # Wait, I need to be careful. A is n×m, so A[i,j] with i in [0,n), j in [0,m).
        # For the 2x2 sub-block starting at (oi, oj), A[oi+i', oj+j'] for i',j' in {0,1}.
        # U index is i*m + j. So A[oi+i', oj+j'] -> (oi+i')*m + (oj+j').
        
        # Let me redo. For M1 = (a00+a11)(b00+b11):
        # a00 = A[oi, oj], a11 = A[oi+1, oj+1]
        # b00 = B[oj, ok], b11 = B[oj+1, ok+1]
        # C00 += M1, C11 += M1. C[i,k] where i is row, k is col. C00 = C[oi, ok], C11 = C[oi+1, ok+1].
        # W index is k*n + i. So C[oi, ok] -> ok*n + oi, C[oi+1, ok+1] -> (ok+1)*n + (oi+1).
        
        U1 = {oi*m + oj: 1, (oi+1)*m + (oj+1): 1}
        V1 = {oj*p + ok: 1, (oj+1)*p + (ok+1): 1}
        W1 = {ok*n + oi: 1, (ok+1)*n + (oi+1): 1}
        add_term(U1, V1, W1)
        
        # M2 = (a10+a11)b00
        U2 = {(oi+1)*m + oj: 1, (oi+1)*m + (oj+1): 1}
        V2 = {oj*p + ok: 1}
        W2 = {(ok+1)*n + (oi+1): 1}  # C11 += M2
        add_term(U2, V2, W2)
        
        # M3 = a00(b01-b11)
        U3 = {oi*m + oj: 1}
        V3 = {oj*p + (ok+1): 1, (oj+1)*p + (ok+1): -1}
        W3 = {ok*n + (oi+1): 1}  # C10? No, C[i,k]. C01 = C[oi, ok+1]. W index: (ok+1)*n + oi.
        # Wait, C01 means row 0, col 1. So i=oi, k=ok+1. W index = (ok+1)*n + oi.
        W3 = {(ok+1)*n + oi: 1}
        add_term(U3, V3, W3)
        
        # M4 = a11(b10-b00)
        U4 = {(oi+1)*m + (oj+1): 1}
        V4 = {(oj+1)*p + ok: 1, oj*p + ok: -1}
        W4 = {(ok+1)*n + oi: 1}  # C10 = C[oi+1, ok]. W index: ok*n + (oi+1).
        W4 = {ok*n + (oi+1): 1}
        add_term(U4, V4, W4)
        
        # M5 = (a00+a01)b11
        U5 = {oi*m + oj: 1, oi*m + (oj+1): 1}
        V5 = {(oj+1)*p + (ok+1): 1}
        W5 = {ok*n + oi: 1}  # C00 += M5? No. C00 = M1 - M5 + M7 + M4. So C00 -= M5. W[ok*n+oi] -= 1.
        # Wait, let me recheck Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M5 contributes to C00 with -1 and C01 with +1.
        W5 = {ok*n + oi: -1, (ok+1)*n + oi: 1}
        add_term(U5, V5, W5)
        
        # M6 = (a10-a00)(b00+b01)
        U6 = {(oi+1)*m + oj: 1, oi*m + oj: -1}
        V6 = {oj*p + ok: 1, oj*p + (ok+1): 1}
        W6 = {(ok+1)*n + (oi+1): 1}  # C11 += M6
        add_term(U6, V6, W6)
        
        # M7 = (a01-a11)(b10+b11)
        U7 = {oi*m + (oj+1): 1, (oi+1)*m + (oj+1): -1}
        V7 = {(oj+1)*p + ok: 1, (oj+1)*p + (ok+1): 1}
        W7 = {ok*n + oi: 1}  # C00 += M7
        add_term(U7, V7, W7)
    
    def strassen_4x4x4(oi, oj, ok):
        """Add Strassen decomposition for 4x4x4 block at offset (oi, oj, ok)."""
        # Split into 2x2 blocks of 2x2 matrices
        # 8 sub-blocks, but Strassen uses 7
        # For 4x4x4, split i into [oi:oi+2] and [oi+2:oi+4], etc.
        
        # M1 = (A00+A11)(B00+B11) where A00 is 2x2 block at (oi,oj), A11 at (oi+2,oj+2)
        # This is a 2x2x2 Strassen where each "entry" is a 2x2 matrix.
        # So we need to do Strassen at the 2x2x2 level, where each scalar is replaced by a 2x2 matrix multiplication.
        # The rank is 7 * rank(2x2x2) = 7*7 = 49.
        
        # For each of the 7 Strassen terms at the 4x4 level, we need a 2x2x2 decomposition.
        # Let me define the 7 linear forms at the 4x4 level:
        
        # A blocks: A00=(oi,oj), A01=(oi,oj+2), A10=(oi+2,oj), A11=(oi+2,oj+2)
        # B blocks: B00=(oj,ok), B01=(oj,ok+2), B10=(oj+2,ok), B11=(oj+2,ok+2)
        # C blocks: C00=(oi,ok), C01=(oi,ok+2), C10=(oi+2,ok), C11=(oi+2,ok+2)
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each M, it's a product of a linear form in A-blocks and a linear form in B-blocks.
        # Each A-block is 2x2, each B-block is 2x2.
        # So M = (sum of 2x2 matrices) * (sum of 2x2 matrices) = sum of products of 2x2 matrices.
        # Actually, (A+B)(C+D) = AC + AD + BC + BD, each of which is a 2x2*2x2 multiplication.
        # So each M at the 4x4 level expands to 4 2x2x2 multiplications.
        # Total: 7 * 4 = 28 2x2x2 multiplications, each of rank 7, total 28*7 = 196. That's too much.
        
        # Wait, I'm confusing myself. The Strassen algorithm for 4x4x4 should give rank 49, not 196.
        # The issue is that when I say M1 = (A00+A11)(B00+B11), this is a 4x4*4x4 product? No.
        # A00+A11 is a 4x4 matrix (block diagonal? No, A00 is top-left 2x2, A11 is bottom-right 2x2, so A00+A11 is a 4x4 matrix with A00 in top-left, A11 in bottom-right, and zeros elsewhere).
        # Similarly B00+B11 is 4x4.
        # So M1 is a 4x4*4x4 product, which is a 4x4x4 multiplication! Not 2x2x2.
        
        # So Strassen for 4x4x4: 7 sub-problems, each of which is a 4x4x4 multiplication? No, that's circular.
        
        # I think I'm making an error. Let me reconsider.
        
        # Strassen's algorithm: to multiply two n×n matrices where n=2^k, split into (n/2)×(n/2) blocks.
        # For n=4: split into 2×2 blocks of 2×2 matrices.
        # The 7 Strassen terms each involve multiplying 2×2 matrices (the block entries).
        # So each Strassen term is a 2×2×2 matrix multiplication (at the block level).
        # The rank of a 2×2×2 matrix multiplication is 7.
        # So total rank = 7 * 7 = 49.
        
        # But when I implement this, each "2×2 matrix multiplication" at the block level is actually a 4×4×4 tensor? No.
        # Each block is a 2×2 matrix. Multiplying two 2×2 matrices (at the block level) gives a 2×2 matrix (at the block level).
        # So the tensor for one Strassen term is a 2×2×2 tensor (block indices), but each "entry" is a 2×2 matrix.
        # So the full tensor is (2*2) × (2*2) × (2*2) = 4×4×4, but with a specific structure.
        
        # To decompose this, I need to decompose the 2×2×2 block-level tensor, where each rank-1 term at the block level corresponds to a 4×4×4 sub-tensor.
        # Each block-level rank-1 term: U_block[i_block, j_block] * V_block[j_block, k_block] * W_block[k_block, i_block]
        # where i_block, j_block, k_block are in {0,1}.
        # U_block[i_block, j_block] is a 2×2 matrix (the block of A).
        # So the full rank-1 term is: (2×2 matrix) * (2×2 matrix) * (2×2 matrix) = a 4×4×4 tensor.
        # This 4×4×4 tensor can be decomposed into 4 rank-1 terms (schoolbook for 2×2×2 at the entry level).
        
        # So each block-level rank-1 term gives 4 entry-level rank-1 terms.
        # Total: 7 (block-level terms) * 4 (entry-level per term) = 28 rank-1 terms.
        # Each entry-level term has U, V, W as 2×2 matrices (4 entries each).
        # So the full decomposition has 28 rank-1 terms.
        
        # Wait, that gives rank 28, not 49. Let me recheck.
        
        # I think the confusion is: the rank of the 4×4×4 matrix multiplication tensor.
        # Strassen: T(4) = 7 * T(2) = 7 * 7 = 49.
        # But T(2) = 7 is the rank of the 2×2×2 tensor.
        # When I do Strassen at the 4×4 level, I get 7 sub-tensors, each of which is a 2×2×2 tensor (at the block level), but each block is a 2×2 matrix.
        # So each sub-tensor is actually a 4×4×4 tensor (since each block entry is a 2×2 matrix).
        # No wait, that's not right either.
        
        # Let me think differently. The 4×4×4 matrix multiplication tensor T[i,j,k] for i,j,k in {0,1,2,3}.
        # Strassen splits this into 7 sub-tensors, each corresponding to a 2×2×2 block multiplication.
        # Each sub-tensor S[i',j',k'] where i',j',k' in {0,1} is a 2×2×2 tensor, but the "entries" are 2×2 matrices.
        # So S is a tensor of shape (2,2,2) where each entry is a 2×2 matrix.
        # The rank of S (as a 4×4×4 tensor) is at most 4 (schoolbook for the 2×2 entry level).
        # So total rank ≤ 7 * 4 = 28.
        
        # But that contradicts T(4) = 49. Let me look up the actual rank.
        
        # Actually, I think I'm wrong about Strassen giving 49 for 4×4×4. Let me recompute.
        # Strassen: to multiply two 4×4 matrices, split into 2×2 blocks of 2×2 matrices.
        # The 7 Strassen terms each require multiplying 2×2 matrices.
        # Each 2×2 matrix multiplication has rank 7.
        # So total rank = 7 * 7 = 49.
        
        # But when I say "each Strassen term requires multiplying 2×2 matrices", I mean that each Strassen term is a 2×2×2 matrix multiplication problem (at the block level).
        # The rank of a 2×2×2 matrix multiplication is 7.
        # So each Strassen term contributes rank 7.
        # Total: 7 * 7 = 49.
        
        # Now, how do I implement this? Each Strassen term is a 2×2×2 matrix multiplication.
        # I can use the Strassen decomposition for 2×2×2 (which I already have) to decompose each Strassen term.
        # So for the 4×4×4 block, I call strassen_2x2x2 seven times, each at different block offsets.
        
        # But wait, the 2×2×2 Strassen decomposition I wrote earlier is for a 2×2×2 sub-block where each "entry" is a scalar.
        # For the 4×4×4 case, each Strassen term involves 2×2 matrices, not scalars.
        # So I need a different approach.
        
        # Alternative: just use schoolbook for the 4×4×4 block, which has rank 64.
        # Then total: 64 + 16 + 16 + 4 = 100. No gain.
        
        # Or use a better decomposition for 4×4×4.
        # I know that rank(4×4×4) ≤ 49 (Strassen).
        # Let me implement the 4×4×4 Strassen properly.
        
        # For the 4×4×4 Strassen, I'll define the 7 linear forms at the block level, and for each, I'll decompose the resulting 2×2×2 matrix multiplication using the 2×2×2 Strassen.
        
        # Actually, let me just use the known fact that I can decompose 4×4×4 with rank 49 by applying Strassen recursively.
        # I'll write a recursive function.
        
        pass  # I'll implement this properly below
    
    # Let me just use a simpler approach: use the Kronecker product structure.
    # For the 4×4×4 block, I'll use the fact that it can be decomposed using 7 Strassen terms, each of which is a 2×2×2 multiplication.
    # I'll implement this by calling the 2×2×2 Strassen decomposition for each of the 7 block-level terms.
    
    # For now, let me just use schoolbook for all blocks to ensure correctness, then optimize.
    
    # Actually, let me implement the 4×4×4 Strassen properly.
    
    # I'll define a helper that adds a rank-1 term for a 2×2×2 sub-block, but where the "scalars" are actually 2×2 matrices.
    # This means that instead of adding one rank-1 term, I add 4 rank-1 terms (one for each entry of the 2×2 matrices).
    
    # Let me redefine the approach. I'll create a function that adds the Strassen decomposition for a 2×2×2 block,
    # where each "entry" is a 2×2 sub-block. This will add 7*4 = 28 rank-1 terms for a 4×4×4 block.
    
    # Wait, 7 Strassen terms, each with 4 entry-level rank-1 terms = 28. But I said rank 49 earlier.
    # The discrepancy is because each entry-level rank-1 term in the 2×2×2 Strassen has U, V, W as 2×2 matrices (4 entries each).
    # So each entry-level term is actually 4 scalar rank-1 terms.
    # So 7 (Strassen terms) * 4 (entries per term) = 28 scalar rank-1 terms.
    # But each of these has U, V, W with 4 non-zero entries, so the total number of non-zero entries is 28*4*3 = 336.
    # The rank is 28, not 49.
    
    # I think I've been confusing myself. Let me just implement it and check the rank.
    
    # For now, let me use the following approach:
    # 1. For the 4×4×4 block, use schoolbook (rank 64) to ensure correctness.
    # 2. For the 4×4×1, 4×1×4, 4×1×1 blocks, use schoolbook (rank 16, 16, 4).
    # Total: 64+16+16+4 = 100. Same as before.
    
    # To get a better rank, I need to use Strassen for the 4×4×4 block.
    # Let me implement it properly.
    
    # I'll write a function that decomposes a 2×2×2 block using Strassen, where each entry is a 2×2 sub-block.
    # This will add 7*4 = 28 rank-1 terms for a 4×4×4 block.
    
    # Actually, let me just use the standard Strassen decomposition for 4×4×4 with rank 49.
    # I'll look up the exact decomposition.
    
    # For now, let me just use schoolbook for everything to ensure the code runs, then optimize.
    
    # Schoolbook for the whole 4×5×5:
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
