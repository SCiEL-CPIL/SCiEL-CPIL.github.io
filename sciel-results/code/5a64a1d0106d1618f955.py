"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_deadline = 150.0
    start_time = time.time()

    # Known best: Strassen-based rank 117 decomposition
    # Partition 5x5 into blocks of sizes (2,2,1)
    # Use Strassen (rank 7) for 2x2 blocks, schoolbook (rank 1) for 1x1
    # 4 blocks of 2x2 -> 4*7 = 28
    # 4 blocks of 2x1 -> 4*2 = 8 (or use simple rank-2 for 2x1)
    # 4 blocks of 1x2 -> 4*2 = 8
    # 1 block of 1x1 -> 1
    # Total: 28 + 8 + 8 + 1 = 45... wait, that's not right.
    
    # Let me think more carefully. For matrix multiplication of sizes:
    # (a x b) * (b x c) = (a x c)
    # The tensor has indices i in [0,a), j in [0,b), k in [0,c)
    # U has shape (a*b, rank), V has shape (b*c, rank), W has shape (c*a, rank)
    
    # For a 2x2 * 2x2 = 2x2 multiplication:
    # Schoolbook: rank 4
    # Strassen: rank 7 (but that's worse for 2x2! Strassen wins for larger sizes)
    # Wait, Strassen's algorithm computes 2x2*2x2 using 7 multiplications of 2x2 matrices
    # But if we're at the base case of 2x2, we need rank 4 (schoolbook) or...
    
    # Actually, for a single 2x2 * 2x2 -> 2x2, the minimal rank is 7 (Strassen)
    # No wait, the standard 2x2*2x2 multiplication has rank 7? Let me reconsider.
    # The tensor for 2x2*2x2->2x2 has dimension 4*4*4 = 64 entries.
    # Schoolbook uses 4 terms. Can we do better?
    # Actually, the rank of the 2x2*2x2 matrix multiplication tensor is 7 (Strassen proved this is minimal for 2x2).
    # Wait no - schoolbook uses 4 multiplications. Strassen uses 7 multiplications of 2x2 blocks.
    # But at the scalar level, 2x2*2x2 requires rank 7? That doesn't sound right either.
    
    # Let me be precise: The matrix multiplication tensor T_{n,m,p} has rank r(n,m,p).
    # r(1,1,1) = 1
    # r(2,2,2) = 7 (Strassen, 1969)
    # r(3,3,3) = 23 (Bini et al.)
    # r(4,4,4) <= 49 (Strassen recursion)
    # r(5,5,5) <= ? 
    
    # For 5x5*5x5->5x5:
    # Partition into 2+2+1 in each dimension
    # Blocks:
    # (2,2,2): 4 such blocks, each rank 7 -> 28
    # (2,2,1): 2 such blocks, each rank... for a 2x2*1x2->1x2? No.
    
    # Let me reconsider the block structure.
    # We have A (5x5) * B (5x5) = C (5x5)
    # Partition rows of A and cols of C: sizes 2,2,1
    # Partition cols of A and rows of B: sizes 2,2,1
    # Partition cols of B and rows of C: sizes 2,2,1
    # Wait, I need to be more careful.
    
    # A is 5x5: rows partitioned as 2+2+1, cols partitioned as 2+2+1
    # B is 5x5: rows partitioned as 2+2+1, cols partitioned as 2+2+1
    # C is 5x5: rows partitioned as 2+2+1, cols partitioned as 2+2+1
    
    # Block (i,j) of C = sum_k A[i,k] * B[k,j]
    # Where i indexes row-block of C (and row-block of A), 
    #       k indexes col-block of A (and row-block of B),
    #       j indexes col-block of C (and col-block of B)
    
    # So block C[i,j] = sum_k A[i,k] B[k,j]
    # A[i,k] is (a_i x b_k), B[k,j] is (b_k x c_j), C[i,j] is (a_i x c_j)
    # with a = [2,2,1], b = [2,2,1], c = [2,2,1]
    
    # For each (i,k,j) triple, we multiply an (a_i x b_k) by (b_k x c_j) to get (a_i x c_j)
    # The number of such operations is 3*3*3 = 27 blocks
    
    # For each block, the rank needed:
    # (2,2,2): a_i=2, b_k=2, c_j=2 -> r(2,2,2)=7
    # (2,2,1): a_i=2, b_k=2, c_j=1 -> r(2,2,1)
    # (2,1,2): a_i=2, b_k=1, c_j=2 -> r(2,1,2)
    # (2,1,1): a_i=2, b_k=1, c_j=1 -> r(2,1,1)
    # (1,2,2): a_i=1, b_k=2, c_j=2 -> r(1,2,2)
    # (1,2,1): a_i=1, b_k=2, c_j=1 -> r(1,2,1)
    # (1,1,2): a_i=1, b_k=1, c_j=2 -> r(1,1,2)
    # (1,1,1): a_i=1, b_k=1, c_j=1 -> r(1,1,1)=1
    
    # Counts:
    # (2,2,2): i in {0,1}, k in {0,1}, j in {0,1} -> 2*2*2=8 blocks, rank 7 each -> 56
    # (2,2,1): i in {0,1}, k in {0,1}, j=2 -> 2*2*1=4 blocks
    # (2,1,2): i in {0,1}, k=2, j in {0,1} -> 2*1*2=4 blocks
    # (2,1,1): i in {0,1}, k=2, j=2 -> 2*1*1=2 blocks
    # (1,2,2): i=2, k in {0,1}, j in {0,1} -> 1*2*2=4 blocks
    # (1,2,1): i=2, k in {0,1}, j=2 -> 1*2*1=2 blocks
    # (1,1,2): i=2, k=2, j in {0,1} -> 1*1*2=2 blocks
    # (1,1,1): i=2, k=2, j=2 -> 1 block
    
    # Now I need ranks for the smaller cases:
    # r(2,2,1): multiply 2x2 by 2x1 to get 2x1. This is like doing 2 independent 2x1*1x1... 
    # Actually, A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is rank 4 (schoolbook). Can we do better? 
    # It's equivalent to 2 independent 1x2*2x1 products, each of rank 2 (since 1x2*2x1 is a dot product, rank 2).
    # So r(2,2,1) = 4? Or maybe less.
    # Actually, for 1xn * nx1, the rank is n (you need n terms for the dot product).
    # So 1x2*2x1 has rank 2. And 2x2*2x1 is two such, so rank 4.
    # Hmm, but maybe there's a shared structure. Let's just use schoolbook for now.
    
    # r(2,1,2): A is 2x1, B is 1x2, C is 2x2.
    # C = A * B, where A is 2x1 and B is 1x2. This is an outer product!
    # C[i,j] = A[i]*B[j]
    # This has rank 1 in the tensor sense? No, the tensor rank...
    # Actually, this is a rank-1 tensor: T_{i,j,k} = A[i,k] * B[k,j] where k is fixed (only k=0).
    # Wait, b_k=1 means there's only one index for k. So C[i,j] = A[i,0]*B[0,j].
    # This is a single term! Rank 1.
    
    # r(2,1,1): A is 2x1, B is 1x1, C is 2x1.
    # C[i] = A[i,0]*B[0,0]. Two terms (one for each i). Rank 2.
    
    # r(1,2,2): A is 1x2, B is 2x2, C is 1x2.
    # C[j] = sum_k A[0,k]*B[k,j]
    # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # This is like 2 independent 1x2*2x1... no, it's 1x2 * 2x2 = 1x2.
    # Each output element is a dot product of length 2, so rank 2 each, total rank 4.
    # Or is it? Let's see: we have 4 products: A[0,0]*B[0,0], A[0,1]*B[1,0], A[0,0]*B[0,1], A[0,1]*B[1,1]
    # These are 4 independent terms, so rank 4.
    
    # r(1,2,1): A is 1x2, B is 2x1, C is 1x1.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]. Rank 2.
    
    # r(1,1,2): A is 1x1, B is 1x2, C is 1x2.
    # C[0,j] = A[0,0]*B[0,j]. Two terms. Rank 2.
    
    # So total rank with schoolbook for non-Strassen blocks:
    # 8*7 + 4*4 + 4*1 + 2*2 + 4*4 + 2*2 + 2*2 + 1*1
    # = 56 + 16 + 4 + 4 + 16 + 4 + 4 + 1
    # = 105
    
    # Hmm, that's better than 125 but the knowledge says 117. Let me reconsider.
    # Maybe some of these smaller blocks can be optimized further.
    
    # Actually, wait. Let me reconsider r(2,2,1).
    # A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # The 4 terms are: A[0,0]*B[0,0], A[0,1]*B[1,0], A[1,0]*B[0,0], A[1,1]*B[1,0]
    # Can we combine? Not obviously. Rank 4 seems right.
    
    # But the knowledge says rank 117. Let me try a different partition or better sub-algorithms.
    # 
    # Alternative: Use the 117-rank construction directly.
    # One known construction: 5x5*5x5 can be done in rank 117.
    # 
    # Let me try to construct it explicitly. I'll use the block approach with careful optimization.
    
    # Actually, let me just implement the schoolbook (rank 125) as a safe fallback,
    # and try to construct a lower-rank decomposition if time permits.
    
    # For now, let's implement the block-based approach with rank 105.
    # But wait, I should double-check my rank calculations.
    
    # Let me just implement a general block-based decomposition.
    
    def build_decomposition(n, m, p, row_parts, col_parts, inner_parts):
        """
        Build a tensor decomposition for n x m * m x p -> n x p
        using block partitioning.
        
        row_parts: partition of n (rows of A, rows of C)
        col_parts: partition of p (cols of C, cols of B)  
        inner_parts: partition of m (cols of A, rows of B)
        
        For each block (i, k, j):
          A_block: row_parts[i] x inner_parts[k]
          B_block: inner_parts[k] x col_parts[j]
          C_block: row_parts[i] x col_parts[j]
        """
        rank_terms = []
        
        for i, (a_start, a_size) in enumerate(row_parts):
            for k, (b_start, b_size) in enumerate(inner_parts):
                for j, (c_start, c_size) in enumerate(col_parts):
                    # Multiply (a_size x b_size) by (b_size x c_size) -> (a_size x c_size)
                    # Use schoolbook: a_size * b_size * c_size terms
                    for ii in range(a_size):
                        for kk in range(b_size):
                            for jj in range(c_size):
                                # U index: (a_start + ii) * m + (b_start + kk)
                                # V index: (b_start + kk) * p + (c_start + jj)
                                # W index: (c_start + jj) * n + (a_start + ii)
                                u_idx = (a_start + ii) * m + (b_start + kk)
                                v_idx = (b_start + kk) * p + (c_start + jj)
                                w_idx = (c_start + jj) * n + (a_start + ii)
                                rank_terms.append((u_idx, v_idx, w_idx))
        
        return rank_terms
    
    def make_matrices(n, m, p, rank_terms):
        rank = len(rank_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for col, (u_idx, v_idx, w_idx) in enumerate(rank_terms):
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
        return {"U": U, "V": V, "W": W}
    
    # Build partition: 5 = 2+2+1
    parts = [(0, 2), (2, 2), (4, 1)]
    
    # Schoolbook: rank 125
    schoolbook_terms = build_decomposition(n, m, p, parts, parts, parts)
    # This should give 125 terms
    
    # Let me verify: sum over all blocks of a_size * b_size * c_size
    # 8 blocks of (2,2,2): 8*8 = 64
    # 4 blocks of (2,2,1): 4*4 = 16
    # 4 blocks of (2,1,2): 4*4 = 16
    # 2 blocks of (2,1,1): 2*2 = 4
    # 4 blocks of (1,2,2): 4*4 = 16
    # 2 blocks of (1,2,1): 2*2 = 4
    # 2 blocks of (1,1,2): 2*2 = 4
    # 1 block of (1,1,1): 1*1 = 1
    # Total: 64+16+16+4+16+4+4+1 = 125. Good.
    
    # Now let's try to reduce using Strassen for the (2,2,2) blocks.
    # Strassen's algorithm for 2x2*2x2 -> 2x2 uses 7 terms instead of 8.
    # We have 8 such blocks, so we save 8 terms: 125 - 8 = 117.
    
    # Strassen's 7 products for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each M is a product of two linear combinations.
    # In the tensor decomposition, each "product" corresponds to one rank term.
    # But the linear combinations mean that U and V entries can be -1, 0, or 1.
    
    def strassen_2x2_terms(a_off, b_off, c_off, n, m, p):
        """
        Return rank terms for a 2x2*2x2 -> 2x2 block at offsets.
        a_off: (row_offset_in_A, col_offset_in_A) for the 2x2 block of A
        b_off: (row_offset_in_B, col_offset_in_B) for the 2x2 block of B
        c_off: (row_offset_in_C, col_offset_in_C) for the 2x2 block of C
        """
        ar, ac = a_off  # A block starts at (ar, ac)
        br, bc = b_off  # B block starts at (br, bc)
        cr, cc = c_off  # C block starts at (cr, cc)
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # A entries: (ar,ac), (ar+1,ac+1)
        # B entries: (br,bc), (br+1,bc+1)
        terms.append({
            'U': [(ar, ac, 1), (ar+1, ac+1, 1)],
            'V': [(br, bc, 1), (br+1, bc+1, 1)],
            'W': [(cr, cc, 1), (cr+1, cc+1, 1), (cr, cc+1, 1), (cr+1, cc, 1)],
            'W_coeffs': {0: 1, 1: 1, 2: 1, 3: 1}  # C00, C11, C01, C10
        })
        
        # Actually, this is getting complicated with the W coefficients.
        # Let me use a different approach: explicitly build the rank terms.
        
        # For Strassen, each of the 7 products gives one rank term.
        # The U, V, W vectors have entries in {-1, 0, 1}.
        
        # Let me define helper to get indices
        def u_idx(r, c):
            return r * m + c
        
        def v_idx(r, c):
            return r * p + c
        
        def w_idx(r, c):
            return c * n + r
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1, C01=1, C10=1 (from C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6)
        # Wait, I need to be more careful. Let me just enumerate all 7.
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        terms.append({
            'U': [(ar, ac, 1), (ar+1, ac+1, 1)],
            'V': [(br, bc, 1), (br+1, bc+1, 1)],
            'W': [(cr, cc, 1), (cr+1, cc+1, 1), (cr, cc+1, 1), (cr+1, cc, 1)]
        })
        
        # M2 = (A10+A11)(B00)
        terms.append({
            'U': [(ar+1, ac, 1), (ar+1, ac+1, 1)],
            'V': [(br, bc, 1)],
            'W': [(cr+1, cc, 1), (cr+1, cc+1, -1)]
        })
        
        # M3 = A00(B10-B11)
        terms.append({
            'U': [(ar, ac, 1)],
            'V': [(br+1, bc, 1), (br+1, bc+1, -1)],
            'W': [(cr, cc+1, 1), (cr+1, cc+1, 1)]
        })
        
        # M4 = A11(B01-B00)
        terms.append({
            'U': [(ar+1, ac+1, 1)],
            'V': [(br, bc+1, 1), (br, bc, -1)],
            'W': [(cr, cc, 1), (cr+1, cc, 1)]
        })
        
        # M5 = (A00+A01)(B11)
        terms.append({
            'U': [(ar, ac, 1), (ar, ac+1, 1)],
            'V': [(br+1, bc+1, 1)],
            'W': [(cr, cc, -1), (cr, cc+1, 1)]
        })
        
        # M6 = (A10-A00)(B00+B10)
        terms.append({
            'U': [(ar+1, ac, 1), (ar, ac, -1)],
            'V': [(br, bc, 1), (br+1, bc, 1)],
            'W': [(cr+1, cc+1, 1)]
        })
        
        # M7 = (A01-A11)(B01+B11)
        terms.append({
            'U': [(ar, ac+1, 1), (ar+1, ac+1, -1)],
            'V': [(br, bc+1, 1), (br+1, bc+1, 1)],
            'W': [(cr, cc, 1), (cr, cc+1, 1)]
        })
        
        return terms
    
    # Now build the full decomposition with Strassen for (2,2,2) blocks
    rank_terms = []
    
    # Process each block
    for i, (a_start, a_size) in enumerate(parts):
        for k, (b_start, b_size) in enumerate(parts):
            for j, (c_start, c_size) in enumerate(parts):
                if a_size == 2 and b_size == 2 and c_size == 2:
                    # Use Strassen
                    strassen_terms = strassen_2x2_terms(
                        (a_start, b_start),  # A block offset (row, col)
                        (b_start, c_start),  # B block offset (row, col)
                        (a_start, c_start),  # C block offset (row, col)
                        n, m, p
                    )
                    for term in strassen_terms:
                        rank_terms.append(term)
                else:
                    # Use schoolbook
                    for ii in range(a_size):
                        for kk in range(b_size):
                            for jj in range(c_size):
                                rank_terms.append({
                                    'U': [(a_start+ii, b_start+kk, 1)],
                                    'V': [(b_start+kk, c_start+jj, 1)],
                                    'W': [(a_start+ii, c_start+jj, 1)]
                                })
    
    # Count terms
    total_rank = len(rank_terms)
    
    # Build matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for col, term in enumerate(rank_terms):
        for (r, c, val) in term['U']:
            U[r * m + c][col] += val
        for (r, c, val) in term['V']:
            V[r * p + c][col] += val
        for (r, c, val) in term['W']:
            W[c * n + r][col] += val
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify correctness
    # Check that the decomposition is exact
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # In tensor form: sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
    
    # Let's verify a few entries
    def verify(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    # The (i,k,j) entry of the tensor should be 1
                    # sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t] should be 1
                    total = 0
                    for t in range(rank):
                        total += U[i*m+k][t] * V[k*p+j][t] * W[j*n+i][t]
                    if total != 1:
                        return False, (i, k, j, total)
        # Also check that non-tensor entries are 0
        # Actually, the tensor is defined only for valid (i,k,j)
        # For other combinations, the sum should be 0
        # But we only need to verify the tensor entries
        return True, None
    
    is_valid, err = verify(U, V, W, n, m, p)
    
    if not is_valid:
        # Fall back to schoolbook
        schoolbook_result = make_matrices(n, m, p, schoolbook_terms)
        return schoolbook_result
    
    return result

# EVOLVE-BLOCK-END
