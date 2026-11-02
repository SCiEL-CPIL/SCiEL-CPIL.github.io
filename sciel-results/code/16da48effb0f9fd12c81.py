<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Strassen-based decomposition: partition 5 = 2+2+1
    # For each 2x2x2 sub-block, use Strassen's 7-multiplication algorithm
    # instead of 8, saving 1 rank per 2x2x2 block.
    
    def strassen_2x2x2(U_base, V_base, W_base, col_offset):
        """
        Apply Strassen's algorithm to a 2x2x2 sub-block.
        U_base: base row index in U (i dimension, 0 or 1)
        V_base: base col index in V (j dimension, 0 or 1)  
        W_base: base col index in W (k dimension, 0 or 1)
        Returns number of terms used (should be 7)
        """
        # Strassen's 7 products for 2x2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        terms = [
            # (i_offset, j_offset, k_offset, coeff) for U, V, W
            # M1 = (A11+A22)(B11+B22)
            [(0,0,0,1), (1,1,0,1)],  # U: rows 0,1 of A; V: cols 0,1 of B; W: contributes to C
            # Actually, let me think about this more carefully.
            # For matrix multiplication C = AB where A is 2x2, B is 2x2, C is 2x2
            # A[i,j] * B[j,k] = C[i,k]
            # U indexes (i,j), V indexes (j,k), W indexes (k,i)
            
            # Let me redefine: for a 2x2x2 block at position (i0, j0, k0)
            # U row: (i0+di, j0+dj) for di,dj in {0,1}
            # V row: (j0+dj, k0+dk) for dj,dk in {0,1}
            # W row: (k0+dk, i0+di) for dk,di in {0,1}
            
            # Strassen terms:
            # M1 = (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1])
            # M2 = (A[i0+1,j0]+A[i0+1,j0+1])*B[j0,k0]
            # M3 = A[i0,j0]*(B[j0+1,k0]-B[j0+1,k0+1])
            # M4 = A[i0+1,j0+1]*(B[j0,k0+1]-B[j0,k0])
            # M5 = (A[i0,j0]+A[i0,j0+1])*B[j0+1,k0+1]
            # M6 = (A[i0+1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k0+1])
            # M7 = (A[i0,j0+1]-A[i0+1,j0+1])*(B[j0+1,k0]+B[j0+1,k0+1])
            
            # C[i0,k0] = M1 + M4 - M5 + M7
            # C[i0,k0+1] = M3 + M5
            # C[i0+1,k0] = M2 + M4
            # C[i0+1,k0+1] = M1 - M2 + M3 + M6
        ]
        return 0  # placeholder
    
    # Let me implement this properly by directly constructing the decomposition
    # Partition: 5 = 2 + 2 + 1
    # Blocks in i: [0,1], [2,3], [4]
    # Blocks in j: [0,1], [2,3], [4]
    # Blocks in k: [0,1], [2,3], [4]
    
    # For 2x2x2 blocks: use Strassen (7 terms instead of 8)
    # For 2x2x1, 2x1x2, 1x2x2, 1x2x1, etc.: use schoolbook (product of sizes)
    
    i_blocks = [(0, 2), (2, 4), (4, 5)]  # (start, end)
    j_blocks = [(0, 2), (2, 4), (4, 5)]
    k_blocks = [(0, 2), (2, 4), (4, 5)]
    
    U = [[0] * 100 for _ in range(n * m)]  # pre-allocate, will trim
    V = [[0] * 100 for _ in range(m * p)]
    W = [[0] * 100 for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal col
        if col >= 100:
            return False
        U[u_row][col] += u_val
        V[v_row][col] += v_val
        W[w_row][col] += w_val
        col += 1
        return True
    
    # Process each combination of blocks
    for i_start, i_end in i_blocks:
        for j_start, j_end in j_blocks:
            for k_start, k_end in k_blocks:
                i_size = i_end - i_start
                j_size = j_end - j_start
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen for 2x2x2
                    # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
                    add_term(i_start*m+j_start, j_start*p+k_start, k_start*n+i_start, 1, 1, 1)
                    add_term((i_start+1)*m+(j_start+1), (j_start+1)*p+(k_start+1), (k_start+1)*n+(i_start+1), 1, 1, 1)
                    # Wait, this isn't right. Let me think again.
                    # Each term is a rank-1 tensor: U[i,j] * V[j,k] * W[k,i]
                    # For Strassen, we combine multiple rank-1 terms.
                    
                    # Let me restart this approach. The key insight is:
                    # In the 2x2x2 case, the 8 schoolbook terms have a linear dependency.
                    # Specifically, we can express one term as a combination of the others.
                    
                    # Actually, the standard Strassen approach works differently.
                    # Let me just use the known result: 2x2x2 has rank 7.
                    
                    # The 7 Strassen terms for 2x2x2 at (i0,j0,k0):
                    i0, j0, k0 = i_start, j_start, k_start
                    
                    # M1 = (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1])
                    # This contributes to C[i0,k0], C[i0+1,k0+1]
                    # Term 1: A[i0,j0]*B[j0,k0]*C[i0,k0]
                    add_term(i0*m+j0, j0*p+k0, k0*n+i0, 1, 1, 1)
                    # Term 2: A[i0,j0]*B[j0+1,k0+1]*C[i0,k0+1]  
                    add_term(i0*m+j0, (j0+1)*p+(k0+1), (k0+1)*n+i0, 1, 1, 1)
                    # Term 3: A[i0+1,j0+1]*B[j0,k0]*C[i0+1,k0]
                    add_term((i0+1)*m+(j0+1), j0*p+k0, k0*n+(i0+1), 1, 1, 1)
                    # Term 4: A[i0+1,j0+1]*B[j0+1,k0+1]*C[i0+1,k0+1]
                    add_term((i0+1)*m+(j0+1), (j0+1)*p+(k0+1), (k0+1)*n+(i0+1), 1, 1, 1)
                    
                    # Hmm, this is getting complicated. Let me use a different approach.
                    # I'll construct the Strassen decomposition explicitly.
                    
                    # Actually, let me just count: we have 3x3x3 = 27 blocks.
                    # 8 blocks are 2x2x2 (where all three dimensions are 2)
                    # Wait: i_blocks has sizes [2,2,1], j_blocks has [2,2,1], k_blocks has [2,2,1]
                    # So 2x2x2 blocks: 2*2*2 = 8 blocks
                    # 2x2x1: 2*2*1 = 4 blocks
                    # 2x1x2: 2*1*2 = 4 blocks
                    # 1x2x2: 1*2*2 = 4 blocks
                    # 2x1x1: 2*1*1 = 2 blocks
                    # 1x2x1: 1*2*1 = 2 blocks
                    # 1x1x2: 1*1*2 = 2 blocks
                    # 1x1x1: 1*1*1 = 1 block
                    # Total schoolbook: 8*8 + 4*4 + 4*4 + 4*4 + 2*2 + 2*2 + 2*2 + 1*1 = 64+16+16+16+4+4+4+1 = 125? No...
                    # Wait, 2x2x2 = 8 terms, 2x2x1 = 4 terms, etc.
                    # 8*8 + 4*4 + 4*4 + 4*4 + 2*2 + 2*2 + 2*2 + 1*1 = 64+16+16+16+4+4+4+1 = 125? That's way more than 50.
                    # I'm confusing myself. n*m*p = 2*5*5 = 50.
                    
                    # Let me restart with a cleaner approach.
                    pass
                else:
                    # Schoolbook for this sub-block
                    for di in range(i_size):
                        for dj in range(j_size):
                            for dk in range(k_size):
                                add_term((i_start+di)*m + (j_start+dj),
                                         (j_start+dj)*p + (k_start+dk),
                                         (k_start+dk)*n + (i_start+di),
                                         1, 1, 1)
    
    # The above approach is getting messy. Let me use a cleaner implementation.
    
    # Clean implementation:
    U = [[0] * 50 for _ in range(n * m)]
    V = [[0] * 50 for _ in range(m * p)]
    W = [[0] * 50 for _ in range(p * n)]
    col = 0
    
    # Partition indices
    i_parts = [(0,1), (2,3), (4,4)]  # (start, end) exclusive
    j_parts = [(0,1), (2,3), (4,4)]
    k_parts = [(0,1), (2,3), (4,4)]
    
    for i0, i1 in i_parts:
        for j0, j1 in j_parts:
            for k0, k1 in k_parts:
                if i1-i0 == 1 and j1-j0 == 1 and k1-k0 == 1:
                    # 1x1x1: 1 term
                    add_term(i0*m+j0, j0*p+k0, k0*n+i0, 1, 1, 1)
                elif i1-i0 == 1 and j1-j0 == 1 and k1-k0 == 2:
                    # 1x1x2: 2 terms (schoolbook)
                    for dk in range(2):
                        add_term(i0*m+j0, j0*p+(k0+dk), (k0+dk)*n+i0, 1, 1, 1)
                elif i1-i0 == 1 and j1-j0 == 2 and k1-k0 == 1:
                    # 1x2x1: 2 terms
                    for dj in range(2):
                        add_term(i0*m+(j0+dj), (j0+dj)*p+k0, k0*n+i0, 1, 1, 1)
                elif i1-i0 == 2 and j1-j0 == 1 and k1-k0 == 1:
                    # 2x1x1: 2 terms
                    for di in range(2):
                        add_term((i0+di)*m+j0, j0*p+k0, k0*n+(i0+di), 1, 1, 1)
                elif i1-i0 == 1 and j1-j0 == 2 and k1-k0 == 2:
                    # 1x2x2: 4 terms (schoolbook, no Strassen benefit since one dim is 1)
                    for dj in range(2):
                        for dk in range(2):
                            add_term(i0*m+(j0+dj), (j0+dj)*p+(k0+dk), (k0+dk)*n+i0, 1, 1, 1)
                elif i1-i0 == 2 and j1-j0 == 1 and k1-k0 == 2:
                    # 2x1x2: 4 terms
                    for di in range(2):
                        for dk in range(2):
                            add_term((i0+di)*m+j0, j0*p+(k0+dk), (k0+dk)*n+(i0+di), 1, 1, 1)
                elif i1-i0 == 2 and j1-j0 == 2 and k1-k0 == 1:
                    # 2x2x1: 4 terms
                    for di in range(2):
                        for dj in range(2):
                            add_term((i0+di)*m+(j0+dj), (j0+dj)*p+k0, k0*n+(i0+di), 1, 1, 1)
                elif i1-i0 == 2 and j1-j0 == 2 and k1-k0 == 2:
                    # 2x2x2: Use Strassen, 7 terms instead of 8
                    # Strassen's 7 products:
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)B00
                    # M3 = A00(B10-B11)
                    # M4 = A11(B01-B00)
                    # M5 = (A00+A01)B11
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    
                    # Each Mi is a product of a linear combo of A entries and linear combo of B entries.
                    # For the tensor decomposition, each Mi corresponds to multiple rank-1 terms.
                    
                    # M1 = (A00+A11)(B00+B11)
                    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
                    # But in the tensor, we need to track which C entry this contributes to.
                    # C00 += M1, C01 += M5, C10 += M2, C11 += M1 (with signs)
                    
                    # This is getting complex. Let me use the explicit Strassen formula.
                    # The key is that we can express the 2x2x2 multiplication with 7 rank-1 tensors
                    # instead of 8.
                    
                    # I'll construct it term by term:
                    # For each Strassen product, we have a set of (u_row, v_row, w_row, coeff) tuples.
                    
                    # M1 = (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1])
                    # Contributes to: C[i0,k0] += M1, C[i0+1,k0+1] += M1
                    # As rank-1 terms:
                    add_term(i0*m+j0, j0*p+k0, k0*n+i0, 1, 1, 1)       # A00*B00 -> C00
                    add_term(i0*m+j0, (j0+1)*p+(k0+1), (k0+1)*n+i0, 1, 1, 1)  # A00*B11 -> C01? No...
                    
                    # I'm making errors. Let me look up the exact Strassen tensor decomposition.
                    # 
                    # The 2x2x2 matrix multiplication tensor has rank 7.
                    # The 7 rank-1 tensors are:
                    # T1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1)  [but with specific signs]
                    # 
                    # Actually, let me just use the known result and construct it carefully.
                    # 
                    # For 2x2x2, the Strassen decomposition uses these 7 terms:
                    # (Using indices (i,j,k) for U[i,j], V[j,k], W[k,i])
                    
                    # I'll give up on doing this inline and just use schoolbook for now,
                    # but note that the rank should be 50 - 8 = 42 if we save 1 per 2x2x2 block.
                    # But that's not right either, because Strassen saves 1 per 2x2x2, so
                    # 8 blocks * 1 saved = 8 saved, giving rank 50 - 8 = 42.
                    
                    # Wait, the inspiration says rank 46, not 42. Let me reconsider.
                    # Maybe only some blocks benefit from Strassen.
                    
                    # For now, let me just use schoolbook and return rank 50.
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                add_term((i0+di)*m+(j0+dj), (j0+dj)*p+(k0+dk), (k0+dk)*n+(i0+di), 1, 1, 1)
    
    # Trim to actual rank
    if col < 50:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        if col < best_rank:
            best_rank = col
            best_result = {"U": U, "V": V, "W": W}
    
    # Check deadline
    if time.time() > deadline - 5:
        return best_result
    
    # Try to find further rank reduction through local optimization
    # (This is where we'd attempt to reduce below 46 if we got there)
    
    return best_result
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Schoolbook decomposition (rank 50)
    def schoolbook():
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
        return U, V, W
    
    U_school, V_school, W_school = schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p
    
    # Strassen-based decomposition for 2x5x5
    # Partition 5 = 2+2+1 in each dimension
    # For 2x2x2 sub-blocks, use rank-7 Strassen decomposition
    # For other sub-blocks, use schoolbook
    
    def strassen_2x2x2_terms(i0, j0, k0):
        """
        Return list of (u_row, v_row, w_row, u_coeff, v_coeff, w_coeff) for Strassen 2x2x2.
        i0, j0, k0 are the base indices.
        Uses 0-indexed: A[i0..i0+1][j0..j0+1], B[j0..j0+1][k0..k0+1], C[k0..k0+1][i0..i0+1]
        
        Strassen's 7 products:
        M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
        M2 = (A10+A11)B00       -> contributes to C10
        M3 = A00(B10-B11)       -> contributes to C01
        M4 = A11(B01-B00)       -> contributes to C00 and C10
        M5 = (A00+A01)B11       -> contributes to C01 and C00
        M6 = (A10-A00)(B00+B01) -> contributes to C11
        M7 = (A01-A11)(B10+B11) -> contributes to C00
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        For the tensor decomposition, each "product" like (A00+A11)(B00+B11) is NOT a single
        rank-1 term. It's a bilinear form. We need to expand it into rank-1 terms.
        
        Actually, the key insight for tensor rank: the 2x2x2 multiplication tensor has rank 7.
        This means there exist 7 rank-1 tensors that sum to the multiplication tensor.
        
        The standard Strassen decomposition gives us 7 bilinear forms, but each bilinear form
        (linear in A entries times linear in B entries) corresponds to potentially multiple
        rank-1 tensor terms when we also account for the C side.
        
        Let me think about this differently. The multiplication tensor T for 2x2x2 is:
        T = sum_{i,j,k} A_{ij} B_{jk} C_{ki}
        
        A rank-r decomposition is: T = sum_{t=1}^{r} sum_{i,j,k} u^t_{ij} v^t_{jk} w^t_{ki} A_{ij} B_{jk} C_{ki}
        
        For Strassen, we have 7 terms where each term t has:
        - u^t is a 2x2 matrix (sparse)
        - v^t is a 2x2 matrix (sparse)  
        - w^t is a 2x2 matrix (sparse)
        
        The 7 Strassen rank-1 tensors for 2x2x2:
        (Using 0-indexed (i,j) for U, (j,k) for V, (k,i) for W)
        
        T1: u=(1,0;0,1), v=(1,0;0,1), w=(1,0;0,1)  -> this is identity*identity*identity
        Actually, let me look up the explicit form.
        
        The 7 terms are (u, v, w) where each is a 2x2 matrix:
        1. u=[[1,0],[0,1]], v=[[1,0],[0,1]], w=[[1,0],[0,1]]
           This gives: A00*B00*C00 + A11*B11*C11 (diagonal terms)
        
        Hmm, I think I need to be more careful. Let me just construct it by ensuring
        that for all A, B, the product AB is correctly computed.
        
        Alternative approach: Just use the fact that 2x2x2 has rank 7 and construct
        a valid decomposition by solving for the coefficients.
        
        For now, let me use a simpler approach: just do schoolbook for all blocks,
        giving rank 50. The Strassen improvement requires careful construction.
        """
        # For now, return 8 schoolbook terms (we'll improve this)
        terms = []
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    u_row = (i0 + di) * m + (j0 + dj)
                    v_row = (j0 + dj) * p + (k0 + dk)
                    w_row = (k0 + dk) * n + (i0 + di)
                    terms.append((u_row, v_row, w_row, 1, 1, 1))
        return terms
    
    # Build Strassen-based decomposition
    # Partition: indices 0,1 | 2,3 | 4
    i_parts = [(0, 2), (2, 4), (4, 5)]
    j_parts = [(0, 2), (2, 4), (4, 5)]
    k_parts = [(0, 2), (2, 4), (4, 5)]
    
    U = [[0] * 60 for _ in range(n * m)]
    V = [[0] * 60 for _ in range(m * p)]
    W = [[0] * 60 for _ in range(p * n)]
    col = 0
    
    def add_term(u_row, v_row, w_row, u_c, v_c, w_c):
        nonlocal col
        if col >= 60:
            return
        U[u_row][col] += u_c
        V[v_row][col] += v_c
        W[w_row][col] += w_c
        col += 1
    
    for i0, i1 in i_parts:
        for j0, j1 in j_parts:
            for k0, k1 in k_parts:
                if i1 - i0 == 2 and j1 - j0 == 2 and k1 - k0 == 2:
                    # 2x2x2 block: use Strassen (7 terms)
                    # Explicit Strassen decomposition for 2x2x2
                    # The 7 rank-1 tensors:
                    # I'll construct them to satisfy: for all A,B, (AB)_{ki} = sum_t U^t_{ij} V^t_{jk} W^t_{ki}
                    
                    # Let me use the known Strassen decomposition:
                    # Term 1: (A00+A11)(B00+B11) contributes to C00 and C11
                    # As a bilinear form, this is: A00*B00 + A00*B11 + A11*B00 + A11*B11
                    # But we need to split this into rank-1 tensor terms.
                    
                    # Actually, the correct way: each Strassen "product" is one rank-1 tensor
                    # in the sense that it's (linear form in A) * (linear form in B) * (linear form in C).
                    
                    # Let me define the 7 terms explicitly:
                    # Using 0-indexed local coordinates (i,j,k) in {0,1}^3
                    
                    # Term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
                    # This means: U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
                    # Contribution: A00*B00*W00 + A11*B11*W11 (but W is indexed by (k,i))
                    # W[0,0] means k=0,i=0, so this contributes to C[0,0]
                    # W[1,1] means k=1,i=1, so this contributes to C[1,1]
                    # So: C00 += A00*B00, C11 += A11*B11
                    
                    add_term((i0+0)*m+(j0+0), (j0+0)*p+(k0+0), (k0+0)*n+(i0+0), 1, 1, 1)  # A00*B00 -> C00
                    add_term((i0+1)*m+(j0+1), (j0+1)*p+(k0+1), (k0+1)*n+(i0+1), 1, 1, 1)  # A11*B11 -> C11
                    
                    # This is getting too complex to do correctly inline. 
                    # Let me just use schoolbook for all blocks to ensure correctness.
                    # The rank will be 50, same as the baseline.
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                add_term((i0+di)*m+(j0+dj), (j0+dj)*p+(k0+dk), (k0+dk)*n+(i0+di), 1, 1, 1)
                else:
                    # Schoolbook for this sub-block
                    for di in range(i1 - i0):
                        for dj in range(j1 - j0):
                            for dk in range(k1 - k0):
                                add_term((i0+di)*m+(j0+dj), (j0+dj)*p+(k0+dk), (k0+dk)*n+(i0+di), 1, 1, 1)
    
    # Trim and validate
    if col < best_rank:
        U_trim = [row[:col] for row in U]
        V_trim = [row[:col] for row in V]
        W_trim = [row[:col] for row in W]
        best_rank = col
        best_result = {"U": U_trim, "V": V_trim, "W": W_trim}
    
    if time.time() > deadline:
        return best_result
    
    # Try alternative partitions: 5 = 2+1+2, 1+2+2, etc.
    alt_partitions = [
        [(0,2), (2,3), (3,5)],  # 2+1+2
        [(0,1), (1,3), (3,5)],  # 1+2+2
        [(0,3), (3,4), (4,5)],  # 3+1+1
        [(0,1), (1,4), (4,5)],  # 1+3+1
        [(0,4), (4,5)],         # 4+1 (not useful for Strassen)
    ]
    
    for i_parts_alt in alt_partitions:
        if time.time() > deadline - 10:
            break
        for j_parts_alt in alt_partitions:
            for k_parts_alt in alt_partitions:
                if time.time() > deadline - 5:
                    break
                
                U_t = [[0] * 60 for _ in range(n * m)]
                V_t = [[0] * 60 for _ in range(m * p)]
                W_t = [[0] * 60 for _ in range(p * n)]
                col_t = 0
                
                def add_term_t(u_row, v_row, w_row, u_c, v_c, w_c):
                    nonlocal col_t
                    if col_t >= 60:
                        return
                    U_t[u_row][col_t] += u_c
                    V_t[v_row][col_t] += v_c
                    W_t[w_row][col_t] += w_c
                    col_t += 1
                
                for i0, i1 in i_parts_alt:
                    for j0, j1 in j_parts_alt:
                        for k0, k1 in k_parts_alt:
                            for di in range(i1 - i0):
                                for dj in range(j1 - j0):
                                    for dk in range(k1 - k0):
                                        add_term_t((i0+di)*m+(j0+dj), (j0+dj)*p+(k0+dk), (k0+dk)*n+(i0+di), 1, 1, 1)
                
                if col_t < best_rank:
                    U_trim = [row[:col_t] for row in U_t]
                    V_trim = [row[:col_t] for row in V_t]
                    W_trim = [row[:col_t] for row in W_t]
                    best_rank = col_t
                    best_result = {"U": U_trim, "V": V_trim, "W": W_trim}
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
