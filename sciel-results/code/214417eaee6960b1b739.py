"""Tensor decomposition for 3x4x6 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 6]


def _add_term(U, V, W, u_idx, u_val, v_idx, v_val, w_idx, w_val):
    U[u_idx] = u_val
    V[v_idx] = v_val
    W[w_idx] = w_val


def _add_strassen_block(U, V, W, base_u, base_v, base_w, n2, m2, p2, rank_offset):
    """Add a 2x2x2 Strassen block at the given base indices.
    
    For a 2x2x2 block, Strassen uses 7 multiplications.
    The indices:
    - U index: i*m2 + j for i in 0..1, j in 0..1 -> offset by base_u
    - V index: j*p2 + k for j in 0..1, k in 0..1 -> offset by base_v  
    - W index: k*n2 + i for k in 0..1, i in 0..1 -> offset by base_w
    
    Wait, I need to be careful. The block is n2 x m2 x p2 where n2=m2=p2=2.
    Within the block:
    - U row: i*m2 + j where i in 0..n2-1, j in 0..m2-1
    - V row: j*p2 + k where j in 0..m2-1, k in 0..p2-1
    - W row: k*n2 + i where k in 0..p2-1, i in 0..n2-1
    
    For the full problem, these local indices are offset by base_u, base_v, base_w.
    """
    # Strassen's 7 products for 2x2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(f+h)
    # C[0,0] = m1 + m4 - m5 - m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m7
    
    # In tensor terms, for the 2x2x2 block at position (i0, j0, k0):
    # U[i0+m2*i, j0+m2*j] contributes to input A[i,j]
    # V[j0+m2*j, k0+p2*k] contributes to input B[j,k]
    # W[k0+p2*k, i0+n2*i] contributes to output C[i,k]
    
    # The 7 Strassen terms:
    strassen_terms = [
        # (coeff_U, coeff_V, coeff_W) for each of the 7 rank-1 terms
        # Each term is a product u*r*v*r*w*r where the indices are shifted
        # We need to express each Strassen product as a sum of rank-1 terms
        
        # m1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
        # This contributes to C[0,0] and C[1,1]
        
        # Let me use the standard 7-term decomposition:
        # Term 1: (A00+A11)(B00+B11) -> contributes to C00 and C11
        # Term 2: (A10+A11)(B00) -> contributes to C10 and C11  
        # Term 3: (A00)(B01-B11) -> contributes to C01 and C11
        # Term 4: (A11)(B10-A00) -> contributes to C10 and C00
        # Term 5: (A00+A01)(B11) -> contributes to C01 and C00
        # Term 6: (A10-A00)(B00+B10) -> contributes to C10 and C11
        # Term 7: (A01-A11)(B01+B11) -> contributes to C01 and C00
    ]
    
    # Let me define the 7 rank-1 terms explicitly:
    # For a 2x2x2 block, the 7 terms are:
    # t1: U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1); W has 1 at (0,0) and (1,1)
    #     Actually no. Each rank-1 term is a single (u, v, w) triple.
    
    # The standard Strassen decomposition in tensor form:
    # The 2x2x2 matrix multiplication tensor has rank 7.
    # The 7 rank-1 decompositions (u_i, v_i, w_i) where:
    # T = sum_i u_i ⊗ v_i ⊗ w_i
    
    # For A[i,j], B[j,k], C[i,k]:
    # T(i,j,k) = 1 if the triple (i,j,k) appears in the multiplication
    
    # Strassen's 7 terms (using 0-indexed):
    # Term 0: u = (1,0,0,1) for (i,j) in {(0,0),(1,1)}, v = (1,0,0,1) for (j,k) in {(0,0),(1,1)}, w = (1,0,0,1) for (k,i) in {(0,0),(1,1)}
    # Hmm, this is getting confusing. Let me just enumerate the 7 rank-1 terms directly.
    
    # The 7 terms for 2x2x2 (using flat indices):
    # U flat index: i*2 + j
    # V flat index: j*2 + k  
    # W flat index: k*2 + i
    
    terms = [
        # (u_indices, v_indices, w_indices) - each is a list of (index, coefficient)
        # Term 1: (A00+A11)(B00+B11)
        # This product appears in the tensor with coefficient 1 for certain (i,j,k)
        # Actually, let me just list the 7 rank-1 vectors directly:
    ]
    
    # Standard Strassen rank-1 decomposition for 2x2x2:
    # The 7 rank-1 tensors are:
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)  -- no, this isn't right either
    
    # Let me think differently. For the 2x2x2 case:
    # Input A is 2x2: indices (0,0),(0,1),(1,0),(1,1) -> flat 0,1,2,3
    # Input B is 2x2: indices (0,0),(0,1),(1,0),(1,1) -> flat 0,1,2,3
    # Output C is 2x2: indices (0,0),(0,1),(1,0),(1,1) -> flat 0,1,2,3
    # But W uses trace convention: W[k*n+i, r] so for 2x2, W flat index = k*2+i
    
    # The tensor T(i,j,k) = 1 for all i,j,k in {0,1}
    # We need 7 rank-1 decompositions.
    
    # Strassen's 7 terms:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(f+h)
    
    # C00 = m1 + m4 - m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m7
    
    # Now express each mi as a product of linear forms in A and B:
    # m1: u1 = (1,0,0,1) [a+d], v1 = (1,0,0,1) [e+h], contributes to w with:
    #   C00 gets +1, C11 gets +1 -> w1 has 1 at C00 (flat 0) and C11 (flat 3)
    #   In W trace convention: C00 is k=0,i=0 -> flat 0; C11 is k=1,i=1 -> flat 3
    #   So w1 = (1,0,0,1)
    
    # m2: u2 = (0,0,1,1) [c+d], v2 = (1,0,0,0) [e], contributes to:
    #   C10 gets +1, C11 gets -1 -> w2 has 1 at C10 (flat 2) and -1 at C11 (flat 3)
    #   C10: k=0,i=1 -> flat 2; C11: k=1,i=1 -> flat 3
    #   So w2 = (0,0,1,-1)
    
    # m3: u3 = (1,0,0,0) [a], v3 = (0,1,0,-1) [f-h], contributes to:
    #   C01 gets +1, C11 gets +1 -> w3 has 1 at C01 (flat 1) and C11 (flat 3)
    #   C01: k=1,i=0 -> flat 2... wait
    #   W[k*n+i] with n=2: C[i,k] -> W[k*2+i]
    #   C01: i=0,k=1 -> W[1*2+0] = W[2]
    #   C11: i=1,k=1 -> W[1*2+1] = W[3]
    #   So w3 = (0,0,1,1)
    
    # m4: u4 = (0,0,0,1) [d], v4 = (0,0,1,-1) [g-c], contributes to:
    #   C00 gets +1, C10 gets +1 -> w4 has 1 at C00 (W[0]) and C10 (W[2])
    #   So w4 = (1,0,1,0)
    
    # m5: u5 = (1,1,0,0) [a+b], v5 = (0,0,0,1) [h], contributes to:
    #   C00 gets -1, C01 gets +1 -> w5 has -1 at C00 (W[0]) and 1 at C01 (W[2])
    #   So w5 = (-1,0,1,0)
    
    # m6: u6 = (0,0,1,-1) [c-a], v6 = (1,0,1,0) [e+g], contributes to:
    #   C10 gets +1, C11 gets +1 -> w6 has 1 at C10 (W[2]) and C11 (W[3])
    #   So w6 = (0,0,1,1)
    
    # m7: u7 = (0,1,0,-1) [b-d], v7 = (0,1,0,1) [f+h], contributes to:
    #   C00 gets -1, C11 gets +1 -> w7 has -1 at C00 (W[0]) and 1 at C11 (W[3])
    #   So w7 = (-1,0,0,1)
    
    # Now collect all 7 terms:
    strassen_7 = [
        # (u_coeffs, v_coeffs, w_coeffs) for flat 4-vector each
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),    # m1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),   # m2
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 0, 1, 1)),   # m3
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),   # m4
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 0, 1, 0)),   # m5
        ((0, 0, 1, -1), (1, 0, 1, 0), (0, 0, 1, 1)),   # m6
        ((0, 1, 0, -1), (0, 1, 0, 1), (-1, 0, 0, 1)),  # m7
    ]
    
    for u_c, v_c, w_c in strassen_7:
        # u_c is a 4-tuple for U flat indices within the 2x2 block
        # u flat index within block: i*2+j -> offset by base_u
        # v flat index within block: j*2+k -> offset by base_v
        # w flat index within block: k*2+i -> offset by base_w
        
        # For each term, u_c, v_c, w_c each have 4 entries
        # But a rank-1 term means each is a single vector, and the product is outer product
        # Wait no. In tensor decomposition T = sum_r u_r ⊗ v_r ⊗ w_r
        # Each r gives one column in U, V, W
        # u_r is a vector of length n*m, v_r of length m*p, w_r of length p*n
        # For a block, u_r has nonzeros only in the block's U rows
        
        # So for term with u_c = (a0,a1,a2,a3):
        # U[base_u + 0][rank_offset] += a0
        # U[base_u + 1][rank_offset] += a1
        # U[base_u + 2][rank_offset] += a2
        # U[base_u + 3][rank_offset] += a3
        # Similarly for V and W
        
        for idx in range(4):
            U[base_u + idx][rank_offset] += u_c[idx]
            V[base_v + idx][rank_offset] += v_c[idx]
            W[base_w + idx][rank_offset] += w_c[idx]


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: Split n=3 into 1+2
    # Part 1: n1=1, m=4, p=6 -> rank 1*4*6 = 24 (schoolbook)
    # Part 2: n2=2, m=4, p=6
    #   Split m=4 into 2+2
    #   Part 2a: n2=2, m1=2, p=6
    #   Part 2b: n2=2, m2=2, p=6
    #   Each 2x2x6: split p=6 into 2+2+2
    #   Three 2x2x2 blocks each with Strassen rank 7
    #   Each 2x2x6 has rank 21
    # Total rank = 24 + 21 + 21 = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: n1=1, m=4, p=6, schoolbook rank 24
    # i in 0..0, j in 0..3, k in 0..5
    for i in range(1):
        for j in range(4):
            for k in range(6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 2: n2=2, m=4, p=6
    # i in 1..2 (rows 1 and 2 of A), j in 0..3, k in 0..5
    
    # Part 2a: i in 1..2, j in 0..1, k in 0..5
    # Split into three 2x2x2 blocks for k in {0,1}, {2,3}, {4,5}
    for block in range(3):
        k0 = block * 2
        # base_u: i starts at 1, j starts at 0 -> U flat = i*4 + j, so base = 1*4 + 0 = 4
        # But within the 2x2 block, local i in 0..1 maps to global i = 1+local_i
        # local j in 0..1 maps to global j = 0+local_j
        # So U flat = (1+local_i)*4 + (0+local_j) = 4 + local_i*4 + local_j
        # For a 2x2 U block starting at U[4], the flat indices are 4,5,8,9
        # But Strassen expects consecutive flat indices 0,1,2,3 within the block
        # U flat: i*2+j for i,j in 0..1 -> 0,1,2,3
        # Global: (1+li)*4 + (0+lj) = 4 + li*4 + lj
        # li=0,lj=0 -> 4; li=0,lj=1 -> 5; li=1,lj=0 -> 8; li=1,lj=1 -> 9
        # These are not consecutive! Strassen needs the 2x2 submatrix to be contiguous.
        # 
        # I need to handle the indexing differently. The 2x2 block in U is not contiguous
        # in the flat indexing because m=4 not m=2.
        #
        # Let me reconsider. For the 2x2x2 Strassen block, I need to map:
        # local (li, lj, lk) to global (i, j, k)
        # i = 1 + li, j = lj, k = k0 + lk
        # U global flat = i*4 + j = (1+li)*4 + lj
        # V global flat = j*6 + k = lj*6 + (k0+lk)
        # W global flat = k*3 + i = (k0+lk)*3 + (1+li)
        
        # For each of the 7 Strassen terms, I need to add contributions to the correct
        # global U, V, W rows.
        
        strassen_7 = [
            ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
            ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
            ((1, 0, 0, 0), (0, 1, 0, -1), (0, 0, 1, 1)),
            ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),
            ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 0, 1, 0)),
            ((0, 0, 1, -1), (1, 0, 1, 0), (0, 0, 1, 1)),
            ((0, 1, 0, -1), (0, 1, 0, 1), (-1, 0, 0, 1)),
        ]
        
        for u_c, v_c, w_c in strassen_7:
            for li in range(2):
                for lj in range(2):
                    u_flat_local = li * 2 + lj
                    u_coeff = u_c[u_flat_local]
                    if u_coeff != 0:
                        global_i = 1 + li
                        global_j = lj
                        U[global_i * m + global_j][col] += u_coeff
            
            for lj in range(2):
                for lk in range(2):
                    v_flat_local = lj * 2 + lk
                    v_coeff = v_c[v_flat_local]
                    if v_coeff != 0:
                        global_j = lj
                        global_k = k0 + lk
                        V[global_j * p + global_k][col] += v_coeff
            
            for lk in range(2):
                for li in range(2):
                    w_flat_local = lk * 2 + li
                    w_coeff = w_c[w_flat_local]
                    if w_coeff != 0:
                        global_k = k0 + lk
                        global_i = 1 + li
                        W[global_k * n + global_i][col] += w_coeff
            
            col += 1
    
    # Part 2b: i in 1..2, j in 2..3, k in 0..5
    for block in range(3):
        k0 = block * 2
        strassen_7 = [
            ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
            ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
            ((1, 0, 0, 0), (0, 1, 0, -1), (0, 0, 1, 1)),
            ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),
            ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 0, 1, 0)),
            ((0, 0, 1, -1), (1, 0, 1, 0), (0, 0, 1, 1)),
            ((0, 1, 0, -1), (0, 1, 0, 1), (-1, 0, 0, 1)),
        ]
        
        for u_c, v_c, w_c in strassen_7:
            for li in range(2):
                for lj in range(2):
                    u_flat_local = li * 2 + lj
                    u_coeff = u_c[u_flat_local]
                    if u_coeff != 0:
                        global_i = 1 + li
                        global_j = 2 + lj
                        U[global_i * m + global_j][col] += u_coeff
            
            for lj in range(2):
                for lk in range(2):
                    v_flat_local = lj * 2 + lk
                    v_coeff = v_c[v_flat_local]
                    if v_coeff != 0:
                        global_j = 2 + lj
                        global_k = k0 + lk
                        V[global_j * p + global_k][col] += v_coeff
            
            for lk in range(2):
                for li in range(2):
                    w_flat_local = lk * 2 + li
                    w_coeff = w_c[w_flat_local]
                    if w_coeff != 0:
                        global_k = k0 + lk
                        global_i = 1 + li
                        W[global_k * n + global_i][col] += w_coeff
            
            col += 1
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Should not happen, but safety guard
        pass
    
    return {"U": U, "V": V, "W": W}
