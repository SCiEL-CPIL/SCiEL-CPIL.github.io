import numpy as np
import time

# EVOLVE-BLOCK-START
_n, _m, _p = 4, 4, 4
_rank = 49
_DEADLINE = time.time() + 110

def _build_decomposition():
    """
    Constructs the rank-49 Strassen decomposition for 4x4 matrix multiplication.
    Returns U, V, W as lists of lists.
    """
    # U: (16, 49), V: (16, 49), W: (16, 49)
    U = [[0] * _rank for _ in range(_n * _m)]
    V = [[0] * _rank for _ in range(_m * _p)]
    W = [[0] * _rank for _ in range(_p * _n)]
    
    # Indices for 4x4 matrices mapped to 0..15
    # A: 4x4, index i,j -> i*4+j
    # B: 4x4, index i,j -> i*4+j
    # C: 4x4, index i,j -> i*4+j (Note: Prompt says W[k*n+i, r] for trace convention? 
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # Usually C = AB. C_{i,j} = sum_k A_{i,k} B_{k,j}.
    # Tensor T_{i,j,k} = C_{i,j} from A_{i,k} B_{k,j}?
    # Standard tensor decomposition for matmul:
    # C_{i,j} = sum_r U_{i,k}^{(r)} V_{k,j}^{(r)} ... wait.
    # The tensor is T_{i,j,k} where i,j are C indices, k is A col/B row index.
    # T_{i,j,k} = 1 if ... no.
    # T_{i,j,k} corresponds to the term A_{i,k} B_{k,j} contributing to C_{i,j}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # The multiplication tensor is defined such that <T, A \otimes B> = <C, ...>
    # Usually, we look for rank-r decomposition:
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    # where u_r is a 4x4 matrix (A factor), v_r is a 4x4 matrix (B factor), w_r is a 4x4 matrix (C factor).
    # C = sum_r (u_r * A) \circ (v_r * B) \circ w_r? No.
    # C_{i,j} = sum_r (sum_k U_{i,k}^{(r)} A_{k, ?}) ...
    
    # Let's stick to the explicit Strassen construction which is known to be correct.
    # The previous code did this. I will just clean it up.
    
    def get_U_index(i, j): return i * _m + j
    def get_V_index(i, j): return i * _p + j
    # W index: Prompt says "W[k*n+i, r]". This is confusing.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # In many contexts, the C matrix is indexed by row, col.
    # If W is a vector of length p*n, and it corresponds to C_{i,j},
    # and the convention is W[k*n+i], with k=col, i=row?
    # Let's assume W index = col * n + row.
    # For 4x4, n=4. W index = j * 4 + i.
    # Let's verify with the previous code.
    # Previous code: `get_W_index(row, col): return col * _n + row`.
    # Yes, it used col * n + row.
    
    def get_W_index(i, j): return j * _n + i
    
    # Strassen steps for 4x4 (recursive 2x2 of 2x2 blocks)
    # We can define the 7 Strassen products for 2x2, and apply them recursively.
    # Or just hardcode the 49 terms.
    # Recursive approach:
    # Let's represent the terms as (U_coeff, V_coeff, W_coeff) for each of the 49 rank-1 components.
    
    # A 4x4 matrix is 4 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # Strassen for 2x2 blocks uses 7 products of 2x2 matrices.
    # Each 2x2 product is decomposed into 7 products of scalars.
    # Total 49 scalar products.
    
    # We will generate the coefficients for U, V, W.
    
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each of the 7 M's, we have 7 scalar multiplications.
    # We need to map these to the 4x4 indices.
    
    # Let's define the 2x2 block indices for 4x4.
    # Block (bi, bj) covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1.
    
    def block_indices(bi, bj):
        rows = [2*bi, 2*bi+1]
        cols = [2*bj, 2*bj+1]
        return rows, cols
    
    # We will iterate over the 7 top-level Strassen products (M1..M7)
    # and for each, iterate over the 7 inner Strassen products (m1..m7).
    
    # Top-level coefficients for A and B blocks
    # M1: A: A00, A11 ; B: B00, B11
    # M2: A: A10, A11 ; B: B00
    # M3: A: A00 ; B: B01, B11
    # M4: A: A11 ; B: B10, B00
    # M5: A: A00, A01 ; B: B11
    # M6: A: A10, A00 ; B: B00, B01
    # M7: A: A01, A11 ; B: B10, B11
    
    top_level = [
        # (A_blocks, B_blocks) where each block is (bi, bj) and coeff
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]), # M1
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),             # M2
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),            # M3
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),            # M4
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),             # M5
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),# M6
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]) # M7
    ]
    
    # Inner-level coefficients for 2x2 blocks
    # For a 2x2 matrix, indices are 0,1.
    # m1: A: (0,0),(1,1) ; B: (0,0),(1,1)
    # m2: A: (1,0),(1,1) ; B: (0,0)
    # m3: A: (0,0) ; B: (0,1),(1,1)
    # m4: A: (1,1) ; B: (1,0),(0,0)
    # m5: A: (0,0),(0,1) ; B: (1,1)
    # m6: A: (1,0),(0,0) ; B: (0,0),(0,1)
    # m7: A: (0,1),(1,1) ; B: (1,0),(1,1)
    
    inner_level = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]), # m1
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),             # m2
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),            # m3
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),            # m4
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),             # m5
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),# m6
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]) # m7
    ]
    
    # C contributions from inner products for a 2x2 block
    # c00 = m1 - m4 + m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    c_contrib = [
        [(0,0, 1), (1,1, 1)], # m1 -> C00? No.
        # m1 contributes to C00 and C11
        # m2 contributes to C10 and C11
        # m3 contributes to C01 and C11
        # m4 contributes to C00 and C10
        # m5 contributes to C00 and C01
        # m6 contributes to C11
        # m7 contributes to C00
        
        # Let's list (row, col, coeff) for each inner product m_k
        [((0,0), 1), ((1,1), 1)], # m1
        [((1,0), 1), ((1,1), -1)],# m2
        [((0,1), 1), ((1,1), 1)], # m3
        [((0,0), -1), ((1,0), 1)],# m4
        [((0,0), 1), ((0,1), 1)], # m5
        [((1,1), 1)],             # m6
        [((0,0), -1)]             # m7
    ]
    
    # Now, combine top and inner levels.
    # Rank index r = t * 7 + k, where t is top-level (0..6), k is inner-level (0..6).
    
    for t in range(7):
        a_blocks, b_blocks = top_level[t]
        for k in range(7):
            a_inner, b_inner = inner_level[k]
            rank_idx = t * 7 + k
            
            # Build U coefficients
            # U is 16x49.
            # For each block in a_blocks:
            for (bi, bj), coeff_t in a_blocks:
                # For each element in that block:
                rows, cols = block_indices(bi, bj)
                for r in rows:
                    for c in cols:
                        # Apply inner coefficients
                        for (ri, ci), coeff_i in a_inner:
                            # ri, ci are local indices 0,1
                            # Actual row = r + ri? No.
                            # Block (bi, bj) starts at 2*bi, 2*bj.
                            # Local index (ri, ci) maps to (2*bi + ri, 2*bj + ci).
                            # Wait, my block_indices returns absolute rows/cols.
                            # Let's use absolute.
                            actual_r = 2*bi + ri
                            actual_c = 2*bj + ci
                            u_idx = get_U_index(actual_r, actual_c)
                            U[u_idx][rank_idx] += coeff_t * coeff_i
            
            # Build V coefficients
            for (bi, bj), coeff_t in b_blocks:
                for (ri, ci), coeff_i in b_inner:
                    actual_r = 2*bi + ri
                    actual_c = 2*bj + ci
                    v_idx = get_V_index(actual_r, actual_c)
                    V[v_idx][rank_idx] += coeff_t * coeff_i
            
            # Build W coefficients
            # The inner product m_k contributes to specific C elements within the 2x2 block result.
            # But the top-level product M_t contributes to specific 2x2 blocks of C.
            # We need to map the inner C contribution (local 2x2) to the global 4x4 C.
            
            # Which global C blocks does M_t affect?
            # M1 -> C00, C11
            # M2 -> C10, C11
            # M3 -> C01, C11
            # M4 -> C00, C10
            # M5 -> C00, C01
            # M6 -> C11
            # M7 -> C00
            
            top_c_blocks = [
                [(0,0, 1), (1,1, 1)], # M1
                [(1,0, 1), (1,1, -1)],# M2
                [(0,1, 1), (1,1, 1)], # M3
                [(0,0, -1), (1,0, 1)],# M4
                [(0,0, 1), (0,1, 1)], # M5
                [(1,1, 1)],           # M6
                [(0,0, -1)]           # M7
            ]
            
            for (ci, cj), coeff_top_c in top_c_blocks[t]:
                # This block (ci, cj) of C receives contribution from m_k.
                # The inner contribution of m_k to local C is given by c_contrib[k].
                for (ri, ci_local), coeff_inner_c in c_contrib[k]:
                    # ri, ci_local are local indices in the 2x2 block.
                    # Global row = 2*ci + ri
                    # Global col = 2*cj + ci_local
                    global_r = 2*ci + ri
                    global_c = 2*cj + ci_local
                    w_idx = get_W_index(global_r, global_c)
                    W[w_idx][rank_idx] += coeff_top_c * coeff_inner_c

    return U, V, W

_U, _V, _W = _build_decomposition()

def construct():
    return {"U": _U, "V": _V, "W": _W}
# EVOLVE-BLOCK-END
