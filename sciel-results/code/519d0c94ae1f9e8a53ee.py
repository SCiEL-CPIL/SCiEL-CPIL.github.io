"""Matrix multiplication 5x5x5 rank-minimal decomposition using Strassen for 2x2 blocks."""

import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication.
    Each tensor is (u, v, w) where u, v, w are length-2 vectors.
    The decomposition satisfies:
      T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
    where T[i,j,k] = 1 for all i,j,k in {0,1} (since every A[i,j]*B[j,k] contributes to C[i,k]).
    """
    # Strassen's 7 products for 2x2:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For the tensor T[i,j,k] = 1 (all ones 2x2x2), we need:
    # T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
    # But actually the standard formulation is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j u_r[i] A[i,j]) * (sum_k v_r[j] B[j,k]) * w_r[k]
    
    # Let me use the known rank-7 decomposition of the 2x2x2 multiplication tensor.
    # The tensor T has T[i,j,k] = 1 for all i,j,k.
    # We need u_r, v_r, w_r such that sum_r u_r[i]*v_r[j]*w_r[k] = 1 for all i,j,k.
    
    # Known decomposition:
    # r=0: u=(1,1), v=(1,1), w=(1,-1)  -> contributes to make C differences
    # Let me just use the standard Strassen mapping directly.
    
    # Actually, for the purpose of this task, I need to construct U, V, W matrices
    # for the full 5x5x5 problem. Let me think about the block structure.
    
    # For 5x5 with a 2+2+1 partition:
    # A = [[A11(2x2), A12(2x2), A13(2x1)],
    #      [A21(2x2), A22(2x2), A23(2x1)],
    #      [A31(1x2), A32(1x2), A33(1x1)]]
    # Similarly for B and C.
    
    # Each 2x2x2 block uses rank 7, each 2x2x1 or 2x1x2 etc uses rank 4, and 1x1x1 uses rank 1.
    
    # Let me count the blocks:
    # 2x2x2 blocks: 3x3 = 9 blocks, each rank 7 -> 63
    # 2x2x1 blocks: positions where one dim is size 1
    # Let me enumerate all 3x3x3 = 27 blocks by (i-block, j-block, k-block)
    
    # Block sizes:
    # i-blocks: 2, 2, 1 (rows of A, cols of C)
    # j-blocks: 2, 2, 1 (cols of A, rows of B)  
    # k-blocks: 2, 2, 1 (cols of B, rows of C)
    
    # For each (ib, jb, kb) block of size (nib, njb, nkb):
    # rank contribution = nib * njb * nkb if schoolbook, or less if using Strassen for 2x2x2
    
    # Using Strassen for 2x2x2 blocks (rank 7 instead of 8):
    # Blocks with all dims = 2: 7 instead of 8, saving 1 each
    # There are 3*3 = 9 such blocks (ib in {0,1}, jb in {0,1}, kb in {0,1})
    # Wait, ib, jb, kb each range over {0,1,2} where 2 means size 1.
    # So blocks with ib=0 or 1 (size 2), jb=0 or 1 (size 2), kb=0 or 1 (size 2): 2*2*2 = 8 blocks
    # Each saves 1 rank: total saving = 8
    # Base rank = 125, new rank = 125 - 8 = 117
    
    # Let me construct the full U, V, W using this block approach.
    # I'll use a helper to generate Strassen vectors for 2x2x2.
    
    # Standard Strassen rank-1 tensors for 2x2x2:
    # These satisfy sum_r u_r[i]*v_r[j]*w_r[k] = 1 for all i,j,k in {0,1}
    
    # One known decomposition:
    tensors = [
        # (u, v, w)
        ((1, 1), (1, 1), (1, 1)),      # r=0
        ((1, -1), (1, 1), (1, -1)),     # r=1  
        ((1, 1), (1, -1), (1, 1)),      # r=2
        ((-1, 1), (1, 1), (1, 1)),      # r=3
        ((1, 1), (1, 1), (-1, 1)),      # r=4
        ((1, -1), (1, 1), (1, 1)),      # r=5
        ((1, 1), (1, -1), (1, 1)),      # r=6
    ]
    
    # Verify: for each (i,j,k), sum should be 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(t[0][i] * t[1][j] * t[2][k] for t in tensors)
                if s != 1:
                    # This decomposition might not be correct. Let me use a different one.
                    pass
    
    # Let me use the correct Strassen decomposition. The 2x2x2 multiplication tensor
    # T[i,j,k] = 1 for all i,j,k has a known rank-7 decomposition.
    
    # From literature, one such decomposition is:
    tensors = [
        ((1, 0), (1, 0), (1, 0)),
        ((1, 0), (0, 1), (0, 1)),
        ((0, 1), (1, 0), (0, 1)),
        ((0, 1), (0, 1), (1, 0)),
        ((1, 1), (1, 1), (1, 1)),
        ((1, -1), (1, 1), (1, -1)),
        ((1, 1), (1, -1), (1, -1)),
    ]
    
    # Verify again
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(t[0][i] * t[1][j] * t[2][k] for t in tensors)
                if s != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}), sum={s}")
    
    return tensors

# Precompute the Strassen tensors
_STRASSEN_2X2 = _build_strassen_2x2()

def construct():
    """
    Construct rank-117 decomposition of 5x5x5 matrix multiplication.
    Uses Strassen's rank-7 algorithm for the 2x2x2 sub-blocks in a 2+2+1 partition.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Block partition: sizes [2, 2, 1] for each dimension
    block_sizes_i = [2, 2, 1]  # rows of A / cols of C (n dimension)
    block_sizes_j = [2, 2, 1]  # cols of A / rows of B (m dimension)  
    block_sizes_k = [2, 2, 1]  # cols of B / rows of C (p dimension)
    
    # Verify sizes sum correctly
    assert sum(block_sizes_i) == n, f"n block sizes {block_sizes_i} don't sum to {n}"
    assert sum(block_sizes_j) == m, f"m block sizes {block_sizes_j} don't sum to {m}"
    assert sum(block_sizes_k) == p, f"p block sizes {block_sizes_k} don't sum to {p}"
    
    # Compute block offsets
    offsets_i = [0]
    for s in block_sizes_i[:-1]:
        offsets_i.append(offsets_i[-1] + s)
    offsets_j = [0]
    for s in block_sizes_j[:-1]:
        offsets_j.append(offsets_j[-1] + s)
    offsets_k = [0]
    for s in block_sizes_k[:-1]:
        offsets_k.append(offsets_k[-1] + s)
    
    # First pass: compute total rank needed
    total_rank = 0
    block_rank_info = []  # (ib, jb, kb, rank, start_col)
    for ib in range(3):
        for jb in range(3):
            for kb in range(3):
                nib = block_sizes_i[ib]
                njb = block_sizes_j[jb]
                nkb = block_sizes_k[kb]
                if nib == 2 and njb == 2 and nkb == 2:
                    r = 7  # Strassen
                else:
                    r = nib * njb * nkb  # Schoolbook
                block_rank_info.append((ib, jb, kb, r, total_rank))
                total_rank += r
    
    rank = total_rank
    assert rank == 117, f"Expected rank 117, got {rank}"
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]  # 25 x 117
    V = [[0] * rank for _ in range(m * p)]  # 25 x 117
    W = [[0] * rank for _ in range(p * n)]  # 25 x 117
    
    # Fill in each block
    for (ib, jb, kb, r, col_start) in block_rank_info:
        nib = block_sizes_i[ib]
        njb = block_sizes_j[jb]
        nkb = block_sizes_k[kb]
        
        if nib == 2 and njb == 2 and nkb == 2:
            # Use Strassen decomposition
            for rr in range(7):
                col = col_start + rr
                u_vec, v_vec, w_vec = _STRASSEN_2X2[rr]
                
                # U: index = (i_global * m + j_global), where i in A rows, j in A cols
                # For this block, i ranges over offsets_i[ib] to offsets_i[ib]+nib-1
                # j ranges over offsets_j[jb] to offsets_j[jb]+njb-1
                for li in range(nib):
                    i_global = offsets_i[ib] + li
                    for lj in range(njb):
                        j_global = offsets_j[jb] + lj
                        U[i_global * m + j_global][col] = u_vec[li]
                
                # V: index = (j_global * p + k_global), where j in B rows, k in B cols
                for lj in range(njb):
                    j_global = offsets_j[jb] + lj
                    for lk in range(nkb):
                        k_global = offsets_k[kb] + lk
                        V[j_global * p + k_global][col] = v_vec[lj]
                
                # W: index = (k_global * n + i_global), trace convention W[k*n+i]
                for lk in range(nkb):
                    k_global = offsets_k[kb] + lk
                    for li in range(nib):
                        i_global = offsets_i[ib] + li
                        W[k_global * n + i_global][col] = w_vec[lk]
        else:
            # Schoolbook for this block
            col = col_start
            for li in range(nib):
                for lj in range(njb):
                    for lk in range(nkb):
                        i_global = offsets_i[ib] + li
                        j_global = offsets_j[jb] + lj
                        k_global = offsets_k[kb] + lk
                        
                        U[i_global * m + j_global][col] = 1
                        V[j_global * p + k_global][col] = 1
                        W[k_global * n + i_global][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
