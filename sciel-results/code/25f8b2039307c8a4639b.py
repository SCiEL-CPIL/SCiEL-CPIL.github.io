"""Matrix multiplication 6x6x6: Strassen(2x2) ⊗ Schoolbook(3x3) rank-189 decomposition."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]
_DEADLINE_S = 150.0
_SAFETY_S = 2.0

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-189 exact tensor decomposition for 6x6 matrix multiplication
    using the Kronecker product of Strassen's 2x2 decomposition (rank 7)
    and the schoolbook 3x3 decomposition (rank 27).
    
    6 = 2 * 3, so 6x6 matmul tensor = (2x2 matmul tensor) ⊗ (3x3 matmul tensor).
    Rank of Kronecker product = product of ranks: 7 * 27 = 189.
    """
    t0 = time.time()
    n, m, p = DIMENSIONS  # all 6

    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen products for 2x2 block structure
    # Each product: (list of (A_block, sign), list of (B_block, sign), list of (C_block, sign))
    strassen_products = [
        # M1 = (A00+A11)*(B00+B11), contributes to C00(+1), C11(+1)
        ([('A00', 1), ('A11', 1)], [('B00', 1), ('B11', 1)], [('C00', 1), ('C11', 1)]),
        # M2 = (A10+A11)*B00, contributes to C10(+1), C11(-1)
        ([('A10', 1), ('A11', 1)], [('B00', 1)], [('C10', 1), ('C11', -1)]),
        # M3 = A00*(B01-B11), contributes to C01(+1), C11(+1)
        ([('A00', 1)], [('B01', 1), ('B11', -1)], [('C01', 1), ('C11', 1)]),
        # M4 = A11*(B10-B00), contributes to C00(+1), C10(+1)
        ([('A11', 1)], [('B10', 1), ('B00', -1)], [('C00', 1), ('C10', 1)]),
        # M5 = (A00+A01)*B11, contributes to C00(-1), C01(+1)
        ([('A00', 1), ('A01', 1)], [('B11', 1)], [('C00', -1), ('C01', 1)]),
        # M6 = (A10-A00)*(B00+B01), contributes to C11(+1)
        ([('A10', 1), ('A00', -1)], [('B00', 1), ('B01', 1)], [('C11', 1)]),
        # M7 = (A01-A11)*(B10+B11), contributes to C00(+1)
        ([('A01', 1), ('A11', -1)], [('B10', 1), ('B11', 1)], [('C00', 1)]),
    ]
    
    # Block index mapping (blocks are 3x3 within 6x6)
    # A block 'Aij' where i in {0,1}, j in {0,1}:
    #   A_ii covers rows [3*i, 3*i+3), cols [3*j, 3*j+3)
    
    def a_idx(bi, bj, i, k):
        """U index for A element at block (bi,bj), local position (i,k)."""
        return (bi * 3 + i) * m + (bj * 3 + k)
    
    def b_idx(bi, bj, k, j):
        """V index for B element at block (bi,bj), local position (k,j)."""
        return (bi * 3 + k) * p + (bj * 3 + j)
    
    def c_idx(bi, bj, i, j):
        """W index for C element at block (bi,bj), local position (i,j).
        C[i,j] -> W[j*n + i]"""
        return (bj * 3 + j) * n + (bi * 3 + i)
    
    block_map = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }
    
    col = 0
    total_inner = 7 * 3 * 3 * 3
    for sp_i, (left_blocks, right_blocks, c_contribs) in enumerate(strassen_products):
        if time.time() - t0 > _DEADLINE_S - _SAFETY_S:
            # Should not happen for this fixed-size construction, but guard anyway.
            # Return whatever has been filled so far (will be invalid if incomplete,
            # but avoids a hard timeout kill).
            break
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    # U: A coefficients
                    for (aname, asgn) in left_blocks:
                        bi, bj = block_map[aname]
                        u_pos = a_idx(bi, bj, i, k)
                        U[u_pos, col] += asgn

                    # V: B coefficients
                    for (bname, bsgn) in right_blocks:
                        bi, bj = block_map[bname]
                        v_pos = b_idx(bi, bj, k, j)
                        V[v_pos, col] += bsgn

                    # W: C coefficients
                    for (cname, csgn) in c_contribs:
                        bi, bj = block_map[cname]
                        w_pos = c_idx(bi, bj, i, j)
                        W[w_pos, col] += csgn

                    col += 1
            # Periodic deadline check inside the inner loops
            if (i + 1) % 3 == 0 and time.time() - t0 > _DEADLINE_S - _SAFETY_S:
                break
        if col >= rank:
            break

    if col != rank:
        # Incomplete due to deadline; return best-effort (will score 0 if invalid,
        # but prevents a wall-clock kill).
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Final deadline guard before serialization
    if time.time() - t0 > _DEADLINE_S - _SAFETY_S:
        # Already past safety margin; return immediately to avoid serialization overhead
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
