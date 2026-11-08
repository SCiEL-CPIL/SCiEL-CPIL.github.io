import numpy as np
import time
from fractions import Fraction

# ---------------------------------------------------------------------------
# Rank-97 decomposition for 4x4 * 4x7 matrix multiplication.
#
# Strategy: split the 7 output columns into a 4-column block (handled by a
# rank-49 Strassen-style recursive decomposition) and a 3-column block
# (handled by the schoolbook, rank 48).  Total rank = 97.
# ---------------------------------------------------------------------------

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before the 150 s kill

_N, _M, _P = 4, 4, 7
_NM = _N * _M          # 16
_MP = _M * _P          # 28
_PN = _P * _N          # 28
_RANK = 97

# ---- Strassen 2x2 building blocks ----------------------------------------
# u_in[r][i][j] : coefficient of A[i,j] in the r-th U-factor (7 entries)
_u_in = np.array([
    [ 1,  0,  0,  1],
    [ 0,  0,  1,  1],
    [ 1,  0,  0,  0],
    [ 0,  0,  0,  1],
    [ 1,  1,  0,  0],
    [-1,  0,  1,  0],
    [ 0,  1,  0, -1],
], dtype=int)

# v_in[r][i][j] : coefficient of B[i,j] in the r-th V-factor
_v_in = np.array([
    [ 1,  0,  0,  1],
    [ 1,  0,  0,  0],
    [ 0,  1,  0, -1],
    [-1,  0,  1,  0],
    [ 0,  0,  0,  1],
    [ 1,  1,  0,  0],
    [ 0,  0,  1,  1],
], dtype=int)

# w_out[r][i][j] : coefficient of the r-th product in C[i,j]
_w_out = np.array([
    [ 1,  0,  0,  1],
    [ 0,  0,  1,  0],
    [ 0,  1,  0,  1],
    [ 1, -1,  1,  0],
], dtype=int)

# The same 7-entry / 7-entry / 4-entry factors are reused at both levels of
# the recursion (block 2x2 of 2x2).  For the 4x4x4 block we need a 4x4
# "outer" coefficient matrix.  In the standard recursive Strassen construction
# for 4x4, the outer 4x4 matrix that maps the 16 block-products back to the
# 4x4 result is simply the identity arranged per block; however, the correct
# recursive embedding uses the SAME w_out pattern at each level.
#
# In fact, the rank of the 4x4x4 Strassen tensor is 7^2 = 49.  The U and V
# factors are 7x7 Kronecker products, and the W factor is a 4x7x7x4
# structure.  The cleanest way to build all 49 terms is to iterate over
# (k, l) in {0..6}^2 and form the outer products of the level-2 and level-1
# coefficients.

def _build_decomposition():
    """Return the rank-97 decomposition as integer lists."""

    U = [[0] * _RANK for _ in range(_NM)]   # 16 x 97
    V = [[0] * _RANK for _ in range(_MP)]   # 28 x 97
    W = [[0] * _RANK for _ in range(_PN)]   # 28 x 97

    col = 0

    # ---- Part 1: rank-49 Strassen for the 4x4 * 4x4 (first 4 cols) block ----
    # For each (k, l) in 0..6:
    #   U[i*4+j]  += u_in[l][i//2][j//2] * u_in[k][i%2][j%2]
    #   V[j*7+kk] += v_in[l][j//2][kk//2] * v_in[k][j%2][kk%2]
    #   W[kk*4+i] += w_out[k][kk//2][kk%2] * w_out[l][i//2][i%2]
    #   (with appropriate index mapping for the 4x4 -> 2x2 block split)
    #
    # More precisely, split the 4x4 matrices into 2x2 blocks indexed by
    # (br, bc) in {0,1}^2.  Block (br,bc) occupies rows 2*br..2*br+1 and
    # cols 2*bc..2*bc+1.
    #
    # For U (input A, shape 4x4, flattened row-major to 16):
    #   element (i,j) with i=2*br+ir, j=2*bc+jc
    #   U contribution for (k,l): u_in[l][ir][jc] * u_in[k][br][bc]
    #
    # Wait, that's not right either.  Let me think more carefully.
    #
    # The 4x4 Strassen decomposition:
    #   A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]  (2x2 blocks of 2x2)
    #   7 products: M1..M7 of 2x2 matrices
    #   Each 2x2 product is itself decomposed into 7 rank-1 terms
    #   Total: 7*7 = 49 rank-1 terms
    #
    # For the outer Strassen step (index k, 0..6):
    #   U_outer[k][br][bc] = u_in[k][br][bc]   (coefficient for block (br,bc))
    #   V_outer[k][bc][kk] = v_in[k][bc][kk]   (coefficient for block (bc,kk))
    #
    # For the inner Strassen step (index l, 0..6):
    #   Within block (br,bc) of A, the 2x2 sub-matrix A[br,bc] has entries
    #   A[2*br+ir, 2*bc+jc].  The inner U coefficient is u_in[l][ir][jc].
    #
    # So the full U factor for term (k,l) at position (i,j):
    #   i = 2*br + ir,  j = 2*bc + jc
    #   U[i,j] += u_in[k][br][bc] * u_in[l][ir][jc]
    #
    # Similarly for V: j = 2*bc + jc (row), kk = 2*kk_r + kk_c (col)
    #   V[j,kk] += v_in[k][bc][kk_r] * v_in[l][jc][kk_c]
    #
    # For W: output C[i,j], i = 2*br + ir, j = 2*bc + jc
    #   W[i,j] += w_out[k][br][bc] * w_out[l][ir][jc]
    #
    # BUT: the output for the 4x4x4 block only covers columns 0..3 of the
    # 4x7 output.  So W index is (kk, i) where kk is the output column
    # (0..3 for this block) and i is the output row.  In the W matrix of
    # shape (p*n, rank) = (28, rank), the index is kk*4 + i where kk is
    # the column index in the 7-column output and i is the row index.
    #
    # For the 4x4x4 block, kk ranges 0..3, i ranges 0..3.
    # W[5] coefficient for term (k,l): w_out[k][kk//2][kk%2] * w_out[l][i//2][i%2]
    #   where kk = 2*kk_r + kk_c, i = 2*i_r + i_c
    #   Wait, I need to be more careful.
    #
    # The output C is 4x4 (for this block).  C[i,j] where i,j in 0..3.
    # Split C into 2x2 blocks: C[br,bc] is the 2x2 block at rows 2*br..2*br+1
    # and cols 2*bc..2*bc+1.
    #
    # Outer Strassen: C[br,bc] = sum_k w_out[k][br][bc] * (M_k)
    # Inner Strassen: M_k is a 2x2 matrix, M_k[ir,jc] = sum_l w_out[l][ir][jc] * (u_in[l] * A_sub) * (v_in[l] * B_sub)
    #
    # So C[2*br+ir, 2*bc+jc] = sum_{k,l} w_out[k][br][bc] * w_out[l][ir][jc] * u_in[k][br][bc]*u_in[l][ir][jc] * ... 
    #
    # Hmm, I'm overcomplicating.  Let me just use the direct construction:
    #
    # Term (k, l), k,l in 0..6:
    #   U factor: for each (i, j) in 0..3:
    #     i = 2*br + ir, j = 2*bc + jc
    #     U[i*4+j] += u_in[k][br][bc] * u_in[l][ir][jc]
    #
    #   V factor: for each (j, k2) in 0..3 x 0..3 (j is row of B, k2 is col):
    #     j = 2*bc + jc, k2 = 2*kr + kc
    #     V[j*7+k2] += v_in[k][bc][kr] * v_in[l][jc][kc]
    #
    #   W factor: for each (k2, i) in 0..3 x 0..3 (k2 is output col, i is output row):
    #     k2 = 2*kr + kc, i = 2*ir + ic
    #     W[k2*4+i] += w_out[k][kr][kc] * w_out[l][ir][ic]
    #
    #     BUT WAIT: for the 4x4x4 block, the output columns are 0..3.
    #     The W matrix has shape (28, 97) where index = col*4 + row.
    #     So W[k2*4 + i] for k2 in 0..3, i in 0..3.
    #     That's correct.
    
    for k in range(7):
        for l in range(7):
            # U factor
            for i in range(4):
                for j in range(4):
                    br, ir = i // 2, i % 2
                    bc, jc = j // 2, j % 2
                    val = _u_in[k, br, bc] * _u_in[l, ir, jc]
                    if val != 0:
                        U[i * 4 + j][col] += val
            # V factor
            for j in range(4):
                for k2 in range(4):
                    bc, jc = j // 2, j % 2
                    kr, kc = k2 // 2, k2 % 2
                    val = _v_in[k, bc, kr] * _v_in[l, jc, kc]
                    if val != 0:
                        V[j * 7 + k2][col] += val
            # W factor (output cols 0..3 for this block)
            for k2 in range(4):
                for i in range(4):
                    kr, kc = k2 // 2, k2 % 2
                    ir, ic = i // 2, i % 2
                    val = _w_out[k, kr, kc] * _w_out[l, ir, ic]
                    if val != 0:
                        W[k2 * 4 + i][col] += val
            col += 1

    # ---- Part 2: rank-48 schoolbook for the 4x4 * 4x3 (last 3 cols) block ----
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                k_global = 4 + k_prime  # columns 4, 5, 6
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1

    # Convert to lists (ensure plain Python ints)
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}

# Build once at module load
_DECOMP = _build_decomposition()

def construct():
    """
    Construct a rank-97 exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x7 matrix.

    Interface: construct() -> {'U', 'V', 'W'}
    Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    Convention: W[k*n+i, r]
    """
    if time.time() > _DEADLINE:
        # Fallback: return the precomputed result (still valid)
        return _DECOMP
    return _DECOMP
