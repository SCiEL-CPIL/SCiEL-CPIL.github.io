"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W matrices for Strassen 2x2x2 multiplication.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    Condition: sum_t U[(i,j),t] * V[(j,k),t] * W[(k,i),t] = 1 for all i,j,k in {0,1}
    """
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In tensor form, U[(i,j),t] is the coefficient of A_{i,j} in product t
    # V[(j,k),t] is the coefficient of B_{j,k} in product t
    # W[(k,i),t] is the coefficient of product t in C_{i,k}
    #
    # Let's index: A is 2x2 with indices (i,j), i=row, j=col
    # B is 2x2 with indices (j,k), j=row, k=col
    # C is 2x2 with indices (i,k), i=row, k=col
    # W is indexed by (k,i) per the convention W[k*n+i]
    
    # U: (4, 7) where row index = i*2+j
    # V: (4, 7) where row index = j*2+k
    # W: (4, 7) where row index = k*2+i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Product 0 (M1): (A00+A11)*(B00+B11)
    # A part: A00 coeff=1 (i=0,j=0 -> idx 0), A11 coeff=1 (i=1,j=1 -> idx 3)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    # B part: B00 coeff=1 (j=0,k=0 -> idx 0), B11 coeff=1 (j=1,k=1 -> idx 3)
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    # C part: C00 += M1, C11 += M1
    # W[k*2+i]: C00 -> k=0,i=0 -> idx 0; C11 -> k=1,i=1 -> idx 3
    W[0][0] = 1  # contributes to C00
    W[3][0] = 1  # contributes to C11
    
    # Product 1 (M2): (A10+A11)*B00
    # A part: A10 coeff=1 (i=1,j=0 -> idx 2), A11 coeff=1 (i=1,j=1 -> idx 3)
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    # B part: B00 coeff=1 (j=0,k=0 -> idx 0)
    V[0][1] = 1  # B00
    # C part: C10 += M2, C11 -= M2
    # C10 -> k=0,i=1 -> idx 1; C11 -> k=1,i=1 -> idx 3
    W[1][1] = 1  # contributes to C10
    W[3][1] = -1  # contributes to C11 with -1
    
    # Product 2 (M3): A00*(B01-B11)
    # A part: A00 coeff=1 (i=0,j=0 -> idx 0)
    U[0][2] = 1  # A00
    # B part: B01 coeff=1 (j=0,k=1 -> idx 1), B11 coeff=-1 (j=1,k=1 -> idx 3)
    V[1][2] = 1  # B01
    V[3][2] = -1  # B11
    # C part: C01 += M3, C11 += M3
    # C01 -> k=1,i=0 -> idx 2; C11 -> k=1,i=1 -> idx 3
    W[2][2] = 1  # contributes to C01
    W[3][2] = 1  # contributes to C11
    
    # Product 3 (M4): A11*(B10-B00)
    # A part: A11 coeff=1 (i=1,j=1 -> idx 3)
    U[3][3] = 1  # A11
    # B part: B10 coeff=1 (j=1,k=0 -> idx 2), B00 coeff=-1 (j=0,k=0 -> idx 0)
    V[2][3] = 1  # B10
    V[0][3] = -1  # B00
    # C part: C00 += M4, C10 += M4
    # C00 -> k=0,i=0 -> idx 0; C10 -> k=0,i=1 -> idx 1
    W[0][3] = 1  # contributes to C00
    W[1][3] = 1  # contributes to C10
    
    # Product 4 (M5): (A00+A01)*B11
    # A part: A00 coeff=1 (i=0,j=0 -> idx 0), A01 coeff=1 (i=0,j=1 -> idx 1)
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    # B part: B11 coeff=1 (j=1,k=1 -> idx 3)
    V[3][4] = 1  # B11
    # C part: C00 -= M5, C01 += M5
    # C00 -> k=0,i=0 -> idx 0; C01 -> k=1,i=0 -> idx 2
    W[0][4] = -1  # contributes to C00 with -1
    W[2][4] = 1  # contributes to C01
    
    # Product 5 (M6): (A10-A00)*(B00+B01)
    # A part: A10 coeff=1 (i=1,j=0 -> idx 2), A00 coeff=-1 (i=0,j=0 -> idx 0)
    U[2][5] = 1  # A10
    U[0][5] = -1  # A00
    # B part: B00 coeff=1 (j=0,k=0 -> idx 0), B01 coeff=1 (j=0,k=1 -> idx 1)
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    # C part: C11 += M6
    # C11 -> k=1,i=1 -> idx 3
    W[3][5] = 1  # contributes to C11
    
    # Product 6 (M7): (A01-A11)*(B10+B11)
    # A part: A01 coeff=1 (i=0,j=1 -> idx 1), A11 coeff=-1 (i=1,j=1 -> idx 3)
    U[1][6] = 1  # A01
    U[3][6] = -1  # A11
    # B part: B10 coeff=1 (j=1,k=0 -> idx 2), B11 coeff=1 (j=1,k=1 -> idx 3)
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    # C part: C00 += M7
    # C00 -> k=0,i=0 -> idx 0
    W[0][6] = 1  # contributes to C00
    
    return U, V, W


def _schoolbook_2x2x2():
    """Return U, V, W for schoolbook 2x2x2 (rank 4)."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W


def _block_compose(n_blocks, m_blocks, p_blocks, sub_U, sub_V, sub_W):
    """
    Compose a block decomposition.
    n_blocks, m_blocks, p_blocks: number of blocks in each dimension
    sub_U: (n_sub*m_sub, r_sub) for one block
    sub_V: (m_sub*p_sub, r_sub) for one block
    sub_W: (p_sub*n_sub, r_sub) for one block
    Returns U: (n*m, total_r), V: (m*p, total_r), W: (p*n, total_r)
    """
    n_sub, m_sub, p_sub = n_blocks, m_blocks, p_blocks
    # Determine sub-dimensions from sub_U shape
    # sub_U has shape (n_sub*m_sub, r_sub)
    n_sub_dim = sub_U.shape[0] // m_sub if m_sub > 0 else sub_U.shape[0]
    # Actually, let's be explicit:
    # sub_U shape: (n_s * m_s, r) where n_s, m_s are the sub-dimensions
    # For 2x2x2 Strassen: sub_U is (4, 7), so n_s=2, m_s=2, r=7
    
    n_s = sub_U.shape[0] // m_sub  # This assumes m_sub divides
    # Better: infer from the problem
    # For our use case, we'll pass explicit dimensions
    
    r_sub = sub_U.shape[1]
    total_r = n_blocks * m_blocks * p_blocks * r_sub
    
    n = n_blocks * n_s
    m = m_blocks * m_sub
    p = p_blocks * p_sub
    
    U = [[0]*total_r for _ in range(n*m)]
    V = [[0]*total_r for _ in range(m*p)]
    W = [[0]*total_r for _ in range(p*n)]
    
    for ib in range(n_blocks):
        for jb in range(m_blocks):
            for kb in range(p_blocks):
                offset = (ib * m_blocks + jb) * p_blocks + kb
                base = offset * r_sub
                for t in range(r_sub):
                    # U: global index (i,j) where i = ib*n_s + i_s, j = jb*m_s + j_s
                    for i_s in range(n_s):
                        for j_s in range(m_s):
                            u_val = sub_U[i_s*m_s + j_s][t]
                            if u_val != 0:
                                gi = ib * n_s + i_s
                                gj = jb * m_s + j_s
                                U[gi*m + gj][base + t] = u_val
                    
                    # V: global index (j,k) where j = jb*m_s + j_s, k = kb*p_s + k_s
                    for j_s in range(m_sub):
                        for k_s in range(p_sub):
                            v_val = sub_V[j_s*p_sub + k_s][t]
                            if v_val != 0:
                                gj = jb * m_sub + j_s
                                gk = kb * p_sub + k_s
                                V[gj*p + gk][base + t] = v_val
                    
                    # W: global index (k,i) where k = kb*p_s + k_s, i = ib*n_s + i_s
                    for k_s in range(p_sub):
                        for i_s in range(n_s):
                            w_val = sub_W[k_s*n_s + i_s][t]
                            if w_val != 0:
                                gk = kb * p_sub + k_s
                                gi = ib * n_s + i_s
                                W[gk*n + gi][base + t] = w_val
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, verbose=False):
    """Verify that sum_t U[(i,j),t] * V[(j,k),t] * W[(k,i),t] = 1 for all i,j,k."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for t in range(r):
                    total += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if total != 1:
                    if verbose:
                        print(f"FAIL: i={i}, j={j}, k={k}, total={total}")
                    return False
    if verbose:
        print(f"Verification PASSED for {n}x{m}x{p} with rank {r}")
    return True


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    start_time = time.time()
    deadline = start_time + 130  # 20s margin
    
    # Strategy: Decompose 4x4x7 as:
    # - 4x4x4 part using 2x2 block Strassen (4 blocks of 2x2x2, rank 4*7=28... 
    #   wait, 2x2x2 Strassen has rank 7, so 2x2 blocks of 2x2x2 gives rank 7 per block)
    #   4x4x4 = 2x2 blocks in each dim = 8 blocks, each rank 7, total rank 56?
    #   No! The block composition for 4x4x4 using 2x2 Strassen:
    #   We have 2 block rows, 2 block cols, 2 block "depths"
    #   Each block is a 2x2x2 Strassen with rank 7
    #   Total rank = 2*2*2*7 = 56
    #   But known rank(4,4,4) = 49, so this isn't optimal but is valid.
    #   
    #   Actually, let me reconsider. The standard result is rank(4,4,4)=49.
    #   But a block Strassen gives 56. Let's use what we can construct easily.
    #
    #   Alternative: 4x4x4 = 4 blocks of 2x2x2 (treating the 4x4 as 2x2 blocks of 2x2)
    #   But the "depth" dimension is also 4, so we need 2x2x2 blocks.
    #   Number of 2x2x2 blocks in 4x4x4: (4/2)*(4/2)*(4/2) = 8 blocks
    #   Each has rank 7, total = 56.
    #
    #   For 4x4x7:
    #   4x4x4 part: 8 blocks of 2x2x2 Strassen, rank 56
    #   4x4x3 part: schoolbook, rank 4*4*3 = 48
    #   Total: 56 + 48 = 104
    #
    #   Better: 4x4x7 as 2x2 blocks:
    #   n=4 -> 2 blocks of size 2
    #   m=4 -> 2 blocks of size 2  
    #   p=7 -> 3 blocks of size 2 (last one is size 1) + 1 block of size 1
    #   This gets complicated with non-uniform blocks.
    #
    #   Simplest valid approach:
    #   4x4x4 using 2x2 Strassen blocks: rank 56
    #   4x4x3 using schoolbook: rank 48
    #   Total: 104
    #
    #   Even better: use 2x2x2 Strassen for the full 4x4x4 and handle 4x4x3
    #   with 2x2x2 Strassen + 2x2x1 schoolbook:
    #   4x4x2: 4 blocks of 2x2x2, rank 4*7=28
    #   4x4x1: schoolbook, rank 4*4*1=16
    #   4x4x4 total: 28+16=44? No, 4x4x2 means p=2, so 2x2 blocks in n,m and 1 block in p
    #   (4/2)*(4/2)*(2/2) = 2*2*1 = 4 blocks, rank 4*7=28
    #   4x4x1: (4/2)*(4/2)*(1/1) = 2*2*1 = 4 blocks of 2x2x1 schoolbook, rank 4*4=16
    #   Total 4x4x4: 28+16=44
    #   Hmm, but that's not right either. Let me think more carefully.
    
    # Let me just do the straightforward thing:
    # Part 1: 4x4x4 using 2x2x2 Strassen blocks
    #   n=4, m=4, p=4, block sizes 2,2,2
    #   Number of blocks: (4/2)*(4/2)*(4/2) = 8
    #   Each block rank: 7
    #   Total rank: 56
    #
    # Part 2: 4x4x3 using schoolbook
    #   Rank: 4*4*3 = 48
    #
    # Combined rank: 56 + 48 = 104
    #
    # Can we do better for 4x4x3?
    # 4x4x2 with Strassen: (4/2)*(4/2)*(2/2) = 4 blocks * 7 = 28
    # 4x4x1 with schoolbook: 4*4*1 = 16
    # Total for 4x4x3: 28 + 16 = 44
    # Combined: 56 + 44 = 100
    #
    # Even better for 4x4x4:
    # 4x4x2 with Strassen: 4 blocks * 7 = 28
    # 4x4x2 with Strassen: 4 blocks * 7 = 28
    # Total: 56 (same)
    #
    # Or: 4x4x4 all Strassen: 8 blocks * 7 = 56
    #
    # Let's try: 4x4x7 = 4x4x4 (Strassen, rank 56) + 4x4x3 (Strassen+schoolbook, rank 44)
    # Total: 100
    
    # Actually, let me reconsider the block composition more carefully.
    # For 2x2x2 Strassen, we have a valid decomposition of rank 7.
    # To compose 4x4x4 from 2x2x2 blocks:
    # We partition A (4x4) into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # B (4x4) into 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # C (4x4) into 2x2 blocks: C = [[C00, C01], [C10, C11]]
    # C_{ab} = sum_{c} A_{ac} * B_{cb} for block indices a,b,c in {0,1}
    # Each block multiplication is a 2x2x2 problem.
    # There are 2*2*2 = 8 block multiplications.
    # Each uses 7 scalar multiplications (Strassen).
    # Total: 56.
    
    # For 4x4x7:
    # A is 4x4, B is 4x7, C is 4x7
    # Partition A into 2x2 blocks (2x2 grid)
    # Partition B into 2x2 blocks for first 4 cols, and 2x1 + 2x1 for last 3 cols
    # This is getting messy. Let me just use uniform 2x2 blocks and handle the remainder.
    
    # Cleanest approach:
    # 4x4x4: 8 Strassen 2x2x2 blocks, rank 56
    # 4x4x3: 4 Strassen 2x2x2 blocks (for 4x4x2) + 4 schoolbook 2x2x1 blocks (for 4x4x1)
    #   4x4x2: (4/2)*(4/2)*(2/2) = 4 blocks of 2x2x2, rank 4*7 = 28
    #   4x4x1: (4/2)*(4/2)*(1/1) = 4 blocks of 2x2x1, rank 4*4 = 16
    #   Total 4x4x3: 44
    # Grand total: 56 + 44 = 100
    
    # Let's implement this step by step.
    
    # First, get the 2x2x2 Strassen decomposition
    sub_U, sub_V, sub_W = _strassen_2x2x2()
    
    # Get 2x2x1 schoolbook decomposition
    # 2x2x1: A is 2x2, B is 2x1, C is 2x1
    # U: (2*2, 4) = (4, 4), V: (2*1, 4) = (2, 4), W: (1*2, 4) = (2, 4)
    # Schoolbook: 4 terms
    U_221 = [[0]*4 for _ in range(4)]
    V_221 = [[0]*4 for _ in range(2)]
    W_221 = [[0]*4 for _ in range(2)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U_221[i*2+j][col] = 1
                V_221[j*1+k][col] = 1
                W_221[k*2+i][col] = 1
                col += 1
    
    # Now compose 4x4x4 using 2x2x2 Strassen blocks
    # n_blocks=2, m_blocks=2, p_blocks=2, sub dims n_s=2, m_s=2, p_s=2, r=7
    U_444, V_444, W_444 = _block_compose(2, 2, 2, sub_U, sub_V, sub_W)
    # U_444: (16, 56), V_444: (16, 56), W_444: (16, 56)
    
    if time.time() > deadline:
        # Fallback to schoolbook
        return _schoolbook_full(n, m, p)
    
    # Verify 4x4x4
    if not _verify_decomposition(U_444, V_444, W_444, 4, 4, 4):
        if time.time() < deadline:
            # Try to debug or fallback
            return _schoolbook_full(n, m, p)
    
    # Compose 4x4x2 using 2x2x2 Strassen blocks
    # n_blocks=2, m_blocks=2, p_blocks=1, sub dims 2,2,2, r=7
    U_442, V_442, W_442 = _block_compose(2, 2, 1, sub_U, sub_V, sub_W)
    # U_442: (16, 28), V_442: (8, 28), W_442: (8, 28)
    
    # Compose 4x4x1 using 2x2x1 schoolbook blocks
    # n_blocks=2, m_blocks=2, p_blocks=1, sub dims 2,2,1, r=4
    U_441, V_441, W_441 = _block_compose(2, 2, 1, U_221, V_221, W_221)
    # U_441: (16, 16), V_441: (8, 16), W_441: (8, 16)
    
    # Now combine 4x4x4 and 4x4x3 (= 4x4x2 + 4x4x1) to get 4x4x7
    # Total rank: 56 + 28 + 16 = 100
    
    total_rank = 56 + 28 + 16
    
    # U: (16, 100)
    U_full = [[0]*total_rank for _ in range(16)]
    # V: (28, 100)  [m*p = 4*7 = 28]
    V_full = [[0]*total_rank for _ in range(28)]
    # W: (28, 100)  [p*n = 7*4 = 28]
    W_full = [[0]*total_rank for _ in range(28)]
    
    # Part 1: 4x4x4 (k=0..3)
    # U_444: (16, 56) -> copy to U_full[:, 0:56]
    # V_444: (16, 56) -> V_full rows 0..15 (j*7+k for j=0..3, k=0..3)
    #   In V_444, row index is j*4+k (local p=4)
    #   In V_full, row index is j*7+k (global p=7)
    #   So for local row j*4+k, global row is j*7+k
    # W_444: (16, 56) -> W_full rows for k=0..3
    #   In W_444, row index is k*4+i (local p=4, n=4)
    #   In W_full, row index is k*7+i... wait, W is (p*n, r), indexed by (k,i)
    #   So row index = k*n + i = k*4 + i
    #   For local k=0..3, global k=0..3, so W_full[k*4+i] = W_444[k*4+i]
    
    for t in range(56):
        for row in range(16):
            U_full[row][t] = U_444[row][t]
        # V: local (j,k) with p=4 -> global (j,k) with p=7
        for j in range(4):
            for k in range(4):
                V_full[j*7 + k][t] = V_444[j*4 + k][t]
        # W: local (k,i) with p=4 -> global (k,i) with p=7
        for k in range(4):
            for i in range(4):
                W_full[k*4 + i][t] = W_444[k*4 + i][t]
    
    # Part 2: 4x4x2 (k=4..5)
    # U_442: (16, 28) -> U_full[:, 56:84]
    # V_442: (8, 28) -> local (j,k) with p=2, j=0..3, k=0..1
    #   Global: j*7 + (k+4) for k=0..1
    # W_442: (8, 28) -> local (k,i) with p=2, k=0..1, i=0..3
    #   Global: (k+4)*4 + i
    
    offset2 = 56
    for t in range(28):
        for row in range(16):
            U_full[row][offset2 + t] = U_442[row][t]
        for j in range(4):
            for k in range(2):
                V_full[j*7 + (k+4)][offset2 + t] = V_442[j*2 + k][t]
        for k in range(2):
            for i in range(4):
                W_full[(k+4)*4 + i][offset2 + t] = W_442[k*2 + i][t]
    
    # Part 3: 4x4x1 (k=6)
    # U_441: (16, 16) -> U_full[:, 84:100]
    # V_441: (8, 16) -> local (j,k) with p=1, j=0..3, k=0
    #   Global: j*7 + 6
    # W_441: (8, 16) -> local (k,i) with p=1, k=0, i=0..3
    #   Global: 6*4 + i
    
    offset3 = 84
    for t in range(16):
        for row in range(16):
            U_full[row][offset3 + t] = U_441[row][t]
        for j in range(4):
            V_full[j*7 + 6][offset3 + t] = V_441[j*1 + 0][t]
        for i in range(4):
            W_full[6*4 + i][offset3 + t] = W_441[0*2 + i][t]
    
    # Final verification
    if time.time() < deadline:
        if not _verify_decomposition(U_full, V_full, W_full, n, m, p, verbose=False):
            # Fallback to schoolbook if verification fails
            return _schoolbook_full(n, m, p)
    
    return {"U": U_full, "V": V_full, "W": W_full}


def _schoolbook_full(n, m, p):
    """Fallback schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
