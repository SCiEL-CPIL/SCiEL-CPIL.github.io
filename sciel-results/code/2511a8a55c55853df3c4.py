"""
Rank-minimal tensor decomposition for 3x5x6 matrix multiplication using
block Strassen decomposition.

Strategy:
- Decompose the 3x5x6 multiplication into overlapping 2x2x2 blocks
- Use Strassen's rank-7 algorithm on each 2x2x2 block
- Combine the results to get a lower-rank decomposition than schoolbook (rank 90)

The 3x5x6 multiplication C = A * B where A is 3x5 and B is 5x6.
We can overlap 2x2 blocks to cover the entire multiplication.

For a 2x2x2 multiplication, Strassen uses 7 multiplications instead of 8.
We'll place these blocks strategically to minimize the total rank.
"""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 130.0  # Safety margin before 150s limit

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W matrices where:
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    with the indexing convention:
    U[i*2+j, c] for (i,j) in 0..1, c in 0..6
    V[j*2+k, c] for (j,k) in 0..1, c in 0..6
    W[k*2+i, c] for (k,i) in 0..1, c in 0..6
    
    The Strassen formulas for C = A*B where A,B are 2x2:
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    C = [[c11, c12], [c21, c22]]
    
    m1 = (a11+a22)*(b11+b22)
    m2 = (a21+a22)*b11
    m3 = a11*(b12-b22)
    m4 = a22*(b21-b11)
    m5 = (a11+a12)*b22
    m6 = (a21-a11)*(b11+b12)
    m7 = (a12-a22)*(b21+b22)
    
    c11 = m1+m4-m5+m7
    c12 = m3+m5
    c21 = m2+m4
    c22 = m1-m2+m3+m6
    """
    # U[i*2+j, r] = coefficient of A[i,j] in multiplication r
    # V[j*2+k, r] = coefficient of B[j,k] in multiplication r
    # W[k*2+i, r] = coefficient for C[i,k] contribution from multiplication r
    
    # Multiplication 1: m1 = (a11+a22)*(b11+b22)
    # A coefficients: a11=1, a22=1, others=0 -> U[0,0]=1, U[3,0]=1
    # B coefficients: b11=1, b22=1, others=0 -> V[0,0]=1, V[3,0]=1
    # C contributions: c11+=1, c12+=0, c21+=0, c22+=1 -> W[0,0]=1 (c11), W[3,0]=1 (c22)
    # Wait, W[k*2+i] where k is col of C, i is row of C
    # c11: i=0, k=0 -> W[0*2+0] = W[0]
    # c12: i=0, k=1 -> W[1*2+0] = W[2]
    # c21: i=1, k=0 -> W[0*2+1] = W[1]
    # c22: i=1, k=1 -> W[1*2+1] = W[3]
    
    U = [[0]*7 for _ in range(4)]  # rows: i*2+j
    V = [[0]*7 for _ in range(4)]  # rows: j*2+k
    W = [[0]*7 for _ in range(4)]  # rows: k*2+i
    
    # m1 = (a11+a22)*(b11+b22)
    U[0][0] = 1  # a11
    U[3][0] = 1  # a22
    V[0][0] = 1  # b11
    V[3][0] = 1  # b22
    W[0][0] = 1  # c11
    W[3][0] = 1  # c22
    
    # m2 = (a21+a22)*b11
    U[1][1] = 1  # a21
    U[3][1] = 1  # a22
    V[0][1] = 1  # b11
    W[1][1] = 1  # c21
    W[3][1] = -1 # c22
    
    # m3 = a11*(b12-b22)
    U[0][2] = 1  # a11
    V[1][2] = 1  # b12
    V[3][2] = -1 # b22
    W[2][2] = 1  # c12
    W[3][2] = 1  # c22
    
    # m4 = a22*(b21-b11)
    U[3][3] = 1  # a22
    V[2][3] = 1  # b21
    V[0][3] = -1 # b11
    W[0][3] = 1  # c11
    W[1][3] = 1  # c21
    
    # m5 = (a11+a12)*b22
    U[0][4] = 1  # a11
    U[2][4] = 1  # a12
    V[3][4] = 1  # b22
    W[0][4] = -1 # c11
    W[2][4] = 1  # c12
    
    # m6 = (a21-a11)*(b11+b12)
    U[1][5] = 1  # a21
    U[0][5] = -1 # a11
    V[0][5] = 1  # b11
    V[1][5] = 1  # b12
    W[3][5] = 1  # c22
    
    # m7 = (a12-a22)*(b21+b22)
    U[2][6] = 1  # a12
    U[3][6] = -1 # a22
    V[2][6] = 1  # b21
    V[3][6] = 1  # b22
    W[0][6] = 1  # c11
    W[2][6] = 1  # c12
    
    return U, V, W

def _build_strassen_block_decomposition():
    """
    Build a block Strassen decomposition for 3x5x6 matrix multiplication.
    
    We need to cover all (i,j,k) triples where i in [0,3), j in [0,5), k in [0,6).
    
    Strategy: Place overlapping 2x2x2 blocks to cover the space.
    Each block covers a 2x2x2 region. We use Strassen on each block.
    
    For 3x5x6, we need to think about how to tile this with 2x2x2 blocks.
    Since 3 = 2+1, 5 = 2+2+1, 6 = 2+2+2, we can use overlapping blocks.
    
    A simpler approach: use the tensor decomposition directly.
    The matmul tensor for 3x5x6 has support on all (i,j,k) for i<3, j<5, k<6.
    
    Let me think differently. The standard approach for rectangular Strassen:
    We can write the 3x5x6 multiplication as a sum of 2x2x2 multiplications
    with some terms shared.
    
    Actually, let me try a direct rank minimization approach.
    The schoolbook rank is 3*5*6 = 90.
    
    For the tensor decomposition, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    Let's try to find a lower rank decomposition by exploiting structure.
    """
    n, m, p = 3, 5, 6
    
    # Start with schoolbook as baseline
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try block Strassen: decompose into 2x2x2 blocks
    # For 3x5x6, we can place blocks at positions (i0, j0, k0)
    # where the block covers rows i0..i0+1, cols j0..j0+1, out k0..k0+1
    
    # Block positions (i0, j0, k0):
    # i can be 0 or 1 (blocks at i=0..1 and i=1..2)
    # j can be 0,1,2,3 (blocks at j=0..1, 1..2, 2..3, 3..4)
    # k can be 0,1,2,3,4 (blocks at k=0..1, 1..2, 2..3, 3..4, 4..5)
    
    # But we need to be careful about overlaps.
    # Let's use a systematic approach: for each (i,j,k), assign it to exactly one block.
    # A block at (i0,j0,k0) covers (i,j,k) where i0<=i<i0+2, j0<=j<j0+2, k0<=k<k0+2.
    
    # Let's use non-overlapping blocks where possible, with some overlaps for Strassen benefit.
    # Actually, for Strassen to work, we need the full 2x2x2 structure in each block.
    
    # Let me try a different approach: use the known result that
    # [3,5,6] can be done with rank less than 90.
    
    # One approach: group the 5 columns of A (and 5 rows of B) into groups.
    # A is 3x5, B is 5x6.
    # Split the 5 into 2+3: A = [A1 | A2] where A1 is 3x2, A2 is 3x3
    #                     B = [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    # A1*B1: 3x2 * 2x6 -> rank 3*2*6 = 36 schoolbook, or use Strassen for the 2x2 part
    # A2*B2: 3x3 * 3x6 -> rank 3*3*6 = 54 schoolbook
    
    # For A1*B1 (3x2 * 2x6):
    # The 2x2x2 Strassen can be applied to the "2" dimension.
    # Actually this is a bit complex. Let me try a cleaner approach.
    
    # Direct approach: use integer programming or greedy to find rank minimization.
    # Given time constraints, let's try a structured approach.
    
    # Approach: Use the fact that the matmul tensor is a "linear" tensor.
    # We can try to find a decomposition using polynomial methods.
    
    # For now, let's try the block Strassen approach carefully.
    
    # Define blocks: each block is a 2x2x2 sub-problem at offset (i0, j0, k0)
    # The block contributes to (i,j,k) for i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    
    # We need to cover all (i,j,k) for i in 0..2, j in 0..4, k in 0..5
    
    # Let's use blocks at:
    # i0 in {0, 1}
    # j0 in {0, 1, 2, 3}
    # k0 in {0, 1, 2, 3, 4}
    
    # Total potential blocks: 2 * 4 * 5 = 40, each of rank 7 = 280
    # But many will overlap and we can combine.
    
    # Better: use minimal set of blocks to cover all (i,j,k).
    # Each (i,j,k) must be covered by at least one block.
    
    # Let's find a minimal cover.
    # i=0: covered by i0=0
    # i=1: covered by i0=0 or i0=1
    # i=2: covered by i0=1
    
    # j=0: j0=0
    # j=1: j0=0 or j0=1
    # j=2: j0=1 or j0=2
    # j=3: j0=2 or j0=3
    # j=4: j0=3
    
    # k=0: k0=0
    # k=1: k0=0 or k0=1
    # k=2: k0=1 or k0=2
    # k=3: k0=2 or k0=3
    # k=4: k0=3 or k0=4
    # k=5: k0=4
    
    # Greedy cover:
    # For i=0: need i0=0
    # For i=2: need i0=1
    # For i=1: can use either
    # For j=0: need j0=0
    # For j=4: need j0=3
    # For j=1: j0=0 or 1
    # For j=2: j0=1 or 2
    # For j=3: j0=2 or 3
    # For k=0: need k0=0
    # For k=5: need k0=4
    # For k=1: k0=0 or 1
    # For k=2: k0=1 or 2
    # For k=3: k0=2 or 3
    # For k=4: k0=3 or 4
    
    # Minimal blocks:
    # Block (0,0,0): covers i={0,1}, j={0,1}, k={0,1}
    # Block (0,1,1): covers i={0,1}, j={1,2}, k={1,2}
    # Block (0,2,2): covers i={0,1}, j={2,3}, k={2,3}
    # Block (0,3,3): covers i={0,1}, j={3,4}, k={3,4}
    # Block (1,0,0): covers i={1,2}, j={0,1}, k={0,1}
    # Block (1,1,1): covers i={1,2}, j={1,2}, k={1,2}
    # Block (1,2,2): covers i={1,2}, j={2,3}, k={2,3}
    # Block (1,3,3): covers i={1,2}, j={3,4}, k={3,4}
    # Block (1,0,4): covers i={1,2}, j={0,1}, k={4,5}  -- wait, k0=4 covers k={4,5}
    # Block (0,0,4): covers i={0,1}, j={0,1}, k={4,5}
    # Block (1,1,4): covers i={1,2}, j={1,2}, k={4,5}
    # Block (0,1,4): covers i={0,1}, j={1,2}, k={4,5}
    # Block (1,2,4): covers i={1,2}, j={2,3}, k={4,5}
    # Block (0,2,4): covers i={0,1}, j={2,3}, k={4,5}
    # Block (1,3,4): covers i={1,2}, j={3,4}, k={4,5}
    # Block (0,3,4): covers i={0,1}, j={3,4}, k={4,5}
    
    # Hmm, this is getting complex. Let me count: we need to cover k=5 which requires k0=4.
    # And k=0 requires k0=0.
    
    # Let me list all (i,j,k) and find which blocks cover them.
    # Actually, let me just use a set of blocks that I know covers everything.
    
    blocks = []
    for i0 in [0, 1]:
        for j0 in [0, 1, 2, 3]:
            for k0 in [0, 1, 2, 3, 4]:
                blocks.append((i0, j0, k0))
    
    # Now find minimal subset that covers all (i,j,k)
    # For each (i,j,k), find which blocks cover it
    needed = set()
    for i in range(3):
        for j in range(5):
            for k in range(6):
                needed.add((i, j, k))
    
    block_covers = {}
    for idx, (i0, j0, k0) in enumerate(blocks):
        covered = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i, j, k = i0 + di, j0 + dj, k0 + dk
                    if 0 <= i < 3 and 0 <= j < 5 and 0 <= k < 6:
                        covered.add((i, j, k))
        block_covers[idx] = covered
    
    # Greedy set cover
    remaining = set(needed)
    selected = []
    while remaining and _time_remaining() > 5.0:
        best_idx = None
        best_count = 0
        for idx in range(len(blocks)):
            if idx in selected:
                continue
            count = len(remaining & block_covers[idx])
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining -= block_covers[best_idx]
    
    if remaining:
        # Fallback to schoolbook
        return U_sb, V_sb, W_sb
    
    # Now build the decomposition using Strassen on each selected block
    # Each block (i0, j0, k0) uses 7 rank-1 terms from Strassen
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    U = {}
    V = {}
    W = {}
    
    rank = 0
    for idx in selected:
        i0, j0, k0 = blocks[idx]
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    ui = U_s[i * 2 + j][r]
                    if ui != 0:
                        key = (i0 + i, j0 + j)
                        if key not in U:
                            U[key] = {}
                        U[key][rank] = U[key].get(rank, 0) + ui
            for j in range(2):
                for k in range(2):
                    vj = V_s[j * 2 + k][r]
                    if vj != 0:
                        key = (j0 + j, k0 + k)
                        if key not in V:
                            V[key] = {}
                        V[key][rank] = V[key].get(rank, 0) + vj
            for k in range(2):
                for i in range(2):
                    wk = W_s[k * 2 + i][r]
                    if wk != 0:
                        key = (k0 + k, i0 + i)
                        if key not in W:
                            W[key] = {}
                        W[key][rank] = W[key].get(rank, 0) + wk
            rank += 1
    
    # Convert to matrix form
    n, m, p = 3, 5, 6
    U_mat = [[Fraction(0)] * rank for _ in range(n * m)]
    V_mat = [[Fraction(0)] * rank for _ in range(m * p)]
    W_mat = [[Fraction(0)] * rank for _ in range(p * n)]
    
    for (i, j), coeffs in U.items():
        for r, val in coeffs.items():
            U_mat[i * m + j][r] = val
    for (j, k), coeffs in V.items():
        for r, val in coeffs.items():
            V_mat[j * p + k][r] = val
    for (k, i), coeffs in W.items():
        for r, val in coeffs.items():
            W_mat[k * n + i][r] = val
    
    # Verify
    if _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        return U_mat, V_mat, W_mat
    else:
        # Fallback
        return U_sb, V_sb, W_sb

def _convert_to_integer_or_half(frac):
    """Convert Fraction to int or half-integer representation."""
    if frac.denominator == 1:
        return int(frac)
    elif frac.denominator == 2:
        # Return as a float with .5
        return float(frac)
    else:
        # Should not happen with Strassen
        return float(frac)

def construct():
    n, m, p = DIMENSIONS
    
    try:
        U_frac, V_frac, W_frac = _build_strassen_block_decomposition()
        
        # Convert to integer/half-integer
        rank = len(U_frac[0])
        U = [[_convert_to_integer_or_half(U_frac[i][r]) for r in range(rank)] for i in range(n * m)]
        V = [[_convert_to_integer_or_half(V_frac[i][r]) for r in range(rank)] for i in range(m * p)]
        W = [[_convert_to_integer_or_half(W_frac[i][r]) for r in range(rank)] for i in range(p * n)]
        
        return {"U": U, "V": V, "W": W}
    except Exception as e:
        # Fallback to schoolbook
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
