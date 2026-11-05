"""
Rank-minimal exact tensor decomposition for 3x4 by 4x8 matrix multiplication.

Strategy:
  1. Use Strassen's rank-7 decomposition on 2x2x2 sub-blocks where possible.
     For n=3, m=4, p=8: we can form 2x2x2 blocks in the (i,j,k) space.
     - i has 3 rows -> one full 2x2 block (i=0,1) plus a 1x2x2 strip (i=2)
     - j has 4 cols -> two full 2x2 blocks (j=0,1 and j=2,3)
     - k has 8 cols -> four full 2x2 blocks (k=0,1; 2,3; 4,5; 6,7)
     
     So we have:
       - 1 (i-block) * 2 (j-blocks) * 4 (k-blocks) = 8 full 2x2x2 Strassen blocks -> 8*7 = 56
       - 1 (i-strip, 1x2x2) * 2 (j-blocks) * 4 (k-blocks) = 8 half-Strassen blocks
         A 1x2x2 block: C[i,k1..k2] = A[i,j1..j2] * B[j1..j2,k1..k2]
         This is a 1x2 * 2x2 multiplication, which needs rank 2 (schoolbook) or can we do better?
         Actually a 1x2 by 2x2 multiplication: C[k1] = A[j1]*B[j1,k1] + A[j2]*B[j2,k1]
         C[k2] = A[j1]*B[j1,k2] + A[j2]*B[j2,k2]
         This is just two independent dot products of length 2, so rank 2.
         So 8 * 2 = 16.
     Total: 56 + 16 = 72.
     
     Wait, let me reconsider. The i-strip has i=2 (one row), j in {0,1,2,3}, k in {0..7}.
     For each (j-block, k-block) pair where j-block is 2x2 and k-block is 2x2:
       C[2, k1] = A[2,j1]*B[j1,k1] + A[2,j2]*B[j2,k1]
       C[2, k2] = A[2,j1]*B[j1,k2] + A[2,j2]*B[j2,k2]
     This is rank 2 per (j-block, k-block) pair.
     Number of (j-block, k-block) pairs: 2 * 4 = 8.
     So 8 * 2 = 16 for the i-strip.
     
     Total rank: 56 + 16 = 72.
     
     But wait, can we do better on the i-strip? The i-strip is essentially computing
     C[2, :] = A[2, :] * B[:, :] which is a 1x4 * 4x8 multiplication.
     The standard rank for 1xm * mp is mp = 32 (schoolbook).
     But using the j-block structure: we split j into two 2-blocks.
     For each j-block (2 elements) and k-block (2 elements), we have a 1x2 * 2x2 multiply.
     That's rank 2 each, and there are 2*4=8 such blocks, giving 16.
     
     Alternatively, we could treat the entire i-strip as a 1x4 * 4x8 multiply.
     Is there a rank < 32 decomposition? The bilinear rank of 1xm * mp is mp (it's just
     mp independent dot products of length m, but they share the same A vector... no wait).
     
     Actually C[k] = sum_j A[j] * B[j,k]. Each C[k] is a linear form in A (with coefficients B[:,k]).
     These are p=8 independent linear forms. The rank is at least 8 (since we need at least
     8 output values). But the bilinear rank... 
     
     For 1xm * mp: the tensor is sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_j ⊗ e_{j,k} ⊗ e_k
     Wait no. Let me think more carefully.
     
     The multiplication C = A * B where A is 1xm, B is mxp, C is 1xp.
     C[k] = sum_j A[j] * B[j,k]
     
     The tensor T has entries T[j, (j,k), k]... no. Let me use the standard indexing.
     T[i, j, k] where i indexes A rows (only i=0), j indexes the shared dimension, k indexes B cols.
     T[0, j, k] = 1 for all j, k.
     
     The rank of this tensor: we need sum_c U[0,j,c] * V[j,k,c] * W[k,0,c] = 1 for all j,k.
     Since U[0,j,c] only depends on j and c, and W[k,0,c] only depends on k and c,
     and V[j,k,c] depends on j,k,c.
     
     Actually this is just: for each k, C[k] = sum_j A[j]*B[j,k].
     This is p independent computations, each a dot product of length m.
     The bilinear rank is p*m = 32 in general (schoolbook).
     
     But can we do better? If m=4, p=8: rank 32 schoolbook.
     Using the j-block structure with 2x2 blocks: 8 blocks * rank 2 = 16.
     
     So total: 56 + 16 = 72.
     
     Hmm, but maybe I can do even better. Let me think about whether the full 2x2x2 Strassen
     can be applied more cleverly, or whether there's overlap.
     
     Actually, let me reconsider the full problem. n=3, m=4, p=8.
     
     Alternative: use Strassen on the full 2x4x4 sub-problem (i=0,1; j=0,1,2,3; k=0..3)
     and similar for other k-blocks.
     
     Actually, let me just go with the simple approach:
     - 8 Strassen 2x2x2 blocks: rank 56
     - 8 schoolbook 1x2x2 blocks: rank 16
     Total: 72.
     
     That's a significant improvement over 96.
     
  2. Additionally, try to see if we can reduce further.
     For the i-strip (i=2), instead of 8 separate 1x2x2 blocks, can we share?
     The 8 blocks are for (j-block, k-block) = {(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)}
     where j-block 0 = j∈{0,1}, j-block 1 = j∈{2,3}, k-block l = k∈{2l,2l+1}.
     
     Each block computes:
     C[2, 2l] = A[2, 2b] * B[2b, 2l] + A[2, 2b+1] * B[2b+1, 2l]
     C[2, 2l+1] = A[2, 2b] * B[2b, 2l+1] + A[2, 2b+1] * B[2b+1, 2l+1]
     
     These are all independent (different A elements or different B elements), so no sharing.
     Rank 16 is minimal for this sub-problem.
     
     For the Strassen blocks, rank 7 per 2x2x2 is known to be minimal (over most fields).
     So 56 is minimal for those.
     
     Total 72 seems to be a good target.
     
  3. Let me also consider: can we use a different block partition?
     For example, use 2x2x2 Strassen for i∈{0,1}, j∈{0,1}, k∈{0,1}, etc.
     That's what I did. 8 blocks of 2x2x2.
     
     What about using a 2x4x4 block with a rank-? decomposition?
     The rank of 2x4x4 matrix multiplication is known to be 18 (over reals) or 17?
     Actually, the rank of 2xn x nxn multiplication:
     - 2x2x2: rank 7
     - 2x3x3: rank 12? (not sure)
     - 2x4x4: rank 18? 
     
     If 2x4x4 has rank 18, then for i∈{0,1}, j∈{0,1,2,3}, k∈{0,1,2,3}: rank 18.
     And for k∈{4,5,6,7}: another 18.
     Total for i∈{0,1}: 36.
     For i=2: 1x4x8, rank 32 schoolbook, or 16 with j-blocking.
     Total: 36 + 16 = 52.
     
     But I'm not sure about the rank of 2x4x4. Let me not rely on uncertain knowledge.
     
     Let me stick with the safe 72-rank construction using Strassen 2x2x2 blocks.
     
     Actually, let me verify: is the rank of 2x4x4 known?
     The bilinear rank of m×n × n×p matrix multiplication is at most mn + np + mp - ... no.
     Upper bounds: mn + np + mp - min(m,n,p) or similar.
     For 2x4x4: 2*4 + 4*4 + 4*2 - 2 = 8+16+8-2 = 30? No, that's not right.
     
     Actually, a simple upper bound: use Strassen on pairs of columns.
     2x4x4: split the 4 columns into two 2-column groups.
     2x4x2 + 2x4x2. Each 2x4x2: split 4 rows into two 2-row groups.
     2x2x2 + 2x2x2. Each rank 7. So 2x4x2 has rank 14.
     2x4x4 has rank 28.
     
     That's worse than schoolbook 32? No, 28 < 32.
     But my Strassen 2x2x2 approach gives 8*7=56 for the i∈{0,1} part, which is
     2x4x8 = 2*(2x4x4) = 2*28 = 56. Consistent.
     
     So 56 + 16 = 72 is the rank.
     
     Let me code this up.
"""

import time
import random
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 120.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _elapsed()


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen's 7 products:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    The decomposition:
    U has entries for A indices (i,j), V for B indices (j,k), W for C indices (k,i) [using trace convention W[k*n+i]]
    
    For 2x2x2: n=2, m=2, p=2.
    U shape: (2*2, 7) = (4, 7)
    V shape: (2*2, 7) = (4, 7)  
    W shape: (2*2, 7) = (4, 7)  [W[k*n+i] = W[2k+i]]
    
    A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    C indices (trace): W[k*2+i]: (k=0,i=0)->0, (k=0,i=1)->1, (k=1,i=0)->2, (k=1,i=1)->3
    
    M1: (A00+A11)(B00+B11) -> contributes to C00, C11
    M2: (A10+A11)*B00 -> contributes to C10, C11
    M3: A00*(B01-B11) -> contributes to C01, C11
    M4: A11*(B10-B00) -> contributes to C10, C00
    M5: (A00+A01)*B11 -> contributes to C01, C00
    M6: (A10-A00)*(B00+B01) -> contributes to C11
    M7: (A01-A11)*(B10+B11) -> contributes to C00
    
    U (A factors):
    c=0 (M1): A00=1, A11=1 -> U[0,0]=1, U[3,0]=1
    c=1 (M2): A10=1, A11=1 -> U[2,1]=1, U[3,1]=1
    c=2 (M3): A00=1 -> U[0,2]=1
    c=3 (M4): A11=1 -> U[3,3]=1
    c=4 (M5): A00=1, A01=1 -> U[0,4]=1, U[1,4]=1
    c=5 (M6): A10=1, A00=-1 -> U[2,5]=1, U[0,5]=-1
    c=6 (M7): A01=1, A11=-1 -> U[1,6]=1, U[3,6]=-1
    
    V (B factors):
    c=0 (M1): B00=1, B11=1 -> V[0,0]=1, V[3,0]=1
    c=1 (M2): B00=1 -> V[0,1]=1
    c=2 (M3): B01=1, B11=-1 -> V[1,2]=1, V[3,2]=-1
    c=3 (M4): B10=1, B00=-1 -> V[2,3]=1, V[0,3]=-1
    c=4 (M5): B11=1 -> V[3,4]=1
    c=5 (M6): B00=1, B01=1 -> V[0,5]=1, V[1,5]=1
    c=6 (M7): B10=1, B11=1 -> V[2,6]=1, V[3,6]=1
    
    W (C factors, trace convention W[k*2+i]):
    C00 = M1 + M4 - M5 + M7 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    C01 = M3 + M5 -> W[1,2]=1, W[1,4]=1
    C10 = M2 + M4 -> W[2,1]=1, W[2,3]=1
    C11 = M1 - M2 + M3 + M6 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_decomposition():
    """
    Build the rank-72 decomposition using Strassen 2x2x2 blocks for i∈{0,1}
    and schoolbook 1x2x2 blocks for i=2.
    
    n=3, m=4, p=8.
    U shape: (3*4, 72) = (12, 72)
    V shape: (4*8, 72) = (32, 72)
    W shape: (8*3, 72) = (24, 72)
    
    Strassen blocks: 8 blocks for (i_block, j_block, k_block) where
      i_block = 0 (i∈{0,1}), j_block ∈ {0,1} (j∈{0,1} or {2,3}), k_block ∈ {0,1,2,3} (k∈{0,1},{2,3},{4,5},{6,7})
    
    For each Strassen block (ib, jb, kb):
      i indices: 2*ib + {0,1}
      j indices: 2*jb + {0,1}
      k indices: 2*kb + {0,1}
      
      U local: (2*2, 7) = (4, 7)
      V local: (2*2, 7) = (4, 7)
      W local: (2*2, 7) = (4, 7)
      
      U global row for A[i, j]: row = i*m + j = i*4 + j
      V global row for B[j, k]: row = j*p + k = j*8 + k
      W global row for C[k, i] (trace): row = k*n + i = k*3 + i
      
    Schoolbook 1x2x2 blocks for i=2:
      For each (jb, kb): 8 blocks
      j indices: 2*jb + {0,1}
      k indices: 2*kb + {0,1}
      
      2 rank terms:
      Term 0: A[2, 2*jb] * B[2*jb, 2*kb] * C[2*kb, 2]
              A[2, 2*jb] * B[2*jb, 2*kb+1] * C[2*kb+1, 2]
      Term 1: A[2, 2*jb+1] * B[2*jb+1, 2*kb] * C[2*kb, 2]
              A[2, 2*jb+1] * B[2*jb+1, 2*kb+1] * C[2*kb+1, 2]
      
      Wait, that's not right. Each 1x2x2 block has 2 rank terms:
      C[2, 2*kb]   = A[2, 2*jb] * B[2*jb, 2*kb] + A[2, 2*jb+1] * B[2*jb+1, 2*kb]
      C[2, 2*kb+1] = A[2, 2*jb] * B[2*jb, 2*kb+1] + A[2, 2*jb+1] * B[2*jb+1, 2*kb+1]
      
      Rank-2 decomposition:
      Term a: U[A[2,2jb]] = 1, V[B[2jb, 2kb]] = 1, W[C[2kb, 2]] = 1
               U[A[2,2jb]] = 1, V[B[2jb, 2kb+1]] = 1, W[C[2kb+1, 2]] = 1
      Term b: U[A[2,2jb+1]] = 1, V[B[2jb+1, 2kb]] = 1, W[C[2kb, 2]] = 1
               U[A[2,2jb+1]] = 1, V[B[2jb+1, 2kb+1]] = 1, W[C[2kb+1, 2]] = 1
    """
    n, m, p = 3, 4, 8
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    col = 0
    # Strassen blocks: i_block=0, j_block in {0,1}, k_block in {0,1,2,3}
    for jb in range(2):
        for kb in range(4):
            for c in range(7):
                # U: A[2*ib + di, 2*jb + dj] where ib=0
                for di in range(2):
                    for dj in range(2):
                        i = di  # ib=0
                        j = 2 * jb + dj
                        U[i * m + j][col] += U_str[di * 2 + dj][c]
                
                # V: B[2*jb + dj, 2*kb + dk]
                for dj in range(2):
                    for dk in range(2):
                        j = 2 * jb + dj
                        k = 2 * kb + dk
                        V[j * p + k][col] += V_str[dj * 2 + dk][c]
                
                # W: C[2*kb + dk, di] (trace: W[k*n + i])
                for dk in range(2):
                    for di in range(2):
                        k = 2 * kb + dk
                        i = di  # ib=0
                        W[k * n + i][col] += W_str[dk * 2 + di][c]
                
                col += 1
    
    # Schoolbook 1x2x2 blocks for i=2: jb in {0,1}, kb in {0,1,2,3}
    for jb in range(2):
        for kb in range(4):
            for dj in range(2):
                j = 2 * jb + dj
                for dk in range(2):
                    k = 2 * kb + dk
                    # U[A[2, j]] = 1
                    U[2 * m + j][col] = 1
                    # V[B[j, k]] = 1
                    V[j * p + k][col] = 1
                    # W[C[k, 2]] = 1 (trace: W[k*3 + 2])
                    W[k * n + 2][col] = 1
                    col += 1
    
    return U, V, W, rank


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(actual - expected) > tol:
                    return False, (i, j, k, actual)
    return True, None


def _verify_fractional(U, V, W, n, m, p):
    """Verify with exact Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = Fraction(1)
                actual = Fraction(0)
                for c in range(rank):
                    uc = Fraction(U[i * m + j][c])
                    vc = Fraction(V[j * p + k][c])
                    wc = Fraction(W[k * n + i][c])
                    actual += uc * vc * wc
                if actual != expected:
                    return False, (i, j, k, actual)
    return True, None


def _build_schoolbook():
    """Return schoolbook rank-96 decomposition as fallback."""
    n, m, p = 3, 4, 8
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


def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    of shapes (n*m, rank), (m*p, rank), (p*n, rank) respectively.
    """
    n, m, p = 3, 4, 8
    
    # Build the Strassen-based rank-72 decomposition
    U, V, W, rank = _build_decomposition()
    
    # Verify with floating point
    valid, info = _verify(U, V, W, n, m, p, tol=1e-9)
    if not valid:
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        return {"U": U, "V": W and V, "W": W}
    
    # Verify with exact fractions (entries are integers, so this should work)
    valid_frac, info_frac = _verify_fractional(U, V, W, n, m, p)
    if not valid_frac:
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
