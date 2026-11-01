"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that no spurious terms exist
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_block(n, m, p):
    """
    For multiplying (n x m) by (m x p) where n=m=2, p=16.
    The 2x16 matrix can be seen as 8 independent 2x2 blocks.
    Each 2x2x2 multiplication has rank 7 (Strassen).
    Total rank = 8 * 7 = 56.
    
    For block t (0..7), the columns in B are at indices 2t and 2t+1.
    Output rows in C are at indices 2t*n+0, 2t*n+1, (2t+1)*n+0, (2t+1)*n+1.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # V indices for B entries in this block:
        # B[0, 2t] -> V[2t]
        # B[0, 2t+1] -> V[2t+1]
        # B[1, 2t] -> V[16+2t]
        # B[1, 2t+1] -> V[16+2t+1]
        
        # W indices for C entries in this block:
        # C[0, 2t] -> W[4t]
        # C[1, 2t] -> W[4t+1]
        # C[0, 2t+1] -> W[4t+2]
        # C[1, 2t+1] -> W[4t+3]
        
        # U indices: A00=0, A01=1, A10=2, A11=3
        
        intermediates = [
            # M1 = (A00 + A11) * (B00 + B11)
            {'U': [(0, 1), (3, 1)],
             'V': [(2*t, 1), (16+2*t+1, 1)],
             'W': [(4*t, 1), (4*t+3, 1)]},
            # M2 = (A10 + A11) * B00
            {'U': [(2, 1), (3, 1)],
             'V': [(2*t, 1)],
             'W': [(4*t+1, 1), (4*t+3, -1)]},
            # M3 = A00 * (B01 - B11)
            {'U': [(0, 1)],
             'V': [(2*t+1, 1), (16+2*t+1, -1)],
             'W': [(4*t+2, 1), (4*t+3, 1)]},
            # M4 = A11 * (B10 - B00)
            {'U': [(3, 1)],
             'V': [(16+2*t, 1), (2*t, -1)],
             'W': [(4*t, 1), (4*t+1, 1)]},
            # M5 = (A00 + A01) * B11
            {'U': [(0, 1), (1, 1)],
             'V': [(16+2*t+1, 1)],
             'W': [(4*t, -1), (4*t+2, 1)]},
            # M6 = (A10 - A00) * (B00 + B01)
            {'U': [(2, 1), (0, -1)],
             'V': [(2*t, 1), (2*t+1, 1)],
             'W': [(4*t+3, 1)]},
            # M7 = (A01 - A11) * (B10 + B11)
            {'U': [(1, 1), (3, -1)],
             'V': [(16+2*t, 1), (16+2*t+1, 1)],
             'W': [(4*t, 1)]}
        ]
        
        for inter in intermediates:
            for idx, coeff in inter['U']:
                U[idx][col] += coeff
            for idx, coeff in inter['V']:
                V[idx][col] += coeff
            for idx, coeff in inter['W']:
                W[idx][col] += coeff
            col += 1
    
    return U, V, W

def _strassen_4x4x16(n, m, p):
    """
    Structural alternative: treat the 2x2 * 2x16 as a 4x4 * 4x16 problem
    by duplicating rows/columns, then use Strassen on 4x4x4 blocks.
    
    Actually, a cleaner structural change: use the "block Strassen" approach
    where we group the 16 columns into 4 groups of 4, and for each group
    use a 2x2 * 2x4 multiplication. The 2x2 * 2x4 can be seen as two
    2x2 * 2x2 multiplications, but we can potentially share intermediates
    across the two 2x2 blocks within a group.
    
    However, the most promising structural change is to use the fact that
    the tensor T[a,b,c] for 2x2*2x16 has a specific algebraic structure
    that might allow rank < 56. Let's try a "shared Strassen" approach
    where we share some intermediate multiplications across adjacent blocks.
    
    For now, let's try a different grouping: 4 groups of 4 columns each.
    For each group of 4 columns (2x4 B block), we compute 2x2 * 2x4 -> 2x4.
    This is equivalent to two 2x2*2x2 multiplications, but we can use
    a 2x2*2x4 algorithm. The rank of 2x2*2x4 is at most 2*7=14,
    but might be lower. Total: 4 * 14 = 56 (same).
    
    Let me try yet another approach: use the "ladder" or "shifted" Strassen
    where we exploit the linear structure across k values.
    
    Actually, the cleanest structural change: try to find a rank-48 or
    rank-40 decomposition using a different algebraic framework.
    
    For this iteration, let's try a "hybrid" approach: use Strassen for
    pairs of blocks (4 pairs of 2 blocks each), and see if we can save
    rank by sharing intermediates between paired blocks.
    
    For blocks t and t+1 (t even), the B entries are at columns 2t, 2t+1,
    2t+2, 2t+3. We can form a 2x4 B block and compute 2x2 * 2x4.
    
    The 2x2 * 2x4 multiplication:
    C[:, 2t:2t+4] = A @ B[:, 2t:2t+4]
    
    This is a 2x2 * 2x4 -> 2x4 multiplication. We can use:
    - Split B into B_left (2x2) and B_right (2x2)
    - Use Strassen on each: 7 + 7 = 14 multiplications
    - But we can potentially share: e.g., (A00+A11) is used in both
      Strassen patterns, so we can compute it once.
    
    Let's implement a shared-Strassen for 2x2 * 2x4:
    
    For B_left (columns 2t, 2t+1) and B_right (columns 2t+2, 2t+3):
    
    Strassen on B_left:
    M1_L = (A00+A11)(B00_L + B11_L)
    M2_L = (A10+A11) B00_L
    M3_L = A00 (B01_L - B11_L)
    M4_L = A11 (B10_L - B00_L)
    M5_L = (A00+A01) B11_L
    M6_L = (A10-A00)(B00_L + B01_L)
    M7_L = (A01-A11)(B10_L + B11_L)
    
    Strassen on B_right:
    M1_R = (A00+A11)(B00_R + B11_R)
    M2_R = (A10+A11) B00_R
    M3_R = A00 (B01_R - B11_R)
    M4_R = A11 (B10_R - B00_R)
    M5_R = (A00+A01) B11_R
    M6_R = (A10-A00)(B00_R + B01_R)
    M7_R = (A01-A11)(B10_R + B11_R)
    
    Shared terms: (A00+A11), (A10+A11), (A00+A01), (A10-A00), (A01-A11)
    These are 5 shared "A-side" combinations. But the rank is determined
    by the number of scalar multiplications, not by the number of A-side
    terms. So we still need 14 multiplications unless we can merge some.
    
    One idea: M1_L and M1_R both use (A00+A11). We can compute:
    M1_combined = (A00+A11) * (B00_L + B11_L + B00_R + B11_R)
    But then we need to subtract to get M1_L and M1_R separately, which
    requires additional terms. This doesn't save rank.
    
    So the shared approach doesn't obviously save rank. Let's stick with
    rank 56 but verify it's correct, and try a completely different
    structural approach for potential improvement.
    
    Alternative structural change: use the "binary splitting" or "divide
    and conquer" approach on the k dimension. Split 16 columns into
    two groups of 8, compute 2x2 * 2x8 for each, then combine.
    The rank of 2x2 * 2x8 is at most 2 * rank(2x2*2x4) = 2 * 14 = 28.
    Total: 2 * 28 = 56. Same.
    
    Let's try a rank-48 approach using a different algebraic identity.
    """
    n, m, p = 2, 2, 16
    # For now, return the standard Strassen (rank 56) as a safe baseline
    return _strassen_block(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Try Strassen-based construction (rank 56 = 8 * 7)
    U, V, W = _strassen_block(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Try alternative structural approach
    U, V, W = _strassen_4x4x16(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
