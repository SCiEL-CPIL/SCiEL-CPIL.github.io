import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# 1x3x4 block:  A (1x3) * B (3x4) = C (1x4),  rank 8, integer entries.
# Index convention: U1[j, c], V1[l*4 + k, c], W1[k, c] with i=0.
# Note: V1 in the current program is 3x8, but for 1x3x4, V should be (3*4, 8) = (12, 8)?
# The current program's _V1 is 3x8. This suggests it might be for a different structure or I need to reshape.
# Let's check the current program's _V1 shape: 3 rows, 8 cols.
# If it's for 1x3x4, V should contract with B (3x4).
# If V is 3x8, it contracts with B's rows?
# Let's re-read the current program's _mul_block.
# It uses V1[j, c] * B[j, 0]. This implies B is treated as 3x1?
# The current program's block logic is confusing.
# I will define a standard 1x3x4 rank 8 block if I can, or fall back to schoolbook.
# Since I don't have a verified rank 8 1x3x4 block immediately available in memory,
# I will use the Rank 14 2x3x3 algorithm if I can recall it, or fall back to Rank 28 via 2x Rank 14.
#
# Actually, the safest bet for a score improvement is to implement the Rank 28 decomposition
# using two copies of a known Rank 14 2x3x3 algorithm.
#
# Rank 14 Algorithm for 2x3x3:
# The coefficients are known. I will hardcode a valid set if possible, 
# or use a simpler construction if Rank 14 is too complex to recall accurately.
#
# Alternative: Use Rank 28 via 2x Rank 14.
# If I cannot guarantee the Rank 14 coefficients, I will use Schoolbook for 2x3x3 (Rank 18)
# which gives 2*18 = 36. This is no improvement.
#
# Let's try to construct a Rank 24 decomposition for 2x3x6 directly?
#
# Given the uncertainty about the exact coefficients of the advanced algorithms,
# and the fact that the current program (score 1.0) is effectively Rank 36,
# any valid decomposition with Rank < 36 is an improvement.
#
# I will implement the Rank 28 decomposition using the following logic:
# 1. Split 2x3x6 into two 2x3x3 problems.
# 2. Use a known Rank 14 algorithm for 2x3x3.
#
# To ensure correctness, I will include a verification step.
# If the Rank 14 algorithm fails verification, I fall back to Schoolbook (Rank 36).
#
# Here is a known Rank 14 decomposition for 2x3x3 (from literature):
# It is complex to hardcode without error.
#
# However, I can use the **Rank 7 Strassen** for 2x2x2 and pad?
# 2x3x3 contains 2x2x2.
#
# Let's try a different, simpler improvement:
# Schoolbook is 36.
# Can we get Rank 30?
#
# I will implement the Rank 28 construction using the Rank 14 algorithm for 2x3x3.
# I will use the coefficients from a standard reference.
#
# Due to the risk of incorrect coefficients, I will implement a **Rank 24** construction
# if I can find one, otherwise I will stick to a robust **Rank 28** if the 14-rank block is verified.
#
# Actually, the most robust improvement I can guarantee without external lookup is
# to verify the Rank 8 1x3x4 block from the current program.
# If it is valid, I can build a Rank 28 solution.
# If not, I return Schoolbook.
#
# Let's verify the current program's 1x3x4 block.
# ---------------------------------------------------------------------------

_U1 = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1],
]
_V1 = [
    [1, 1, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 1],
]
_W1 = [
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 1, 0, 1],
]

def _verify_1x3x4_block(U, V, W):
    """
    Verify 1x3x4 block.
    U: 3xR, V: 3xR (contracting with B rows?), W: 4xR.
    Condition: sum_r U[j,r] * V[l,r] * W[k,r] == delta_{j,l} * delta_{k, ?}
    For 1x3x4: C_k = sum_j A_j B_{j k}.
    Tensor T_{j, l, k} = delta_{j,l}. (l is B row, k is B col).
    So sum_r U[j,r] V[l,r] W[k,r] should be delta_{j,l} for all k?
    No, W depends on k.
    So sum_r U[j,r] V[l,r] W[k,r] = delta_{j,l} * (something depending on k)?
    Actually, the tensor is T_{j, l, k} = delta_{j,l}.
    So sum_r U[j,r] V[l,r] W[k,r] = delta_{j,l}.
    This must hold for all j, l, k.
    """
    n_j = 3
    n_l = 3
    n_k = 4
    R = len(U[0])
    for j in range(n_j):
        for l in range(n_l):
            for k in range(n_k):
                total = Fraction(0)
                for r in range(R):
                    total += U[j][r] * V[l][r] * W[k][r]
                expected = Fraction(1) if j == l else Fraction(0)
                if total != expected:
                    return False
    return True

def _schoolbook_2x3x6():
    n, m, p = DIMENSIONS
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

def _assemble_rank28():
    """
    Assemble Rank 28 decomposition using two Rank 8 1x3x4 blocks 
    and two Rank 6 1x3x2 schoolbook blocks.
    """
    n, m, p = DIMENSIONS
    # 1x3x4 block rank 8
    # 1x3x2 block rank 6 (schoolbook)
    rank_1x3x4 = 8
    rank_1x3x2 = 6
    total_rank = 2 * rank_1x3x4 + 2 * rank_1x3x2 # 16 + 12 = 28
    
    U = [[Fraction(0) for _ in range(total_rank)] for _ in range(n * m)]
    V = [[Fraction(0) for _ in range(total_rank)] for _ in range(m * p)]
    W = [[Fraction(0) for _ in range(total_rank)] for _ in range(p * n)]
    
    # Block 1: i=0, k=0..3 (1x3x4)
    # A row 0, B cols 0..3
    col_offset = 0
    for r in range(rank_1x3x4):
        for j in range(3):
            U[0 * 3 + j][col_offset + r] = _U1[j][r]
            V[j * 6 + 0][col_offset + r] = 0 # V is 3x6, we use cols 0..3 for this block?
            # V index is j*p + k. Here k is 0..3.
            # The block V1 is 3x8. It contracts with B rows?
            # In 1x3x4, V is 3xR. It multiplies B (3x4).
            # So V[j, r] * B[j, k].
            # So V in the full matrix should be at V[j*6 + k, r].
            # But the block V1 is only 3x8.
            # This implies the block V1 is used for ALL k in 0..3?
            # No, the tensor condition sum_r U[j,r] V[l,r] W[k,r] = delta_{j,l}
            # means V does NOT depend on k.
            # So for the 1x3x4 block, V is 3x8.
            # In the full 2x3x6 matrix, V is 18x28.
            # For i=0, k=0..3, the V part is V[j*6 + k, r].
            # But the block V1 is 3x8.
            # This implies V[j*6 + k, r] = V1[j, r] for k=0..3?
            # Let's check the tensor condition again.
            # If V1[j, r] is used for all k, then sum_r U[j,r] V1[j,r] W[k,r] = delta_{j,j}?
            # No, sum_r U[j,r] V1[l,r] W[k,r] = delta_{j,l}.
            # So V1[l, r] is used for all k.
            # So in the full matrix, V[j*6 + k, r] should be V1[j, r] for k=0..3.
            
            for k in range(4):
                V[j * 6 + k][col_offset + r] = _V1[j][r]
            W[k * 2 + 0][col_offset + r] = _W1[k][r] # W index is k*n + i. i=0.
            
    col_offset += rank_1x3x4
    
    # Block 2: i=1, k=0..3 (1x3x4)
    for r in range(rank_1x3x4):
        for j in range(3):
            U[1 * 3 + j][col_offset + r] = _U1[j][r]
            for k in range(4):
                V[j * 6 + k][col_offset + r] = _V1[j][r]
            W[k * 2 + 1][col_offset + r] = _W1[k][r]
            
    col_offset += rank_1x3x4

    # Block 3: i=0, k=4..5 (1x3x2 schoolbook)
    # A row 0, B cols 4,5
    for j in range(3):
        for k in range(2):
            U[0 * 3 + j][col_offset + j * 2 + k] = 1
            V[j * 6 + 4 + k][col_offset + j * 2 + k] = 1
            W[(4 + k) * 2 + 0][col_offset + j * 2 + k] = 1
            
    col_offset += rank_1x3x2

    # Block 4: i=1, k=4..5 (1x3x2 schoolbook)
    for j in range(3):
        for k in range(2):
            U[1 * 3 + j][col_offset + j * 2 + k] = 1
            V[j * 6 + 4 + k][col_offset + j * 2 + k] = 1
            W[(4 + k) * 2 + 1][col_offset + j * 2 + k] = 1

    return {"U": U, "V": V, "W": W}

def _verify_full(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i * m + j][r] * V[l * p + k][r] * W[k * n + i][r]
                    expected = Fraction(1) if j == l else Fraction(0)
                    if total != expected:
                        return False
    return True

def construct():
    deadline = time.time() + 110.0
    try:
        if _verify_1x3x4_block(_U1, _V1, _W1):
            result = _assemble_rank28()
            if time.time() < deadline and _verify_full(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    return _schoolbook_2x3x6()

# EVOLVE-BLOCK-END
