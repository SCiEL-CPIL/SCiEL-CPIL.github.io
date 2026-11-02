"""Matrix multiplication 2x5x6 rank-minimal decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

_start_time = time.time()
_DEADLINE = 135  # safety margin before 150s kill

def _build_strassen_2x2x2():
    """Return the 7 Strassen terms for 2x2x2 as lists of (idx_u, idx_v, idx_w, coeff).
    U index: i*m+j with m=2 -> i*2+j
    V index: j*p+k with p=2 -> j*2+k  
    W index: k*n+i with n=2 -> k*2+i
    """
    # Standard Strassen for 2x2x2:
    # M1=(A00+A11)(B00+B11)
    # M2=(A10+A11)B00
    # M3=A00(B01-B11)
    # M4=A11(B10-B00)
    # M5=(A00+A01)B11
    # M6=(A10-A00)(B00+B01)
    # M7=(A01-A11)(B10+B11)
    # C00=M1-M5+M4+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    terms = []
    # Term 1: M1
    terms.append([(0, 0, 0, 1), (1, 3, 0, 1), (0, 3, 1, 1)])
    # Term 2: M2
    terms.append([(2, 0, 0, 1), (3, 0, 1, 1), (0, 1, -1, 1)])
    # Term 3: M3
    terms.append([(0, 0, 0, 1), (1, 1, 1, 1), (0, 1, 1, 1)])
    # Term 4: M4
    terms.append([(3, 0, 0, 1), (1, 0, -1, 1), (0, 1, 1, 1)])
    # Term 5: M5
    terms.append([(0, 1, 0, 1), (3, 1, 1, 1), (0, 0, -1, 1)])
    # Term 6: M6
    terms.append([(2, 0, -1, 1), (0, 0, 1, 1), (3, 0, 1, 1)])
    # Term 7: M7
    terms.append([(1, 3, 1, 1), (0, 1, 1, 1), (0, 0, 1, 1)])
    return terms

def _build_block_2x4x6():
    """Build the 2x4x6 block using 3 Strassen 2x2x2 sub-blocks (splitting k=6 into 3x2).
    Wait, that's 2x4x2 three times = 3*14 = 42 terms.
    Actually let me reconsider: 2x4x6. Split j=4 into 2+2, k=6 into 2+2+2.
    Each 2x2x2 block uses Strassen (rank 7). Total: 2*3*7 = 42.
    """
    strassen = _build_strassen_2x2x2()
    entries = []  # list of (u_idx, v_idx, w_idx, coeff)
    n, m, p = 2, 4, 6
    for j_block in range(2):  # j in {0,1} or {2,3}
        j_base = j_block * 2
        for k_block in range(3):  # k in {0,1}, {2,3}, {4,5}
            k_base = k_block * 2
            for term in strassen:
                for (local_u, local_v, local_w, coeff) in term:
                    # local_u: i*2 + j_local (j_local in {0,1})
                    i_local = local_u // 2
                    j_local = local_u % 2
                    u_idx = i_local * m + (j_base + j_local)
                    
                    # local_v: j_local2 * 2 + k_local
                    j_local2 = local_v // 2
                    k_local = local_v % 2
                    v_idx = (j_base + j_local2) * p + (k_base + k_local)
                    
                    # local_w: k_local2 * 2 + i_local2
                    k_local2 = local_w // 2
                    i_local2 = local_w % 2
                    w_idx = (k_base + k_local2) * n + i_local2
                    
                    entries.append((u_idx, v_idx, w_idx, coeff))
    return entries

def _build_block_2x1x6():
    """Build the 2x1x6 block (standard multiplication, rank 12)."""
    entries = []
    n, m, p = 2, 1, 6
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            entries.append((u_idx, v_idx, w_idx, 1))
    return entries

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: Split m=5 into 1+4.
    # Block 1: 2x1x6, rank 12
    # Block 2: 2x4x6, rank 42 (3 Strassen 2x2x2 blocks)
    # Total rank: 54
    
    block1_entries = _build_block_2x1x6()
    block2_entries = _build_block_2x4x6()
    
    rank = len(block1_entries) + len(block2_entries)
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    for (u_idx, v_idx, w_idx, coeff) in block1_entries:
        U[u_idx][col] += Fraction(coeff)
        V[v_idx][col] += Fraction(coeff)
        W[w_idx][col] += Fraction(coeff)
        col += 1
    
    for (u_idx, v_idx, w_idx, coeff) in block2_entries:
        U[u_idx][col] += Fraction(coeff)
        V[v_idx][col] += Fraction(coeff)
        W[w_idx][col] += Fraction(coeff)
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify correctness
    _verify(U, V, W, n, m, p)
    
    # Convert to regular lists (integers or half-integers)
    U_out = []
    for row in U:
        U_out.append([float(x) if x.denominator != 1 else int(x) for x in row])
    V_out = []
    for row in V:
        V_out.append([float(x) if x.denominator != 1 else int(x) for x in row])
    W_out = []
    for row in W:
        W_out.append([float(x) if x.denominator != 1 else int(x) for x in row])
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        raise TimeoutError(f"Exceeded deadline: {elapsed:.1f}s")
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1
                actual = Fraction(0)
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if actual != expected:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {actual}, expected {expected}")

# EVOLVE-BLOCK-END
