import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    Strassen's 7 products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)(B00)
    M3 = (A00)(B11-B00)
    M4 = (A11)(B10-B11)
    M5 = (A00+A01)(B11)
    M6 = (A10-A11)(B01+B11)
    M7 = (A01-A10)(B00+B10)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 - M7
    """
    # For each product Mc, we need U, V, W vectors such that
    # U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] gives the coefficient
    # of A[i,j]*B[j,k] in Mc.
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A00 (idx 0), A11 (idx 3)
    # B part: B00 (idx 0), B11 (idx 3)
    # C part: C00 (idx 0), C11 (idx 3)
    # T[0,0,0] += 1: U[0,0]*V[0,0]*W[0,0]
    # T[0,0,1] += 1: U[0,0]*V[1,0]*W[2,0]  -- but V[1,0] is B01, not B11
    # Wait, V[j*2+k]. For j=0,k=0: V[0]. For j=0,k=1: V[1]. For j=1,k=0: V[2]. For j=1,k=1: V[3].
    # B00 is j=0,k=0 -> V[0]. B11 is j=1,k=1 -> V[3].
    # So M1 has B00 and B11. V[0,0]=1, V[3,0]=1.
    # A00 is i=0,j=0 -> U[0]. A11 is i=1,j=1 -> U[3].
    # So U[0,0]=1, U[3,0]=1.
    # C00 is k=0,i=0 -> W[0]. C11 is k=1,i=1 -> W[3].
    # So W[0,0]=1, W[3,0]=1.
    # Check: T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. Correct (A00*B00 -> C00).
    # T[0,0,1]: U[0,0]*V[1,0]*W[2,0] = 1*0*0 = 0. But M1 should contribute to C01 via A00*B01? No, M1 is (A00+A11)(B00+B11). It doesn't involve B01.
    # T[0,1,0]: U[1,0]*V[2,0]*W[0,0] = 0*0*1 = 0. M1 doesn't involve A01 or B10.
    # T[0,1,1]: U[1,0]*V[3,0]*W[2,0] = 0*1*0 = 0.
    # T[1,0,0]: U[2,0]*V[0,0]*W[0,1] = 0*1*0 = 0.
    # T[1,0,1]: U[2,0]*V[1,0]*W[1,1] = 0*0*0 = 0.
    # T[1,1,0]: U[3,0]*V[2,0]*W[0,1] = 1*0*0 = 0. But M1 should contribute to C10 via A11*B10? No, M1 is (A00+A11)(B00+B11). It involves A11 and B00, B11.
    # A11*B00: i=1,j=1,k=0. T[1,1,0]: U[3,0]*V[2,0]*W[0,1] = 1*0*0 = 0. But V[2,0] should be 1 for B10? No, B00 is j=0,k=0 -> V[0]. So V[2,0]=0 is correct.
    # Wait, I'm confused. Let me redo.
    # V[j*2+k]: j is row of B, k is column of B.
    # B00: j=0,k=0 -> V[0]
    # B01: j=0,k=1 -> V[1]
    # B10: j=1,k=0 -> V[2]
    # B11: j=1,k=1 -> V[3]
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # A00*B00: i=0,j=0,k=0 -> T[0,0,0]
    # A00*B11: i=0,j=0,k=1 -> T[0,0,1]  (j=0 for A, k=1 for B... wait)
    # A[i,j]*B[j,k]: A00 is i=0,j=0. B11 is j=1,k=1. But j must match! A00 has j=0, B11 has j=1. They don't multiply!
    # I made an error. (A00+A11)(B00+B11) means we're computing a product of two linear combinations.
    # A00*A00? No. The product is (A00+A11)*(B00+B11) which is a scalar product.
    # This contributes to the output C. But which C?
    # In Strassen, M1 = (A00+A11)(B00+B11) is used to compute parts of C.
    # The tensor decomposition requires that for each (i,j,k), the sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
    # So each Mc contributes to specific (i,j,k) entries.
    
    # Let me use a known correct rank-7 decomposition.
    # From literature, one valid decomposition is:
    
    # c=0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]  (diagonal-like)
    # This gives T[0,0,0]=1, T[1,1,1]=1, others 0.
    
    # Actually, let me just use the standard Strassen and verify.
    # I'll use a different known valid decomposition.
    
    U = np.array([
        [1,  0,  0,  1,  0,  0,  0],
        [0,  1,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  1,  0,  0],
        [1,  0,  0,  1,  0,  0,  0]
    ], dtype=float)
    
    V = np.array([
        [1,  0,  0,  1,  0,  0,  0],
        [0,  1,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  1,  0,  0],
        [1,  0,  0,  1,  0,  0,  0]
    ], dtype=float)
    
    W = np.array([
        [1,  0,  0,  1,  0,  0,  0],
        [0,  1,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  1,  0,  0],
        [1,  0,  0,  1,  0,  0,  0]
    ], dtype=float)
    
    # This is just schoolbook with 4 terms, not rank 7. Let me use the actual Strassen.
    
    # Correct Strassen rank-7 decomposition:
    # Reference: standard Strassen algorithm
    U = np.array([
        [1,  0,  1,  0,  1,  0,  0],
        [0,  0,  0,  0,  1,  1,  1],
        [0,  1,  0,  0,  0,  1, -1],
        [1,  1,  0,  1,  0, -1, 0]
    ], dtype=float)
    
    V = np.array([
        [1,  1,  0,  0,  0,  0,  1],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  1,  1],
        [1,  0,  0,  0,  1,  0,  0]
    ], dtype=float)
    
    W = np.array([
        [1,  0,  0,  0,  0,  0,  1],
        [0,  0,  0,  1,  1,  0,  0],
        [1,  1,  0,  0,  0,  0,  0],
        [0,  0,  1,  1,  0,  1, -1]
    ], dtype=float)
    
    if not _check_exact(U, V, W, 2, 2, 2, tol=1e-9):
        # If Strassen fails, use schoolbook rank 8
        U = np.zeros((4, 8), dtype=float)
        V = np.zeros((4, 8), dtype=float)
        W = np.zeros((4, 8), dtype=float)
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j, c] = 1
                    V[j*2+k, c] = 1
                    W[k*2+i, c] = 1
                    c += 1
    return U, V, W

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank

def _build_decomposition(n, m, p):
    """
    Build 3x5x6 decomposition using Strassen 2x2x2 blocks where possible.
    
    Strategy: Split the problem into blocks and use Strassen for 2x2x2 subproblems.
    
    Split m=5 into [2,2,1] and p=6 into [2,2,2].
    
    For each (m_block, p_block) pair:
    - 2x2: use Strassen (rank 7) for i=0,1 + schoolbook (rank 2) for i=2 -> rank 9
    - 2x1: schoolbook rank 3 (n*m*p = 3*2*1 = 6... wait)
    
    Actually, let me think more carefully.
    
    For a subproblem with n rows, m_block cols of A, p_block cols of B:
    - If m_block=2, p_block=2: can use 2x2x2 Strassen for i=0,1 (rank 7) + schoolbook for i=2 (rank 2) = rank 9
    - If m_block=2, p_block=1: schoolbook rank 3*2*1 = 6
    - If m_block=1, p_block=2: schoolbook rank 3*1*2 = 6
    - If m_block=1, p_block=1: schoolbook rank 3*1*1 = 3
    
    Split m=5 as [2,2,1], p=6 as [2,2,2]:
    - (2,2): rank 9, three times = 27
    - (2,2): rank 9, three times = 27
    - (1,2): rank 6, three times = 18
    Total: 27+27+18 = 72
    
    Alternative split m=5 as [2,3], p=6 as [3,3]:
    - (2,3): need to handle 3x2x3. Split p=3 into [2,1].
      - 3x2x2: rank 9
      - 3x2x1: rank 6
      Total for 3x2x3: 15, two times = 30
    - (3,3): need to handle 3x3x3. Split m=3 into [2,1], p=3 into [2,1].
      - 3x2x2: rank 9
      - 3x2x1: rank 6
      - 3x1x2: rank 6
      - 3x1x1: rank 3
      Total for 3x3x3: 24, two times = 48
    Total: 30+48 = 78. Worse.
    
    Let me try m=5 as [2,2,1], p=6 as [2,2,2]: total 72.
    """
    U222, V222, W222 = _strassen_2x2x2()
    r222 = U222.shape[1]
    
    # Split m into [2,2,1] and p into [2,2,2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Precompute schoolbook blocks
    def get_schoolbook(n, mb, pb):
        r = n * mb * pb
        U_s = np.zeros((n * mb, r), dtype=float)
        V_s = np.zeros((mb * pb, r), dtype=float)
        W_s = np.zeros((pb * n, r), dtype=float)
        c = 0
        for i in range(n):
            for j in range(mb):
                for k in range(pb):
                    U_s[i * mb + j, c] = 1
                    V_s[j * pb + k, c] = 1
                    W_s[k * n + i, c] = 1
                    c += 1
        return U_s, V_s, W_s, r
    
    # Build full matrices
    total_rank = 0
    # First pass: compute total rank
    for mi, mb in enumerate(m_splits):
        for pi, pb in enumerate(p_splits):
            if mb == 2 and pb == 2:
                total_rank += r222 + 2  # Strassen for i=0,1 + schoolbook for i=2
            else:
                total_rank += n * mb * pb
    
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    offset = 0
    for mi, mb in enumerate(m_splits):
        m_off = sum(m_splits[:mi])
        for pi, pb in enumerate(p_splits):
            p_off = sum(p_splits[:pi])
            if mb == 2 and pb == 2:
                # Use Strassen for i=0,1
                for i in range(2):
                    for j in range(2):
                        for c in range(r222):
                            U_full[i * m + (m_off + j), offset + c] = U222[i * 2 + j, c]
                for j in range(2):
                    for k in range(2):
                        for c in range(r222):
                            V_full[(m_off + j) * p + (p_off + k), offset + c] = V222[j * 2 + k, c]
                for k in range(2):
                    for i in range(2):
                        for c in range(r222):
                            W_full[(p_off + k) * n + i, offset + c] = W222[k * 2 + i, c]
                offset += r222
                # Schoolbook for i=2
                for j in range(2):
                    for k in range(2):
                        U_full[2 * m + (m_off + j), offset] = 1
                        V_full[(m_off + j) * p + (p_off + k), offset] = 1
                        W_full[(p_off + k) * n + 2, offset] = 1
                        offset += 1
            else:
                U_s, V_s, W_s, r_s = get_schoolbook(n, mb, pb)
                for i in range(n):
                    for j in range(mb):
                        for c in range(r_s):
                            U_full[i * m + (m_off + j), offset + c] = U_s[i * mb + j, c]
                for j in range(mb):
                    for k in range(pb):
                        for c in range(r_s):
                            V_full[(m_off + j) * p + (p_off + k), offset + c] = V_s[j * pb + k, c]
                for k in range(pb):
                    for i in range(n):
                        for c in range(r_s):
                            W_full[(p_off + k) * n + i, offset + c] = W_s[k * n + i, c]
                offset += r_s
    
    return U_full, V_full, W_full, total_rank

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    try:
        U, V, W, rank = _build_decomposition(n, m, p)
        if _check_exact(U, V, W, n, m, p, tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Fallback schoolbook
    U, V, W, rank = _schoolbook_block(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
