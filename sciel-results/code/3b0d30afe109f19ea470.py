"""
Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
  1. Use Strassen's 7-term decomposition for 2x2 sub-blocks.
  2. Partition the 3x8 matrix B into overlapping 2-row blocks to enable
     Strassen on all 2x2 sub-blocks of A, covering all three rows of A.
  3. The 2x3 A matrix has rows a=0,1,2. We split the 8 columns of B into
     four 2-column blocks. For each block, we can use Strassen on the 2x2
     sub-matrix formed by any two rows of A.
  
  A better approach: use the known rank bound. The bilinear complexity of
  m_{2,3,8} is at most 44 (from Strassen + schoolbook on remaining).
  
  Let's think more carefully. The tensor is T[i,a,k] = delta (scalar mult).
  We need rank-r decomposition: sum_r U[i,a,r] * V[a,k,r] * W[k,i,r] = 1 for all i,a,k.
  
  Key insight: The 2x3*3x8 problem can be decomposed as:
    C = A * B where A is 2x3, B is 3x8.
    C[i,k] = sum_a A[i,a] * B[a,k]
    
  We can write this as:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
  Group terms: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + A[i,2]*B[2,k]
  
  The first part is a 2x2 * 2x8 multiplication (rows 0,1 of A and B).
  The second part is a 2x1 * 1x8 multiplication.
  
  For the 2x2*2x8 part: split B's 8 columns into 4 blocks of 2.
  Each 2x2*2x2 block uses Strassen (7 terms). Total: 4*7 = 28 terms.
  The 2x1*1x8 part: 2*8 = 16 terms.
  Total: 28 + 16 = 44 terms.
  
  Can we do better? Let's try to combine the last row more cleverly.
  
  Alternative: Use the fact that we have 3 rows. Consider splitting into
  overlapping Strassen blocks.
  
  Actually, let's try a different partition. Use Strassen on (rows 0,1) and
  also on (rows 1,2), then combine.
  
  Let's define:
    S1 = A[i,0]*B[0,k] + A[i,1]*B[1,k]  (2x2*2x8, uses Strassen: 28 terms)
    S2 = A[i,1]*B[1,k] + A[i,2]*B[2,k]  (2x2*2x8, uses Strassen: 28 terms)
    
  But we can't just add S1 + S2 because that double-counts A[i,1]*B[1,k].
  
  C[i,k] = S1[i,k] + A[i,2]*B[2,k]  -- this is the original decomposition.
  
  Or: C[i,k] = S2[i,k] + A[i,0]*B[0,k] -- same thing, just different split.
  
  Let's try: C[i,k] = (S1[i,k] + S2[i,k] - A[i,1]*B[1,k]) / ... no, this doesn't work directly.
  
  Actually, let's try a more clever combination. Consider:
    M1 = A[i,0]*B[0,k] + A[i,1]*B[1,k]  (Strassen on rows 0,1: 28 terms)
    M2 = A[i,1]*B[1,k] + A[i,2]*B[2,k]  (Strassen on rows 1,2: 28 terms)
    
  We want C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
                 = M1[i,k] + A[i,2]*B[2,k]
                 = M2[i,k] + A[i,0]*B[0,k]
  
  These are equivalent. No savings.
  
  Let's try yet another approach. What if we use a rank-7 decomposition for
  the 2x2x2 tensor and then extend it?
  
  The 2x3*3x8 tensor has 2*3*8 = 48 non-zero entries (all 1s).
  The rank is at most 48 (schoolbook).
  With Strassen on 2x2 blocks: 44 as computed.
  
  Can we get below 44? The known lower bound for m_{2,3,8} is not well-established,
  but let's try to find a better decomposition.
  
  Idea: Use Strassen's trick more globally. The 2x3*3x8 multiplication can be
  viewed as computing 16 scalar products. Each scalar product C[i,k] = sum_a A[i,a]*B[a,k].
  
  Let's try to use a rank-6 or rank-7 decomposition for a 2x3*3x3 sub-block,
  then extend to 8 columns.
  
  Actually, the rank of m_{2,3,3} is known to be 11 (from Bini et al.).
  But we need m_{2,3,8}.
  
  Let me try a direct approach: use the 44-term decomposition and see if we
  can merge terms or find a sparser representation.
  
  For now, let's implement the 44-term solution and validate it.
"""

import time
import sys

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        from fractions import Fraction
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    # For multiplying (2x2) * (2x2) -> (2x2)
    # U indices: (i,a) where i in {0,1}, a in {0,1} -> 4 entries
    # V indices: (a,k) where a in {0,1}, k in {0,1} -> 4 entries  
    # W indices: (k,i) where k in {0,1}, i in {0,1} -> 4 entries
    # Note: W is indexed as W[k*n+i] with n=2, so W[k*2+i]
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3 - M4 + M6
    # C10 = -M1 + M2 + M5 - M7
    # C11 = M1 - M3 + M4 + M5
    
    # U[r][i*2+a] for r in 0..6, i in 0..1, a in 0..1
    S_U = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1]  # M7: A01-A11
    ]
    # V[r][a*2+k] for r in 0..6, a in 0..1, k in 0..1
    S_V = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B10-B11
        [-1, 0, 1, 0], # M4: B01-B10
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B10
        [0, 0, 1, 1]   # M7: B01+B11
    ]
    # W[r][k*2+i] for r in 0..6, k in 0..1, i in 0..1
    S_W = [
        [1, 0, 0, 1],  # M1: C00 + C10
        [0, 1, 0, -1], # M2: C01 - C11
        [0, 0, 1, 1],  # M3: C10 + C11
        [1, 1, 0, 0],  # M4: C00 + C01
        [-1, 0, 1, 0], # M5: -C00 + C10
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0]   # M7: C00
    ]

    # Verify Strassen decomposition internally
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build 44-term candidate
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, :] -> 2x2 * 2x8
    # Split B's 8 columns into 4 blocks of 2
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            # U: i in {0,1}, a in {0,1} (local a = global a for this block)
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] += u_val
            
            # V: a in {0,1}, k in {k0, k1}
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] += v_val
            
            # W: k in {k0, k1}, i in {0,1}
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] += w_val
            
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A[:, 2] * B[2, :] -> 2x1 * 1x8
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
