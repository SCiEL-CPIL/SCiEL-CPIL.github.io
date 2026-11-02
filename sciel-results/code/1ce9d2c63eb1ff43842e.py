"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time as _time

def _build_strassen_2x2x2(p, t, U, V, W, base, n=2):
    """Add Strassen rank-7 terms for 2x2x2 block t into arrays."""
    v_b00 = 0 * p + 2 * t
    v_b01 = 0 * p + 2 * t + 1
    v_b10 = 1 * p + 2 * t
    v_b11 = 1 * p + 2 * t + 1
    
    w_c00 = (2 * t) * n + 0
    w_c01 = (2 * t + 1) * n + 0
    w_c10 = (2 * t) * n + 1
    w_c11 = (2 * t + 1) * n + 1
    
    for s in range(7):
        c = base + s
        if s == 0:
            U[0][c] += 1; U[3][c] += 1
            V[v_b00][c] += 1; V[v_b11][c] += 1
            W[w_c00][c] += 1; W[w_c11][c] += 1
        elif s == 1:
            U[2][c] += 1; U[3][c] += 1
            V[v_b00][c] += 1
            W[w_c10][c] += 1; W[w_c11][c] += -1
        elif s == 2:
            U[0][c] += 1
            V[v_b01][c] += 1; V[v_b11][c] += -1
            W[w_c01][c] += 1; W[w_c11][c] += 1
        elif s == 3:
            U[3][c] += 1
            V[v_b10][c] += 1; V[v_b00][c] += -1
            W[w_c00][c] += 1; W[w_c10][c] += 1
        elif s == 4:
            U[0][c] += 1; U[1][c] += 1
            V[v_b11][c] += 1
            W[w_c00][c] += -1; W[w_c01][c] += 1
        elif s == 5:
            U[2][c] += 1; U[0][c] += -1
            V[v_b00][c] += 1; V[v_b01][c] += 1
            W[w_c11][c] += 1
        elif s == 6:
            U[1][c] += 1; U[3][c] += -1
            V[v_b10][c] += 1; V[v_b11][c] += 1
            W[w_c00][c] += 1

def _build_direct_rank24(n, m, p):
    """
    Direct rank-24 construction for 2x2x12.
    For each output column j and each input row i, create one term.
    Term (i,j): U[i*m+k] = delta_{k,i} (picks A[i,k]... wait)
    
    Actually: C[j,i] = sum_k A[i,k] * B[k,j]
    W index: j*n + i (trace convention W[k*n+i, r] where k=j, i=row)
    U index: i*m + k where i=0,1 (row of A), k=0,1 (col of A)
    V index: k*p + j where k=0,1 (row of B), j=0..11 (col of B)
    
    For term r corresponding to (i, j):
      U[i*m + k, r] = 1 if k == i else 0  -- wait, that doesn't work directly
    """
    # C[j,i] = sum_k A[i,k] * B[k,j]
    # We need: sum_r U[i*m+k, r] * V[k'*p+j', r] * W[j'*n+i', r] = delta conditions
    # 
    # Simple rank-24: for each (i, j) pair, one term
    # Term r = i*12 + j:
    #   U[i*m + i, r] = 1  (only A[i,i]... no that's wrong)
    #
    # Let me think again. C[j,i] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # 
    # Rank-24 construction:
    # For each i in {0,1} and j in {0..11}:
    #   Term r = i*12 + j contributes to C[j,i]
    #   We need: sum_k U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = A[i,k]*B[k,j] summed over k
    #   
    # Simplest: split into 24 terms, each handling one (i,k,j) triple:
    #   But that's 2*2*12 = 48 terms.
    #
    # Better: for each i, j: one term where U picks row i of A, V picks column j of B
    #   Term r = i*12+j:
    #     U[i*m + 0, r] = 1, U[i*m + 1, r] = 1  -- no, we need separate k
    #
    # Actually the cleanest rank-24:
    #   For i in {0,1}, j in {0..11}, k in {0,1}: that's 48.
    #   
    # The rank-24 works by: for each (i,j), use one term with:
    #   U[i*m+0, r] = 1, U[i*m+1, r] = 1 -- NO
    #
    # Let me reconsider. The tensor is:
    #   T[i*m+k, k'*p+j, j'*n+i'] = 1 iff i==i' and k==k' and j==j'
    #   Wait no. T[a, b, c] where a=i*m+k, b=k'*p+j, c=j'*n+i'
    #   T = 1 iff i==i' and k==k' and j==j'
    #
    # So T[i*m+k, k*p+j, j*n+i] = 1 (when all indices match)
    #
    # Rank-24: for each (i,j), one term:
    #   r = i*12 + j
    #   U[i*m+0, r] = 1, U[i*m+1, r] = 1 -- this gives sum over k
    #   V[0*p+j, r] = 1, V[1*p+j, r] = 1 -- this gives sum over k'
    #   W[j*n+i, r] = 1
    #   Product: (1+1)*(1+1)*1 = 4 for matching, but we need 1.
    #   And cross terms: U[i*m+0]*V[1*p+j] = 1*1 = 1 (wrong, k!=k')
    #
    # This doesn't work directly. The rank of this tensor is actually 24.
    # 
    # Correct rank-24: for each (i, k, j) that's 48. But we can do better.
    # 
    # Key insight: fix i. Then C[j,i] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # This is an outer product structure: for fixed i, it's rank 2 in (j) for each k.
    # Total rank = sum over i of rank of (A[i,:] x B[:,j]) = 2 * 2 = 4 per i? No.
    #
    # For fixed i: the sub-tensor has T[i*m+k, k*p+j, j*n+i] = 1 iff k matches
    # This is: sum_k A[i,k] * B[k,j] -> C[j,i]
    # The rank of this 2x2x12 subproblem (for fixed i) is 12:
    #   For each j: term picks A[i,:] dot B[:,j]
    #   U[i*m+0, r_j] = 1, U[i*m+1, r_j] = 1 -- NO, still cross terms
    #
    # I think the minimal rank for 2x2x12 is actually 24, achieved by:
    # For each j in 0..11, and each "mode" in 0..1 (representing the 2x2 matmul rank-2):
    #   Actually 2x2 matmul has rank 7 (Strassen) or 4 (schoolbook).
    #   So 6 blocks * 4 = 24 (schoolbook) or 6*7=42 (Strassen).
    #
    # Schoolbook 2x2: rank 4 per 2x2 block. 6 blocks * 4 = 24.
    # That's the direct rank-24!
    
    rank = 24
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # 6 blocks of 2x2x2, schoolbook rank 4 each
    for t in range(6):
        base = t * 4
        col_start = 2 * t
        # Schoolbook 2x2x2:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        r = base + 0
        U[0*2+0][r] = 1.0  # A[0,0]
        V[0*p + col_start][r] = 1.0  # B[0, col_start]
        W[col_start*n + 0][r] = 1.0  # C[col_start, 0]
        
        # Term 1: A[0,1]*B[1,0] -> C[0,0]
        r = base + 1
        U[0*2+1][r] = 1.0  # A[0,1]
        V[1*p + col_start][r] = 1.0  # B[1, col_start]
        W[col_start*n + 0][r] = 1.0  # C[col_start, 0]
        
        # Term 2: A[0,0]*B[0,1] -> C[0,1]
        r = base + 2
        U[0*2+0][r] = 1.0  # A[0,0]
        V[0*p + col_start+1][r] = 1.0  # B[0, col_start+1]
        W[(col_start+1)*n + 0][r] = 1.0  # C[col_start+1, 0]
        
        # Term 3: A[0,1]*B[1,1] -> C[0,1]
        r = base + 3
        U[0*2+1][r] = 1.0  # A[0,1]
        V[1*p + col_start+1][r] = 1.0  # B[1, col_start+1]
        W[(col_start+1)*n + 0][r] = 1.0  # C[col_start+1, 0]
        
        # Term 4: A[1,0]*B[0,0] -> C[1,0]
        r = base + 4
        U[1*2+0][r] = 1.0  # A[1,0]
        V[0*p + col_start][r] = 1.0  # B[0, col_start]
        W[col_start*n + 1][r] = 1.0  # C[col_start, 1]
        
        # Term 5: A[1,1]*B[1,0] -> C[1,0]
        r = base + 5
        U[1*2+1][r] = 1.0  # A[1,1]
        V[1*p + col_start][r] = 1.0  # B[1, col_start]
        W[col_start*n + 1][r] = 1.0  # C[col_start, 1]
        
        # Term 6: A[1,0]*B[0,1] -> C[1,1]
        r = base + 6
        U[1*2+0][r] = 1.0  # A[1,0]
        V[0*p + col_start+1][r] = 1.0  # B[0, col_start+1]
        W[(col_start+1)*n + 1][r] = 1.0  # C[col_start+1, 1]
        
        # Term 7: A[1,1]*B[1,1] -> C[1,1]
        r = base + 7
        U[1*2+1][r] = 1.0  # A[1,1]
        V[1*p + col_start+1][r] = 1.0  # B[1, col_start+1]
        W[(col_start+1)*n + 1][r] = 1.0  # C[col_start+1, 1]
    
    # Wait, that's 8 terms per block = 48 total. That's schoolbook rank 48.
    # I made an error. Schoolbook 2x2x2 is rank 4, not 8.
    # The 4 terms are the 4 products, but each output needs 2 terms.
    # Actually schoolbook IS rank 8 for 2x2x2 (4 outputs x 2 terms each).
    # No wait: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] -- that's 2 rank-1 terms
    # So total rank = 4*2 = 8? No, that's not right either.
    #
    # The matrix multiplication tensor for 2x2x2 has rank 7 (Strassen) or 8 (schoolbook).
    # Schoolbook: each of the 4 output entries is a sum of 2 products, so 8 rank-1 terms.
    # But actually, the "rank" in tensor decomposition counts independent rank-1 terms.
    # Schoolbook uses 8, Strassen uses 7.
    
    # So my construction above gives rank 8*6 = 48, which is the schoolbook rank.
    # That's worse than Strassen's 42.
    
    # Let me just return the Strassen construction instead.
    return None

def _build_strassen_2x2x4(p, t, U, V, W, base, n=2):
    """
    Build rank-10 decomposition for 2x2x4 block.
    2x2 matrix times 2x4 matrix -> 2x4 result.
    Split into two 2x2 blocks and use shared Strassen terms.
    
    Actually, let me use the known rank-10 for 2x2x4:
    A(2x2) * B(2x4) = C(2x4)
    Write B = [B1 | B2] where B1, B2 are 2x2.
    C = [A*B1 | A*B2]
    
    Using Strassen on both:
    M1 = A*B1 (7 terms), M2 = A*B2 (7 terms) = 14 terms.
    But we can share: M1+M2 = A*(B1+B2) -- 7 more terms.
    Then M1 = (M1+M2) - M2, M2 = (M1+M2) - M1.
    Total: 7 + 7 + 7 = 21? No, that's worse.
    
    Better: use the fact that we compute:
    S1 = (A00+A11)*(B1_00+B2_00+B1_11+B2_11) -- shared
    ... this gets complicated.
    
    Let me just use 2 separate Strassen blocks: rank 14 for 2x2x4.
    """
    # Just use two Strassen 2x2x2 blocks
    _build_strassen_2x2x2(p, 2*t, U, V, W, base, n)
    _build_strassen_2x2x2(p, 2*t+1, U, V, W, base+7, n)

def construct():
    """
    Rank-42 exact tensor decomposition via Strassen on six 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    deadline = _time.time() + 115
    
    # Strassen rank-42: six 2x2x2 blocks
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        _build_strassen_2x2x2(p, t, U, V, W, t * 7, n)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
