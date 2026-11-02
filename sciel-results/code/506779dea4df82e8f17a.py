"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time as _time
import numpy as _np
from scipy.optimize import minimize as _minimize
import numpy as _np

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
            U[1][c] += 1; U[3][c] -= 1
            V[v_b10][c] += 1; V[v_b11][c] += 1
            W[w_c00][c] += 1

def _build_2x2x4_rank12(p, t, U, V, W, base, n=2):
    """
    Build rank-12 decomposition for 2x2x4 block (columns 4t..4t+3).
    
    For 2x2x4: C[j,i] = sum_k A[i,k]*B[k,j] for j=0..3 (local), i=0,1
    Output: 2x4 matrix.
    
    Rank-12 construction using Strassen-like sharing:
    Split B into B1 (cols 0,1) and B2 (cols 2,3).
    
    Use Strassen for A*B1 (7 terms) and A*B2 (7 terms) = 14, but share 2 terms.
    
    Actually, known rank of 2x2x4 is 12. Construction:
    
    Let's use the following 12 terms:
    Row 0 of C (4 entries):
      C[0,0] = A00*B00 + A01*B10
      C[0,1] = A00*B01 + A01*B11
      C[0,2] = A00*B02 + A01*B12
      C[0,3] = A00*B03 + A01*B13
    
    Row 1 of C (4 entries):
      C[1,0] = A10*B00 + A11*B10
      C[1,1] = A10*B01 + A11*B11
      C[1,2] = A10*B02 + A11*B12
      C[1,3] = A10*B03 + A11*B13
    
    12-term construction:
    Terms 0-3: For row 0, use 4 terms (one per column, picking A[0,:] dot B[:,j])
      But that's not rank-1... 
    
    Let me use: 6 terms for row 0 (Strassen-like for 2x1x4) + 6 terms for row 1
    2x1x4 rank is 4 (since C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j], and this is rank 4: 
    for each j, one term with U[0]=1,V[0*p+j]=1,W[j]=1 and U[1]=1,V[1*p+j]=1,W[j]=1)
    That's 8 terms for row 0. Similarly 8 for row 1. Total 16. Not good.
    
    Better: use Strassen structure.
    For 2x2x4, here's a rank-12:
    
    M1 = (A00+A10)*(B00+B10) -> helps C[0,0],C[1,0]
    ... this is getting complex. Let me just use 2 Strassen blocks (rank 14) 
    and accept rank 42 total, or find a better construction.
    
    Actually, let me just use the simple schoolbook which is rank 16 per 2x2x4
    (4 columns * 2 rows * 2 terms = 16), giving total 48. That's the baseline.
    
    For now, use Strassen on 2x2x2 subblocks: 2*7=14 per 2x2x4, total 42.
    """
    col_start = 4 * t
    # Two Strassen 2x2x2 blocks
    _build_strassen_2x2x2(p, 2*t, U, V, W, base, n)
    _build_strassen_2x2x2(p, 2*t+1, U, V, W, base+7, n)

def _build_2x2x4_rank12(p, t, U, V, W, base, n=2):
    """
    Build rank-12 decomposition for 2x2x4 block (columns 4t..4t+3).
    
    Uses the known rank-12 construction for 2x2x4 matrix multiplication.
    C[j,i] = sum_k A[i,k]*B[k,j] for j=0..3 (local), i=0,1.
    
    12-term construction:
    For each column pair, use a modified Strassen.
    """
    c0 = 4 * t
    c1 = 4 * t + 1
    c2 = 4 * t + 2
    c3 = 4 * t + 3
    
    # 12 terms for 2x2x4
    # Terms 0-5: handle columns c0, c1 with Strassen-like (6 terms)
    # Terms 6-11: handle columns c2, c3 with Strassen-like (6 terms)
    # 
    # For 2x2x2 Strassen uses 7 terms. For 2x2x2 with only 2 output columns
    # (i.e., 2x2x2 where we only compute C[:,0] and C[:,1]), we can use 6 terms:
    # 
    # C[0,0] = A00*B00 + A01*B10
    # C[1,0] = A10*B00 + A11*B10
    # C[0,1] = A00*B01 + A01*B11
    # C[1,1] = A10*B01 + A11*B11
    #
    # 6-term construction:
    # T0: (A00+A10)*(B00+B10) -> C[0,0]+C[1,0] + cross
    # T1: A00*B00 -> C[0,0]
    # T2: A10*B10 -> C[1,0]
    # T3: (A00+A10)*(B01+B11) -> C[0,1]+C[1,1] + cross
    # T4: A00*B01 -> C[0,1]
    # T5: A10*B11 -> C[1,1]
    #
    # Check: C[0,0] = T0_contribution + T1 = (A00+A10)*B00 + A00*B00 + A01*B10... 
    # This doesn't work cleanly.
    
    # Let me use a verified construction. The rank of 2x2x2 is 7, not 6.
    # So 2x2x4 with two independent 2x2x2 blocks is rank 14.
    # 
    # For rank 12, we need to share 2 terms between the two blocks.
    # 
    # Shared term idea: (A00+A11) appears in both blocks.
    # 
    # Block 1 (cols c0,c1) Strassen:
    # M1 = (A00+A11)*(B[c0,0]+B[c1,0]+B[c0,1]+B[c1,1])
    # M2 = (A10+A11)*B[c0,0]
    # M3 = A00*(B[c1,1]-B[c0,0])
    # M4 = A11*(B[c0,0]-B[c1,1])
    # M5 = (A00+A01)*B[c1,1]
    # M6 = (A01-A00)*(B[c0,0]+B[c1,0])
    # M7 = (A11-A10)*(B[c1,0]+B[c1,1])
    #
    # This is getting too complex. Let me just use rank 14 per block (2 Strassen)
    # giving total rank 42, same as before. No improvement.
    
    # For now, just use two Strassen blocks.
    _build_strassen_2x2x2(p, 2*t, U, V, W, base, n)
    _build_strassen_2x2x2(p, 2*t+1, U, V, W, base+7, n)

def _try_numerical_rank(n, m, p, target_rank, deadline):
    """
    Try to find a rank-target_rank decomposition numerically.
    Returns (U, V, W) as lists or None.
    """
    if _time.time() > deadline:
        return None
    
    # The tensor T[a,b,c] = 1 if a=i*m+k, b=k*p+j, c=j*n+i with i=i',k=k',j=j'
    # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
    
    # This is a 3-tensor of size (n*m) x (m*p) x (p*n) = 4 x 24 x 24
    # Total entries: 4*24*24 = 2304
    # Each rank-1 term has 4+24+24 = 52 parameters
    # For rank R: 52R parameters, 2304 equations
    
    # Use the Strassen solution as initial guess and try to reduce
    # This is very hard numerically. Let's try a different approach.
    
    # Actually, let's just return the Strassen solution for now.
    return None

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
