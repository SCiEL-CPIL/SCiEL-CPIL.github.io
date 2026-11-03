"""Matrix multiplication 2x3x10 - rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    r = len(U[0])
    if len(V[0]) != r or len(W[0]) != r:
        return False
    
    # Check target entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for col in range(r):
                    total += U[a][col] * V[b][col] * W[c][col]
                if abs(total - 1) > 1e-10:
                    return False
                
    # Check non-target entries (should be 0)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                for c_test in range(p * n):
                    if c_test != k * n + i:
                        total = 0
                        for col in range(r):
                            total += U[a][col] * V[b][col] * W[c_test][col]
                        if abs(total) > 1e-10:
                            return False
    return True

def _construct_schoolbook(n, m, p):
    """Construct the standard schoolbook decomposition of rank n*m*p."""
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

def _construct_strassen_55(n, m, p):
    """
    Construct a rank-55 decomposition using Strassen's algorithm on the 2x2 block
    structure.
    
    A is 2x3: A = [[A0, A1, A2], [B0, B1, B2]] where A0,B0 are 1x1, etc.
    Actually, let's think in terms of the standard tensor indices.
    
    For multiplying A (n x m) by B (m x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    With n=2, m=3, p=10:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    We can group the j-index into blocks. Let's use a 2x2 Strassen-like approach.
    
    Strategy: 
    - Use schoolbook for the first column (j=0): 2*10 = 20 multiplications
    - Use Strassen for the remaining 2x2 block (j=1,2): 
      We have A_sub = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]] (2x2)
      B_sub = [[B[1,k], B[2,k]] for k in 0..9] (2x10)
      
      Strassen for 2x2 * 2x10:
      M1 = (A00 + A11) * (B00 + B11)
      M2 = (A10 + A11) * B00
      M3 = A00 * B10
      M4 = A11 * B11
      M5 = (A00 + A01) * B11
      M6 = (A10 - A00) * (B00 + B01)
      M7 = (A01 - A11) * (B10 + B11)
      
      Where A00=A[0,1], A01=A[0,2], A10=A[1,1], A11=A[1,2]
      B00=B[1,:], B01=B[2,:], B10=B[1,:], B11=B[2,:]
      Wait, let me be more careful.
      
      For 2x2 * 2x10:
      A_sub = [[a00, a01], [a10, a11]]
      B_sub = [[b00, b01, ..., b09], [b10, b11, ..., b19]]
      
      C_sub[0,k] = a00*b0k + a01*b1k
      C_sub[1,k] = a10*b0k + a11*b1k
      
      Strassen:
      M1 = (a00+a11)*(b0k+b1k)
      M2 = (a10+a11)*b0k
      M3 = a00*b1k
      M4 = a11*b1k
      M5 = (a00+a01)*b1k
      M6 = (a10-a00)*(b0k+b0k)  -- wait, this doesn't work for rectangular
      M7 = (a01-a11)*(b1k+b1k)
      
      Actually for 2x2 * 2xp, Strassen still uses 7 multiplications per "block" of p columns.
      So 7*10 = 70 for the 2x2*2x10 part. That's worse than schoolbook 2*2*10=40.
      
      Hmm, let me reconsider. The inspiration says rank 55 = 20 (schoolbook for 2x1x10) + 35 (something for 2x2x10).
      
      Wait, 2x2x10 with Strassen: 7*10 = 70. That's not 35.
      
      Let me re-read: "Part 1 (rank 20): schoolbook for 2×1 × 1×10" and the rest.
      
      Actually, maybe the decomposition is:
      - Split m=3 into 1+2
      - For the 1x2 part (j=0): schoolbook gives 2*1*10 = 20
      - For the 2x2 part (j=1,2): use a rank-35 decomposition for 2x2x10
      
      For 2x2x10, the schoolbook rank is 40. Can we do 35?
      Using Strassen on 2x2x10: we pad to 2x2x16 (rank 7*16=112) or use block Strassen.
      Actually, for 2x2xp, Strassen gives 7p = 70, which is worse.
      
      Alternative: Use the fact that 2x2x10 tensor might have rank 35 by some other means.
      
      Let me try a different approach. Maybe we can use a rank-7 decomposition for 2x2x2 and then apply it to 5 blocks of 2 columns.
      
      2x2x10 = 5 blocks of 2x2x2. Each 2x2x2 has rank 7 (Strassen). Total: 5*7 = 35.
      
      Yes! That makes sense.
      
      So the full decomposition:
      - j=0 (1 column): schoolbook, rank 2*1*10 = 20
      - j=1,2 (2 columns): 5 blocks of 2x2x2, each using Strassen rank 7, total 35
      - Total rank: 20 + 35 = 55
    """
    n, m, p = 2, 3, 10
    
    # Part 1: Schoolbook for j=0 (A[:,0] * B[0,:])
    # This contributes to C[i,k] = A[i,0]*B[0,k]
    # U indices: i*m + j = i*3 + 0 = 0, 3
    # V indices: j*p + k = 0*10 + k = k
    # W indices: k*n + i = k*2 + i
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    # Schoolbook for j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 20
    
    # Part 2: Strassen for j=1,2 in 5 blocks of 2 columns each
    # For each block of 2 columns (k, k+1), we compute:
    #   C[0,k]   += A[0,1]*B[1,k]   + A[0,2]*B[2,k]
    #   C[0,k+1] += A[0,1]*B[1,k+1] + A[0,2]*B[2,k+1]
    #   C[1,k]   += A[1,1]*B[1,k]   + A[1,2]*B[2,k]
    #   C[1,k+1] += A[1,1]*B[1,k+1] + A[1,2]*B[2,k+1]
    
    # Strassen for 2x2x2:
    # a00=A[0,1], a01=A[0,2], a10=A[1,1], a11=A[1,2]
    # b00=B[1,k],  b01=B[1,k+1], b10=B[2,k], b11=B[2,k+1]
    
    # M1 = (a00+a11)*(b00+b10)
    # M2 = (a10+a11)*b00
    # M3 = a00*b10
    # M4 = a11*b11
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b01+b00)
    # M7 = (a01-a11)*(b11+b10)
    
    # C00 = M1 - M4 + M3
    # C01 = M5 - M1 + M7  -- wait, let me recheck
    # Actually:
    # C00 = a00*b00 + a01*b10 = (a00+a11)(b00+b10) - a11*b10 - a01*b10 + a00*b00
    #     = M1 - a11*b10 - a01*b10 + a00*b00
    # Hmm, let me use the standard Strassen formulas.
    
    # Standard Strassen for C = A*B where A,B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M5
    # C10 = M3 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify C00:
    # M1 + M4 - M5 + M7
    # = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + A01*B10 + (A00*B11 - A00*B11) + (A11*B00 - A11*B00) + (A11*B11 - A11*B11) + (-A01*B11 + A01*B11) + (A11*B01 - A11*B10)
    # Wait, that gives A00*B00 + A01*B10 + A11*B01 - A11*B10
    # But we want C00 = A00*B00 + A01*B10
    # So there's an extra term A11*B01 - A11*B10. That's wrong.
    
    # Let me use the correct Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00+B01)
    # M3 = (A00-A10)(B10+B11)
    # M4 = A11(B11-B00)
    # M5 = A00(B01-B10)
    # M6 = (A01+A10)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 - M4 - M7
    # C10 = M3 + M5 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # Verify C00:
    # M1 + M4 - M5 + M7
    # = (A00+A11)(B00+B11) + A11(B11-B00) - A00(B01-B10) + (A01-A11)(B11+B01)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B11 - A11*B00 - A00*B01 + A00*B10 + A01*B11 + A01*B01 - A11*B11 - A11*B01
    # = A00*B00 + A00*B11 + A11*B11 - A00*B01 + A00*B10 + A01*B11 + A01*B01 - A11*B01
    # Hmm, still messy. Let me just use the well-known Strassen formulas.
    
    # The standard Strassen algorithm:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Verify C00 = M1 - M4 + M5 - M7:
    # M1 - M4 + M5 - M7
    # = (A00+A11)(B00+B11) - A11(B01-B00) + (A00+A01)B11 - (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B00 + A00*B11 + A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    # = A00*B00 + A00*B11 + 2*A11*B00 + 2*A11*B11 - A11*B01 + A00*B11 + A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    # This is getting complicated. Let me just trust the standard formulas and verify numerically later.
    
    # For now, I'll use a simpler approach: just use schoolbook for the 2x2x10 part as well,
    # giving rank 20 + 40 = 60, which is the same as full schoolbook.
    # 
    # To get rank 55, I need to find a rank-35 decomposition for 2x2x10.
    # 
    # Alternative: Use the fact that we can share some computations.
    # 
    # Let me try a different block structure. What if we use Strassen on a 2x2x4 block?
    # 2x2x4 has rank 7*4 = 28 with Strassen, vs schoolbook 16. Still worse.
    # 
    # I think the key insight from the inspiration is that there's a specific rank-35 
    # decomposition for 2x2x10 that I need to construct.
    
    # For now, let me just return the schoolbook solution and add a search for rank 55.
    # The search will try to find a valid rank-55 decomposition.
    
    # Fallback: return schoolbook
    return None

def _search_rank_55(n, m, p, deadline):
    """
    Search for a rank-55 decomposition using a structured approach.
    
    The idea: split m=3 into j=0 and j=1,2.
    - j=0: schoolbook, rank 20
    - j=1,2: need rank 35 for 2x2x10
    
    For 2x2x10, we can try to find a rank-35 decomposition by exploiting
    the tensor structure. One approach: use 5 independent 2x2x2 Strassen blocks
    but with shared linear combinations.
    """
    n, m, p = 2, 3, 10
    target_rank = 55
    
    # Initialize with schoolbook for j=0
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # For j=1,2, we need to fill columns 20..54
    # The tensor entries for j=1,2 are:
    # C[i,k] += A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # We'll use a random search to find coefficients that satisfy the constraints.
    # This is a system of linear equations in the unknowns (the entries of U, V, W).
    
    # Number of constraints:
    # - Target entries: 2*2*10 = 40 (should equal 1)
    # - Non-target entries: for each (i,k) in the 2x2x10 sub-tensor, the output
    #   should only affect C[i,k], not other C[i',k']
    #   Total output entries: 2*10 = 20, each should be affected only by its own (i,k)
    #   So 40 target + (20*19) non-target = 40 + 380 = 420 constraints
    #   But some are redundant.
    
    # Number of unknowns: 55 * (6 + 30 + 20) = 55 * 56 = 3080
    # So the system is underdetermined, which is good.
    
    # Let me set up the linear system.
    # For each target entry (i, j, k) with j in {1,2}:
    #   sum_{col=20}^{54} U[i*3+j][col] * V[j*10+k][col] * W[k*2+i][col] = 1
    
    # For each non-target entry:
    #   sum_{col=20}^{54} U[i*3+j][col] * V[j*10+k][col] * W[k'*2+i'][col] = 0
    #   for (i',k') != (i,k)
    
    # This is a system of polynomial equations (quadratic in the unknowns if we fix
    # one matrix, but bilinear if we consider them as separate variables).
    
    # Let me try a greedy approach: fix U and W, solve for V, then iterate.
    
    # Initialize U and W randomly
    import random
    random.seed(42)
    
    for col in range(20, target_rank):
        for i in range(n):
            for j in [1, 2]:
                U[i * m + j][col] = random.choice([-1, 0, 1])
        for k in range(p):
            for i in range(n):
                W[k * n + i][col] = random.choice([-1, 0, 1])
    
    # Now solve for V given U and W
    # For each (j, k) in {1,2} x {0..9}:
    #   sum_{col=20}^{54} U[i*3+j][col] * V[j*10+k][col] * W[k*2+i][col] = 1 for all i
    #   and = 0 for all (i',k') != (i,k)
    
    # This is a system of linear equations in V[j*10+k][col] for col=20..54
    # But it's coupled across j and k.
    
    # Let me simplify: for a fixed j, the equations involve V[j*10+k][col] for all k.
    # For fixed j and k, the equation for output (i,k) is:
    #   sum_{col} U[i*3+j][col] * V[j*10+k][col] * W[k*2+i][col] = 1
    # And for output (i',k') with k' != k:
    #   sum_{col} U[i*3+j][col] * V[j*10+k][col] * W[k'*2+i'][col] = 0
    
    # So for fixed j, we have a system where V[j*10+k][col] appears in equations
    # for all outputs (i',k') where the W factor is W[k*2+i'][col].
    
    # This is complex. Let me just try to verify if a simple structured solution works.
    
    # Alternative: use the fact that 2x2x10 can be decomposed as 5 copies of 2x2x2
    # with rank 7 each, but with shared structure.
    
    # For block b (b=0..4), columns 2b and 2b+1:
    # A_sub = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
    # B_sub = [[B[1,2b], B[1,2b+1]], [B[2,2b], B[2,2b+1]]]
    
    # Strassen for this block:
    # M1_b = (A[0,1]+A[1,2]) * (B[1,2b]+B[2,2b])
    # M2_b = (A[1,1]+A[1,2]) * B[1,2b]
    # M3_b = A[0,1] * (B[2,2b]-B[2,2b+1])
    # M4_b = A[1,2] * (B[1,2b+1]-B[1,2b])
    # M5_b = (A[0,1]+A[0,2]) * B[2,2b+1]
    # M6_b = (A[1,1]-A[0,1]) * (B[1,2b]+B[1,2b+1])
    # M7_b = (A[0,2]-A[1,2]) * (B[2,2b]+B[2,2b+1])
    
    # C[0,2b]   = M1_b - M4_b + M5_b - M7_b
    # C[0,2b+1] = M2_b + M4_b
    # C[1,2b]   = M3_b + M5_b
    # C[1,2b+1] = M1_b - M2_b + M3_b + M6_b
    
    # Let me verify C[0,2b] = M1_b - M4_b + M5_b - M7_b:
    # M1_b - M4_b + M5_b - M7_b
    # = (A00+A11)(B00+B11) - A11(B01-B00) + (A00+A01)B11 - (A01-A11)(B10+B11)
    # where A00=A[0,1], A01=A[0,2], A10=A[1,1], A11=A[1,2]
    #       B00=B[1,2b], B01=B[1,2b+1], B10=B[2,2b], B11=B[2,2b+1]
    
    # = (A00+A11)(B00+B11) - A11*B01 + A11*B00 + A00*B11 + A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B00 + A00*B11 + A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    # = A00*B00 + 2*A00*B11 + 2*A11*B00 + 2*A11*B11 - A11*B01 - A01*B10 + A11*B10
    # This doesn't simplify to A00*B00 + A01*B10.
    
    # I think I'm using the wrong Strassen formulas. Let me look up the correct ones.
    # 
    # The correct Strassen formulas are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify C00 again:
    # M1 - M4 + M5 - M7
    # = (A00+A11)(B00+B11) - A11(B01-B00) + (A00+A01)B11 - (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B00 + A00*B11 + A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    # = A00*B00 + A00*B11 + 2*A11*B00 + 2*A11*B11 - A11*B01 + A00*B11 + A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    # = A00*B00 + 2*A00*B11 + 2*A11*B00 + 3*A11*B11 - A11*B01 - A01*B10 + A11*B10
    # Still not simplifying correctly. I must be making an algebraic error.
    
    # Let me just trust that Strassen works (it's a well-known algorithm) and 
    # construct the decomposition, then verify it numerically.
    
    # For now, let me just return the schoolbook solution.
    return None

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # 1. Generate schoolbook as the baseline (rank 60)
    schoolbook = _construct_schoolbook(n, m, p)
    best_U, best_V, best_W = schoolbook["U"], schoolbook["V"], schoolbook["W"]
    best_rank = n * m * p
    
    # 2. Try to construct a rank-55 decomposition
    # Based on the inspiration, this should be possible.
    # For now, we return the schoolbook solution.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
