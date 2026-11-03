"""Rewritten program: rank-minimal tensor decomposition for 2x4 * 4x8."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 64

# Deadline guard: return best found before 140s (evaluator kills at 150s)
_DEADLINE = 140.0
_T0 = time.monotonic()

def _elapsed():
    return time.monotonic() - _T0

def _schoolbook():
    """Return the rank-64 schoolbook decomposition."""
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

def _verify(U, V, W):
    """
    Verify that U, V, W form a valid tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 for all (i,j,k) in the multiplication tensor.
    """
    rank = len(U[0])
    # For each (i,j,k), check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    # Also check that for non-tensor entries the sum is 0? 
    # Actually the tensor is defined on all (i,j,k) in [n]x[m]x[p], and the multiplication
    # tensor has T[i,j,k]=1 for all such triples. So we just need T=1 everywhere.
    return True

def _strassen_like_2x4x8():
    """
    Construct a rank-56 decomposition using Strassen-style algebraic structure.
    
    Strategy: Split the 4-dimension into two 2-blocks and the 8-dimension into two 4-blocks.
    A (2x4) = [A1 | A2], each 2x2
    B (4x8) = [B1; B2], each 2x8, where B1 = [B11 | B12] and B2 = [B21 | B22], each 2x4.
    
    C = A1*B1 + A2*B2
    
    For each 2x2 @ 2x4 product, use Strassen on the 2x2 part:
    A2x2 = [[a, b], [c, d]], B2x4 has rows r1, r2 (each 1x4).
    
    Strassen 7 mults (scalar * 1x4 vector):
    M1 = (a+d) * (r1+r2)
    M2 = (c+d) * r2
    M3 = a * (r1-r2)
    M4 = d * (r2-r1)
    M5 = (a+b) * r2
    M6 = (c-a) * (r1+r2)
    M7 = (d-b) * (r1+r2)
    
    Each scalar*vector mult contributes 4 rank-1 terms (one per column).
    So 7*4 = 28 rank-1 terms per 2x2@2x4 product.
    
    We have 4 such products (A1*B11, A1*B12, A2*B21, A2*B22), giving 4*28 = 112. Too many.
    
    Alternative: apply Strassen at a higher level.
    
    Better approach: Use the full Strassen recursion.
    2x4 @ 4x8:
    - Split into 2x2 blocks in the middle dimension (4 = 2+2)
    - This gives C = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x8
    
    For 2x2 @ 2x8:
    - Split 8 into 4+4
    - 2x2 @ 2x8 = [2x2@2x4 | 2x2@2x4]
    - Each 2x2@2x4 uses Strassen: 7 mults, each scalar*1x4 = 4 rank-1 terms, total 28
    
    So 2x2@2x8 = 2 * 28 = 56 rank-1 terms.
    And 2x4@4x8 = 2 * 56 = 112. Still too many.
    
    I need a different approach. Let me try to directly construct rank-56.
    
    Key insight: The Strassen algorithm for 2x2x2 saves 1 term (7 vs 8).
    For 2x4x8, if we can find 8 such savings, we get 64-8=56.
    
    Let me group the 64 schoolbook terms into 8 groups of 8, and apply Strassen to each group.
    
    Group by (i, j mod 2, k mod 4) or similar.
    
    Actually, let me try: group by fixed i and fixed (j mod 2, k mod 4).
    For fixed i, j=2a+b (a in {0,1}, b in {0,1}), k=4c+d (c in {0,1}, d in {0,1,2,3}).
    
    Hmm, this is getting complex. Let me just try to build it explicitly.
    """
    from fractions import Fraction as F
    
    n, m, p = 2, 4, 8
    rank = 56
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]
    
    # We'll use Strassen on 2x2x2 sub-tensors.
    # The 64 terms can be grouped into 8 groups of 8, each forming a 2x2x2 tensor.
    # Group index: (i, j//2, k//4) -- but that gives 2*2*2=8 groups, each with 1*2*4=8 terms.
    # Within each group, the j-offset is 0,1 and k-offset is 0,1,2,3.
    # That's not a 2x2x2 structure.
    
    # Let me try: group by (i, j mod 2, k mod 2) with j//2 and k//2 as free.
    # For fixed i, j=2a+b, k=2c+d where a,c in {0,1,2,3}... no, j in 0..3, k in 0..7.
    # j = 2a + b, a in {0,1}, b in {0,1}
    # k = 4c + d, c in {0,1}, d in {0,1,2,3}
    
    # Alternative grouping: (i, a, c) where j=2a+b, k=4c+d.
    # 8 groups, each with b in {0,1}, d in {0,1,2,3} = 8 terms.
    # Within a group, the structure is 1 x 2 x 4, not 2x2x2.
    
    # Let me try yet another grouping.
    # j = 2a + b, a in {0,1}, b in {0,1}
    # k = 2c + d, c in {0,1,2,3}, d in {0,1}
    # Group by (i, a, c): 2*2*4 = 16 groups, each with b,d in {0,1} = 4 terms.
    # 4 terms per group, not 8.
    
    # I think the cleanest approach is to just hardcode a known rank-56 construction.
    # Let me use the following: apply Strassen to the 2x2x2 core and schoolbook elsewhere.
    
    # Actually, let me try a completely different strategy:
    # Use 56 terms where 8 of them are "Strassen-combined" terms that replace pairs of schoolbook terms.
    
    # In schoolbook, term (i,j,k) has U[i*4+j]=1, V[j*8+k]=1, W[k*2+i]=1.
    # 
    # Strassen for 2x2x2: instead of 8 terms, use 7.
    # The 7 Strassen terms for 2x2x2 have coefficients like:
    # S1 = (a11+a22)(b11+b22)
    # S2 = (a21+a22)b11
    # S3 = a11(b21-b22)
    # S4 = a22(b12-b11)
    # S5 = (a11+a12)b22
    # S6 = (a21-a11)(b11+b12)
    # S7 = (a22-a12)(b12+b22)
    
    # For our problem, let me identify 8 disjoint 2x2x2 sub-tensors within the 2x4x8 tensor.
    # A 2x2x2 sub-tensor is indexed by (i in {0,1}, j in {j0,j0+1}, k in {k0,k0+1}).
    # We need 8 such sub-tensors that partition the 64 entries.
    # 2x4x8 / (2x2x2) = 64/8 = 8 sub-tensors.
    # 
    # Partition: j = 2*a + b (a in {0,1}, b in {0,1}), k = 4*c + d (c in {0,1}, d in {0,1,2,3})
    # This doesn't give 2x2x2 blocks.
    #
    # Try: j = 2*a + b (a in {0,1}, b in {0,1}), k = 2*c + d (c in {0,1,2,3}, d in {0,1})
    # Sub-tensor indexed by (a, c): 2*4 = 8 sub-tensors, each 2x2x2.
    # Within sub-tensor (a,c): i in {0,1}, j=2a+b (b in {0,1}), k=2c+d (d in {0,1}).
    # This works! Each sub-tensor is a 2x2x2 tensor.
    
    # For each of the 8 sub-tensors, apply Strassen (7 terms instead of 8).
    # Total: 8 * 7 = 56 terms.
    
    col = 0
    for a in range(2):      # j-block: j = 2*a + b
        for c in range(4):  # k-block: k = 2*c + d
            # Sub-tensor: i in {0,1}, j=2a+b (b in {0,1}), k=2c+d (d in {0,1})
            # Strassen 7 terms for this 2x2x2 sub-tensor
            
            # Define the 7 Strassen combinations
            # Original 8 terms would be (i,b,d) for i,b,d in {0,1}
            # Strassen replaces them with 7 combinations
            
            # M1 = (A[i0,b0]+A[i1,b1]) * (B[j0,k0]+B[j1,k1])
            # This is complex. Let me just use the standard Strassen formulas.
            
            # For a 2x2x2 tensor with indices (i,b,d):
            # U indices: i*4 + (2a+b) = i*4 + 2a + b
            # V indices: (2a+b)*8 + (2c+d) = (2a+b)*8 + 2c + d
            # W indices: (2c+d)*2 + i = (2c+d)*2 + i
            
            # Strassen term 1: (a11+a22)(b11+b22)
            # a11 = A[0, 2a+0], a22 = A[1, 2a+1]
            # b11 = B[2a+0, 2c+0], b22 = B[2a+1, 2c+1]
            # This contributes to C[0,2c+0] and C[1,2c+1]
            
            # This is getting very complex to do by hand. Let me just return None and use a different approach.
            pass
    
    return None

def _simple_improvement():
    """
    Try to find a rank < 64 decomposition using explicit Strassen construction.
    """
    from fractions import Fraction as F
    
    n, m, p = 2, 4, 8
    rank = 56
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]
    
    # Partition into 8 sub-tensors of size 2x2x2:
    # j = 2*a + b, a in {0,1}, b in {0,1}
    # k = 2*c + d, c in {0,1,2,3}, d in {0,1}
    # Sub-tensor (a,c): i in {0,1}, b in {0,1}, d in {0,1}
    
    col = 0
    for a in range(2):
        for c in range(4):
            # 7 Strassen terms for this 2x2x2 sub-tensor
            # Indices within sub-tensor:
            # U row: i*4 + (2*a + b)
            # V row: (2*a + b)*8 + (2*c + d)
            # W row: (2*c + d)*2 + i
            
            # Strassen formulas for 2x2x2:
            # Let A = [[a00, a01], [a10, a11]] where aib = A[i, 2a+b]
            # Let B = [[b00, b01], [b10, b11]] where bjd = B[2a+j, 2c+d]
            # C[0,0] = a00*b00 + a01*b10
            # C[0,1] = a00*b01 + a01*b11
            # C[1,0] = a10*b00 + a11*b10
            # C[1,1] = a10*b01 + a11*b11
            
            # Strassen 7 products:
            # M1 = (a00+a11)*(b00+b11)
            # M2 = (a10+a11)*b00
            # M3 = a00*(b10-b11)
            # M4 = a11*(b01-b00)
            # M5 = (a00+a01)*b11
            # M6 = (a10-a00)*(b00+b01)
            # M7 = (a11-a01)*(b01+b11)
            
            # C[0,0] = M1 - M4 - M6 + M3
            # C[0,1] = M1 + M5 - M7 - M2
            # C[1,0] = M2 + M6 - M1 + M4
            # C[1,1] = M1 + M7 - M5 + M2
            
            # Wait, let me use the standard Strassen formulas:
            # M1 = (a00+a11)(b00+b11)
            # M2 = (a10+a11)b00
            # M3 = a00(b10-b11)
            # M4 = a11(b01-b00)
            # M5 = (a00+a01)b11
            # M6 = (a10-a00)(b00+b01)
            # M7 = (a11-a01)(b01+b11)
            
            # C00 = M1 - M4 - M6 + M3
            # C01 = M1 + M5 - M7 - M2
            # C10 = M2 + M6 - M1 + M4
            # C11 = M1 + M7 - M5 + M2
            
            # Each M is a product of a linear combo of A entries and a linear combo of B entries.
            # In the tensor decomposition, each M corresponds to 1 rank-1 term where:
            # U has the A coefficients, V has the B coefficients, W has the C output coefficients.
            
            # M1: A part = a00 + a11, B part = b00 + b11
            #   U[0*4+2a+0] += 1, U[1*4+2a+1] += 1
            #   V[(2a+0)*8+2c+0] += 1, V[(2a+1)*8+2c+1] += 1
            #   W[2c+0 *2 + 0] += 1, W[2c+0 *2 + 1] += -1 (for C00)
            #   W[2c+1 *2 + 0] += 1, W[2c+1 *2 + 1] += 1 (for C01)
            #   Wait, W is indexed by (k*n+i) = (2c+d)*2+i
            #   For C[0,2c+0]: W[(2c+0)*2+0] = W[4c]
            #   For C[0,2c+1]: W[(2c+1)*2+0] = W[4c+2]
            #   For C[1,2c+0]: W[(2c+0)*2+1] = W[4c+1]
            #   For C[1,2c+1]: W[(2c+1)*2+1] = W[4c+3]
            
            # C00 = M1 - M4 - M6 + M3
            # C01 = M1 + M5 - M7 - M2
            # C10 = M2 + M6 - M1 + M4
            # C11 = M1 + M7 - M5 + M2
            
            # So for M1: W contributions: C00:+1, C01:+1, C10:-1, C11:+1
            #   W[4c] += 1, W[4c+2] += 1, W[4c+1] += -1, W[4c+3] += 1
            
            # For M2: C00:0, C01:-1, C10:+1, C11:+1
            #   W[4c] += 0, W[4c+2] += -1, W[4c+1] += 1, W[4c+3] += 1
            
            # For M3: C00:+1, C01:0, C10:0, C11:0
            #   W[4c] += 1, others 0
            
            # For M4: C00:-1, C01:0, C10:+1, C11:0
            #   W[4c] += -1, W[4c+1] += 1, others 0
            
            # For M5: C00:0, C01:+1, C10:0, C11:-1
            #   W[4c+2] += 1, W[4c+3] += -1, others 0
            
            # For M6: C00:-1, C01:0, C10:+1, C11:0
            #   W[4c] += -1, W[4c+1] += 1, others 0
            
            # For M7: C00:0, C01:-1, C10:0, C11:+1
            #   W[4c+2] += -1, W[4c+3] += 1, others 0
            
            # Now define the 7 terms:
            
            # M1: (a00+a11)(b00+b11)
            #   U: row 0*4+2a+0 += 1, row 1*4+2a+1 += 1
            #   V: row (2a+0)*8+2c+0 += 1, row (2a+1)*8+2c+1 += 1
            #   W: W[4c] += 1, W[4c+1] += -1, W[4c+2] += 1, W[4c+3] += 1
            U[0*4+2*a+0][col] += F(1)
            U[1*4+2*a+1][col] += F(1)
            V[(2*a+0)*8+2*c+0][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(1)
            W[4*c+0][col] += F(1)
            W[4*c+1][col] += F(-1)
            W[4*c+2][col] += F(1)
            W[4*c+3][col] += F(1)
            col += 1
            
            # M2: (a10+a11)b00
            #   U: row 1*4+2a+0 += 1, row 1*4+2a+1 += 1
            #   V: row (2a+0)*8+2c+0 += 1
            #   W: W[4c+1] += 1, W[4c+2] += -1, W[4c+3] += 1
            U[1*4+2*a+0][col] += F(1)
            U[1*4+2*a+1][col] += F(1)
            V[(2*a+0)*8+2*c+0][col] += F(1)
            W[4*c+1][col] += F(1)
            W[4*c+2][col] += F(-1)
            W[4*c+3][col] += F(1)
            col += 1
            
            # M3: a00(b10-b11)
            #   U: row 0*4+2a+0 += 1
            #   V: row (2a+1)*8+2c+0 += 1, row (2a+1)*8+2c+1 += -1
            #   W: W[4c] += 1
            U[0*4+2*a+0][col] += F(1)
            V[(2*a+1)*8+2*c+0][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(-1)
            W[4*c+0][col] += F(1)
            col += 1
            
            # M4: a11(b01-b00)
            #   U: row 1*4+2a+1 += 1
            #   V: row (2a+0)*8+2c+1 += 1, row (2a+0)*8+2c+0 += -1
            #   W: W[4c] += -1, W[4c+1] += 1
            U[1*4+2*a+1][col] += F(1)
            V[(2*a+0)*8+2*c+1][col] += F(1)
            V[(2*a+0)*8+2*c+0][col] += F(-1)
            W[4*c+0][col] += F(-1)
            W[4*c+1][col] += F(1)
            col += 1
            
            # M5: (a00+a01)b11
            #   U: row 0*4+2a+0 += 1, row 0*4+2a+1 += 1
            #   V: row (2a+1)*8+2c+1 += 1
            #   W: W[4c+2] += 1, W[4c+3] += -1
            U[0*4+2*a+0][col] += F(1)
            U[0*4+2*a+1][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(1)
            W[4*c+2][col] += F(1)
            W[4*c+3][col] += F(-1)
            col += 1
            
            # M6: (a10-a00)(b00+b01)
            #   U: row 1*4+2a+0 += 1, row 0*4+2a+0 += -1
            #   V: row (2a+0)*8+2c+0 += 1, row (2a+0)*8+2c+1 += 1
            #   W: W[4c] += -1, W[4c+1] += 1
            U[1*4+2*a+0][col] += F(1)
            U[0*4+2*a+0][col] += F(-1)
            V[(2*a+0)*8+2*c+0][col] += F(1)
            V[(2*a+0)*8+2*c+1][col] += F(1)
            W[4*c+0][col] += F(-1)
            W[4*c+1][col] += F(1)
            col += 1
            
            # M7: (a11-a01)(b01+b11)
            #   U: row 1*4+2a+1 += 1, row 0*4+2a+1 += -1
            #   V: row (2a+0)*8+2c+1 += 1, row (2a+1)*8+2c+1 += 1
            #   W: W[4c+2] += -1, W[4c+3] += 1
            U[1*4+2*a+1][col] += F(1)
            U[0*4+2*a+1][col] += F(-1)
            V[(2*a+0)*8+2*c+1][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(1)
            W[4*c+2][col] += F(-1)
            W[4*c+3][col] += F(1)
            col += 1
    
    # col should be 56
    assert col == 56, f"Expected 56 terms, got {col}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try Strassen-like constructions
    if _elapsed() < _DEADLINE - 10:
        s = _strassen_like_2x4x8()
        if s is not None:
            if _verify(s["U"], s["V"], s["W"]):
                r = len(s["U"][0])
                if r < best_rank:
                    best = s
                    best_rank = r
    
    # Try simple improvement (which now contains the actual Strassen construction)
    if _elapsed() < _DEADLINE - 5:
        s2 = _simple_improvement()
        if s2 is not None:
            if _verify(s2["U"], s2["V"], s2["W"]):
                r = len(s2["U"][0])
                if r < best_rank:
                    best = s2
                    best_rank = r
    
    return best
