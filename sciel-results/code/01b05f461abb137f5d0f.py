import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 140.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49():
    if _remaining() < 15:
        return None
    
    n, m, p = 2, 5, 5
    
    # Block decomposition: split m=5 and p=5 into blocks of size 2,2,1
    # A is 2x5, B is 5x5, C is 2x5
    # A = [A1 | A2 | A3] where A1,A2 are 2x2, A3 is 2x1
    # B = [B11 B12 B13; B21 B22 B23; B31 B32 B33] where B11,B12,B21,B22 are 2x2, etc.
    # C = A*B = [A1*B11 + A2*B21 + A3*B31, ...]
    
    # For the 2x2*2x2 blocks, use Strassen (rank 7 instead of 8)
    # There are 4 such blocks: (A1,B11), (A1,B21), (A2,B11), (A2,B21)
    # Wait, let me think more carefully.
    
    # A is 2x5: rows are i=0,1; cols are j=0..4
    # B is 5x5: rows are j=0..4; cols are k=0..4
    # C is 2x5: rows are i=0,1; cols are k=0..4
    
    # Split j into [0,1], [2,3], [4]
    # Split k into [0,1], [2,3], [4]
    
    # A_j blocks: A[:, 0:2] (2x2), A[:, 2:4] (2x2), A[:, 4:5] (2x1)
    # B_jk blocks: B[0:2, 0:2] (2x2), B[0:2, 2:4] (2x2), B[0:2, 4:5] (2x1)
    #               B[2:4, 0:2] (2x2), B[2:4, 2:4] (2x2), B[2:4, 4:5] (2x1)
    #               B[4:5, 0:2] (1x2), B[4:5, 2:4] (1x2), B[4:5, 4:5] (1x1)
    
    # C_k = sum_j A_j * B_jk
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2] + A[:, 4:5] * B[4:5, 0:2]
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4] + A[:, 4:5] * B[4:5, 2:4]
    # C[:, 4:5] = A[:, 0:2] * B[0:2, 4:5] + A[:, 2:4] * B[2:4, 4:5] + A[:, 4:5] * B[4:5, 4:5]
    
    # The 2x2*2x2 products can use Strassen (7 mults each)
    # The 2x2*1x2, 2x1*2x2, 2x1*1x2, 2x1*1x1 products are small
    
    # Let me count:
    # For C[:, 0:2]: 
    #   A[:,0:2] * B[0:2,0:2]: 2x2*2x2 -> 7 (Strassen)
    #   A[:,2:4] * B[2:4,0:2]: 2x2*2x2 -> 7 (Strassen)
    #   A[:,4:5] * B[4:5,0:2]: 2x1*1x2 -> 2 (each element of A[:,4] times each row of B[4,0:2])
    #   Total: 16
    # For C[:, 2:4]:
    #   A[:,0:2] * B[0:2,2:4]: 7
    #   A[:,2:4] * B[2:4,2:4]: 7
    #   A[:,4:5] * B[4:5,2:4]: 2
    #   Total: 16
    # For C[:, 4:5]:
    #   A[:,0:2] * B[0:2,4:5]: 2x2*2x1 -> 4
    #   A[:,2:4] * B[2:4,4:5]: 2x2*2x1 -> 4
    #   A[:,4:5] * B[4:5,4:5]: 2x1*1x1 -> 2
    #   Total: 10
    # Grand total: 16+16+10 = 42
    
    # But wait, we can share Strassen intermediates across the different C blocks!
    # A[:,0:2] is used in both C[:,0:2] and C[:,2:4]
    # A[:,2:4] is used in both C[:,0:2] and C[:,2:4]
    
    # Let me define Strassen for 2x2*2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For A[:,0:2]*B[0:2,0:2] (call it P1) and A[:,0:2]*B[0:2,2:4] (call it P2):
    # They share the same A matrix but different B matrices.
    # Strassen intermediates depend on both A and B, so they can't be directly shared.
    # Unless we restructure...
    
    # Actually, let me just build it directly with rank 42 and verify.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 10 rows
    V = [[0] * rank for _ in range(m * p)]  # 25 rows
    W = [[0] * rank for _ in range(p * n)]  # 10 rows
    
    c = 0
    
    # Helper: add a scalar multiplication term
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        nonlocal c
        U[u_idx][c] = coeff_u
        V[v_idx][c] = coeff_v
        W[w_idx][c] = coeff_w
        c += 1
    
    # Strassen for 2x2*2x2: A is 2x2 (rows a0,a1; cols b0,b1 in A-space)
    # B is 2x2 (rows c0,c1; cols d0,d1 in B-space)
    # Result C is 2x2
    # U indices: i*m + j where i in {0,1}, j in {j0, j0+1}
    # V indices: j*p + k where j in {j0, j0+1}, k in {k0, k0+1}
    # W indices: k*n + i where k in {k0, k0+1}, i in {0,1}
    
    def strassen_2x2(j0, k0):
        nonlocal c
        # A block: rows i=0,1; cols j=j0, j0+1
        # B block: rows j=j0, j0+1; cols k=k0, k0+1
        # C block: rows i=0,1; cols k=k0, k0+1
        
        # M1 = (A[0,j0]+A[1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        # This contributes to C[0,k0], C[0,k0+1], C[1,k0], C[1,k0+1]
        # But Strassen combines multiple products.
        
        # Let me just do 7 separate scalar multiplications for the Strassen intermediates.
        # Each intermediate is a sum of products of linear combinations.
        
        # M1 = (A00+A11)*(B00+B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # This is 4 scalar mults, but we want to count it as 1 "Strassen mult"
        # In the tensor decomposition, each scalar mult is a separate term.
        # So Strassen 2x2*2x2 = 7 scalar mults = 7 terms.
        
        # Wait, that's not right. In tensor decomposition, each term c corresponds to one
        # scalar multiplication. Strassen says we can compute 2x2*2x2 with 7 scalar mults
        # instead of 8. So it's 7 terms.
        
        # Let me define the 7 Strassen intermediates:
        # A00 = A[0,j0], A01 = A[0,j0+1], A10 = A[1,j0], A11 = A[1,j0+1]
        # B00 = B[j0,k0], B01 = B[j0,k0+1], B10 = B[j0+1,k0], B11 = B[j0+1,k0+1]
        
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # Each Mk is a product of a linear combination of A entries and a linear combination of B entries.
        # In the tensor decomposition, each Mk corresponds to one term c.
        # For term c:
        #   U[i*m+j][c] = coefficient of A[i,j] in the A-linear-combination for Mk
        #   V[j*p+k][c] = coefficient of B[j,k] in the B-linear-combination for Mk
        #   W[k*n+i][c] = coefficient of C[i,k] in the output
        
        # Let me enumerate:
        # M1: A-part = A00 + A11, B-part = B00 + B11
        #   U[0*m+j0][c] = 1, U[1*m+j0+1][c] = 1
        #   V[j0*p+k0][c] = 1, V[(j0+1)*p+k0+1][c] = 1
        #   W[k0*n+0][c] = 1 (for C00), W[(k0+1)*n+0][c] = 1 (for C01),
        #   W[k0*n+1][c] = 1 (for C10), W[(k0+1)*n+1][c] = 1 (for C11)
        # Wait, that's not right. Each Mk contributes to multiple C entries.
        # C00 gets M1, C01 gets M1, C10 gets M1, C11 gets M1 (with different signs/coeffs)
        
        # Actually, in the tensor decomposition, each term c has fixed U, V, W vectors.
        # The product U[i,j]*V[j,k]*W[k,i] gives the contribution to C[i,k].
        # So for M1, we need:
        #   U[0,j0][c] = 1, U[1,j0+1][c] = 1 (A00 and A11)
        #   V[j0,k0][c] = 1, V[j0+1,k0+1][c] = 1 (B00 and B11)
        #   W[k0,0][c] = 1, W[k0+1,0][c] = 1, W[k0,1][c] = 1, W[k0+1,1][c] = 1
        # But this would make M1 contribute to ALL of C00, C01, C10, C11 equally,
        # which is not what we want. We want C00 += M1, C01 += M1, C10 += M1, C11 += M1,
        # but C00 -= M5, etc.
        
        # Hmm, this is the issue. In Strassen, each intermediate Mk is used in multiple
        # output entries with different coefficients. In the tensor decomposition, each term c
        # has a single W vector, so it contributes the same coefficient to all C[i,k] that
        # are "connected" through the U and V vectors.
        
        # Actually, I think the way to do this is: each Strassen intermediate Mk
        # corresponds to one term c. The U vector has nonzeros at the A entries involved,
        # the V vector has nonzeros at the B entries involved, and the W vector has nonzeros
        # at the C entries that Mk contributes to, with the appropriate coefficients.
        
        # For M1: A-part = A00+A11, B-part = B00+B11
        # C-part: C00 += 1*M1, C01 += 1*M1, C10 += 1*M1, C11 += 1*M1
        # So W[k0,0][c] = 1, W[k0+1,0][c] = 1, W[k0,1][c] = 1, W[k0+1,1][c] = 1
        
        # For M2: A-part = A10+A11, B-part = B00
        # C-part: C10 += 1*M2, C11 -= 1*M2
        # W[k0,1][c] = 1, W[k0+1,1][c] = -1
        
        # etc.
        
        # Let me implement this.
        
        def add_strassen_term(a_coeffs, b_coeffs, c_coeffs):
            """a_coeffs: dict {(i,j): coeff}, b_coeffs: dict {(j,k): coeff}, c_coeffs: dict {(i,k): coeff}"""
            nonlocal c
            for (i, j), coeff in a_coeffs.items():
                U[i * m + j][c] = coeff
            for (j, k), coeff in b_coeffs.items():
                V[j * p + k][c] = coeff
            for (i, k), coeff in c_coeffs.items():
                W[k * n + i][c] = coeff
            c += 1
        
        # M1 = (A00+A11)*(B00+B11)
        add_strassen_term(
            {(0, j0): 1, (1, j0+1): 1},
            {(j0, k0): 1, (j0+1, k0+1): 1},
            {(0, k0): 1, (0, k0+1): 1, (1, k0): 1, (1, k0+1): 1}
        )
        
        # M2 = (A10+A11)*B00
        add_strassen_term(
            {(1, j0): 1, (1, j0+1): 1},
            {(j0, k0): 1},
            {(1, k0): 1, (1, k0+1): -1}
        )
        
        # M3 = A00*(B01-B11)
        add_strassen_term(
            {(0, j0): 1},
            {(j0, k0+1): 1, (j0+1, k0+1): -1},
            {(0, k0+1): 1, (1, k0+1): 1}
        )
        
        # M4 = A11*(B10-B00)
        add_strassen_term(
            {(1, j0+1): 1},
            {(j0+1, k0): 1, (j0, k0): -1},
            {(0, k0): 1, (1, k0): 1}
        )
        
        # M5 = (A00+A01)*B11
        add_strassen_term(
            {(0, j0): 1, (0, j0+1): 1},
            {(j0+1, k0+1): 1},
            {(0, k0): -1, (0, k0+1): 1}
        )
        
        # M6 = (A10-A00)*(B00+B01)
        # C11 += M6
        add_strassen_term(
            {(1, j0): 1, (0, j0): -1},
            {(j0, k0): 1, (j0, k0+1): 1},
            {(1, k0+1): 1}
        )
        
        # M7 = (A01-A11)*(B10+B11)
        add_strassen_term(
            {(0, j0+1): 1, (1, j0+1): -1},
            {(j0+1, k0): 1, (j0+1, k0+1): 1},
            {(0, k0): 1}
        )
        
        return 7  # 7 terms used
    
    # Now let's apply Strassen to the four 2x2*2x2 blocks:
    # (j0=0, k0=0): A[:,0:2] * B[0:2,0:2] -> contributes to C[:,0:2]
    # (j0=2, k0=0): A[:,2:4] * B[2:4,0:2] -> contributes to C[:,0:2]
    # (j0=0, k0=2): A[:,0:2] * B[0:2,2:4] -> contributes to C[:,2:4]
    # (j0=2, k0=2): A[:,2:4] * B[2:4,2:4] -> contributes to C[:,2:4]
    
    strassen_2x2(0, 0)  # 7 terms, c=0..6
    strassen_2x2(2, 0)  # 7 terms, c=7..13
    strassen_2x2(0, 2)  # 7 terms, c=14..20
    strassen_2x2(2, 2)  # 7 terms, c=21..27
    
    # Now the remaining products:
    # A[:,4:5] * B[4:5,0:2]: 2x1 * 1x2 -> 2 scalar mults
    # A[:,4:5] * B[4:5,2:4]: 2x1 * 1x2 -> 2 scalar mults
    # A[:,0:2] * B[0:2,4:5]: 2x2 * 2x1 -> 4 scalar mults
    # A[:,2:4] * B[2:4,4:5]: 2x2 * 2x1 -> 4 scalar mults
    # A[:,4:5] * B[4:5,4:5]: 2x1 * 1x1 -> 2 scalar mults
    # Total: 14 terms
    # Grand total: 28 + 14 = 42. 
    
    # A[:,4:5] * B[4:5,0:2]: A[i,4] * B[4,k] for i in {0,1}, k in {0,1}
    for i in range(2):
        for k in range(2):
            add_term(i * m + 4, 4 * p + k, k * n + i, 1, 1, 1)
    # c=28..31
    
    # A[:,4:5] * B[4:5,2:4]: A[i,4] * B[4,k] for i in {0,1}, k in {2,3}
    for i in range(2):
        for k in range(2, 4):
            add_term(i * m + 4, 4 * p + k, k * n + i, 1, 1, 1)
    # c=32..35
    
    # A[:,0:2] * B[0:2,4:5]: A[i,j] * B[j,4] for i in {0,1}, j in {0,1}
    for i in range(2):
        for j in range(2):
            add_term(i * m + j, j * p + 4, 4 * n + i, 1, 1, 1)
    # c=36..39
    
    # A[:,2:4] * B[2:4,4:5]: A[i,j] * B[j,4] for i in {0,1}, j in {2,3}
    for i in range(2):
        for j in range(2, 4):
            add_term(i * m + j, j * p + 4, 4 * n + i, 1, 1, 1)
    # c=40..43
    
    # A[:,4:5] * B[4:5,4:5]: A[i,4] * B[4,4] for i in {0,1}
    for i in range(2):
        add_term(i * m + 4, 4 * p + 4, 4 * n + i, 1, 1, 1)
    # c=44..45
    
    # Total: 46 terms. But I allocated rank=42. Let me fix.
    # 28 + 4 + 4 + 4 + 4 + 2 = 46.
    
    # I need to adjust the rank.
    if c > rank:
        # Extend matrices
        for row in U:
            row.extend([0] * (c - rank))
        for row in V:
            row.extend([0] * (c - rank))
        for row in W:
            row.extend([0] * (c - rank))
        rank = c
    
    if _verify(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    r49 = _try_rank_49()
    if r49 is not None:
        return r49
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
