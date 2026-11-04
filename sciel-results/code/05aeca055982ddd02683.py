"""Matrix multiplication 3,5,7 — rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal delta_{i,j,k} (1 if consistent indices else 0)
    Actually the tensor for matrix mult C = A*B where A is n x m, B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the trilinear form is sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] ... 
    The tensor T_{i,j,k} = 1 for all i,j,k (since each product term A[i,j]*B[j,k] contributes to C[i,k]).
    Wait, the standard tensor decomposition: C[i,k] = sum_j A[i,j] B[j,k]
    As a bilinear map: (A,B) -> C. The tensor has entries T_{(i,k), (i',j),(j',k')} ... 
    Let me re-derive. The multiplication tensor M_{n,m,p} has entries indexed by (i, j, k) where
    i in [0,n), j in [0,m), k in [0,p), and M[i,j,k] = 1.
    A decomposition of rank r gives U (n*m x r), V (m*p x r), W (p*n x r) such that
    M[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U[0].__len__() if isinstance(U[0], list) else len(U[0])
    # Build the tensor from U,V,W
    # M[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    val += u_val * v_val * w_val
                expected = 1.0
                if abs(val - expected) > tol:
                    return False
    return True


def _strassen_block(a, b):
    """Strassen 2x2 matrix multiplication. a, b are 2x2 lists."""
    # Returns 2x2 result
    p1 = [a[0][0] + a[1][1], a[0][1] + a[1][0]]
    p2 = [a[1][0] + a[1][1], 0]
    p3 = [a[0][0], 0]
    p4 = [a[1][1], 0]
    p5 = [a[0][0] + a[0][1], 0]
    p6 = [0, b[0][1] - b[1][1]]
    p7 = [a[0][1] + a[1][1], 0]
    p8 = [0, b[0][0] - b[1][0]]
    
    # This is getting complicated with lists. Let me use a cleaner approach.
    # Actually for the decomposition we need symbolic/numeric entries, not actual multiplication.
    # Let me think differently.
    pass


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    Uses Strassen's algorithm on 2x2 sub-blocks to reduce rank below schoolbook 105.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook baseline: rank = 3*5*7 = 105
    # Try to find a lower rank decomposition.
    
    # Approach: Use Strassen on overlapping 2x2 blocks.
    # The key insight: we can partition the computation into blocks and apply Strassen.
    
    # For 3x5 * 5x7:
    # We can view this as combining several 2x2 Strassen multiplications.
    
    # Let me try a direct approach: enumerate possible decompositions with small integer entries
    # and use the tensor verification to check validity.
    
    # Actually, let's use a known result: for M_{3,5,7}, the Strassen-based approach
    # can achieve rank 99 (as noted in the knowledge: 105 -> 99).
    
    # The strategy: partition the 3x5 matrix into blocks and the 5x7 matrix into blocks,
    # then apply Strassen to 2x2 sub-problems.
    
    # One approach: 
    # A (3x5) can be split as [A1(2x2) | A2(2x3); A3(1x2) | A4(1x3)]
    # B (5x7) can be split as [B1(2x3); B2(3x7)] ... this is getting complex.
    
    # Let me use a simpler strategy: use the fact that M_{n,m,p} can be decomposed
    # by applying Strassen to 2x2x2 sub-tensors.
    
    # For 3x5x7, we can cover it with:
    # - One 2x2x2 Strassen block (rank 7 instead of 8) saving 1
    # - Plus the remaining schoolbook terms
    
    # Actually the measured result says 99, which is 105 - 6. So 6 terms are saved.
    # This suggests 3 Strassen 2x2 blocks each saving 1 term (3*2=6? No, each saves 1, so 3 blocks save 3).
    # Or maybe 6 individual savings.
    
    # Let me try a different approach: use random search / optimization to find a low-rank decomposition.
    
    import numpy as np
    from scipy.optimize import minimize
    
    # The tensor M has entries M[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,7).
    # We want to find U (15 x r), V (35 x r), W (21 x r) with rational entries
    # such that M[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # This is a very hard problem in general. Let me try the Strassen approach more carefully.
    
    # Strassen approach for M_{3,5,7}:
    # Consider the indices i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5,6}
    
    # We can group:
    # i=0,1 together (2 rows), i=2 separate (1 row)
    # j=0,1,2,3,4 - group as (0,1), (2,3), (4) 
    # k=0,1,2,3,4,5,6 - group as (0,1), (2,3), (4,5), (6)
    
    # For the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}): use Strassen (rank 7)
    # For all other combinations: use schoolbook (rank 1 each)
    
    # Total rank = 7 + (number of remaining terms)
    # Remaining terms: total 3*5*7 = 105, minus 2*2*2 = 8 schoolbook terms replaced by 7 Strassen
    # So rank = 105 - 8 + 7 = 104. That's only saving 1.
    
    # To save 6, we need 6 more Strassen-like savings.
    # Each 2x2x2 block saves 1 term. So we need 6 such blocks? But they overlap.
    
    # Let me try: use multiple non-overlapping 2x2x2 blocks.
    # In 3x5x7, how many non-overlapping 2x2x2 blocks can we fit?
    # i: 3 -> can fit 1 block of size 2 (using i=0,1)
    # j: 5 -> can fit 2 blocks of size 2 (using j=0,1 and j=2,3)
    # k: 7 -> can fit 3 blocks of size 2 (using k=0,1, k=2,3, k=4,5)
    # So 1*2*3 = 6 non-overlapping 2x2x2 blocks, each saving 1 term.
    # Total saving: 6 terms.
    # Rank = 105 - 6 = 99. This matches the measured result!
    
    # Now let me construct this decomposition.
    
    # For each 2x2x2 block at (i0, j0, k0) where i0 in {0}, j0 in {0,2}, k0 in {0,2,4}:
    # The block covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # Strassen's 7 products for 2x2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # In tensor terms, for a 2x2x2 block with local indices (ii, jj, kk) in {0,1}^3:
    # The schoolbook would have 8 terms: for each (ii,jj,kk), a term with U,V,W all = 1.
    # Strassen replaces these 8 with 7 terms where U,V,W entries are in {-1, 0, 1}.
    
    # Let me define the Strassen decomposition for a 2x2x2 block.
    # Local indices: ii, jj, kk in {0,1}
    # U_local[ii*2+jj, c], V_local[jj*2+kk, c], W_local[kk*2+ii, c]
    
    # Strassen products (7 of them):
    # c=0: M1 = (a00+a11)(b00+b11)
    #   U: a00=1, a11=1 -> U[0,0]=1, U[3,0]=1 (ii=0,jj=0 and ii=1,jj=1)
    #   V: b00=1, b11=1 -> V[0,0]=1, V[3,0]=1 (jj=0,kk=0 and jj=1,kk=1)
    #   W: c00=1, c11=1 -> W[0,0]=1, W[5,0]=1 (kk=0,ii=0 and kk=1,ii=1)
    #   Wait, W index is kk*2+ii. So W[0*2+0,0]=W[0,0]=1, W[1*2+1,0]=W[3,0]=1
    #   Hmm, let me be more careful.
    
    # For 2x2x2, the tensor entry T[ii,jj,kk] = 1 for all 8 combinations.
    # We need: T[ii,jj,kk] = sum_c U[ii,jj,c] * V[jj,kk,c] * W[kk,ii,c]
    
    # Strassen's 7 terms:
    # c=0: (a00+a11)*(b00+b11) -> contributes to c00 and c11
    #   U[0,0]=1 (a00), U[3,0]=1 (a11) [ii*2+jj: 0*2+0=0, 1*2+1=3]
    #   V[0,0]=1 (b00), V[3,0]=1 (b11) [jj*2+kk: 0*2+0=0, 1*2+1=3]
    #   W[0,0]=1 (c00), W[3,0]=1 (c11) [kk*2+ii: 0*2+0=0, 1*2+1=3]
    #   Check T[0,0,0] = U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
    #   Check T[1,1,1] = U[3,0]*V[3,0]*W[3,0] = 1*1*1 = 1 ✓
    #   But we also get cross terms: T[0,1,0] = U[0,0]*V[1,0]*W[0,0] = 1*0*1 = 0 ✓
    #   T[0,0,1] = U[0,0]*V[0,0]*W[1,0] = 1*1*0 = 0 ✓
    #   T[1,0,1] = U[2,0]*V[0,0]*W[3,0] = 0*1*1 = 0 ✓
    #   T[1,1,0] = U[3,0]*V[3,0]*W[2,0] = 1*1*0 = 0 ✓
    #   T[0,1,1] = U[1,0]*V[3,0]*W[3,0] = 0*1*1 = 0 ✓
    #   T[1,0,0] = U[2,0]*V[0,0]*W[0,0] = 0*1*1 = 0 ✓
    #   Great, only T[0,0,0] and T[1,1,1] get 1 from this term.
    
    # c=1: (a10+a11)*b00 -> contributes to c10
    #   U[2,1]=1 (a10: ii=1,jj=0), U[3,1]=1 (a11: ii=1,jj=1)
    #   V[0,1]=1 (b00: jj=0,kk=0)
    #   W[2,1]=1 (c10: kk=0,ii=1)
    #   Check T[1,0,0] = U[2,1]*V[0,1]*W[2,1] = 1*1*1 = 1 ✓
    #   Other entries: T[1,1,0] = U[3,1]*V[2,1]*W[2,1] = 1*0*1 = 0 ✓
    #   Good.
    
    # c=2: a00*(b10-b11) -> contributes to c01
    #   U[0,2]=1 (a00)
    #   V[2,2]=1 (b10: jj=1,kk=0), V[3,2]=-1 (b11: jj=1,kk=1)
    #   W[4,2]=1 (c01: kk=1,ii=0)
    #   Check T[0,1,1] = U[1,2]*V[3,2]*W[4,2] = 0*(-1)*1 = 0 
    #   Hmm, that's wrong. T[0,1,1] should be 1.
    #   Wait, I think I have the indexing wrong. Let me reconsider.
    
    # Actually, let me just use the known Strassen decomposition and verify it numerically.
    
    # Let me define the 7 Strassen terms for 2x2x2 more carefully.
    # Using the standard Strassen formula for C = A*B where A,B are 2x2:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # In tensor form, C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form: for each (i,j,k), the coefficient is 1.
    # So we need to express the identity tensor as a sum of 7 rank-1 tensors.
    
    # Let me just hardcode the Strassen 2x2x2 decomposition and verify.
    
    def strassen_2x2x2():
        """Return U (4x7), V (4x7), W (4x7) for 2x2x2 Strassen decomposition."""
        # U[ii*2+jj, c], V[jj*2+kk, c], W[kk*2+ii, c]
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2 = (A10+A11)B00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # M3 = A00(B10-B11)
        U[0,2] = 1
        V[2,2] = 1; V[3,2] = -1
        W[4,2] = 1
        
        # M4 = A11(B01-B00)
        U[3,3] = 1
        V[1,3] = 1; V[0,3] = -1
        W[2,3] = 1
        
        # M5 = (A00+A01)B11
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[4,4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[2,5] = 1
        W[0,5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        U[1,6] = 1; U[3,6] = -1
        V[1,6] = 1; V[3,6] = 1
        W[3,6] = 1
        
        return U, V, W
    
    # Verify the Strassen 2x2x2 decomposition
    sU, sV, sW = strassen_2x2x2()
    valid = True
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                val = 0.0
                for c in range(7):
                    val += sU[ii*2+jj][c] * sV[jj*2+kk][c] * sW[kk*2+ii][c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
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
    
    # Now build the full 3x5x7 decomposition using 6 Strassen blocks + schoolbook for the rest.
    # Strassen blocks at:
    # i0=0, j0 in {0, 2}, k0 in {0, 2, 4}
    # Each block covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    
    # For terms NOT covered by any Strassen block, use schoolbook (rank 1 each).
    
    # Let's track which (i,j,k) triples are covered by Strassen blocks.
    strassen_blocks = []
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            strassen_blocks.append((0, j0, k0))  # i0=0 always
    
    # Check if (i,j,k) is in a Strassen block
    def in_strassen_block(i, j, k):
        for (i0, j0, k0) in strassen_blocks:
            if i0 <= i < i0 + 2 and j0 <= j < j0 + 2 and k0 <= k < k0 + 2:
                return True
        return False
    
    # Count schoolbook terms (not in any Strassen block)
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not in_strassen_block(i, j, k):
                    schoolbook_terms.append((i, j, k))
    
    # Total rank = 6 * 7 (Strassen) + len(schoolbook_terms) (schoolbook)
    rank = len(strassen_blocks) * 7 + len(schoolbook_terms)
    
    # Build U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Add Strassen blocks
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = strassen_2x2x2()
        for ii in range(2):
            for jj in range(2):
                for c in range(7):
                    if sU[ii*2+jj][c] != 0:
                        U[(i0+ii)*m + (j0+jj)][col+c] = sU[ii*2+jj][c]
        for jj in range(2):
            for kk in range(2):
                for c in range(7):
                    if sV[jj*2+kk][c] != 0:
                        V[(j0+jj)*p + (k0+kk)][col+c] = sV[jj*2+kk][c]
        for kk in range(2):
            for ii in range(2):
                for c in range(7):
                    if sW[kk*2+ii][c] != 0:
                        W[(k0+kk)*n + (i0+ii)][col+c] = sW[kk*2+ii][c]
        col += 7
    
    # Add schoolbook terms
    for (i, j, k) in schoolbook_terms:
        U[i*m + j][col] = 1
        V[j*p + k][col] = 1
        W[k*n + i][col] = 1
        col += 1
    
    # Verify the full decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
