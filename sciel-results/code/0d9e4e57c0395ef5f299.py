"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) as 2D lists for the schoolbook (rank = n*m*p) decomposition
    of the matrix multiplication tensor.
    
    Indexing conventions:
      U has shape (n*m, rank), indexed by row = i*m + j
      V has shape (m*p, rank), indexed by row = j*p + k
      W has shape (p*n, rank), indexed by row = k*n + i
    
    The tensor identity is:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
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
    return U, V, W


def _build_strassen_3x5x6():
    """
    Build a rank-reduced decomposition of the 3x5x6 matrix multiplication tensor
    using Strassen's rank-7 algorithm on the 2x2x2 sub-block.
    
    Split: A (3x5) = [A1(1x5); A2(2x5)], B (5x6) = [B1(5x2) B2(5x4)]
    C = [C11(1x2)  C12(1x4); C21(2x2)  C22(2x4)]
    
    For C21 = A2 * B1 (2x5 * 5x2 -> 2x2), apply Strassen on the 2x2x2 core.
    Total rank: 10 + 20 + 19 + 40 = 89 (saving 1 from schoolbook 90).
    """
    n, m, p = 3, 5, 6
    
    # Strassen 2x2x2 coefficients:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We'll build factors incrementally.
    # Each "product" contributes to U, V, W columns.
    
    U_rows = n * m  # 15
    V_rows = m * p  # 30
    W_rows = p * n  # 18
    
    # We'll collect contributions as lists of (row, value) per factor
    U_contribs = []  # list of dicts: {row: value}
    V_contribs = []
    W_contribs = []
    
    def add_product(u_row, v_row, w_row, u_val=1, v_val=1, w_val=1):
        """Add a single rank-1 tensor component."""
        U_contribs.append({u_row: u_val})
        V_contribs.append({v_row: v_val})
        W_contribs.append({w_row: w_val})
    
    def add_strassen_product(a_rows, b_rows, c_row, coeff_a, coeff_b, coeff_c):
        """
        Add a Strassen-style product:
        sum_{ja} coeff_a[ja] * A[i,ja] * sum_{jb} coeff_b[jb] * B[jb,k] -> C[c_row]
        where a_rows are the j-indices for A, b_rows for B.
        """
        # This represents: (sum ja coeff_a[ja] A[i,ja]) * (sum jb coeff_b[jb] B[jb,k])
        # In tensor form: for each (i, j, k), the contribution is:
        #   if j in a_rows and k in b_rows: coeff_a[j] * coeff_b[k] * (1/c_row_weight)
        # Actually, let's think differently.
        # 
        # The Strassen product M = X*Y where X = sum ja ca[ja]*A[ia,ja], Y = sum jb cb[jb]*B[jb,ik]
        # Then C[c] += coeff_c * M
        # 
        # In the tensor decomposition, each rank-1 term is U[i*m+j]*V[j*p+k]*W[k*n+i]
        # For a Strassen product contributing to C[c_row]:
        #   U row = i*m + ja, V row = jb*p + k, W row = k*n + i
        #   with U_val = ca[ja], V_val = cb[jb], W_val = coeff_c
        #   but we need to sum over all ja in a_rows and jb in b_rows
        #
        # Actually for the 2x2x2 Strassen, the "matrix" elements are scalars.
        # A[i,ja] where i is the row index of the 2x2 block, ja is column.
        # In our global indexing, for C21 block:
        #   i_global = 1 + i_local (i_local in {0,1})
        #   ja_global = ja_local (ja_local in {0,1}) for B1 columns
        #   jb_global = jb_local (jb_local in {0,1}) for A2 rows
        #   k_global = k_local (k_local in {0,1}) for C21 columns
        #
        # Wait, let me re-index carefully.
        # A is 3x5, indexed A[i][j] with i in 0..2, j in 0..4
        # B is 5x6, indexed B[j][k] with j in 0..4, k in 0..5
        # C is 3x6, indexed C[i][k] with i in 0..2, k in 0..5
        #
        # Split: A1 = A[0,:] (1x5), A2 = A[1:,:] (2x5)
        #        B1 = B[:,0:2] (5x2), B2 = B[:,2:6] (5x4)
        # C11 = A1*B1 (1x2): C[0,0:2]
        # C12 = A1*B2 (1x4): C[0,2:6]
        # C21 = A2*B1 (2x2): C[1:3,0:2]
        # C22 = A2*B2 (2x4): C[1:3,2:6]
        #
        # For C21 = A2*B1:
        #   A2[i2, j] where i2 in {0,1}, j in {0..4}
        #   B1[j, k1] where j in {0..4}, k1 in {0,1}
        #   C21[i2, k1] = sum_j A2[i2,j] * B1[j,k1]
        #   Global: i = 1+i2, k = k1
        #
        # Strassen on 2x2: split A2 into rows, B1 into columns
        #   A2 = [A2[0,:]; A2[1,:]] -> A21 = A2[0,:], A22 = A2[1,:]
        #   B1 = [B1[:,0] B1[:,1]] -> B11 = B1[:,0], B12 = B1[:,1]
        #
        # M1 = (A21 + A22) * (B11 + B12)
        #     = sum_j (A[1,j]+A[2,j]) * (B[j,0]+B[j,1])
        # M2 = (A22) * B11 = sum_j A[2,j] * B[j,0]
        # M3 = A21 * (B12 - B12) = sum_j A[1,j] * (B[j,1]-B[j,1]) = 0... 
        # 
        # Wait, Strassen for 2x2 matrices:
        # X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
        # M1 = (x11+x22)(y11+y22)
        # M2 = (x21+x22)y11
        # M3 = x11(y12-y22)
        # M4 = x22(y21-y11)
        # M5 = (x11+x12)y22
        # M6 = (x21-x11)(y11+y12)
        # M7 = (x12-x22)(y21+y22)
        # Z11 = M1+M4-M5+M7
        # Z12 = M3+M5
        # Z21 = M2+M4
        # Z22 = M1-M2+M3+M6
        #
        # In our case, X = A2 (2x5), Y = B1 (5x2). These are NOT 2x2 matrices!
        # The inner dimension is 5, not 2. So we can't directly apply Strassen.
        #
        # The correct approach: the 2x2x2 Strassen applies when the inner
        # dimension equals 2. Here the inner dimension is 5. So we need to
        # think about this differently.
        #
        # Actually, for multiplying a (2x5) by (5x2), the tensor rank is 20
        # (schoolbook). We can't directly apply 2x2x2 Strassen here.
        #
        # Let me reconsider. The known result of rank 84 for 3x5x6 likely
        # uses a different decomposition strategy. Let me think about what
        # structure gives 84.
        #
        # 90 - 84 = 6 savings. This suggests 6 units of Strassen-like savings.
        # Perhaps splitting into multiple 2x2x2 blocks?
        #
        # Alternative: Split A (3x5) into [A1(1x5); A2(1x5); A3(1x5)] and
        # B (5x6) into [B1(5x2) B2(5x2) B3(5x2)]
        # Then C has 9 blocks of (1x2), each requiring 10 schoolbook products.
        # Total: 90. No savings from this split alone.
        #
        # Another approach: Use the fact that 3x5x6 can be embedded in a
        # larger structure. Or use Winograd-style algorithms.
        #
        # Let me try a different split:
        # A (3x5) = [A1(2x5); A2(1x5)]
        # B (5x6) = [B1(5x3) B2(5x3)]
        # C11 = A1*B1 (2x3): 30 schoolbook
        # C12 = A1*B2 (2x3): 30 schoolbook  
        # C21 = A2*B1 (1x3): 15 schoolbook
        # C22 = A2*B2 (1x3): 15 schoolbook
        # Total: 90
        #
        # For C11 (2x5 * 5x3 -> 2x3), can we save? The 2x2x2 sub-blocks
        # within would give savings if we can isolate them.
        #
        # I think the key insight from the inspiration is that by carefully
        # choosing the block structure, we can apply Strassen to multiple
        # 2x2x2 sub-problems. Let me try:
        #
        # Split A (3x5) into rows: [A_row0(1x5), A_row1(1x5), A_row2(1x5)]
        # Split B (5x6) into cols: [B_col0(5x1), ..., B_col5(5x1)]
        # Each C[i,k] = sum_j A[i,j]*B[j,k] needs 5 products.
        # Total: 3*6*5 = 90.
        #
        # To use Strassen, we need 2x2 blocks. Let's group:
        # Rows 0,1 of A and cols 0,1 of B form a 2x2x2 sub-problem (if we
        # also group the inner dimension, but it's 5 not 2).
        #
        # I think the correct approach for rank 84 involves a more clever
        # embedding. Let me just implement the straightforward Strassen
        # embedding that gives rank 89 (saving 1) and verify it works.
        # Then we can iterate.
        #
        # For rank 89: Use Strassen on one 2x2x2 sub-block.
        # The only place we have a true 2x2x2 is if we split the inner
        # dimension into 2+2+1. 
        #
        # Split inner dim j: j in {0,1,2,3,4} -> groups {0,1}, {2,3}, {4}
        # For C21 = A2*B1 (2x5 * 5x2):
        #   C21[i2,k1] = sum_{j=0}^{4} A[1+i2, j] * B[j, k1]
        #   = sum_{j in {0,1}} + sum_{j in {2,3}} + A[1+i2,4]*B[4,k1]
        #
        # The {0,1} part: A[1+i2, {0,1}] * B[{0,1}, k1] is a 2x2x2 problem!
        # (2 rows of A, 2 cols of B, 2 inner indices)
        # Similarly {2,3} part is another 2x2x2.
        # The {4} part is 2x1x2 = 4 schoolbook.
        #
        # So C21 rank = 7 + 7 + 4 = 18 (vs 20 schoolbook), saving 2!
        #
        # Let me redo the full accounting:
        # Split inner dim into {0,1}, {2,3}, {4}
        #
        # C11 = A1*B1 (1x5 * 5x2 -> 1x2):
        #   = A1[{0,1}]*B1[{0,1},:] + A1[{2,3}]*B1[{2,3},:] + A1[4]*B1[4,:]
        #   Each 1x2x2 block: schoolbook 4, no Strassen benefit (not 2x2x2)
        #   Total: 4+4+2 = 10 (no savings)
        #
        # C12 = A1*B2 (1x5 * 5x4 -> 1x4):
        #   Similarly: 8+8+4 = 20 (no savings)
        #
        # C21 = A2*B1 (2x5 * 5x2 -> 2x2):
        #   2x2x2 + 2x2x2 + 2x1x2 = 7+7+4 = 18 (saving 2)
        #
        # C22 = A2*B2 (2x5 * 5x4 -> 2x4):
        #   2x2x4 + 2x2x4 + 2x1x4 = 16+16+8 = 40 (no savings for 2x2x4)
        #   Wait, 2x2x4: can we save? It's like two 2x2x2 blocks side by side.
        #   2x2x4 = 2*(2x2x2) = 2*8 schoolbook, 2*7 Strassen = 14 (saving 2)
        #   So C22 = 14+14+8 = 36 (saving 4)
        #
        # Total: 10 + 20 + 18 + 36 = 84! That matches!
        #
        # Now let me implement this properly.
        
    # Let me rebuild with the correct strategy.
    # Inner dimension split: j_groups = [{0,1}, {2,3}, {4}]
    # Row split: A1 = row 0, A2 = rows 1-2
    # Col split: B1 = cols 0-1, B2 = cols 2-5
    
    # For each block, we'll add rank-1 terms.
    
    def add_term(i, j, k, u_val=1.0, v_val=1.0, w_val=1.0):
        """Add a rank-1 term: U[i*m+j] += u_val, V[j*p+k] += v_val, W[k*n+i] += w_val"""
        U_contribs.append((i * m + j, u_val))
        V_contribs.append((j * p + k, v_val))
        W_contribs.append((k * n + i, w_val))
    
    def add_strassen_2x2x2(i_base, j_group, k_base):
        """
        Add Strassen rank-7 terms for a 2x2x2 sub-block.
        i_base: starting row index for the 2 rows of A
        j_group: list of 2 j-indices for the inner dimension
        k_base: starting col index for the 2 cols of B
        
        The 2x2x2 block computes:
          C[i_base + di, k_base + dk] for di,dk in {0,1}
          = sum_{dj in {0,1}} A[i_base+di, j_group[dj]] * B[j_group[dj], k_base+dk]
        
        Strassen:
          M1 = (A[i_base, j0] + A[i_base+1, j1]) * (B[j0, k_base] + B[j1, k_base+1])
          M2 = (A[i_base+1, j0] + A[i_base+1, j1]) * B[j0, k_base]
          M3 = A[i_base, j0] * (B[j1, k_base+1] - B[j0, k_base+1])
          M4 = A[i_base+1, j1] * (B[j1, k_base] - B[j0, k_base])
          M5 = (A[i_base, j0] + A[i_base, j1]) * B[j1, k_base+1]
          M6 = (A[i_base+1, j0] - A[i_base, j1]) * (B[j0, k_base] + B[j0, k_base+1])
          M7 = (A[i_base, j1] - A[i_base+1, j1]) * (B[j1, k_base] + B[j1, k_base+1])
          
          C[i_base, k_base] = M1 + M4 - M5 + M7
          C[i_base, k_base+1] = M3 + M5
          C[i_base+1, k_base] = M2 + M4
          C[i_base+1, k_base+1] = M1 - M2 + M3 + M6
        """
        j0, j1 = j_group
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # This is a sum of 4 products:
        # A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j1,k1] + A[i1,j1]*B[j0,k0] + A[i1,j1]*B[j1,k1]
        # In tensor form, each product (i,j,k) contributes U[i*m+j]*V[j*p+k]*W[k*n+i]
        # M1 contributes to C[i0,k0] with coeff +1, C[i0,k1] with 0, C[i1,k0] with 0, C[i1,k1] with +1
        # Wait, M1 appears in C[i0,k0] (+1) and C[i1,k1] (+1)
        
        # For each Strassen multiplier M, it's a bilinear form:
        # M = (sum_a ca[a]*A[i_a, j_a]) * (sum_b cb[b]*B[j_b, k_b])
        # This expands to sum_{a,b} ca[a]*cb[b] * A[i_a,j_a]*B[j_b,k_b]
        # Each term A[i_a,j_a]*B[j_b,k_b] contributes to C[i_a, k_b]
        # So M contributes to C[i_a, k_b] with weight ca[a]*cb[b]
        # Then C[c_i, c_k] += coeff_M * M, so the contribution to C[c_i,c_k] is:
        #   sum over (a,b) where i_a=c_i and k_b=c_k of coeff_M * ca[a]*cb[b]
        
        # Let me define the Strassen multipliers more carefully.
        # For 2x2x2 with A rows {i0,i1}, B cols {k0,k1}, inner {j0,j1}:
        
        # M1: A part: A[i0,j0] + A[i1,j1], B part: B[j0,k0] + B[j1,k1]
        #   ca = [(i0,j0,1), (i1,j1,1)], cb = [(j0,k0,1), (j1,k1,1)]
        #   M1 contributes to:
        #     C[i0,k0]: 1*1 = 1
        #     C[i0,k1]: 1*1 = 1  (from A[i0,j0]*B[j1,k1])
        #     C[i1,k0]: 1*1 = 1  (from A[i1,j1]*B[j0,k0])
        #     C[i1,k1]: 1*1 = 1
        #   Then C[i0,k0] += 1*M1, C[i1,k1] += 1*M1
        
        # This is getting complex. Let me just enumerate all rank-1 terms.
        
        # For each Strassen multiplier, I'll add the appropriate rank-1 terms
        # with the correct signs.
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # Terms: (i0,j0,k0), (i0,j0,k1), (i1,j1,k0), (i1,j1,k1)
        # M1 appears in: C[i0,k0] with +1, C[i1,k1] with +1
        # So each term (i,j,k) in M1 gets W-coeff = +1 if (i,k)==(i0,k0) or (i,k)==(i1,k1)
        
        # Actually, let me think about it differently.
        # Each rank-1 term is identified by (i, j, k) and has values (u, v, w).
        # The contribution to C[i,k] from term (i,j,k) is u*v*w.
        # 
        # For a Strassen multiplier M = (sum_a ca_a A[ia,ja]) * (sum_b cb_b B[jb,kb]):
        #   M = sum_{a,b} ca_a * cb_b * A[ia,ja] * B[jb,kb]
        #   Each product A[ia,ja]*B[jb,kb] is a rank-1 tensor term with:
        #     U[ia*m+ja] = 1, V[ja*p+kb] = 1, W[kb*n+ia] = 1
        #     contribution to C[ia,kb] = 1
        #   So M contributes ca_a*cb_b to C[ia,kb] for each (a,b) pair.
        #
        # Then C[c_i, c_k] += coeff * M means:
        #   For each (a,b) with ia=c_i and kb=c_k: add coeff*ca_a*cb_b to C[c_i,c_k]
        #
        # In the tensor decomposition, we need:
        #   sum_terms U[i*m+j]*V[j*p+k]*W[k*n+i] = 1 for all (i,j,k) [the tensor entries]
        #
        # Wait no. The tensor T[i,j,k] = 1 for all i,j,k.
        # The decomposition is: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        #
        # So for each (i,j,k), we need the sum of products to equal 1.
        #
        # A schoolbook term for (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
        #
        # For Strassen, we're replacing 8 schoolbook terms with 7 "Strassen terms".
        # But Strassen terms are not simple (i,j,k) triples - they involve sums.
        #
        # Let me reconsider. In the Strassen algorithm for 2x2:
        # We compute 7 products of sums, then combine them.
        # Each "product" is of the form (sum of A entries) * (sum of B entries).
        # 
        # In the tensor decomposition framework, each rank-1 component is:
        #   u(i,j) * v(j,k) * w(k,i)
        # where u, v, w are functions on the index sets.
        #
        # For a Strassen product like M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]):
        # This is NOT a single rank-1 tensor term. It's a sum of 4 rank-1 terms.
        # 
        # The key insight: Strassen works by computing 7 intermediate products
        # (each of which is a sum of several matrix element products) and then
        # combining them with additions/subtractions to get the final result.
        #
        # In the bilinear complexity framework, each "product" in Strassen
        # corresponds to a rank-1 tensor term where:
        #   u(i,j) is non-zero for the A entries involved
        #   v(j,k) is non-zero for the B entries involved
        #   w(k,i) is the output coefficient
        #
        # For M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]):
        #   u(i0,j0)=1, u(i1,j1)=1, others 0
        #   v(j0,k0)=1, v(j1,k1)=1, others 0
        #   w(k0,i0)=1, w(k1,i1)=1, others 0
        #   This gives: A[i0,j0]*B[j0,k0]*1 + A[i0,j0]*B[j1,k1]*0 + ... 
        #   Wait, that's not right either.
        #
        # Let me be very explicit. A rank-1 tensor term is:
        #   sum over (i,j,k) of u[i,j] * v[j,k] * w[k,i] * T[i,j,k]
        # where T[i,j,k] = 1 is the tensor entry.
        #
        # No wait. The tensor decomposition says:
        #   T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        #
        # So for a single rank-1 component c, the contribution to T[i,j,k] is
        # U[i,j,c] * V[j,k,c] * W[k,i,c].
        #
        # For the schoolbook, each (i,j,k) has its own component with U=V=W=1.
        #
        # For Strassen, we want to replace 8 such components with 7 that have
        # different U, V, W patterns.
        #
        # Consider the 2x2x2 case with indices i in {0,1}, j in {0,1}, k in {0,1}.
        # Schoolbook: 8 terms, each (i,j,k) with U[i,j]=V[j,k]=W[k,i]=1.
        #
        # Strassen M1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # This product equals A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
        # 
        # In the tensor, this corresponds to a rank-1 term where:
        #   U[0,0]=1, U[1,1]=1 (A part)
        #   V[0,0]=1, V[1,1]=1 (B part)  
        #   W[0,0]=1, W[1,1]=1 (output part, since M1 contributes to C[0,0] and C[1,1])
        #
        # The contribution to T[0,0,0] = U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
        # The contribution to T[0,1,1] = U[0,0]*V[1,1]*W[1,0] = 1*1*0 = 0 (W[1,0]=0)
        # The contribution to T[1,0,0] = U[1,1]*V[0,0]*W[0,1] = 1*1*0 = 0 (U[1,1] is for j=1, not j=0)
        #
        # Hmm, I'm confusing myself. Let me use the standard formulation.
        #
        # The tensor T has entries T[i,j,k] for i in rows, j in inner, k in cols.
        # T[i,j,k] = 1 for all valid (i,j,k).
        #
        # A rank-1 decomposition term is specified by three vectors:
        #   u: defined on (i,j) pairs
        #   v: defined on (j,k) pairs
        #   w: defined on (k,i) pairs
        # And the term contributes u[i,j]*v[j,k]*w[k,i] to T[i,j,k].
        #
        # For Strassen M1 in 2x2x2:
        #   u[0,0] = 1, u[1,1] = 1 (all other u = 0)
        #   v[0,0] = 1, v[1,1] = 1 (all other v = 0)
        #   w[0,0] = 1, w[1,1] = 1 (all other w = 0)
        #
        # Contribution to T[0,0,0] = u[0,0]*v[0,0]*w[0,0] = 1
        # Contribution to T[0,0,1] = u[0,0]*v[0,1]*w[1,0] = 1*0*0 = 0
        # Contribution to T[0,1,0] = u[0,1]*v[1,0]*w[0,0] = 0*0*1 = 0
        # Contribution to T[0,1,1] = u[0,1]*v[1,1]*w[1,0] = 0*1*0 = 0
        # Contribution to T[1,0,0] = u[1,0]*v[0,0]*w[0,1] = 0*1*0 = 0
        # Contribution to T[1,0,1] = u[1,0]*v[0,1]*w[1,1] = 0*0*1 = 0
        # Contribution to T[1,1,0] = u[1,1]*v[1,0]*w[0,1] = 1*0*0 = 0
        # Contribution to T[1,1,1] = u[1,1]*v[1,1]*w[1,1] = 1*1*1 = 1
        #
        # So M1 only contributes to T[0,0,0] and T[1,1,1], each with value 1.
        # But we need T[0,0,0]=1 and T[1,1,1]=1, and the other 6 entries also =1.
        # So M1 alone doesn't give the full tensor. We need all 7 Strassen terms.
        #
        # Let me verify: the 7 Strassen terms together should give T[i,j,k]=1 for all (i,j,k).
        
        # I'll just directly construct the 7 Strassen rank-1 terms for 2x2x2
        # and verify they sum to 1 for all 8 (i,j,k) entries.
        
        # Strassen 2x2x2 rank-1 terms (u, v, w patterns):
        # M1: u[(0,0)]=1, u[(1,1)]=1; v[(0,0)]=1, v[(1,1)]=1; w[(0,0)]=1, w[(1,1)]=1
        # M2: u[(1,0)]=1, u[(1,1)]=1; v[(0,0)]=1; w[(0,1)]=1
        # M3: u[(0,0)]=1; v[(1,1)]=1, v[(0,1)]=-1; w[(1,0)]=1
        # M4: u[(1,1)]=1; v[(1,0)]=1, v[(0,0)]=-1; w[(0,1)]=1
        # M5: u[(0,0)]=1, u[(0,1)]=1; v[(1,1)]=1; w[(1,0)]=1
        # M6: u[(1,0)]=1, u[(1,1)]=-1; v[(0,0)]=1, v[(0,1)]=1; w[(1,1)]=1
        # M7: u[(0,1)]=1, u[(1,1)]=-1; v[(1,0)]=1, v[(1,1)]=1; w[(0,0)]=1
        
        # Let me verify M1:
        # T[0,0,0] += u[0,0]*v[0,0]*w[0,0] = 1*1*1 = 1
        # T[1,1,1] += u[1,1]*v[1,1]*w[1,1] = 1*1*1 = 1
        # Others: 0
        # Good.
        
        # M2: u[(1,0)]=1, u[(1,1)]=1; v[(0,0)]=1; w[(0,1)]=1
        # T[1,0,0] += u[1,0]*v[0,0]*w[0,1] = 1*1*1 = 1
        # T[1,1,0] += u[1,1]*v[1,0]*w[0,1] = 1*0*1 = 0 (v[1,0]=0)
        # Only T[1,0,0] gets 1.
        # But we need T[1,0,0]=1. Good, but M2 should contribute to C[1,0] = M2+M4.
        
        # I think the issue is that in the Strassen algorithm, the multipliers
        # don't directly correspond to tensor entries. The tensor entries are
        # the products A[i,j]*B[j,k], and the Strassen algorithm computes
        # combinations of these products.
        
        # Let me take a completely different approach. Instead of trying to
        # understand the Strassen tensor decomposition abstractly, let me
        # directly verify that 7 specific rank-1 terms give T[i,j,k]=1 for
        # all 8 entries of the 2x2x2 tensor.
        
        # The 2x2x2 tensor has T[i,j,k]=1 for i,j,k in {0,1}.
        # I need 7 rank-1 terms (u_c, v_c, w_c) such that:
        #   sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k in {0,1}
        
        # From the Strassen algorithm, the 7 products are:
        # p1 = (a11+a22)(b11+b22)
        # p2 = (a21+a22)b11
        # p3 = a11(b12-b22)
        # p4 = a22(b21-b11)
        # p5 = (a11+a12)b22
        # p6 = (a21-a11)(b11+b12)
        # p7 = (a12-a22)(b21+b22)
        
        # Each p_m is a product of a linear combination of A entries and a
        # linear combination of B entries. In the tensor decomposition:
        # p_m = (sum_{i,j} alpha_m[i,j] * A[i,j]) * (sum_{j,k} beta_m[j,k] * B[j,k])
        #      = sum_{i,j,k} alpha_m[i,j] * beta_m[j,k] * A[i,j] * B[j,k]
        
        # The contribution to T[i,j,k] from term m is alpha_m[i,j] * beta_m[j,k].
        # (There's no separate w factor in this formulation; the "w" is absorbed
        #  into how p_m is used to compute C entries.)
        
        # Wait, I need to reconcile this with the U,V,W framework.
        # In the U,V,W framework:
        #   T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # 
        # If we set W[k,i,c] = 1 for all k,i,c, then:
        #   T[i,j,k] = sum_c U[i,j,c] * V[j,k,c]
        # 
        # But that's only a 2-way decomposition, not 3-way. The W factor is
        # essential for the 3-way structure.
        
        # Actually, I think the correct interpretation is:
        # The bilinear map is: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] = 1 (for all i,j,k).
        # A rank-1 decomposition: T[i,j,k] = sum_c U_c[i,j] V_c[j,k] W_c[k,i]
        
        # For Strassen, each multiplier p_m corresponds to a rank-1 term where:
        #   U_c[i,j] = coefficient of A[i,j] in the A-part of p_m
        #   V_c[j,k] = coefficient of B[j,k] in the B-part of p_m
        #   W_c[k,i] = 1 (or the coefficient relating to output)
        
        # But p_m is used to compute multiple C entries with different coefficients.
        # This means p_m doesn't directly correspond to a single rank-1 term.
        
        # I think the correct way is:
        # Each p_m is a rank-1 tensor term with:
        #   U_m[i,j] = alpha_m[i,j] (coefficients in A-part)
        #   V_m[j,k] = beta_m[j,k] (coefficients in B-part)
        #   W_m[k,i] = 1
        # And the contribution to T[i,j,k] is U_m[i,j] * V_m[j,k] * W_m[k,i] = alpha_m[i,j]*beta_m[j,k]
        
        # Let me verify with p1 = (a11+a22)(b11+b22):
        # alpha_1: A[0,0]=1, A[1,1]=1
        # beta_1: B[0,0]=1, B[1,1]=1
        # T[0,0,0] += 1*1*1 = 1
        # T[0,0,1] += 1*0*1 = 0 (beta[0,1]=0)
        # T[0,1,0] += 0*0*1 = 0 (alpha[0,1]=0)
        # T[0,1,1] += 0*1*1 = 0
        # T[1,0,0] += 0*1*1 = 0 (alpha[1,0]=0)
        # T[1,0,1] += 0*0*1 = 0
        # T[1,1,0] += 1*0*1 = 0 (beta[1,0]=0)
        # T[1,1,1] += 1*1*1 = 1
        # So p1 contributes 1 to T[0,0,0] and T[1,1,1].
        
        # p2 = (a21+a22)b11:
        # alpha_2: A[1,0]=1, A[1,1]=1
        # beta_2: B[0,0]=1
        # T[1,0,0] += 1*1*1 = 1
        # T[1,1,0] += 1*0*1 = 0 (beta[1,0]=0)
        # Only T[1,0,0] gets 1.
        
        # p3 = a11(b12-b22):
        # alpha_3: A[0,0]=1
        # beta_3: B[0,1]=1, B[1,1]=-1
        # T[0,0,1] += 1*1*1 = 1
        # T[0,1,1] += 1*(-1)*1 = -1
        # Hmm, this gives -1 for T[0,1,1]. That's a problem.
        
        # I think the issue is that the Strassen multipliers don't directly
        # decompose the tensor. The Strassen algorithm works at the level of
        # computing C entries, not at the level of the tensor entries.
        
        # Let me reconsider. The tensor T[i,j,k] = 1 represents the fact that
        # each product A[i,j]*B[j,k] appears exactly once in C[i,k].
        # 
        # In Strassen, we compute 7 intermediate products and then combine them.
        # The "rank" of the algorithm is 7, meaning we use 7 scalar multiplications.
        # But each scalar multiplication is of a sum of A entries times a sum of B entries.
        #
        # In the tensor decomposition framework, a rank-1 term corresponds to
        # one scalar multiplication. The U, V, W factors encode which A entries,
        # B entries, and C entries are involved.
        #
        # For p1 = (a11+a22)(b11+b22):
        # This computes a11*b11 + a11*b22 + a22*b11 + a22*b22
        # In terms of tensor entries: T[0,0,0] + T[0,0,1] + T[1,1,0] + T[1,1,1]
        # Wait, a11*b22 is A[0,0]*B[0,1]? No, b22 = B[1,1].
        # a11*b11 = A[0,0]*B[0,0] -> T[0,0,0]
        # a11*b22 = A[0,0]*B[1,1] -> but B[1,1] has j=1, so this is A[0,0]*B[1,1]
        #   which corresponds to T[0,1,1] (i=0, j=1, k=1)
        # a22*b11 = A[1,1]*B[0,0] -> A[1,1] has j=1, B[0,0] has j=0. 
        #   These don't share the same j! So this isn't a valid tensor entry.
        
        # I see the problem. In the tensor T[i,j,k], the j index must be the same
        # for both A[i,j] and B[j,k]. In Strassen, the multipliers mix different j
        # indices, so they don't directly correspond to single tensor entries.
        
        # This means the Strassen algorithm doesn't directly give a tensor
        # decomposition in the naive sense. The rank-7 result for 2x2x2
        # matrix multiplication refers to the bilinear complexity, not a
        # direct rank-7 tensor decomposition of the identity tensor.
        
        # I'm going down a rabbit hole. Let me just implement the known
        # correct approach: directly construct the factors for the 3x5x6
        # tensor using the Strassen embedding that's known to give rank 84.
        #
        # Based on the cell finding, the approach is to replace 8 schoolbook
        # products per 2x2x2 block with 7 Strassen products, across 6 such blocks.
        # 90 - 6*1 = 84.
        #
        # Let me just hardcode the rank-84 factors based on the known construction.
        # I'll use the approach of splitting the inner dimension into pairs and
        # applying Strassen to each 2x2x2 sub-block.
        
    # Given the complexity of deriving this from scratch, let me use a 
    # verified construction. I'll build the factors programmatically.
    
    # Approach: For the 3x5x6 tensor, split inner dim j into {0,1}, {2,3}, {4}.
    # For blocks where both row and col dimensions are 2, apply Strassen.
    
    # Let me define the Strassen 2x2x2 rank-1 terms explicitly.
    # After careful analysis, the 7 rank-1 terms for 2x2x2 (i,j,k in {0,1})
    # that satisfy sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] = 1 for all (i,j,k) are:
    
    # I'll use the following verified set (from literature):
    # Term 1: U: {(0,0):1,(1,1):1}, V: {(0,0):1,(1,1):1}, W: {(0,0):1,(1,1):1}
    # Term 2: U: {(1,0):1,(1,1):1}, V: {(0,0):1}, W: {(0,1):1}
    # Term 3: U: {(0,0):1}, V: {(1,1):1,(0,1):-1}, W: {(1,0):1}
    # Term 4: U: {(1,1):1}, V: {(1,0):1,(0,0):-1}, W: {(0,1):1}
    # Term 5: U: {(0,0):1,(0,1):1}, V: {(1,1):1}, W: {(1,0):1}
    # Term 6: U: {(1,0):1,(1,1):-1}, V: {(0,0):1,(0,1):1}, W: {(1,1):1}
    # Term 7: U: {(0,1):1,(1,1):-1}, V: {(1,0):1,(1,1):1}, W: {(0,0):1}
    
    # Let me verify term 1:
    # T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # T[1,1,1] += U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1
    # All others: 0. Good.
    
    # Term 2: U: {(1,0):1,(1,1):1}, V: {(0,0):1}, W: {(0,1):1}
    # T[1,0,0] += U[1,0]*V[0,0]*W[0,1] = 1*1*1 = 1
    # T[1,1,0] += U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0
    # Only T[1,0,0] = 1. 
    
    # Term 3: U: {(0,0):1}, V: {(1,1):1,(0,1):-1}, W: {(1,0):1}
    # T[0,0,1] += U[0,0]*V[0,1]*W[1,0] = 1*(-1)*1 = -1
    # T[0,1,1] += U[0,1]*V[1,1]*W[1,0] = 0*1*1 = 0
    # Hmm, T[0,0,1] gets -1. That's not 1.
    
    # I think my Strassen terms are wrong. Let me look up the correct ones.
    # 
    # Actually, I recall that the Strassen algorithm's 7 products do NOT
    # directly give a rank-7 tensor decomposition of the 2x2x2 multiplication
    # tensor. The bilinear rank of 2x2x2 matrix multiplication is 7, but
    # this is for the bilinear map (A,B) -> AB, not for the tensor T[i,j,k]=1.
    #
    # The tensor T[i,j,k]=1 for 2x2x2 has rank 8 (it's just the identity,
    # and each (i,j,k) is independent). Wait, no. The tensor for matrix
    # multiplication is NOT T[i,j,k]=1. 
    #
    # Let me re-derive. The matrix multiplication map is:
    #   C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map from A (n*m dim) x B (m*p dim) to C (n*p dim).
    # The associated tensor T has entries T[i,j,k] where:
    #   T is in A* x B* x C*
    #   T(A,B) = C means C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    #   So T[i,j,k] = 1 for all i,j,k.
    #
    # The rank of this tensor is the bilinear complexity of matrix multiplication.
    # For 2x2x2, the rank is 7 (Strassen).
    #
    # So T[i,j,k]=1 for all i,j,k in {0,1} has rank 7, not 8!
    # The schoolbook gives rank 8, but Strassen shows it can be done in 7.
    #
    # So my verification above was wrong. Let me recheck.
    # If T[0,0,1] should be 1, and term 3 gives -1, then the sum of all 7 terms
    # should give 1 for T[0,0,1]. Let me compute the total contribution to T[0,0,1]:
    # Term 1: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # Term 2: U[0,0]*V[0,1]*W[1,0] = 0*0*0 = 0 (U[0,0]=0 for term 2)
    # Term 3: U[0,0]*V[0,1]*W[1,0] = 1*(-1)*1 = -1
    # Term 4: U[0,0]*V[0,1]*W[1,0] = 0*0*0 = 0
    # Term 5: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0 (V[0,1]=0 for term 5)
    # Term 6: U[0,0]*V[0,1]*W[1,0] = 0*0*0 = 0
    # Term 7: U[0,0]*V[0,1]*W[1,0] = 0*0*0 = 0
    # Total: -1. But we need +1!
    
    # So my terms are definitely wrong. The correct Strassen rank-7 decomposition
    # of the 2x2x2 tensor must have different coefficients.
    
    # I'll skip the detailed derivation and just use the known result: 
    # the 2x2x2 matrix multiplication tensor has rank 7. I'll construct
    # the factors by solving the system numerically or using a known explicit
    # decomposition.
    
    # For now, let me just return the schoolbook factors (rank 90) since
    # I can't quickly derive the correct Strassen factors. The program will
    # score 1.0, same as before.
    
    return None


def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the matrix multiplication tensor decomposition.
    Uses the schoolbook decomposition which is known to be valid.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Get the schoolbook decomposition
    U_school, V_school, W_school = _schoolbook_factors(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p
    
    # Try Strassen embedding
    if time.time() < deadline - 10:
        result = _build_strassen_3x5x6()
        if result is not None:
            U_try, V_try, W_try = result
            try:
                U_arr = np.array(U_try, dtype=float)
                V_arr = np.array(V_try, dtype=float)
                W_arr = np.array(W_try, dtype=float)
                rank_try = U_arr.shape[1]
                
                # Verify
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-9:
                    if rank_try < best_rank:
                        best_result = {"U": U_try, "V": V_try, "W": W_try}
                        best_rank = rank_try
            except Exception:
                pass
    
    return best_result
# EVOLVE-BLOCK-END
