"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Decompose into 8 independent 2x2*2x2 Strassen products.
    Each 2x2*2x2 multiplication has rank 7 (Strassen), so total rank = 8*7 = 56.
    
    The 2x4 matrix A is split into 4 blocks of 2x2: A = [A00 A01; A10 A11]
    Wait, 2x4 means 2 rows, 4 columns. Let's think in terms of the tensor.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Here n=2, m=4, p=8.
    
    We can view this as 4 independent 2x2 * 2x2 multiplications if we group appropriately.
    Actually, let's think of it differently.
    
    A is 2x4: rows i=0,1; cols j=0,1,2,3
    B is 4x8: rows j=0,1,2,3; cols k=0,...,7
    C is 2x8: rows i=0,1; cols k=0,...,7
    
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can split the j dimension into 2 groups of 2:
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    So C[i,k] = C1[i,k] + C2[i,k]
    where C1 = A1 * B1 (A1 is 2x2, B1 is 2x2) and C2 = A2 * B2 (A2 is 2x2, B2 is 2x2)
    
    But wait, B1 has columns k=0..7, so B1 is actually 2x8, not 2x2.
    
    Let me reconsider. We can split the output columns k into 4 groups of 2:
    For each group g (g=0,1,2,3), columns k=2g and k=2g+1.
    
    C[i, 2g] = sum_j A[i,j] * B[j, 2g]
    C[i, 2g+1] = sum_j A[i,j] * B[j, 2g+1]
    
    So for each g, we have a 2x2 output (rows i=0,1; cols 2g, 2g+1) computed from
    A (2x4) and B_g (4x2, columns 2g and 2g+1 of B).
    
    This is a 2x4 * 4x2 multiplication producing 2x2. This isn't directly Strassen.
    
    Alternative: Split both j and k.
    Let's split j into 2 groups of 2, and k into 4 groups of 2.
    
    Actually, the cleanest way: Think of the 4x4 middle as 2x2 blocks.
    
    Let me use a different approach. The tensor T[i,j,k] = A[i,j]*B[j,k] for C[i,k].
    We want to decompose this into rank-1 tensors.
    
    For Strassen on 2x2*2x2:
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    Strassen:
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-d)(g+h)
    m7 = (b-d)(e+f)
    
    C[0,0] = m1 - m4 + m5 - m7
    C[0,1] = m3 + m5
    C[1,0] = m2 + m4
    C[1,1] = m1 - m2 + m3 + m6
    
    Now for 2x4 * 4x8:
    A[i,j] for i in {0,1}, j in {0,1,2,3}
    B[j,k] for j in {0,1,2,3}, k in {0,...,7}
    
    Split j into two halves: j1 = {0,1}, j2 = {2,3}
    Split k into four quarters: k1={0,1}, k2={2,3}, k3={4,5}, k4={6,7}
    
    C[i,k] = sum_{j in j1} A[i,j]*B[j,k] + sum_{j in j2} A[i,j]*B[j,k]
    
    For each (half of j, quarter of k) pair, we have a 2x2 * 2x2 multiplication:
    A_half is 2x2 (rows i, cols in j_half)
    B_quarter is 2x2 (rows in j_half, cols in k_quarter)
    
    There are 2 j-halves * 4 k-quarters = 8 such 2x2*2x2 products.
    Each has rank 7 via Strassen.
    Total rank = 8 * 7 = 56.
    
    Let's construct this.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Split j into 2 groups of 2: J0={0,1}, J1={2,3}
    # Split k into 4 groups of 2: K0={0,1}, K1={2,3}, K2={4,5}, K3={6,7}
    
    j_groups = [[0, 1], [2, 3]]
    k_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    
    # Strassen coefficients for 2x2*2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(g+h)
    # m7 = (b-d)(e+f)
    
    # For each (jg, kg) pair, we have a 2x2*2x2 subproblem.
    # A_sub[i, ja] = A[i, jg[ja]] for i in {0,1}, ja in {0,1}
    # B_sub[jb, kb] = B[jg[jb], kg[kb]] for jb in {0,1}, kb in {0,1}
    # C_sub[i, kb] = C[i, kg[kb]]
    
    # Strassen factorizations:
    # Each m_t is a product of linear combinations of A_sub entries and B_sub entries.
    # We need to express each m_t as U_vec * V_vec where U_vec is linear in A entries, V_vec in B entries.
    
    # For a single 2x2*2x2:
    # m1 = (a+d)(e+h): U = [1,0,1,0] (for a,b,c,d), V = [1,1,0,0] (for e,f,g,h)
    #   Actually: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    #   e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    #   m1 = (a+d)*(e+h) = (1*a + 0*b + 0*c + 1*d) * (1*e + 1*f + 0*g + 0*h)
    #   U_vec = [1,0,0,1], V_vec = [1,1,0,0]
    # m2 = (c+d)*e = (0*a + 0*b + 1*c + 1*d) * (1*e + 0*f + 0*g + 0*h)
    #   U_vec = [0,0,1,1], V_vec = [1,0,0,0]
    # m3 = a*(f-h) = (1*a + 0*b + 0*c + 0*d) * (0*e + 1*f + 0*g + -1*h)
    #   U_vec = [1,0,0,0], V_vec = [0,1,0,-1]
    # m4 = d*(g-e) = (0*a + 0*b + 0*c + 1*d) * (-1*e + 0*f + 1*g + 0*h)
    #   U_vec = [0,0,0,1], V_vec = [-1,0,1,0]
    # m5 = (a+b)*h = (1*a + 1*b + 0*c + 0*d) * (0*e + 0*f + 0*g + 1*h)
    #   U_vec = [1,1,0,0], V_vec = [0,0,0,1]
    # m6 = (c-d)*(g+h) = (0*a + 0*b + 1*c + -1*d) * (0*e + 0*f + 1*g + 1*h)
    #   U_vec = [0,0,1,-1], V_vec = [0,0,1,1]
    # m7 = (b-d)*(e+f) = (0*a + 1*b + 0*c + -1*d) * (1*e + 1*f + 0*g + 0*h)
    #   U_vec = [0,1,0,-1], V_vec = [1,1,0,0]
    
    # C[0,0] = m1 - m4 + m5 - m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    strassen_U = [
        [1, 0, 0, 1],  # m1
        [0, 0, 1, 1],  # m2
        [1, 0, 0, 0],  # m3
        [0, 0, 0, 1],  # m4
        [1, 1, 0, 0],  # m5
        [0, 0, 1, -1], # m6
        [0, 1, 0, -1], # m7
    ]
    
    strassen_V = [
        [1, 1, 0, 0],  # m1
        [1, 0, 0, 0],  # m2
        [0, 1, 0, -1], # m3
        [-1, 0, 1, 0], # m4
        [0, 0, 0, 1],  # m5
        [0, 0, 1, 1],  # m6
        [1, 1, 0, 0],  # m7
    ]
    
    strassen_W = [
        # W[k*n+i, c] for the 2x2 output
        # C[0,0] = m1 - m4 + m5 - m7
        # C[0,1] = m3 + m5
        # C[1,0] = m2 + m4
        # C[1,1] = m1 - m2 + m3 + m6
        # For k in {0,1}, i in {0,1}:
        # C[i,k] -> index k*2 + i
        # C[0,0]: k=0,i=0 -> index 0
        # C[0,1]: k=1,i=0 -> index 2
        # C[1,0]: k=0,i=1 -> index 1
        # C[1,1]: k=1,i=1 -> index 3
        # W[0] = C[0,0] = m1 - m4 + m5 - m7
        # W[1] = C[1,0] = m2 + m4
        # W[2] = C[0,1] = m3 + m5
        # W[3] = C[1,1] = m1 - m2 + m3 + m6
        [1, 0, 0, -1, 1, 0, -1],  # W[0] = C[0,0]
        [0, 1, 0, 1, 0, 0, 0],    # W[1] = C[1,0]
        [0, 0, 1, 0, 1, 0, 0],    # W[2] = C[0,1]
        [1, -1, 1, 0, 0, 1, 0],   # W[3] = C[1,1]
    ]
    
    rank = 8 * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 8*2 = 16 rows
    
    col = 0
    for jg_idx, jg in enumerate(j_groups):
        for kg_idx, kg in enumerate(k_groups):
            for t in range(7):
                # U: A[i, j] for i in {0,1}, j in jg
                # A_sub[i, ja] where ja is index within jg
                for i in range(n):  # i in {0,1}
                    for ja in range(2):
                        j = jg[ja]
                        U[i * m + j][col] = strassen_U[t][i * 2 + ja]
                
                # V: B[j, k] for j in jg, k in kg
                # B_sub[jb, kb] where jb is index within jg, kb is index within kg
                for jb in range(2):
                    j = jg[jb]
                    for kb in range(2):
                        k = kg[kb]
                        V[j * p + k][col] = strassen_V[t][jb * 2 + kb]
                
                # W: C[i, k] for i in {0,1}, k in kg
                # C_sub[i, kb]
                for i in range(n):  # i in {0,1}
                    for kb in range(2):
                        k = kg[kb]
                        W[k * n + i][col] = strassen_W[t][kb * 2 + i]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
