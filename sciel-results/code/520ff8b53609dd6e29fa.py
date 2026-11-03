"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import numpy as np
import time
import itertools
from scipy.optimize import linprog

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy:
    - Use a known construction: split the 6 output columns into groups.
    - For the first 4 columns: use a 2x4x4 decomposition.
      The 2x4x4 can be done by splitting the 4 middle indices into two groups of 2,
      giving two 2x2x2 blocks. Each 2x2x2 uses Strassen (rank 7), so total 14.
    - For the last 2 columns: use a 2x4x2 decomposition.
      The 2x4x2 can be done by splitting the 4 middle indices into two groups of 2,
      giving two 2x2x2 blocks (but the output is 2x2 each). Each 2x2x2 uses Strassen (rank 7),
      so total 14.
    - Total rank: 14 + 14 = 28.
    
    However, we can potentially do better by using a more global approach.
    Let's construct the decomposition explicitly.
    
    Tensor: T_{i,j,k} = sum over middle indices.
    A is 2x4 (indices i=0,1; j=0,1,2,3)
    B is 4x6 (indices j=0,1,2,3; k=0,1,2,3,4,5)
    C is 2x6 (indices i=0,1; k=0,1,2,3,4,5)
    
    C_{i,k} = sum_{j=0}^{3} A_{i,j} * B_{j,k}
    
    We want to decompose this into rank-r terms.
    Each term r corresponds to:
      U_{i,j}^{r} * V_{j,k}^{r} * W_{k,i}^{r}  (summed over r)
    
    Actually, the standard form is:
      sum_r U_r ⊗ V_r ⊗ W_r = T
    
    where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}.
    
    Let's use the following construction:
    
    For k in {0,1,2,3}:
      C_{i,k} = sum_{j=0}^{3} A_{i,j} B_{j,k}
    
    Split j into {0,1} and {2,3}.
    For the first pair (j=0,1):
      C_{i,k} += A_{i,0} B_{0,k} + A_{i,1} B_{1,k}
    
    This is a 2x2x2 sub-problem with A part [A_{i,0}, A_{i,1}], B part [B_{0,k}, B_{1,k}],
    and C part [C_{i,k}].
    
    Using Strassen for 2x2x2, we need 7 terms per (j-group, k-group).
    
    For k in {0,1}: one 2x2x2 (7 terms)
    For k in {2,3}: one 2x2x2 (7 terms)
    For j in {0,1}: multiply the above
    For j in {2,3}: multiply the above
    
    So for k in {0,1,2,3}: 2 (j-groups) * 2 (k-groups) * 7 = 28 terms? No, that's not right.
    
    Let me reconsider. The total computation is:
      C_{i,k} = sum_{j=0}^{3} A_{i,j} B_{j,k}
    
    We can view this as: for each k, we have a 2x4 matrix multiplication (row vector A_i * B_{*,k}).
    
    Alternative: use bilinear rank.
    
    Let's just use the schoolbook method but optimize.
    Schoolbook: n*m*p = 2*4*6 = 48 terms.
    
    We want to find a better decomposition.
    
    Known results: 
      - 2x2x2: rank 7 (Strassen)
      - 2x2x3: rank ?
      - 2x3x3: rank ?
    
    For 2x4x6, we can write it as:
      C = A @ B
    
    where A is 2x4, B is 4x6.
    
    We can use block Strassen. Write A = [A1 A2] where A1, A2 are 2x2.
    Write B = [B1; B2] where B1, B2 are 2x6.
    
    C = A1 B1 + A2 B2
    
    Each 2x2x6 product can be computed using Strassen-like methods.
    
    For 2x2x6: A1 (2x2) * B1 (2x6).
    C1 = A1 B1, where C1 is 2x6.
    C1_{i,k} = sum_{j=0,1} A1_{i,j} B1_{j,k}
    
    For each k, this is a 2x2x1 product. Using Strassen for 2x2x2, we can handle 2 columns at a time.
    So for 6 columns, we need 3 Strassen blocks of 2x2x2.
    Each block has rank 7.
    So 2x2x6 has rank 3*7 = 21? No, that's not right either.
    
    Actually, for 2x2x6, we can use:
      C1_{i,k} = A1_{i,0} B1_{0,k} + A1_{i,1} B1_{1,k}
    
    This is a sum of 2 products for each (i,k). The total bilinear rank is at most 2*2*6 = 24.
    Using Strassen for each pair of columns: 3 * 7 = 21.
    
    Similarly for A2 B2: 21.
    
    Total: 42. That's the same as the seed.
    
    Can we do better? Let's think about the structure.
    
    Actually, the minimum rank for 2x4x6 might be less than 42.
    
    Let's try a different approach: use the fact that n=2 is small.
    
    For each k, C_{:,k} = A @ B_{:,k} where A is 2x4 and B_{:,k} is 4x1.
    
    This is a linear map from R^4 to R^2 for each k.
    
    The bilinear rank of the 2x4x6 tensor is what we want.
    
    Known: rank(2x4x4) <= 14 (using two 2x2x2 Strassen blocks for the middle dimension split into 2+2, and the output split into 2+2).
    Wait, 2x4x4: A is 2x4, B is 4x4.
    C_{i,k} = sum_{j=0}^{3} A_{i,j} B_{j,k}
    
    Split j into {0,1} and {2,3}.
    C_{i,k} = sum_{j=0,1} A_{i,j} B_{j,k} + sum_{j=2,3} A_{i,j} B_{j,k}
    
    For the first sum, for each k, we have a 2x2x1 product.
    Using Strassen for 2x2x2, we can handle 2 k-values at a time.
    So for k in {0,1}: 7 terms, for k in {2,3}: 7 terms. Total 14 for the first j-group.
    Similarly for the second j-group: 14 terms.
    Total: 28 for 2x4x4.
    
    But we can do better: use Strassen on the 2x4x4 by viewing it as two 2x2x4 products.
    2x2x4: A (2x2) * B (2x4).
    C_{i,k} = sum_{j=0,1} A_{i,j} B_{j,k}
    
    For each pair of k's, use Strassen: 7 terms per pair.
    4 k's = 2 pairs, so 14 terms for 2x2x4.
    Two such products (for j-group 1 and j-group 2): 28 terms.
    
    Hmm, same.
    
    Let's just implement the rank-42 solution correctly and verify it.
    Then try to see if we can reduce it.
    """
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # We'll use the block Strassen approach with rank 42.
    # But first, let's verify the correct Strassen coefficients.
    
    # Strassen for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C11 = m1 - m4 + m5 - m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m6 - m7
    
    # In our indexing:
    # A indices: (i, j) where i in {0,1}, j in {0,1}
    # B indices: (j, k) where j in {0,1}, k in {0,1}
    # C indices: (i, k) where i in {0,1}, k in {0,1}
    
    # U coefficients for each m_r:
    # m1: a + d -> (0,0):1, (1,1):1
    # m2: c + d -> (1,0):1, (1,1):1
    # m3: a -> (0,0):1
    # m4: d -> (1,1):1
    # m5: a + b -> (0,0):1, (0,1):1
    # m6: c - a -> (1,0):1, (0,0):-1
    # m7: b - d -> (0,1):1, (1,1):-1
    
    # V coefficients for each m_r:
    # m1: e + h -> (0,0):1, (1,1):1
    # m2: e -> (0,0):1
    # m3: f - h -> (0,1):1, (1,1):-1
    # m4: g - f -> (1,0):1, (0,1):-1
    # m5: h -> (1,1):1
    # m6: e + f -> (0,0):1, (0,1):1
    # m7: g + h -> (1,0):1, (1,1):1
    
    # W coefficients for each m_r (C = sum m_r * W_r):
    # m1: C11 + C22 -> (0,0):1, (1,1):1
    # m2: C21 -> (1,0):1
    # m3: C12 - C22 -> (0,1):1, (1,1):-1
    # m4: C11 + C21 -> (0,0):1, (1,0):1
    # m5: C11 + C12 -> (0,0):1, (0,1):1
    # m6: C22 - C21 -> (1,1):1, (1,0):-1
    # m7: C11 - C12 + C21 - C22 -> (0,0):1, (0,1):-1, (1,0):1, (1,1):-1
    
    U_coeff = [
        {(0,0):1, (1,1):1},   # m1
        {(1,0):1, (1,1):1},   # m2
        {(0,0):1},             # m3
        {(1,1):1},             # m4
        {(0,0):1, (0,1):1},   # m5
        {(1,0):1, (0,0):-1},  # m6
        {(0,1):1, (1,1):-1},  # m7
    ]
    
    V_coeff = [
        {(0,0):1, (1,1):1},   # m1
        {(0,0):1},             # m2
        {(0,1):1, (1,1):-1},  # m3
        {(1,0):1, (0,1):-1},  # m4
        {(1,1):1},             # m5
        {(0,0):1, (0,1):1},   # m6
        {(1,0):1, (1,1):1},   # m7
    ]
    
    W_coeff = [
        {(0,0):1, (1,1):1},   # m1
        {(1,0):1},             # m2
        {(0,1):1, (1,1):-1},  # m3
        {(0,0):1, (1,0):1},   # m4
        {(0,0):1, (0,1):1},   # m5
        {(1,1):1, (1,0):-1},  # m6
        {(0,0):1, (0,1):-1, (1,0):1, (1,1):-1},  # m7
    ]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42
    
    col = 0
    
    # Part 1: k in {0,1,2,3}
    # Split j into {0,1} and {2,3}
    # For each j-group and each k-pair {0,1} or {2,3}, use Strassen
    
    for j_group in range(2):  # 0: j in {0,1}, 1: j in {2,3}
        for k_group in range(2):  # 0: k in {0,1}, 1: k in {2,3}
            for r in range(7):
                u_c = U_coeff[r]
                v_c = V_coeff[r]
                w_c = W_coeff[r]
                
                for (li, lj), coeff in u_c.items():
                    i = li
                    j = 2 * j_group + lj
                    U[i * m + j][col] += coeff
                
                for (lj2, lk), coeff in v_c.items():
                    j = 2 * j_group + lj2
                    k = 2 * k_group + lk
                    V[j * p + k][col] += coeff
                
                for (li2, lk2), coeff in w_c.items():
                    i = li2
                    k = 2 * k_group + lk2
                    W[k * n + i][col] += coeff
                
                col += 1
    
    # Part 2: k in {4,5}
    # Split j into {0,1} and {2,3}
    # For each j-group, use Strassen for 2x2x2 (k in {4,5})
    
    for j_group in range(2):  # 0: j in {0,1}, 1: j in {2,3}
        for r in range(7):
            u_c = U_coeff[r]
            v_c = V_coeff[r]
            w_c = W_coeff[r]
            
            for (li, lj), coeff in u_c.items():
                i = li
                j = 2 * j_group + lj
                U[i * m + j][col] += coeff
            
            for (lj2, lk), coeff in v_c.items():
                j = 2 * j_group + lj2
                k = 4 + lk
                V[j * p + k][col] += coeff
            
            for (li2, lk2), coeff in w_c.items():
                i = li2
                k = 4 + lk2
                W[k * n + i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
