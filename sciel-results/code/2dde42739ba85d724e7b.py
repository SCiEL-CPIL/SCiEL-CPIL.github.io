import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We will build a decomposition of rank 56.
    # Rank 20 for A2 B
    # Rank 14 for A1 B2
    # Rank 14 for A1 B11
    # Rank 8 for A1 B12
    # Total 56.
    
    U = [[0] * 56 for _ in range(nm)]
    V = [[0] * 56 for _ in range(mp)]
    W = [[0] * 56 for _ in range(pn)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1

    # 1. A2 B: i=2, j=0..3, k=0..4
    # Rank 20
    for j in range(4):
        for k in range(5):
            add_term(2, j, k, 1, 1, 1)
            
    # 2. A1 B2: i=0,1, j=0..3, k=3,4
    # Split j into {0,1} and {2,3}
    # Block 1: i in {0,1}, j in {0,1}, k in {3,4}
    # Strassen
    # Map local indices (li, lj, lk) to global (i, j, k)
    # li=0->i=0, li=1->i=1
    # lj=0->j=0, lj=1->j=1
    # lk=0->k=3, lk=1->k=4
    
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to express these as sums of U, V, W terms.
        # Each M is a product of a linear combo of A and a linear combo of B.
        # We expand each M into basic terms.
        
        # M1
        # (A[i_base+0, j_base+0] + A[i_base+1, j_base+1]) * (B[j_base+0, k_base+0] + B[j_base+1, k_base+1])
        # = A00 B00 + A00 B11 + A11 B00 + A11 B11
        # But Strassen uses pre-computed products.
        # We add terms for each multiplication in the Strassen algorithm.
        
        # We will add 7 terms for the 7 multiplications.
        # Then we add the additions for C.
        # Wait, the tensor decomposition requires U, V, W such that sum U V W = C.
        # Strassen defines C as linear combo of Ms.
        # So we need to represent each M as a rank-1 term?
        # No, M is a product of two linear forms.
        # M1 = (A00+A11)(B00+B11).
        # This is NOT a rank-1 term in the tensor sense if we consider A and B as variables.
        # The tensor decomposition requires $C_{ik} = \sum_c U_{i,j,c} V_{j,k,c} W_{i,k,c}$.
        # Here $j$ is summed.
        # In Strassen, we compute $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This $M_1$ is then used in $C_{00} = M_1 + \dots$.
        # So we need to decompose the computation of $M_1$ and then the additions.
        # But the additions are free? No, the additions are part of the U, V, W coefficients.
        # Actually, the tensor rank counts the number of scalar multiplications.
        # The additions are absorbed into the coefficients of U, V, W.
        
        # So for each Strassen step, we add 7 terms.
        # The coefficients in U and V will be the linear combinations.
        
        # Let's define the 7 multiplications.
        # 1. (A00+A11) * (B00+B11)
        # 2. (A10+A11) * B00
        # 3. A00 * (B10-B11)
        # 4. A11 * (B10-B00)
        # 5. (A00+A01) * B11
        # 6. (A10-A00) * (B00+B10)
        # 7. (A01-A11) * (B10+B11)
        
        # We need to map these to U, V, W.
        # For term 1:
        # U part: A00 + A11. This involves j=0 and j=1.
        # U[i_base+0, j_base+0] += 1
        # U[i_base+1, j_base+1] += 1
        # V part: B00 + B11. This involves j=0 and j=1.
        # V[j_base+0, k_base+0] += 1
        # V[j_base+1, k_base+1] += 1
        # W part: This product contributes to C00, C01, C10, C11.
        # In Strassen, M1 is added to C00 and C11.
        # C00 += M1
        # C11 += M1
        # So W[k_base+0, i_base+0] += 1
        # W[k_base+1, i_base+1] += 1
        
        # We will implement this logic.
        
        # Term 1
        U[i_base * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # Term 2: (A10+A11) * B00
        # A part: A10 (i=1, j=0) + A11 (i=1, j=1)
        U[(i_base + 1) * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        # B part: B00 (j=0, k=0)
        V[j_base * p + k_base][col] += 1
        # W part: C10 += M2
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # Term 3: A00 * (B10-B11)
        # A part: A00 (i=0, j=0)
        U[i_base * m + j_base][col] += 1
        # B part: B10 (j=1, k=0) - B11 (j=1, k=1)
        V[(j_base + 1) * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] -= 1
        # W part: C01 += M3, C11 += M3
        W[k_base * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # Term 4: A11 * (B10-B00)
        # A part: A11 (i=1, j=1)
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        # B part: B10 (j=1, k=0) - B00 (j=0, k=0)
        V[(j_base + 1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        # W part: C00 += M4, C10 += M4
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # Term 5: (A00+A01) * B11
        # A part: A00 (i=0, j=0) + A01 (i=0, j=1)
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base + 1)][col] += 1
        # B part: B11 (j=1, k=1)
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        # W part: C00 -= M5, C01 += M5
        W[k_base * n + i_base][col] -= 1
        W[k_base * n + i_base][col] += 1 # Wait, C00 has -M5, C01 has +M5.
        # C00 index: k=0, i=0. W[0, 0] -= 1
        # C01 index: k=1, i=0. W[1, 0] += 1
        # Correction:
        # W[k_base * n + i_base][col] -= 1
        # W[(k_base + 1) * n + i_base][col] += 1
        # I already added 1 to W[k_base * n + i_base] above? No, I am in a new term.
        # So:
        W[k_base * n + i_base][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        
        # Term 6: (A10-A00) * (B00+B10)
        # A part: A10 (i=1, j=0) - A00 (i=0, j=0)
        U[(i_base + 1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        # B part: B00 (j=0, k=0) + B10 (j=1, k=0)
        V[j_base * p + k_base][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        # W part: C11 += M6
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # Term 7: (A01-A11) * (B10+B11)
        # A part: A01 (i=0, j=1) - A11 (i=1, j=1)
        U[i_base * m + (j_base + 1)][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] -= 1
        # B part: B10 (j=1, k=0) + B11 (j=1, k=1)
        V[(j_base + 1) * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        # W part: C00 += M7
        W[k_base * n + i_base][col] += 1
        col += 1

    add_strassen(0, 0, 3)
    add_strassen(0, 2, 3)
    
    # 3. A1 B11: i=0,1, j=0..3, k=0,1
    # Split j into {0,1} and {2,3}
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 0)
    
    # 4. A1 B12: i=0,1, j=0..3, k=2
    # Split A into A11 (j=0,1) and A12 (j=2,3)
    # Split B into B121 (j=0,1) and B122 (j=2,3)
    # Block 5: i in {0,1}, j in {0,1}, k=2. M_{2,2,1}
    # Block 6: i in {0,1}, j in {2,3}, k=2. M_{2,2,1}
    
    def add_m221(i_base, j_base, k_base):
        nonlocal col
        # M_{2,2,1}
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        # 4 terms
        # 1. A00 B00 -> C0
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        # 2. A01 B10 -> C0
        U[i_base * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        # 3. A10 B00 -> C1
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        # 4. A11 B10 -> C1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1

    add_m221(0, 0, 2)
    add_m221(0, 2, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
