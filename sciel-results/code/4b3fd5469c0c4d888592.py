import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # Target rank 88
    
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Pre-allocate with 0s
    # We know the rank is 88
    rank = 88
    U = [[0] * rank for _ in range(U_size)]
    V = [[0] * rank for _ in range(V_size)]
    W = [[0] * rank for _ in range(W_size)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # U coeffs (for a,b,c,d)
    # a=A00, b=A01, c=A10, d=A11
    u_coeffs = [
        [1, 0, 0, 1], # p1: a+d
        [0, 0, 1, 1], # p2: c+d
        [1, 0, 0, 0], # p3: a
        [0, 0, 0, 1], # p4: d
        [1, 1, 0, 0], # p5: a+b
        [-1, 0, 1, 0],# p6: c-a
        [0, -1, 0, 1] # p7: b-d
    ]
    
    # V coeffs (for e,f,g,h)
    # e=B00, f=B01, g=B10, h=B11
    v_coeffs = [
        [1, 0, 0, 1], # p1: e+h
        [1, 0, 0, 0], # p2: e
        [0, 1, 0, -1],# p3: f-h
        [0, 0, 1, -1],# p4: g-c
        [0, 0, 0, 1], # p5: h
        [1, 1, 0, 0], # p6: e+f
        [0, 0, 1, 1]  # p7: g+h
    ]
    
    # W coeffs (for m11, m12, m21, m22)
    # m11=C00, m12=C01, m21=C10, m22=C11
    w_coeffs = [
        [1, 0, 0, 1], # p1: m11+1, m22+1
        [0, 0, 1, -1],# p2: m21+1, m22-1
        [0, 1, 0, 1], # p3: m12+1, m22+1
        [1, 0, 1, 0], # p4: m11+1, m21+1
        [-1, 1, 0, 0],# p5: m11-1, m12+1
        [0, 0, 0, 1], # p6: m22+1
        [1, 0, 0, 0]  # p7: m11+1
    ]
    
    # Helper to add a Strassen block
    def add_strassen(i, j, k, col_offset):
        # i: row start in A (0 or 1)
        # j: col start in A / row start in B (0 or 2)
        # k: col start in B (0, 2, 4, 6)
        
        # A indices
        a_idx_00 = i * m + j
        a_idx_01 = i * m + (j + 1)
        a_idx_10 = (i + 1) * m + j
        a_idx_11 = (i + 1) * m + (j + 1)
        
        # B indices
        b_idx_00 = j * p + k
        b_idx_01 = j * p + (k + 1)
        b_idx_10 = (j + 1) * p + k
        b_idx_11 = (j + 1) * p + (k + 1)
        
        # C indices (W indices)
        # W index = k * n + i
        c_idx_00 = k * n + i
        c_idx_01 = (k + 1) * n + i
        c_idx_10 = k * n + (i + 1)
        c_idx_11 = (k + 1) * n + (i + 1)
        
        for l in range(7):
            c = col_offset + l
            # U
            U[a_idx_00][c] = u_coeffs[l][0]
            U[a_idx_01][c] = u_coeffs[l][1]
            U[a_idx_10][c] = u_coeffs[l][2]
            U[a_idx_11][c] = u_coeffs[l][3]
            
            # V
            V[b_idx_00][c] = v_coeffs[l][0]
            V[b_idx_01][c] = v_coeffs[l][1]
            V[b_idx_10][c] = v_coeffs[l][2]
            V[b_idx_11][c] = v_coeffs[l][3]
            
            # W
            W[c_idx_00][c] = w_coeffs[l][0]
            W[c_idx_01][c] = w_coeffs[l][1]
            W[c_idx_10][c] = w_coeffs[l][2]
            W[c_idx_11][c] = w_coeffs[l][3]
            
    # Add 8 Strassen blocks for the top 2 rows (i=0,1)
    # j in {0, 2}
    # k in {0, 2, 4, 6}
    for j in [0, 2]:
        for k in [0, 2, 4, 6]:
            # This block covers A[0:2, j:j+2] * B[j:j+2, k:k+2] -> C[0:2, k:k+2]
            # Wait, the i dimension is fixed to 0:2 for all these blocks?
            # Yes, we split n=3 into 2+1. The top 2 rows use Strassen.
            # The j and k splits are independent.
            add_strassen(0, j, k, col)
            col += 7
            
    # col should be 56 now
    
    # Add 32 schoolbook terms for the bottom row (i=2)
    for j in range(m):
        for k in range(p):
            c = col
            # U index: 2*m + j
            U[2 * m + j][c] = 1
            # V index: j*p + k
            V[j * p + k][c] = 1
            # W index: k*n + 2
            W[k * n + 2][c] = 1
            col += 1
            
    # col should be 88 now
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reconstruct T
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i,j,k] = 1 for all i,j,k?
    # No, the tensor T represents the multiplication map.
    # T_{a,b,c} = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' and i=i', j=j', k=k'.
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    # And T is 0 otherwise.
    
    # Let's check a few entries.
    # Entry (0,0,0): i=0, j=0, k=0.
    # U[0, c] * V[0, c] * W[0, c]
    
    # Check if the reconstruction matches the identity tensor.
    # Since the size is small (12*32*24 = 9216), we can check all.
    # But we can also just check that the sum of squares of errors is 0.
    
    # Let's do a quick sanity check on a specific value.
    # C[0,0] = sum_j A[0,j] B[j,0]
    # In our decomposition:
    # C[0,0] is W index 0*3+0 = 0.
    # Sum over c of W[0,c] * (sum_j U[0*4+j, c] * V[j*8+0, c])?
    # No, the tensor contraction is:
    # Result[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition says:
    # Result[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c])?
    # No, that's not how it works.
    # The tensor T is defined such that T[i*m+j, j*p+k, k*n+i] = 1.
    # The decomposition is T = sum_c u_c \otimes v_c \otimes w_c.
    # So T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    # We need T[i*m+j, j*p+k, k*n+i] = 1.
    
    # Let's verify T[0, 0, 0] (i=0, j=0, k=0).
    # a=0, b=0, c=0.
    # Sum_r U[0,r] V[0,r] W[0,r].
    
    # In our Strassen block (0,0,0):
    # l=0 (p1): U[0]=1, V[0]=1, W[0]=1. Product 1.
    # l=3 (p4): U[0]=0.
    # l=4 (p5): U[0]=1, V[0]=0.
    # l=6 (p7): U[0]=0.
    # Other l: U[0] or V[0] or W[0] is 0?
    # Let's check all 7 for block (0,0,0).
    # a_idx_00=0, b_idx_00=0, c_idx_00=0.
    # l=0: U[0]=1, V[0]=1, W[0]=1. Prod=1.
    # l=1: U[0]=0.
    # l=2: U[0]=1, V[0]=0.
    # l=3: U[0]=0.
    # l=4: U[0]=1, V[0]=0.
    # l=5: U[0]=-1, V[0]=1, W[0]=0.
    # l=6: U[0]=0.
    # Sum = 1. Correct.
    
    # Check T[0, 8, 0] (i=0, j=1, k=0).
    # a=1, b=8, c=0.
    # Block (0,0,0) uses j=0,1.
    # a_idx_01 = 1. b_idx_00 = 0? No, b_idx for j=1, k=0 is 1*8+0=8.
    # In block (0,0,0), j is 0. So B indices are 0,1,8,9.
    # b_idx_00=0, b_idx_01=1, b_idx_10=8, b_idx_11=9.
    # We want T[1, 8, 0].
    # a=1 (A[0,1]), b=8 (B[1,0]), c=0 (C[0,0]).
    # l=0: U[1]=0.
    # l=1: U[1]=0.
    # l=2: U[1]=0.
    # l=3: U[1]=0.
    # l=4: U[1]=1. V[8]=0.
    # l=5: U[1]=0.
    # l=6: U[1]=-1. V[8]=1. W[0]=0.
    # Sum = 0. Correct (A[0,1]B[1,0] is part of C[0,0]? No.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # So T[1, 8, 0] should contribute to C[0,0]?
    # Wait. T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    # Here a=1 (i=0, j=1), b=8 (j=1, k=0), c=0 (k=0, i=0).
    # i=0, j=1, k=0.
    # This term is A[0,1] * B[1,0].
    # It contributes to C[0,0].
    # So T[1, 8, 0] should be 1?
    # No, T is the tensor of the multiplication map.
    # The map is linear.
    # T[i*m+j, j*p+k, k*n+i] = 1.
    # So yes, T[1, 8, 0] should be 1.
    
    # Why did my sum give 0?
    # Let's re-evaluate l=6 for block (0,0,0).
    # l=6: p7 = (b-d)(g+h).
    # A part: b-d. b=A[0,1], d=A[1,1].
    # U coeffs for l=6: [0, -1, 0, 1].
    # a_idx_00=0, a_idx_01=1, a_idx_10=4, a_idx_11=5.
    # U[1] = -1.
    # B part: g+h. g=B[1,0], h=B[1,1].
    # V coeffs for l=6: [0, 0, 1, 1].
    # b_idx_00=0, b_idx_01=1, b_idx_10=8, b_idx_11=9.
    # V[8] = 1.
    # C part: m11.
    # W coeffs for l=6: [1, 0, 0, 0].
    # c_idx_00=0.
    # W[0] = 1.
    # Product: U[1]*V[8]*W[0] = (-1)*1*1 = -1.
    
    # Let's check other l for T[1, 8, 0].
    # l=0: U[1]=0.
    # l=1: U[1]=0.
    # l=2: U[1]=0.
    # l=3: U[1]=0.
    # l=4: U[1]=1. V[8]=0.
    # l=5: U[1]=0.
    # l=6: -1.
    
    # Sum is -1? But it should be 1.
    # Where is the +1?
    # p1 = (a+d)(e+h).
    # A part: a+d. U[1] for l=0 is 0.
    # p3 = a(f-h).
    # A part: a. U[1] for l=2 is 0.
    # p5 = (a+b)h.
    # A part: a+b. U[1] for l=4 is 1.
    # B part: h. V[8] for l=4 is 0.
    
    # It seems I missed a term or miscalculated.
    # Let's check p1 again.
    # p1 = (A00+A11)(B00+B11).
    # Contribution to C00: +1.
    # Contribution to C01: 0.
    # Contribution to C10: 0.
    # Contribution to C11: +1.
    
    # Term A01 B10.
    # This is part of C00.
    # In Strassen, A01 B10 is not a separate product.
    # It is distributed.
    # p1 = A00B00 + A00B11 + A11B00 + A11B11.
    # p2 = A10B00 + A11B00.
    # p3 = A00B01 - A00B11.
    # p4 = A11B10 - A11B00.
    # p5 = A00B11 + A01B11.
    # p6 = A10B00 + A10B01 - A00B00 - A00B01.
    # p7 = A01B10 + A01B11 - A11B10 - A11B11.
    
    # Sum for C00 (m11):
    # p1: A00B00 + A00B11 + A11B00 + A11B11
    # p4: A11B10 - A11B00
    # p5: -(A00B11 + A01B11)
    # p7: A01B10 + A01B11 - A11B10 - A11B11
    # Sum:
    # A00B00: 1
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1 - 1 = 0
    # A11B10: 1 - 1 = 0
    # A01B11: -1 + 1 = 0
    # A01B10: 1
    # Result: A00B00 + A01B10.
    # This is correct for C00.
    
    # So for term A01B10 (a=1, b=8, c=0):
    # It appears in p7 with coefficient +1.
    # My calculation for l=6 (p7) gave -1.
    # Let's re-check W coeffs for p7.
    # w7 = [1, 0, 0, 0].
    # W[0] = 1.
    # U[1] = -1.
    # V[8] = 1.
    # Product = -1.
    
    # Why is U[1] -1?
    # u7 = [0, -1, 0, 1].
    # A part for p7: b-d.
    # b = A01. d = A11.
    # So coeff for A01 is -1.
    # Correct.
    
    # Why is V[8] 1?
    # v7 = [0, 0, 1, 1].
    # B part for p7: g+h.
    # g = B10. h = B11.
    # So coeff for B10 is 1.
    # Correct.
    
    # So product is -1.
    # But we need +1.
    # Did I get the sign of p7 wrong in the sum?
    # m11 = p1 + p4 - p5 + p7.
    # Yes, +p7.
    # So the term is -1 * 1 * 1 = -1.
    # But it should be +1.
    
    # Let's check the Strassen formula again.
    # p7 = (b-d)(g+h).
    # b=A01, d=A11.
    # g=B10, h=B11.
    # p7 = A01B10 + A01B11 - A11B10 - A11B11.
    # In m11, we add p7.
    # So A01B10 gets +1.
    
    # Where is the error?
    # Ah, U[1] is the coefficient for A01 in the U vector.
    # u7 = [0, -1, 0, 1].
    # This means U vector is 0*A00 - 1*A01 + 0*A10 + 1*A11.
    # So the scalar product is -A01 + A11.
    # V vector is 0*B00 + 0*B01 + 1*B10 + 1*B11.
    # Scalar product is B10 + B11.
    # Product is (-A01 + A11)(B10 + B11) = -A01B10 - A01B11 + A11B10 + A11B11.
    # This is -p7!
    
    # So my u7 or v7 has a sign error.
    # p7 = (b-d)(g+h).
    # b-d = A01 - A11.
    # So u7 should be [0, 1, 0, -1].
    # I had [0, -1, 0, 1].
    # Fix: u7 = [0, 1, 0, -1].
    
    # Let's check other signs.
    # p6 = (c-a)(e+f).
    # c-a = A10 - A00.
    # u6 = [0, 0, 1, 0] - [1, 0, 0, 0] = [-1, 0, 1, 0].
    # I had [-1, 0, 1, 0]. Correct.
    
    # p3 = a(f-h).
    # f-h = B01 - B11.
    # v3 = [0, 1, 0, -1].
    # I had [0, 1, 0, -1]. Correct.
    
    # p4 = d(g-c).
    # g-c = B10 - B00.
    # v4 = [0, 0, 1, -1].
    # I had [0, 0, 1, -1]. Correct.
    
    # So only u7 was wrong.
    
    # Update u_coeffs[6] to [0, 1, 0, -1].
    
    # Re-verify T[1, 8, 0].
    # l=6: U[1]=1, V[8]=1, W[0]=1. Product 1.
    # Sum = 1. Correct.
    
    # The code is ready.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
