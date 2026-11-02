"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank
    rank_schoolbook = n * m * p
    
    # We will try to reduce rank by replacing a 2x2x2 block with Strassen's rank 7 decomposition
    # Block indices: i in {0, 1}, j in {0, 1}, k in {0, 1} (0-indexed)
    # This corresponds to i=1,2; j=1,2; k=1,2 in 1-indexed
    
    # The standard decomposition has terms for all (i, j, k)
    # We will construct U, V, W matrices
    
    # We will use a list of tuples (i, j, k, coeff_u, coeff_v, coeff_w)
    # But it's easier to just build the matrices directly.
    
    # Let's define the Strassen decomposition for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # In terms of tensor decomposition:
    # The block contributes to C[i,k] for i in {0,1}, k in {0,1}
    # The inputs are A[i,j] for i in {0,1}, j in {0,1} and B[j,k] for j in {0,1}, k in {0,1}
    
    # We will replace the 8 standard terms for this block with 7 Strassen terms.
    # The other 76 terms remain standard.
    
    # Total rank = 7 + 76 = 83
    
    rank = 83
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the Strassen terms for the block i in {0,1}, j in {0,1}, k in {0,1}
    # We need to map the linear combinations to the U, V, W matrices.
    # Each Strassen term m_r corresponds to a rank-1 tensor u_r \otimes v_r \otimes w_r
    # u_r is a vector in R^{nm}, v_r in R^{mp}, w_r in R^{pn}
    
    # m1 = (a+d)(e+h)
    # u1: A[0,0]=1, A[1,1]=1
    # v1: B[0,0]=1, B[0,1]=1, B[1,0]=1, B[1,1]=1
    # w1: C[0,0]=1, C[0,1]=1, C[1,0]=1, C[1,1]=1?
    # Wait, the output of m1 is added to C11 and C22.
    # The rank-1 tensor for m1 is u1 \otimes v1 \otimes w1 where w1 has 1s at (0,0) and (1,1).
    # Actually, the Strassen algorithm computes the product.
    # The decomposition is:
    # T_block = sum_{r=1}^7 u_r \otimes v_r \otimes w_r
    
    # Let's define the vectors for the 7 terms.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # m1: (A00+A11)(B00+B01+B10+B11)
    # u1: U[0*m+0]=1, U[1*m+1]=1
    # v1: V[0*p+0]=1, V[0*p+1]=1, V[1*p+0]=1, V[1*p+1]=1
    # w1: W[0*n+0]=1, W[1*n+1]=1  (Contributes to C00 and C11)
    
    # m2: (A10+A11)B00
    # u2: U[1*m+0]=1, U[1*m+1]=1
    # v2: V[0*p+0]=1
    # w2: W[0*n+1]=1 (Contributes to C10)
    
    # m3: A00(B01-B11)
    # u3: U[0*m+0]=1
    # v3: V[0*p+1]=1, V[1*p+1]=-1
    # w3: W[1*n+0]=1 (Contributes to C01)
    
    # m4: A11(B10-B01)
    # u4: U[1*m+1]=1
    # v4: V[1*p+0]=1, V[0*p+1]=-1
    # w4: W[0*n+0]=1 (Contributes to C00)
    
    # m5: (A00+A01)B11
    # u5: U[0*m+0]=1, U[0*m+1]=1
    # v5: V[1*p+1]=1
    # w5: W[1*n+0]=1 (Contributes to C01)
    
    # m6: (A10-A00)(B00+B01)
    # u6: U[1*m+0]=1, U[0*m+0]=-1
    # v6: V[0*p+0]=1, V[0*p+1]=1
    # w6: W[1*n+1]=1 (Contributes to C11)
    
    # m7: (A01-A11)(B10+B11)
    # u7: U[0*m+1]=1, U[1*m+1]=-1
    # v7: V[1*p+0]=1, V[1*p+1]=1
    # w7: W[0*n+0]=1 (Contributes to C00)
    
    # Let's verify the contributions:
    # C00: m1 + m4 + m7
    # C01: m3 + m5
    # C10: m2 + m4
    # C11: m1 + m6
    
    # Wait, standard Strassen:
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # My w vectors:
    # w1: (0,0)=1, (1,1)=1
    # w2: (1,0)=1
    # w3: (0,1)=1
    # w4: (0,0)=1
    # w5: (0,1)=1
    # w6: (1,1)=1
    # w7: (0,0)=1
    
    # Sum for C00 (i=0, k=0):
    # m1 (1) + m4 (1) + m7 (1) = 3?
    # Strassen C11 (which is C00 in 0-indexed) = m1 + m4 - m5 + m7.
    # So w5 should have -1 at (0,1)? No, w5 contributes to C01.
    # In Strassen, m5 is (A1+A2)B22.
    # m5 contributes to C11 (C00) with -1 and C12 (C01) with +1?
    # No, m5 is used in C11 and C12.
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # So m5 has coefficient -1 in C00 and +1 in C01.
    # This means the rank-1 tensor for m5 is u5 \otimes v5 \otimes w5 where w5 has -1 at (0,0) and 1 at (0,1).
    
    # Let's redefine the w vectors with correct coefficients.
    # The rank-1 tensor is u \otimes v \otimes w.
    # The contribution to C[i,k] is u[i,j] v[j,k] w[k,i].
    # We want the sum over r of these to be 1 for the block.
    
    # Term 1: m1
    # u1: A00=1, A11=1
    # v1: B00=1, B01=1, B10=1, B11=1
    # w1: C00=1, C11=1
    # Contribution:
    # C00: A00B00 + A00B01 + A11B10 + A11B11 (from u1,v1) * 1 (from w1 at 0,0)
    # Wait, the product u1 \otimes v1 \otimes w1 gives:
    # Sum_{j} u1[i,j] v1[j,k] w1[k,i]
    # For i=0, k=0:
    # j=0: u1[0,0] v1[0,0] w1[0,0] = 1 * 1 * 1 = 1
    # j=1: u1[0,1] v1[1,0] w1[0,0] = 0 * 1 * 1 = 0 (u1[0,1] is 0)
    # Wait, u1 has 1 at (0,0) and (1,1).
    # So u1[0,0]=1, u1[0,1]=0, u1[1,0]=0, u1[1,1]=1.
    # v1 has 1 at all 4 positions.
    # For i=0, k=0:
    # j=0: u1[0,0] v1[0,0] w1[0,0] = 1 * 1 * 1 = 1
    # j=1: u1[0,1] v1[1,0] w1[0,0] = 0 * 1 * 1 = 0
    # Sum = 1.
    # But we want A00B00 + A11B11?
    # No, m1 = (A00+A11)(B00+B01+B10+B11).
    # The term in the sum for C00 should be A00B00 + A11B11?
    # No, the product is A B.
    # C00 = A00B00 + A01B10.
    # The Strassen algorithm computes this using m1..m7.
    # The tensor decomposition must sum to the standard multiplication tensor.
    # The standard tensor for the block is sum_{i,j,k in {0,1}} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}.
    # This sum is 8 rank-1 tensors.
    # The Strassen decomposition is a sum of 7 rank-1 tensors that equals this sum.
    # So we just need to provide the 7 rank-1 tensors that make up the Strassen decomposition.
    
    # The vectors u_r, v_r, w_r for the 7 terms are as derived in Strassen's algorithm,
    # but mapped to the tensor product space.
    
    # Let's list the 7 terms (u_r, v_r, w_r)
    # r=1:
    # u1: (1,0,0,1) in (0,0),(0,1),(1,0),(1,1) order
    # v1: (1,1,1,1)
    # w1: (1,0,0,1) in (0,0),(0,1),(1,0),(1,1) order for W?
    # W index is k*n + i.
    # For block, k in {0,1}, i in {0,1}.
    # W[0*n+0] is (0,0). W[0*n+1] is (1,0)? No.
    # W index: k*n + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> n
    # k=1, i=1 -> n+1
    
    # The Strassen terms are linear combinations.
    # We will just hardcode the 7 terms for the block.
    
    block_terms = [
        # u, v, w dicts
        ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # m1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # m2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}), # m3
        ({(1,1):1}, {(1,0):1, (0,1):-1}, {(0,0):1}), # m4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # m5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # m6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}) # m7
    ]
    
    # Fill U, V, W for the block terms
    for u_dict, v_dict, w_dict in block_terms:
        for (i, j), val in u_dict.items():
            U[i*m + j][col] += val
        for (j, k), val in v_dict.items():
            V[j*p + k][col] += val
        for (k, i), val in w_dict.items():
            W[k*n + i][col] += val
        col += 1
    
    # Now fill the remaining 76 terms
    # These are all (i,j,k) NOT in the block {0,1}x{0,1}x{0,1}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
