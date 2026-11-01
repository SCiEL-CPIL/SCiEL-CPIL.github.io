import numpy as np

def construct():
    n, m, p = 2, 3, 5
    # Tensor T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i
    # T[a,b,c] = 1 iff j==j' and i==i' and k==k' (i.e., a=i*m+j, b=j*p+k, c=k*n+i)
    # Rank-minimal decomposition: T = sum_r u_r(v_r, w_r)
    
    # Known: rank of 2x3*3x5 matmul tensor is 28.
    # Construction via splitting j into {0,1} (Laderman) and {2} (schoolbook).
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    
    # Part 1: j in {0,1}, k in {0,1}: 2x2 * 2x2, use Laderman (rank 7)
    # Laderman's 7 rank-1 terms for 2x2*2x2:
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]], C = A*B
    # C00 = (a00+a11)(b00+b11) - (a00-a11)(b00-b11) + ... 
    # Standard Laderman:
    # t0 = (a00+a11)(b00+b11)
    # t1 = (a00+a11)(b00-b11)  
    # t2 = (a10+a11)b01
    # t3 = (a00+a01)b10
    # t4 = (a10+a01)(b10-b01)
    # t5 = (a10-a01)(b00+b01)
    # t6 = (a11-a10)(b10+b11)
    # C00 = t0 - t1 + t5 - t6  (need to verify)
    # Actually, let me use a verified Laderman construction:
    # f0 = (a00 + a11)(b00 + b11)  -> contributes to C00 and C11
    # f1 = (a00 + a11)(b00 - b11)  -> contributes to C00 and C11
    # f2 = (a10 + a11) * b01       -> contributes to C01
    # f3 = (a00 + a01) * b10       -> contributes to C10
    # f4 = (a10 + a01)(b10 - b01)  -> contributes to C01, C10
    # f5 = (a10 - a01)(b00 + b01)  -> contributes to C00
    # f6 = (a11 - a10)(b10 + b11)  -> contributes to C11
    # C00 = f0 - f1 + f5 - f6
    # C01 = f2 - f4
    # C10 = f3 + f4
    # C11 = f0 + f1 + f6 - f5  (need to double-check signs)
    
    # Let me use a cleaner known Laderman:
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a00+a11)(b00-b11)
    # m2 = (a10+a11)b01
    # m3 = (a00+a01)b10
    # m4 = (a10+a01)(b10-b01)
    # m5 = (a10-a01)(b00+b01)
    # m6 = (a11-a10)(b10+b11)
    # C00 = m0 - m1 + m5 - m6
    # C01 = m2 - m4
    # C10 = m3 + m4
    # C11 = m0 + m1 + m6 - m5
    
    # For our problem: i' in {0,1} corresponds to i, j' in {0,1} to j, k' in {0,1} to k.
    # A[i', j'] = A[i, j], B[j', k'] = B[j, k], C[k', i'] = C[k, i]
    
    laderman_terms = [
        # (u_coeffs on A indices [i0j0, i0j1, i1j0, i1j1],
        #  v_coeffs on B indices [j0k0, j0k1, j1k0, j1k1],
        #  w_coeffs on C indices [k0i0, k0i1, k1i0, k1i1])
        # m0: (a00+a11)(b00+b11)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # m1: (a00+a11)(b00-b11)
        ([1,0,0,1], [1,0,0,-1], [1,0,0,-1]),
        # m2: (a10+a11)*b01
        ([0,0,1,1], [0,1,0,0], [0,1,0,0]),
        # m3: (a00+a01)*b10
        ([1,1,0,0], [0,0,1,0], [0,0,1,0]),
        # m4: (a10+a01)(b10-b01)
        ([0,1,1,0], [0,-1,1,0], [0,-1,1,0]),
        # m5: (a10-a01)(b00+b01)
        ([0,-1,1,0], [1,1,0,0], [1,1,0,0]),
        # m6: (a11-a10)(b10+b11)
        ([0,0,-1,1], [0,0,1,1], [0,0,1,1]),
    ]
    
    # Signs for C:
    # C00 = m0 - m1 + m5 - m6
    # C01 = m2 - m4
    # C10 = m3 + m4
    # C11 = m0 + m1 + m6 - m5
    signs = {
        0: [1, -1, 0, 0, 0, 1, -1],   # C00 coefficients for m0..m6
        1: [0, 0, 1, 0, -1, 0, 0],    # C01
        2: [0, 0, 0, 1, 1, 0, 0],     # C10
        3: [1, 1, 0, 0, 0, -1, 1],    # C11
    }
    # C index mapping: C00 -> [k0i0]=0, C01 -> [k0i1]=1, C10 -> [k1i0]=2, C11 -> [k1i1]=3
    
    for r in range(7):
        u_coeff, v_coeff, w_coeff = laderman_terms[r]
        # A indices: i0j0=0, i0j1=1, i1j0=3, i1j1=4  (since a = i*m+j, m=3)
        # Wait: a = i*m + j. i in {0,1}, j in {0,1}.
        # i0j0: a=0*3+0=0, i0j1: a=0*3+1=1, i1j0: a=1*3+0=3, i1j1: a=1*3+1=4
        # B indices: b = j*p + k. j in {0,1}, k in {0,1}.
        # j0k0: b=0, j0k1: b=1, j1k0: b=5, j1k1: b=6
        # C indices: c = k*n + i. k in {0,1}, i in {0,1}.
        # k0i0: c=0, k0i1: c=1, k1i0: c=2, k1i1: c=3
        
        a_idx = [0, 1, 3, 4]   # i0j0, i0j1, i1j0, i1j1
        b_idx = [0, 1, 5, 6]   # j0k0, j0k1, j1k0, j1k1
        c_idx = [0, 1, 2, 3]   # k0i0, k0i1, k1i0, k1i1
        
        # For this Laderman term r, the contribution to C[c_idx] is signs[c][r] * w_coeff[c]
        # Actually, the way it works: each m_r produces a scalar, and it contributes
        # to multiple C entries with different signs.
        # The tensor entry T[a,b,c] for this term is: u_coeff[a_pos] * v_coeff[b_pos] * (sum over c' signs[c'][r] * w_coeff[c'])
        # No, that's not right either. Let me think again.
        
        # The Laderman construction says:
        # C[c] = sum_r (u_coeff_r * v_coeff_r) * s_r_c
        # where s_r_c is the sign/coefficient.
        # Equivalently, T[a,b,c] = sum_r u_coeff_r[a] * v_coeff_r[b] * s_r_c
        # where u_coeff_r[a] picks out the A entry, v_coeff_r[b] picks out the B entry.
        
        # So for term r, the rank-1 tensor is:
        # U[a] = u_coeff_r[position of a in a_idx]
        # V[b] = v_coeff_r[position of b in b_idx]  
        # W[c] = s_r_c (the sign for output c)
        
        # But wait, that's not a rank-1 tensor in the standard sense. A rank-1 tensor
        # T = u ⊗ v ⊗ w means T[a,b,c] = u[a]*v[b]*w[c].
        # In Laderman, the same scalar m_r = (u_a)*(v_b) contributes to multiple c's
        # with different weights. So T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
        # where w_r[c] = s_r_c.
        
        for a_pos, a_val in enumerate(a_idx):
            U[a_val, col] = u_coeff[a_pos]
        for b_pos, b_val in enumerate(b_idx):
            V[b_val, col] = v_coeff[b_pos]
        for c_pos, c_val in enumerate(c_idx):
            W[c_val, col] = signs[c_pos][r]
        col += 1
    
    # Part 2: j in {0,1}, k in {2,3,4}: 2x2 * 2x3
    # For each k in {2,3,4}: C[:,k] = A[:,0:2] @ B[0:2,k]
    # This is a 2x2 * 2x1 multiplication for each k.
    # 2x2 * 2x1: C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    # Tensor: T[i,j,k] for i in {0,1}, j in {0,1}, k fixed.
    # This is a 2x2x1 tensor. Its rank is 2 (can be written as 2 rank-1 terms).
    # Term 1: u=[1,0,0,1] (A00,A10), v=[1,0] (B0), w=[1,0] (C0) -> no, that's not right.
    # C0 = A00*B0 + A01*B1: two products.
    # C1 = A10*B0 + A11*B1: two products.
    # Total 4 products, but we can share: B0 appears in both C0 and C1.
    # Rank-1 decomposition:
    # T1: A00 * B0 * C0
    # T2: A01 * B1 * C0
    # T3: A10 * B0 * C1
    # T4: A11 * B1 * C1
    # Can we do it in 3? 
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Let m1 = A00*B0, m2 = A01*B1, m3 = (A10-A00)*B0 + A11*B1... no.
    # Actually, 2x2*2x1 has tensor rank 2? Let's see:
    # T[i,j,k] where k is fixed. T[0,0]=1, T[0,1]=1, T[1,0]=1, T[1,1]=1 (all 4 entries are 1).
    # Wait no: T[i,j] = 1 for all i,j means the tensor is just the all-ones 2x2 matrix
    # for each k. But that's the tensor for the specific bilinear map, not the identity.
    # The tensor for 2x2*2x1 matmul: C[i] = sum_j A[i,j]*B[j].
    # T[i,j] = 1 for all i,j (since every A[i,j] contributes to C[i]).
    # So T is a 2x2x1 tensor with all entries 1. Rank of all-ones 2x2 matrix is 1!
    # So T = [1,1]^T ⊗ [1,1]^T ⊗ [1]. That's rank 1!
    # Wait, that means for fixed k, C[:,k] = A @ B[:,k] can be done in rank 1?
    # C[:,k] = A @ B[:,k]. If A is 2x2 and B[:,k] is 2x1, then C[:,k] is 2x1.
    # The bilinear map (A, B[:,k]) -> C[:,k] has tensor T[i,j] = 1 for all i,j.
    # This tensor has rank 1: T = u ⊗ v where u=[1,1], v=[1,1].
    # So C[:,k] = sum_j A[i,j]*B[j,k] = (sum over i,j) ... 
    # Hmm, let me be careful. T[i,j,k] = 1 means A[i,j] and B[j,k] both contribute to C[i,k].
    # For fixed k, T[i,j] = 1 for all i,j. This is the 2x2 all-ones matrix.
    # Its rank is 1. So yes, rank 1 per k.
    # For 3 k's (k=2,3,4), that's rank 3.
    
    # But wait, I can do better by grouping. For k in {2,3,4}:
    # C[:,k] = A @ B[:,k] for k=2,3,4.
    # This is C[:,2:5] = A @ B[:,2:5].
    # The tensor for this is: T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {2,3,4}.
    # That's a 2x2x3 tensor with all entries 1. Rank is 1!
    # T = u ⊗ v ⊗ w where u=[1,1], v=[1,1], w=[1,1,1].
    # So one rank-1 term handles all of j in {0,1}, k in {2,3,4}.
    
    U[0, col] = 1; U[1, col] = 1; U[3, col] = 1; U[4, col] = 1  # A entries for i in {0,1}, j in {0,1}
    V[0, col] = 1; V[1, col] = 1; V[5, col] = 1; V[6, col] = 1  # B entries for j in {0,1}, k in {2,3,4}
    # Wait, V index: b = j*p + k. j in {0,1}, k in {2,3,4}.
    # j=0: b = 0*5+2=2, 0*5+3=3, 0*5+4=4
    # j=1: b = 1*5+2=7, 1*5+3=8, 1*5+4=9
    V[2, col] = 1; V[3, col] = 1; V[4, col] = 1
    V[7, col] = 1; V[8, col] = 1; V[9, col] = 1
    # W index: c = k*n + i. k in {2,3,4}, i in {0,1}.
    # k=2: c=4,5; k=3: c=6,7; k=4: c=8,9
    W[4, col] = 1; W[5, col] = 1
    W[6, col] = 1; W[7, col] = 1
    W[8, col] = 1; W[9, col] = 1
    col += 1
    
    # Part 3: j=2, k in {0,1,2,3,4}: 2x1 * 1x5
    # C[i,k] = A[i,2] * B[2,k] for all i,k.
    # T[i,j,k] = 1 for i in {0,1}, j=2, k in {0..4}.
    # This is a 2x1x5 tensor with all entries 1. Rank 1.
    U[0, col] = 1; U[3, col] = 1  # A[i,2]: a = i*3+2, i=0->2, i=1->5
    U[2, col] = 1; U[5, col] = 1
    V[10, col] = 1; V[11, col] = 1; V[12, col] = 1; V[13, col] = 1; V[14, col] = 1  # B[2,k]: b=2*5+k
    W[0, col] = 1; W[1, col] = 1  # k=0: c=0,1
    W[2, col] = 1; W[3, col] = 1  # k=1: c=2,3
    W[4, col] = 1; W[5, col] = 1  # k=2: c=4,5
    W[6, col] = 1; W[7, col] = 1  # k=3: c=6,7
    W[8, col] = 1; W[9, col] = 1  # k=4: c=8,9
    col += 1
    
    # Total rank so far: 7 + 1 + 1 = 9? That seems too low.
    # Let me verify this is correct.
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(col):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    # Check
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                max_err = max(max_err, abs(T[a,b,c] - 1.0))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i = a // m; j = a % m
                j2 = b // p; k = b % p
                k2 = c // n; i2 = c % n
                if not (j == j2 and i == i2 and k == k2):
                    max_err = max(max_err, abs(T[a,b,c]))
    
    if max_err < 1e-6:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

BET: {mechanism: "Explicit rank-9 construction via Laderman + shared rank-1 terms; if correct, massive improvement over rank-30", q: 0.5, delta: [0.6, 0.7]}
