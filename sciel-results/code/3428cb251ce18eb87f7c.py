"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    # We will build the tensor decomposition via block partitioning with Strassen
    # Split: n=3 = 2+1, m=5 = 2+2+1, p=6 = 2+2+2
    # For each triple of blocks (n_i, m_j, p_k) where n_i*m_j*p_k, we use:
    #   - Strassen (rank 7) if all dims are 2
    #   - schoolbook if any dim is 1
    # Actually, let me think more carefully.
    
    # The matrix multiplication tensor T_{n,m,p} has indices (i,j,k) for i in [n], j in [m], k in [p].
    # A rank-one term corresponds to choosing specific (i,j,k) and a scalar coefficient.
    # The decomposition is: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to 1 if (i,j,k) is valid, else 0.
    # Actually, the standard formulation: for the multiplication tensor, the (i,j,k)-entry is 1.
    # The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    # Wait, that's not quite right. Let me re-derive.
    
    # The matrix multiplication tensor M_{n,m,p} ∈ R^{(n*m) x (m*p) x (p*n)} is defined by:
    # M[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j' and i=i', else 0? No.
    
    # Actually, the standard convention: M[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # M[a,b,c] = 1 if j = j' and i = i', else 0.
    # Wait, that doesn't look right either. Let me think again.
    
    # The matrix multiplication C = A * B where A is n×m, B is m×p, C is n×p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is over indices (i,j) for A, (j,k) for B, (i,k) for C.
    # But the task says: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # And W[k*n+i, r] follows the trace convention.
    
    # So the tensor T has dimensions (n*m) × (m*p) × (p*n).
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'.
    # The value is 1 if j = j' and i = i', else 0.
    # Wait, let me check: for a valid triple (i, j, k) contributing to C[i,k]:
    # a = i*m+j, b = j*p+k, c = k*n+i.
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    # For other combinations, T = 0.
    
    # The decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k),
    # and 0 for all other (a,b,c) combinations.
    
    # For the schoolbook: each (i,j,k) gives one term with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    
    # Now, for Strassen on 2x2x2:
    # The 7-term Strassen decomposition. Let me recall it.
    # For A (2x2), B (2x2), C = A*B (2x2).
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    # c11 = a11*b11 + a12*b21
    # c12 = a11*b12 + a12*b22
    # c21 = a21*b11 + a22*b21
    # c22 = a21*b12 + a22*b22
    
    # Strassen:
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # So for each of the 7 terms, we have combinations of A entries, B entries, and C entries.
    # In the tensor decomposition framework:
    # For a 2x2x2 block with local indices (i,j,k) where i,j,k ∈ {0,1}:
    # We need 7 rank-one terms.
    
    # Let me define the 7 terms for 2x2x2. Each term r has:
    # U_part: values for (i,j) pairs
    # V_part: values for (j,k) pairs  
    # W_part: values for (k,i) pairs
    
    # Term 1 (m1): 
    # A entries: (0,0):1, (1,1):1 → U[0*m+0]=1, U[1*m+1]=1 (local m=2)
    # B entries: (0,0):1, (0,1):1, (1,0):1, (1,1):1 → V[0*p+0]=1, V[0*p+1]=1, V[1*p+0]=1, V[1*p+1]=1 (local p=2)
    # C entries: (0,0):1, (1,0):-1, (0,1):-1, (1,1):1 → W[0*n+0]=1, W[0*n+1]=-1, W[1*n+0]=-1, W[1*n+1]=1 (local n=2)
    # Wait, let me be more careful.
    
    # Actually, for the 2x2x2 tensor, the indices are:
    # A index: i*2+j, B index: j*2+k, C index: k*2+i
    # where i,j,k ∈ {0,1}
    
    # The 8 nonzero entries of the tensor are at:
    # (i,j,k) = (0,0,0): A[0], B[0], C[0]
    # (i,j,k) = (0,0,1): A[0], B[1], C[2]  (C index: k*n+i = 1*2+0 = 2)
    # (i,j,k) = (0,1,0): A[1], B[0], C[1]  (C index: 0*2+0 = 0... wait)
    
    # Let me redo. C index = k*n + i. With n=2:
    # (i=0, k=0): C index 0
    # (i=0, k=1): C index 1
    # (i=1, k=0): C index 2
    # (i=1, k=1): C index 3
    
    # A index = i*m + j, m=2:
    # (0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3
    # B index = j*p + k, p=2:
    # (0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3
    
    # The 8 valid triples:
    # (i,j,k) -> (A_idx, B_idx, C_idx)
    # (0,0,0) -> (0, 0, 0)
    # (0,0,1) -> (0, 1, 1)
    # (0,1,0) -> (1, 2, 0)
    # (0,1,1) -> (1, 3, 1)
    # (1,0,0) -> (2, 0, 2)
    # (1,0,1) -> (2, 1, 3)
    # (1,1,0) -> (3, 2, 2)
    # (1,1,1) -> (3, 3, 3)
    
    # Strassen terms for 2x2x2:
    # I'll use the standard Strassen with the sign conventions.
    
    # Let me define the 7 terms as (U_vector, V_vector, W_vector) over the 4x4x4 space.
    
    # Term 1: m1 = (a00+a11)(b00+b11)
    # U: a00:1, a11:1 → [1,0,0,1]
    # V: b00:1, b01:1, b10:1, b11:1 → [1,1,1,1]
    # W: c00:1, c01:-1, c10:-1, c11:1 → W index: c00=0, c01=1, c10=2, c11=3
    #    W: [1, -1, -1, 1]
    
    # Term 2: m2 = (a10+a11)b00
    # U: a10:1, a11:1 → [0,0,1,1]
    # V: b00:1 → [1,0,0,0]
    # W: c10:1, c11:-1 → W[2]:1, W[3]:-1 → [0,0,1,-1]
    
    # Term 3: m3 = a00(b01-b11)
    # U: a00:1 → [1,0,0,0]
    # V: b01:1, b11:-1 → [0,1,0,-1]
    # W: c01:1, c11:1 → W[1]:1, W[3]:1 → [0,1,0,1]
    
    # Term 4: m4 = a11(b10-b00)
    # U: a11:1 → [0,0,0,1]
    # V: b10:1, b00:-1 → [-1,0,1,0]
    # W: c10:1, c00:1 → W[2]:1, W[0]:1 → [1,0,1,0]
    
    # Term 5: m5 = (a00+a01)b11
    # U: a00:1, a01:1 → [1,1,0,0]
    # V: b11:1 → [0,0,0,1]
    # W: c01:-1, c11:1 → W[1]:-1, W[3]:1 → [0,-1,0,1]
    
    # Term 6: m6 = (a10-a00)(b00+b01)
    # U: a10:1, a00:-1 → [-1,0,1,0]
    # V: b00:1, b01:1 → [1,1,0,0]
    # W: c11:1 → [0,0,0,1]
    
    # Term 7: m7 = (a01-a11)(b10+b11)
    # U: a01:1, a11:-1 → [0,1,0,-1]
    # V: b10:1, b11:1 → [0,0,1,1]
    # W: c00:1, c01:-1 → W[0]:1, W[1]:-1 → [1,-1,0,0]
    
    # Let me verify: sum of products should give the 8 entries.
    # Entry (0,0,0): A[0]*B[0]*W[0]
    # = 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*1 + 1*0*0 + (-1)*1*1 + 0*0*1
    # = 1 + 0 + 0 + 0 + 0 - 1 + 0 = 0. But should be 1!
    
    # Hmm, something is wrong. Let me recheck.
    # The tensor entry T[A_idx, B_idx, C_idx] = 1 if the indices correspond to a valid (i,j,k) triple.
    # T[0,0,0] corresponds to (i=0,j=0,k=0), so it should be 1.
    
    # Let me recheck Strassen.
    # c00 = m1 + m4 - m5 + m7
    # m1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # m4 = a11*(b10-b00) = a11*b10 - a11*b00
    # -m5 = -(a00+a01)*b11 = -a00*b11 - a01*b11
    # m7 = (a01-a11)(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
    # Sum: a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b10 - a11*b00 - a00*b11 - a01*b11 + a01*b10 + a01*b11 - a11*b10 - a11*b11
    # = a00*b00 + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + (a11*b11 - a11*b11) + (a11*b10 - a11*b10) + (-a01*b11 + a01*b11) + a01*b10
    # = a00*b00 + a01*b10
    
    # But c00 should be a00*b00 + a01*b10. Yes! That's correct for C[0,0].
    
    # So the issue is in how I'm mapping to the tensor. Let me reconsider.
    # The tensor entry T[A_idx, B_idx, C_idx] where:
    # A_idx = i*2+j, B_idx = j*2+k, C_idx = k*2+i
    # T is nonzero only when the (i,j) in A, (j,k) in B, and (i,k) in C are consistent.
    # Actually, T[A_idx, B_idx, C_idx] = 1 if A_idx = i*2+j, B_idx = j*2+k, C_idx = k*2+i for some valid i,j,k.
    # So T is 1 at exactly 8 positions and 0 elsewhere.
    
    # The decomposition: T[A,B,C] = sum_r U[A,r] * V[B,r] * W[C,r]
    # For T[0,0,0] = 1:
    # sum_r U[0,r] * V[0,r] * W[0,r] = 1
    # U[0,:] = [1, 0, 1, 0, 1, -1, 0] (from terms 1-7)
    # V[0,:] = [1, 1, 0, -1, 0, 1, 0]
    # W[0,:] = [1, 0, 0, 1, 0, 1, 1]
    # Product: 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*1 + 1*0*0 + (-1)*1*1 + 0*0*1 = 1 + 0 + 0 + 0 + 0 - 1 + 0 = 0
    
    # That's 0, not 1. So my Strassen mapping is wrong.
    
    # The problem: in Strassen, the same "m" term contributes to multiple C entries with different coefficients.
    # In the tensor decomposition, each rank-one term has fixed U, V, W vectors.
    # The product U[A]*V[B]*W[C] for a given (A,B,C) triple gives the contribution.
    
    # Let me reconsider. For Strassen to work as a tensor decomposition, I need to verify that:
    # sum_r U[A,r] * V[B,r] * W[C,r] = 1 for the 8 valid triples, 0 otherwise.
    
    # Let me re-derive the Strassen terms more carefully in tensor form.
    # Each "m" in Strassen is a product of a linear combination of A entries and a linear combination of B entries.
    # Then C entries are linear combinations of the m's.
    
    # In the tensor decomposition:
    # For each m_r, we have:
    # U_r[i*m+j] = coefficient of A[i,j] in the A-part of m_r
    # V_r[j*p+k] = coefficient of B[j,k] in the B-part of m_r
    # W_r[k*n+i] = coefficient of C[i,k] in the C-part where m_r appears
    
    # For m1 = (a00+a11)(b00+b11):
    # A-part: a00:1, a11:1
    # B-part: b00:1, b01:1, b10:1, b11:1
    # C-part: m1 appears in c00 with coeff +1, c11 with coeff +1
    # c00 = C[0,0], index k*n+i = 0*2+0 = 0
    # c11 = C[1,1], index k*n+i = 1*2+1 = 3
    # W_r[0] = 1, W_r[3] = 1
    
    # For m2 = (a10+a11)b00:
    # A-part: a10:1, a11:1
    # B-part: b00:1
    # C-part: m2 appears in c10 with coeff +1, c11 with coeff -1
    # c10 = C[1,0], index 0*2+1 = 2
    # c11 = C[1,1], index 3
    # W_r[2] = 1, W_r[3] = -1
    
    # For m3 = a00(b01-b11):
    # A-part: a00:1
    # B-part: b01:1, b11:-1
    # C-part: m3 appears in c01 with coeff +1, c11 with coeff +1
    # c01 = C[0,1], index 1*2+0 = 1
    # c11 = C[1,1], index 3
    # W_r[1] = 1, W_r[3] = 1
    
    # For m4 = a11(b10-b00):
    # A-part: a11:1
    # B-part: b10:1, b00:-1
    # C-part: m4 appears in c10 with coeff +1, c00 with coeff +1
    # c10 = C[1,0], index 2
    # c00 = C[0,0], index 0
    # W_r[2] = 1, W_r[0] = 1
    
    # For m5 = (a00+a01)b11:
    # A-part: a00:1, a01:1
    # B-part: b11:1
    # C-part: m5 appears in c01 with coeff -1, c00 with coeff -1
    # c01 = C[0,1], index 1
    # c00 = C[0,0], index 0
    # W_r[1] = -1, W_r[0] = -1
    
    # For m6 = (a10-a00)(b00+b01):
    # A-part: a10:1, a00:-1
    # B-part: b00:1, b01:1
    # C-part: m6 appears in c11 with coeff +1
    # c11 = C[1,1], index 3
    # W_r[3] = 1
    
    # For m7 = (a01-a11)(b10+b11):
    # A-part: a01:1, a11:-1
    # B-part: b10:1, b11:1
    # C-part: m7 appears in c00 with coeff +1, c01 with coeff -1
    # c00 = C[0,0], index 0
    # c01 = C[0,1], index 1
    # W_r[0] = 1, W_r[1] = -1
    
    # Now let me verify T[0,0,0] = 1:
    # sum_r U[0,r] * V[0,r] * W[0,r]
    # U[0,:] = [1, 0, 1, 0, 1, -1, 0]
    # V[0,:] = [1, 1, 0, -1, 0, 1, 0]
    # W[0,:] = [1, 0, 0, 1, -1, 0, 1]
    # = 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*1 + 1*0*(-1) + (-1)*1*0 + 0*0*1
    # = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1 ✓
    
    # T[0,1,0]: A[0], B[1], C[0]
    # U[0,:] * V[1,:] * W[0,:]
    # V[1,:] = [0, 1, 0, 0, 0, 1, 0] (b01 is at B index 1)
    # = 1*0*1 + 0*1*0 + 1*0*0 + 0*0*1 + 1*0*(-1) + (-1)*1*0 + 0*0*1 = 0 ✓
    
    # T[0,1,1]: A[0], B[1], C[1]
    # W[1,:] = [0, 1, 0, 1, 0, 0, -1]
    # = 1*0*0 + 0*1*1 + 1*0*0 + 0*0*1 + 1*0*0 + (-1)*1*0 + 0*0*(-1) = 0 ✓
    
    # T[1,0,0]: A[1], B[0], C[0]
    # U[1,:] = [0, 1, 0, 0, 1, 0, 1]
    # = 0*1*1 + 1*1*0 + 0*0*0 + 0*(-1)*1 + 1*0*(-1) + 0*1*0 + 1*0*1 = 0 ✓
    
    # T[1,0,1]: A[1], B[0], C[1]
    # W[1,:] = [0, 1, 0, 1, 0, 0, -1]
    # = 0*1*0 + 1*1*1 + 0*0*0 + 0*(-1)*1 + 1*0*0 + 0*1*0 + 1*0*(-1) = 1 ✓
    
    # T[1,1,0]: A[1], B[1], C[0]
    # W[0,:] = [1, 0, 0, 1, -1, 0, 1]
    # = 0*1*1 + 1*1*0 + 0*0*0 + 0*(-1)*1 + 1*0*(-1) + 0*1*0 + 1*0*1 = 0 ✓
    
    # T[1,1,1]: A[1], B[1], C[1]
    # = 0*1*0 + 1*1*1 + 0*0*0 + 0*(-1)*1 + 1*0*0 + 0*1*0 + 1*0*(-1) = 1 ✓
    
    # T[2,0,0]: A[2], B[0], C[0]
    # U[2,:] = [0, 0, 0, 0, 0, 1, 0]
    # = 0 ✓
    
    # T[2,0,2]: A[2], B[0], C[2]
    # W[2,:] = [0, 0, 1, 0, 0, 0, 0]
    # = 0*0*1 + 0*0*0 + 0*0*0 + 0*0*0 + 0*0*0 + 1*1*0 + 0*0*0 = 0 ✓
    
    # T[2,1,2]: A[2], B[1], C[2]
    # = 0 ✓
    
    # T[2,1,3]: A[2], B[1], C[3]
    # W[3,:] = [1, 1, -1, 1, 1, 0, 0]
    # = 0*1*1 + 0*1*1 + 0*0*(-1) + 0*0*1 + 0*0*1 + 1*1*0 + 0*0*0 = 0 ✓
    
    # T[3,0,2]: A[3], B[0], C[2]
    # U[3,:] = [1, 1, 0, 0, 0, 0, -1]
    # = 1*1*0 + 1*1*0 + 0*0*0 + 0*(-1)*0 + 0*0*0 + 0*1*0 + (-1)*0*0 = 0 ✓
    
    # T[3,0,3]: A[3], B[0], C[3]
    # = 1*1*1 + 1*1*1 + 0 + 0 + 0 + 0 + (-1)*0*0 = 2. Should be 1! ✗
    
    # Hmm, T[3,0,3] should be 1 (i=1,j=0,k=1 → C index 1*2+1=3).
    # Let me recheck.
    # U[3,:] = [1, 1, 0, 0, 0, 0, -1] (a11:1 from m1, a11:1 from m2, a11:-1 from m7)
    # V[0,:] = [1, 1, 0, -1, 0, 1, 0] (b00:1 from m1, b00:1 from m2, b00:-1 from m4, b00:1 from m6)
    # W[3,:] = [1, -1, 1, 1, 1, 1, 0] (c11:1 from m1, c11:-1 from m2, c11:1 from m3, c11:1 from m6, c11:1 from... wait)
    
    # Let me recheck W[3]:
    # m1: c11 coeff +1
    # m2: c11 coeff -1
    # m3: c11 coeff +1
    # m6: c11 coeff +1
    # m4: no c11
    # m5: no c11
    # m7: no c11
    # So W[3] = [1, -1, 1, 0, 0, 1, 0]
    
    # U[3,:] * V[0,:] * W[3,:]
    # = 1*1*1 + 1*1*(-1) + 0*0*1 + 0*(-1)*0 + 0*0*0 + 0*1*1 + (-1)*0*0
    # = 1 - 1 + 0 + 0 + 0 + 0 + 0 = 0. But should be 1!
    
    # I'm making errors. Let me be more systematic. I'll code it up and verify with NumPy.
    
    # Let me just code the Strassen terms and verify numerically.
    
    def strassen_2x2x2():
        """Return the 7 rank-one terms for 2x2x2 matrix multiplication."""
        # Each term is (U_list, V_list, W_list) of length 4 each.
        # U[i*2+j], V[j*2+k], W[k*2+i]
        terms = []
        # m1
        terms.append(([1,0,0,1], [1,1,1,1], [1,0,0,1]))
        # m2
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
        # m3
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
        # m4
        terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
        # m5
        terms.append(([1,1,0,0], [0,0,0,1], [0,-1,0,1]))
        # m6
        terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
        # m7
        terms.append(([0,1,0,-1], [0,0,1,1], [1,-1,0,0]))
        return terms
    
    # Verify
    import numpy as np
    terms = strassen_2x2x2()
    # Build the decomposed tensor
    T = np.zeros((4,4,4))
    for U_t, V_t, W_t in terms:
        U_t = np.array(U_t)
        V_t = np.array(V_t)
        W_t = np.array(W_t)
        T += np.einsum('a,b,c->abc', U_t, V_t, W_t)
    
    # Expected tensor: T[i*2+j, j*2+k, k*2+i] = 1
    T_expected = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_expected[i*2+j, j*2+k, k*2+i] = 1
    
    if not np.allclose(T, T_expected):
        print("Strassen verification failed!")
        print("T:\n", T)
        print("T_expected:\n", T_expected)
        # Fall back to schoolbook for 2x2x2
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_t = [0]*4; V_t = [0]*4; W_t = [0]*4
                    U_t[i*2+j] = 1
                    V_t[j*2+k] = 1
                    W_t[k*2+i] = 1
                    terms.append((U_t, V_t, W_t))
    
    # Now build the full 3x5x6 decomposition.
    # Strategy: partition n=3 into [2,1], m=5 into [2,2,1], p=6 into [2,2,2]
    # For each block combination (n_block, m_block, p_block), use the appropriate decomposition.
    # If all are 2: use Strassen (7 terms)
    # If any is 1: use schoolbook (n_block * m_block * p_block terms)
    
    # Actually, let me think about this differently.
    # The full tensor has indices (i,j,k) with i∈[3], j∈[5], k∈[6].
    # I'll partition:
    # i: block 0 = {0,1} (size 2), block 1 = {2} (size 1)
    # j: block 0 = {0,1} (size 2), block 1 = {2,3} (size 2), block 2 = {4} (size 1)
    # k: block 0 = {0,1} (size 2), block 1 = {2,3} (size 2), block 2 = {4,5} (size 2)
    
    # For each combination of (i_block, j_block, k_block), the sub-tensor is of size
    # n_sub × m_sub × p_sub.
    # If n_sub=2, m_sub=2, p_sub=2: use Strassen (7 terms)
    # Otherwise: use schoolbook (n_sub * m_sub * p_sub terms)
    
    # Total rank:
    # (2,2,2) blocks: i_block=0, j_block in {0,1}, k_block in {0,1,2} → 3 blocks, each 7 terms → 21
    # (2,2,1) blocks: none (no p_block of size 1)
    # (1,2,2) blocks: i_block=1, j_block in {0,1}, k_block in {0,1,2} → 3 blocks, each 1*2*2=4 terms → 12
    # (1,1,2) blocks: i_block=1, j_block=2, k_block in {0,1,2} → 3 blocks, each 1*1*2=2 terms → 6
    # (2,1,2) blocks: i_block=0, j_block=2, k_block in {0,1,2} → 3 blocks, each 2*1*2=4 terms → 12
    # (1,2,1): none
    # (2,1,1): none
    # (1,1,1): none
    
    # Total: 21 + 12 + 6 + 12 = 51
    
    # That's a big improvement from 90!
    
    # Now let's implement this.
    # For each block, I need to map the local indices to global indices.
    
    # Define the partitions
    i_blocks = [0, 1]  # block 0: indices 0,1; block 1: index 2
    i_block_sizes = [2, 1]
    i_block_starts = [0, 2]
    
    j_blocks = [0, 1, 2]
    j_block_sizes = [2, 2, 1]
    j_block_starts = [0, 2, 4]
    
    k_blocks = [0, 1, 2]
    k_block_sizes = [2, 2, 2]
    k_block_starts = [0, 2, 4]
    
    # Collect all terms
    all_terms = []  # Each term is (U_dict, V_dict, W_dict) where keys are global indices
    
    for ib in range(2):
        for jb in range(3):
            for kb in range(3):
                n_sub = i_block_sizes[ib]
                m_sub = j_block_sizes[jb]
                p_sub = k_block_sizes[kb]
                i_start = i_block_starts[ib]
                j_start = j_block_starts[jb]
                k_start = k_block_starts[kb]
                
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    # Use Strassen
                    for U_t, V_t, W_t in strassen_2x2x2():
                        U_dict = {}
                        V_dict = {}
                        W_dict = {}
                        for local_idx, val in enumerate(U_t):
                            if val != 0:
                                li = local_idx // 2
                                lj = local_idx % 2
                                gi = i_start + li
                                gj = j_start + lj
                                U_dict[gi * m + gj] = val
                        for local_idx, val in enumerate(V_t):
                            if val != 0:
                                lj = local_idx // 2
                                lk = local_idx % 2
                                gj = j_start + lj
                                gk = k_start + lk
                                V_dict[gj * p + gk] = val
                        for local_idx, val in enumerate(W_t):
                            if val != 0:
                                lk = local_idx // 2
                                li = local_idx % 2
                                gk = k_start + lk
                                gi = i_start + li
                                W_dict[gk * n + gi] = val
                        all_terms.append((U_dict, V_dict, W_dict))
                else:
                    # Use schoolbook
                    for li in range(n_sub):
                        for lj in range(m_sub):
                            for lk in range(p_sub):
                                gi = i_start + li
                                gj = j_start + lj
                                gk = k_start + lk
                                U_dict = {gi * m + gj: 1}
                                V_dict = {gj * p + gk: 1}
                                W_dict = {gk * n + gi: 1}
                                all_terms.append((U_dict, V_dict, W_dict))
    
    rank = len(all_terms)
    
    # Convert to matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for idx, val in U_dict.items():
            U[idx][c] = val
        for idx, val in V_dict.items():
            V[idx][c] = val
        for idx, val in W_dict.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
