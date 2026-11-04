"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Correct rank-7 decomposition for 2x2 * 2x2 -> 2x2
    # Verified by direct expansion. Each entry is (u[i,j], v[j,k], w[i,k])
    # where indices are 0-based: (0,0)=(a11), (0,1)=(a12), (1,0)=(a21), (1,1)=(a22)
    #
    # Term 0: u[0,0]=1, v[0,0]=1, v[0,1]=1, w[0,0]=1, w[0,1]=1  => a11*(b11+b12) contributes to c11,c12
    # Term 1: u[1,0]=1, v[0,0]=1, v[1,1]=1, w[1,0]=1, w[1,1]=1  => a21*(b11+b22) contributes to c21,c22
    # Term 2: u[0,1]=1, v[1,0]=1, v[1,1]=1, w[0,0]=1, w[0,1]=1  => a12*(b21+b22) contributes to c11,c12
    # Term 3: u[1,1]=1, v[1,0]=1, v[0,1]=1, w[1,0]=1, w[1,1]=1  => a22*(b21+b12) contributes to c21,c22
    # Term 4: u[0,0]=1, v[1,1]=1, w[0,0]=1                          => a11*b22 contributes to c11
    # Term 5: u[1,1]=1, v[0,0]=1, w[1,1]=1                          => a22*b11 contributes to c22
    # Term 6: u[0,1]=1, v[1,0]=1, w[0,1]=1                          => a12*b21 contributes to c12
    # Term 7: u[1,0]=1, v[0,1]=1, w[1,0]=1                          => a21*b12 contributes to c21
    #
    # Verification:
    # c11 = t0 + t2 - t4 = a11*b11 + a11*b12 + a12*b21 + a12*b22 - a11*b22
    #     = a11*b11 + a12*b21 + a11*b12 + a12*b22 - a11*b22
    #     Hmm, that has extra terms. Let me re-verify.
    #
    # Actually let me be very careful. The decomposition must satisfy:
    # For each (i,k) in C: sum_r w_r[i,k] * u_r[i,j] * v_r[j,k] = sum_j A[i,j]*B[j,k]
    #
    # Let me define the 7 terms explicitly as sparse dicts and verify computationally.
    #
    # Known correct rank-7 2x2 MM decomposition (Bini et al. 1979, or equivalent):
    # Using the "Strassen-style" rank-7:
    #
    # I'll define it as a list of 7 tuples, each with (u_dict, v_dict, w_dict)
    # where keys are (row, col) indices.

    def _build_rank7_2x2():
        """Return list of 7 (u, v, w) dicts for 2x2 matrix multiplication.
        u: {(i,j): val}, v: {(j,k): val}, w: {(i,k): val}
        """
        terms = []
        # Term 0: a11 * (b11 + b12) -> c11, c12
        u0 = {(0,0): 1}
        v0 = {(0,0): 1, (0,1): 1}
        w0 = {(0,0): 1, (0,1): 1}
        terms.append((u0, v0, w0))
        
        # Term 1: a21 * (b11 + b22) -> c21, c22
        u1 = {(1,0): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(1,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: a12 * (b21 + b22) -> c11, c12
        u2 = {(0,1): 1}
        v2 = {(1,0): 1, (1,1): 1}
        w2 = {(0,0): 1, (0,1): 1}
        terms.append((u2, v2, w2))
        
        # Term 3: a22 * (b21 + b12) -> c21, c22
        u3 = {(1,1): 1}
        v3 = {(1,0): 1, (0,1): 1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: a11 * b22 -> c11
        u4 = {(0,0): 1}
        v4 = {(1,1): 1}
        w4 = {(0,0): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: a22 * b11 -> c22
        u5 = {(1,1): 1}
        v5 = {(0,0): 1}
        w5 = {(1,1): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: a12 * b21 -> c12, AND a21 * b12 -> c21 (combined with sign)
        # We need: c12 gets a12*b21, c21 gets a21*b12
        # But we only have 1 term left. We need to encode both.
        # Trick: use the fact that we can have w with entries in different positions.
        # u6: a12 component, v6: b21 component -> but we also need a21*b12 for c21.
        # 
        # Actually, the 7th term in standard rank-7 is:
        # (a12 - a21) * (b21 - b12) which expands to:
        # a12*b21 - a12*b12 - a21*b21 + a21*b12
        # This contributes to multiple c entries.
        #
        # Let me instead just verify my 7 terms above give the right result.
        # For c[0,0] (c11): sum of w[0,0] * u[0,j] * v[j,0] over j, summed over terms
        # Term 0: w[0,0]=1, u[0,0]=1, v[0,0]=1 => 1*1*1 = 1 (contributes a11*b11)
        # Term 2: w[0,0]=1, u[0,1]=1, v[1,0]=1 => 1*1*1 = 1 (contributes a12*b21)
        # Term 4: w[0,0]=1, u[0,0]=1, v[0,0]=0 (v4 only has (1,1)) => 0
        # Other terms: w[0,0]=0
        # So c11 = a11*b11 + a12*b21. CORRECT!
        #
        # For c[0,1] (c12): 
        # Term 0: w[0,1]=1, u[0,0]=1, v[0,1]=1 => a11*b12
        # Term 0: w[0,1]=1, u[0,1]=0 => 0
        # Term 2: w[0,1]=1, u[0,1]=1, v[1,1]=1 => a12*b22
        # Term 2: w[0,1]=1, u[0,0]=0 => 0
        # So c12 = a11*b12 + a12*b22. CORRECT!
        #
        # For c[1,0] (c21):
        # Term 1: w[1,0]=1, u[1,0]=1, v[0,0]=1 => a21*b11
        # Term 3: w[1,0]=1, u[1,1]=1, v[1,0]=1 => a22*b21
        # So c21 = a21*b11 + a22*b21. CORRECT!
        #
        # For c[1,1] (c22):
        # Term 1: w[1,1]=1, u[1,0]=1, v[0,1]=0 (v1 has (0,0) and (1,1)) => 0
        # Term 1: w[1,1]=1, u[1,1]=0 => 0
        # Term 3: w[1,1]=1, u[1,1]=1, v[0,1]=1 => a22*b12
        # Term 3: w[1,1]=1, u[1,0]=0 => 0
        # Term 5: w[1,1]=1, u[1,1]=1, v[0,0]=1 => a22*b11
        # So c22 = a22*b12 + a22*b11. 
        # Should be: a21*b12 + a22*b22. INCORRECT!
        
        # I made an error. Let me fix term 1 and term 3.
        # The issue is that my terms don't correctly capture all 8 schoolbook products.
        #
        # Let me restart with a verified construction.
        # The 8 schoolbook terms are:
        # c11: a11*b11, a12*b21
        # c12: a11*b12, a12*b22
        # c21: a21*b11, a22*b21
        # c22: a21*b12, a22*b22
        #
        # Rank-7 means we save 1 term. The standard saving is:
        # Instead of having separate terms for a11*b12 and a22*b22 (or similar),
        # we combine them.
        #
        # Correct rank-7 (verified in multiple sources):
        # m1 = a11 * (b11 + b12)
        # m2 = (a11 + a21) * b12
        # m3 = a21 * (b21 + b22)
        # m4 = (a12 + a22) * b21
        # m5 = a12 * (b11 + b22)
        # m6 = (a12 - a21) * (b21 - b12)
        # m7 = a22 * (b11 + b22)
        #
        # c11 = m1 - m2 + m5 - m6
        # c12 = m2 + m7 - m5 + m6  [hmm, let me verify]
        # c21 = m3 - m4 + m5 + m6
        # c22 = m4 + m7 - m3 - m6  [needs verification]
        #
        # This is getting complicated. Let me just use a computational approach:
        # I'll set up the 7 terms and verify they work.

        # Let me use a different, simpler verified rank-7:
        # From Bini, Capovani, Lotti, Romani (1979):
        # 
        # Actually, let me just carefully construct it.
        # We need 7 bilinear forms m_r such that each c[i,k] is a linear combo.
        #
        # Simplest correct rank-7 (I'll verify each c entry):
        #
        # t0: u=(a11), v=(b11+b12), w contributes to c11 with +1, c12 with +1
        # t1: u=(a21), v=(b11+b22), w contributes to c21 with +1, c22 with +1  
        # t2: u=(a12), v=(b21+b22), w contributes to c11 with +1, c12 with +1
        # t3: u=(a22), v=(b12+b21), w contributes to c21 with +1, c22 with +1
        # t4: u=(a11), v=(b22), w contributes to c11 with -1
        # t5: u=(a22), v=(b11), w contributes to c22 with -1
        # t6: u=(a12), v=(b21), w contributes to c12 with +1
        #     AND u=(a21), v=(b12), w contributes to c21 with +1
        #     But this is TWO rank-1 terms in one slot! Not allowed.
        #
        # So I can't combine t6 that way. I need exactly 7 separate rank-1 terms.
        #
        # Let me count: I have t0-t5 (6 terms) and need 1 more.
        # The 8 schoolbook products are:
        # a11*b11 (in t0), a11*b12 (in t0), a21*b11 (in t1), a21*b22 (in t1)
        # a12*b21 (in t2), a12*b22 (in t2), a22*b12 (in t3), a22*b21 (in t3)
        #
        # Wait, t0 gives a11*b11 AND a11*b12. That's 2 products in 1 term.
        # t1 gives a21*b11 AND a21*b22. 2 products in 1 term.
        # t2 gives a12*b21 AND a12*b22. 2 products in 1 term.
        # t3 gives a22*b12 AND a22*b21. 2 products in 1 term.
        # That's 8 products in 4 terms. Then t4, t5 are corrections.
        #
        # Let me verify c11 = a11*b11 + a12*b21:
        # From t0: w[0,0]=1, so a11*b11 + a11*b12
        # From t2: w[0,0]=1, so a12*b21 + a12*b22
        # From t4: w[0,0]=-1, so -a11*b22
        # Total c11 = a11*b11 + a11*b12 + a12*b21 + a12*b22 - a11*b22
        # = a11*b11 + a12*b21 + a11*b12 + a12*b22 - a11*b22
        # This has extra terms a11*b12, a12*b22, -a11*b22. NOT correct.
        #
        # I see the problem. My "combined" terms don't work because the w vector
        # multiplies the ENTIRE product. I need to be more careful.
        #
        # Let me use a completely different, known-correct rank-7 decomposition.
        # 
        # From the literature, one valid rank-7 for 2x2 MM:
        # (using 0-based indexing)
        #
        # I'll define it as: for each of 7 terms, specify which (i,j) of A,
        # which (j,k) of B, and which (i,k) of C are involved.
        #
        # A valid rank-7 decomposition (verified computationally):
        
        # Let me just hardcode a verified one. I'll use the following:
        # Each term is (u_dict, v_dict, w_dict)
        
        # Term 0: a11*b11 -> c11
        t0 = ({(0,0):1}, {(0,0):1}, {(0,0):1})
        # Term 1: a12*b21 -> c11
        t1 = ({(0,1):1}, {(1,0):1}, {(0,0):1})
        # Term 2: a11*b12 -> c12
        t2 = ({(0,0):1}, {(0,1):1}, {(0,1):1})
        # Term 3: a12*b22 -> c12
        t3 = ({(0,1):1}, {(1,1):1}, {(0,1):1})
        # Term 4: a21*b11 -> c21
        t4 = ({(1,0):1}, {(0,0):1}, {(1,0):1})
        # Term 5: a22*b21 -> c21
        t5 = ({(1,1):1}, {(1,0):1}, {(1,0):1})
        # Term 6: a21*b12 -> c22
        t6 = ({(1,0):1}, {(0,1):1}, {(1,1):1})
        # This is only 7 terms but misses a22*b22 for c22!
        # So this is rank 8 schoolbook with one missing. Not valid.
        #
        # The key insight for rank-7: one of the 8 products must be "free" or
        # derivable from the others via linear combination. But matrix multiplication
        # products are linearly independent as bilinear forms, so we can't just drop one.
        #
        # The rank-7 saving comes from combining two products into one bilinear form
        # using the structure of the w matrix.
        #
        # Let me look at this differently. The rank-7 decomposition exists because
        # the 2x2x2 multiplication tensor has rank 7. One explicit decomposition:
        #
        # I'll use the one from "Tensor rank and arithmetic complexity" by Landsberg:
        #
        # Actually, let me just implement it correctly by construction.
        # 
        # The 2x2 MM tensor T has T[i,j,k] = delta[i,k] * delta[j,j] (in appropriate basis).
        # No, T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k], which is always
        # the case, so T[i,j,k] = 1 for all i,j,k in {0,1}. That's the 8-dimensional space.
        #
        # Wait no. The tensor is T in A* x B* x C* where
        # T(A,B,C) = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
        # The rank-1 tensors are of the form u(A)*v(B)*w(C) where
        # u(A) = sum_{i,j} u[i,j]*A[i,j], v(B) = sum_{j,k} v[j,k]*B[j,k], w(C) = sum_{i,k} w[i,k]*C[i,k]
        #
        # For rank-1 term to contribute to the multiplication tensor, we need:
        # sum_{i,j,k} w[i,k]*u[i,j]*v[j,k] = sum_{i,j,k} delta_{ik} * A[i,j]*B[j,k]
        # Wait, that's not right either. The multiplication tensor is:
        # T = sum_{i,j,k} e_i^C ⊗ e_j^A ⊗ e_k^B (in appropriate tensor product space)
        # where e_i^C is the functional extracting C[i,?], etc.
        #
        # More precisely: T(A,B) = C where C[i,k] = sum_j A[i,j]*B[j,k]
        # As a trilinear form: T(A,B,C) = sum_{i,j,k} C[i,k]*A[i,j]*B[j,k]
        #
        # A rank-1 decomposition term is: u(A)*v(B)*w(C) = (sum_{i,j} u[i,j]*A[i,j]) * (sum_{j,k} v[j,k]*B[j,k]) * (sum_{i,k} w[i,k]*C[i,k])
        #
        # For this to equal part of T, we need the trilinear form to match.
        # The contribution to T from one rank-1 term is:
        # sum_{i,j} u[i,j] * sum_{j',k} v[j',k] * sum_{i',k'} w[i',k'] * A[i,j]*B[j',k]*C[i',k']
        # = sum_{i,j,j',k,i',k'} u[i,j]*v[j',k]*w[i',k'] * A[i,j]*B[j',k]*C[i',k']
        #
        # For this to be part of T = sum_{i,j,k} C[i,k]*A[i,j]*B[j,k],
        # we need: the coefficient of A[i,j]*B[j',k]*C[i',k'] in the rank-1 term
        # should be 1 if (i',k')=(i,k) and j'=j, and 0 otherwise (for the full sum over all rank-1 terms).
        #
        # This is complex. Let me just use a numerically verified decomposition.
        # I'll set up the problem as finding 7 rank-1 terms and verify.
        
        # Known correct rank-7 2x2 MM (from Bini et al., verified):
        # I'll use the following which I can verify:
        
        # Let me use a different approach: construct it from Strassen's 7 products
        # and verify each output entry.
        
        # Strassen's 7 products for 2x2:
        # p1 = (a11 + a22) * (b11 + b22)
        # p2 = (a21 + a22) * b11
        # p3 = a11 * (b12 - b22)
        # p4 = a22 * (b21 - b11)
        # p5 = (a11 + a12) * b22
        # p6 = (a21 - a11) * (b11 + b12)
        # p7 = (a12 - a22) * (b21 + b22)
        
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p6
        
        # Let me verify c12 = a11*b12 + a12*b22:
        # p3 + p5 = a11*(b12-b22) + (a11+a12)*b22 = a11*b12 - a11*b22 + a11*b22 + a12*b22 = a11*b12 + a12*b22. CORRECT!
        
        # Verify c21 = a21*b11 + a22*b21:
        # p2 + p4 = (a21+a22)*b11 + a22*(b21-b11) = a21*b11 + a22*b11 + a22*b21 - a22*b11 = a21*b11 + a22*b21. CORRECT!
        
        # Verify c11 = a11*b11 + a12*b21:
        # p1 + p4 - p5 + p7 = (a11+a22)(b11+b22) + a22(b21-b11) - (a11+a12)b22 + (a12-a22)(b21+b22)
        # = a11*b11 + a11*b22 + a22*b11 + a22*b22 + a22*b21 - a22*b11 - a11*b22 - a12*b22 + a12*b21 + a12*b22 - a22*b21 - a22*b22
        # = a11*b11 + a12*b21. CORRECT!
        
        # Verify c22 = a21*b12 + a22*b22:
        # p1 - p2 + p3 + p6 = (a11+a22)(b11+b22) - (a21+a22)b11 + a11(b12-b22) + (a21-a11)(b11+b12)
        # = a11*b11 + a11*b22 + a22*b11 + a22*b22 - a21*b11 - a22*b11 + a11*b12 - a11*b22 + a21*b11 + a21*b12 - a11*b11 - a11*b12
        # = a22*b22 + a21*b12. CORRECT!
        
        # Great! Now I can build the rank-7 decomposition from Strassen's products.
        # Each p_r is a product of a linear form in A and a linear form in B.
        #
        # p1 = (a11+a22)*(b11+b22): u={(0,0):1,(1,1):1}, v={(0,0):1,(1,1):1}
        # p2 = (a21+a22)*b11: u={(1,0):1,(1,1):1}, v={(0,0):1}
        # p3 = a11*(b12-b22): u={(0,0):1}, v={(0,1):1,(1,1):-1}
        # p4 = a22*(b21-b11): u={(1,1):1}, v={(1,0):1,(0,0):-1}
        # p5 = (a11+a12)*b22: u={(0,0):1,(0,1):1}, v={(1,1):1}
        # p6 = (a21-a11)*(b11+b12): u={(1,0):1,(0,0):-1}, v={(0,0):1,(0,1):1}
        # p7 = (a12-a22)*(b21+b22): u={(0,1):1,(1,1):-1}, v={(1,0):1,(1,1):1}
        
        # w contributions:
        # c11 = p1 + p4 - p5 + p7: w[0,0]: p1:+1, p4:+1, p5:-1, p7:+1
        # c12 = p3 + p5: w[0,1]: p3:+1, p5:+1
        # c21 = p2 + p4: w[1,0]: p2:+1, p4:+1
        # c22 = p1 - p2 + p3 + p6: w[1,1]: p1:+1, p2:-1, p3:+1, p6:+1
        
        terms = []
        # p1
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        # p2
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        # p3
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        # p4
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        # p5
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        # p6
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        # p7
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        
        return terms

    rank7_terms = _build_rank7_2x2()
    
    # Now build the full 3x4 * 4x4 decomposition with rank 36.
    # 
    # Structure:
    # A is 3x4, B is 4x4, C is 3x4.
    # Split A rows: top 2 (i=0,1), bottom 1 (i=2)
    # Split B cols: left 2 (k=0,1), right 2 (k=2,3)
    # Split middle dim j: j=0,1 and j=2,3
    #
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    #   = 2x2*2x2 + 2x2*2x2, each rank 7, total 14
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    #   = 2x2*2x2 + 2x2*2x2, each rank 7, total 14
    # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
    #   = 1x2*2x2 + 1x2*2x2, each rank 2, total 4
    # C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]
    #   = 1x2*2x2 + 1x2*2x2, each rank 2, total 4
    #
    # Total rank: 14 + 14 + 4 + 4 = 36

    total_rank = 36
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 36
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 36
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 36

    def add_rank7_block(a_row_offset, b_col_offset, j_offset, w_row_offset, col_start):
        """Add a rank-7 2x2*2x2 block.
        a_row_offset: row offset in A (0 or 1 for the 2x2 block)
        b_col_offset: col offset in B (0 or 2 for the 2x2 block)
        j_offset: row offset in B / col offset in A for the 2x2 block (0 or 2)
        w_row_offset: row offset in C (0 or 2 for the 2x2 block)
        col_start: starting column in the factor matrices
        """
        for r in range(7):
            u_dict, v_dict, w_dict = rank7_terms[r]
            col = col_start + r
            for (ai, aj), val in u_dict.items():
                U[a_row_offset + ai][col] += val
            for (bj, bk), val in v_dict.items():
                V[(j_offset + bj) * p + (b_col_offset + bk)][col] += val
            for (ci, ck), val in w_dict.items():
                W[(w_row_offset + ck) * n + (w_row_offset + ci)][col] += val
        return col_start + 7

    def add_rank2_block_1x2(a_row, b_col_offset, j_offset, w_row_offset, col_start):
        """Add a rank-2 1x2*2x2 block (for the bottom row of A).
        A[a_row, j_offset:j_offset+2] * B[j_offset:j_offset+2, b_col_offset:b_col_offset+2] -> C[a_row, b_col_offset:b_col_offset+2]
        """
        # Schoolbook: 2 terms
        # Term 0: A[a_row, j_offset] * B[j_offset, b_col_offset] -> C[a_row, b_col_offset]
        #          A[a_row, j_offset] * B[j_offset, b_col_offset+1] -> C[a_row, b_col_offset+1]
        # Term 1: A[a_row, j_offset+1] * B[j_offset+1, b_col_offset] -> C[a_row, b_col_offset]
        #          A[a_row, j_offset+1] * B[j_offset+1, b_col_offset+1] -> C[a_row, b_col_offset+1]
        for r in range(2):
            col = col_start + r
            # u: A[a_row, j_offset + r] = 1
            U[a_row * m + (j_offset + r)][col] = 1
            # v: B[j_offset + r, b_col_offset] = 1, B[j_offset + r, b_col_offset+1] = 1
            V[(j_offset + r) * p + b_col_offset][col] = 1
            V[(j_offset + r) * p + b_col_offset + 1][col] = 1
            # w: C[a_row, b_col_offset] = 1, C[a_row, b_col_offset+1] = 1
            W[b_col_offset * n + a_row][col] = 1
            W[(b_col_offset + 1) * n + a_row][col] = 1
        return col_start + 2

    col = 0
    # C[0:2, 0:2]: A[0:2, 0:2] * B[0:2, 0:2]
    col = add_rank7_block(0, 0, 0, 0, col)
    # C[0:2, 0:2]: A[0:2, 2:4] * B[2:4, 0:2]
    col = add_rank7_block(0, 0, 2, 0, col)
    # C[0:2, 2:4]: A[0:2, 0:2] * B[0:2, 2:4]
    col = add_rank7_block(0, 2, 0, 0, col)
    # C[0:2, 2:4]: A[0:2, 2:4] * B[2:4, 2:4]
    col = add_rank7_block(0, 2, 2, 0, col)
    # C[2:3, 0:2]: A[2:3, 0:2] * B[0:2, 0:2]
    col = add_rank2_block_1x2(2, 0, 0, 2, col)
    # C[2:3, 0:2]: A[2:3, 2:4] * B[2:4, 0:2]
    col = add_rank2_block_1x2(2, 0, 2, 2, col)
    # C[2:3, 2:4]: A[2:3, 0:2] * B[0:2, 2:4]
    col = add_rank2_block_1x2(2, 2, 0, 2, col)
    # C[2:3, 2:4]: A[2:3, 2:4] * B[2:4, 2:4]
    col = add_rank2_block_1x2(2, 2, 2, 2, col)

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify the decomposition
    try:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The decomposition says: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T[i,j,k] = 1 (for the multiplication tensor in the right basis)
        # 
        # Verification: for random A, B, check that C = A*B matches the reconstruction.
        # Reconstruction: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j'*p+k', r] * B[j',k']) * W[k*n+i, r]
        # Wait, that's not right. The reconstruction is:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
        # No, that's a product of three sums, which is cubic.
        #
        # Correct reconstruction: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # Still cubic.
        #
        # Actually, the standard tensor decomposition for MM is:
        # C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # No, that's not right either.
        #
        # The correct form: each rank-1 term r contributes:
        # (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{j,k} V[j*p+k, r] * B[j,k]) * (sum_{i,k} W[k*n+i, r] * C[i,k])
        # And the sum over r of these products equals the multiplication trilinear form.
        #
        # For verification, I'll check: for random A, B, the computed C should match A*B.
        # Using the bilinear form: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Wait, this is still not right. Let me think more carefully.
        #
        # The multiplication tensor T satisfies: T(A,B,C) = sum_{i,j,k} C[i,k]*A[i,j]*B[j,k]
        # A rank-1 decomposition: T = sum_r u_r(A) * v_r(B) * w_r(C)
        # where u_r(A) = sum_{i,j} U[i*m+j,r]*A[i,j], v_r(B) = sum_{j,k} V[j*p+k,r]*B[j,k], w_r(C) = sum_{i,k} W[k*n+i,r]*C[i,k]
        #
        # So: sum_{i,j,k} C[i,k]*A[i,j]*B[j,k] = sum_r (sum_{i,j} U[i*m+j,r]*A[i,j]) * (sum_{j,k} V[j*p+k,r]*B[j,k]) * (sum_{i,k} W[k*n+i,r]*C[i,k])
        #
        # This means: for any A, B, C: LHS = RHS.
        # In particular, if we set C = A*B (the true product), LHS = sum_{i,j,k} (A*B)[i,k]*A[i,j]*B[j,k]
        # And RHS should equal LHS.
        #
        # For verification, I can check that the bilinear form (A,B) -> C computed via the decomposition
        # equals A*B. The way to extract C from the decomposition:
        # Given A and B, compute for each r: a_r = sum_{i,j} U[i*m+j,r]*A[i,j], b_r = sum_{j,k} V[j*p+k,r]*B[j,k]
        # Then C[i,k] = sum_r W[k*n+i,r] * a_r * b_r
        #
        # Wait, that gives C[i,k] as a sum of products a_r*b_r weighted by W. But a_r and b_r are scalars.
        # So C[i,k] = sum_r W[k*n+i,r] * a_r * b_r. This is correct!
        
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(total_rank):
            a_r = sum(U_arr[i*m+j, r] * A_test[i,j] for i in range(n) for j in range(m))
            b_r = sum(V_arr[j*p+k, r] * B_test[j,k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_computed[i,k] += W_arr[k*n+i, r] * a_r * b_r
        
        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-9:
            print(f"Verification failed: max error = {max_err}")
            return {"U": U_school, "V": V_school, "W": W_school}
        print(f"Verification passed: max error = {max_err}")
    except Exception as e:
        print(f"Verification error: {e}")
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
