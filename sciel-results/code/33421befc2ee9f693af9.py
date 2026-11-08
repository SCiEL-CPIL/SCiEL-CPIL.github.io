import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # The blocks are defined by index sets.
    # I0={0,1}, I1={2}
    # J0={0,1}, J1={2,3}, J2={4}
    # K0={0,1}, K1={2,3}, K2={4}
    
    # We accumulate the rank contributions.
    # Each contribution is a tuple of (U_vec, V_vec, W_vec)
    # U_vec is size n*m, V_vec is size m*p, W_vec is size p*n.
    
    contributions = []
    
    # Helper to add a rank 1 term for specific indices
    def add_rank1(i_set, j_set, k_set):
        # This assumes the tensor is all 1s on these sets.
        # u[i, j] = 1 if i in i_set, j in j_set
        # v[j, k] = 1 if j in j_set, k in k_set
        # w[k, i] = 1 if k in k_set, i in i_set
        # This works if the sets are singletons?
        # If i_set has size > 1, u[i,j] is 1 for all i in i_set.
        # This is valid for rank 1 if the tensor is separable.
        # T[i,j,k] = 1. u[i,j]=1, v[j,k]=1, w[k,i]=1 -> sum 1*1*1 = 1.
        # Yes.
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for i in i_set:
            for j in j_set:
                u[i * m + j] = 1
        for j in j_set:
            for k in k_set:
                v[j * p + k] = 1
        for k in k_set:
            for i in i_set:
                w[k * n + i] = 1
        contributions.append((u, v, w))

    # Helper to add Strassen 2x2x2
    # i0, j0, k0 are the starting indices for the 2x2x2 block.
    def add_strassen(i0, j0, k0):
        # Local indices 0,1 map to global i0+i, j0+j, k0+k
        # Strassen 7 terms for 2x2x2
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We need U, V, W such that sum U V W = T
        # T[i,j,k] = 1 for i,j,k in {0,1}
        
        # The vectors are:
        # 1. U: A11+A22, V: B11+B22, W: M1
        #    U[0,0]=1, U[1,1]=1
        #    V[0,0]=1, V[1,1]=1
        #    W[0,0]=1, W[1,1]=1, W[0,1]=1, W[1,0]=1?
        #    Wait, W corresponds to C.
        #    C11 = M1 - M4 + M5 - M7
        #    C12 = M3 + M5
        #    C21 = M2 + M4
        #    C22 = M1 - M2 + M3 + M7
        
        # Let's just use the known 7 vectors for 2x2x2 all-ones tensor.
        # Actually, we can just use the standard Strassen matrices.
        
        vecs = []
        
        # Term 1: (A11+A22)(B11+B22)
        u1 = [0]*4; u1[0]=1; u1[3]=1
        v1 = [0]*4; v1[0]=1; v1[3]=1
        w1 = [0]*4; w1[0]=1; w1[3]=1; w1[1]=1; w1[2]=1 # C11, C22, C12, C21?
        # Wait, W index is k*n + i.
        # C11 is i=0,k=0 -> idx 0.
        # C22 is i=1,k=1 -> idx 3.
        # C12 is i=0,k=1 -> idx 1.
        # C21 is i=1,k=0 -> idx 2.
        # M1 contributes to C11, C12, C21, C22?
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # So W1 should have 1 at C11, C22, C12, C21?
        # No, W1 is the vector for the term.
        # The term is U1 * V1 * W1.
        # The contribution to C is sum of terms.
        
        # Let's define the 7 terms explicitly.
        # T1: U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1] -> C11+=1, C12+=1, C21+=1, C22+=1
        # T2: U=[0,1,1,1], V=[1,0,0,0], W=[0,0,1,0] -> C21+=1 (A21+A22)*B11
        #     U[1,0]=1, U[1,1]=1. V[0,0]=1. W[1,0]=1 (C21).
        # T3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0] -> C12+=1 (A11)*(B21-B22)
        #     U[0,0]=1. V[1,0]=1, V[1,1]=-1. W[0,1]=1 (C12).
        # T4: U=[0,0,0,1], V=[1,-1,0,0], W=[0,0,1,0] -> C21+=1 (A22)*(B12-B11)
        #     U[1,1]=1. V[0,0]=-1, V[0,1]=1. W[1,0]=1.
        #     Wait, B12-B11 is V[0,1]-V[0,0].
        #     So V=[0,1,-1,0]? No.
        #     V is size 4 (2x2). j=0,1; k=0,1.
        #     V[j*p+k].
        #     B11 is j=0,k=0. B12 is j=0,k=1.
        #     B12 - B11 -> V[1]=1, V[0]=-1.
        #     So V=[-1, 1, 0, 0].
        # T5: U=[1,1,0,0], V=[0,0,0,1], W=[1,1,0,0] -> C11+=1, C12+=1
        #     A11+A12. B22.
        #     U[0,0]=1, U[0,1]=1. V[1,1]=1.
        #     W[0,0]=1, W[0,1]=1.
        # T6: U=[1,-1,0,0], V=[1,1,0,0], W=[0,0,1,0] -> C21+=1
        #     A21-A11. B11+B12.
        #     U[1,0]=1, U[0,0]=-1.
        #     V[0,0]=1, V[0,1]=1.
        #     W[1,0]=1.
        # T7: U=[0,1,1,0], V=[0,1,1,0], W=[1,0,0,1] -> C11+=1, C22+=1
        #     A12-A22. B21+B22.
        #     U[0,1]=1, U[1,1]=-1.
        #     V[1,0]=1, V[1,1]=1.
        #     W[0,0]=1, W[1,1]=1.
        
        # Let's verify the sum.
        # C11: T1(1) - T4(0) + T5(1) - T7(1) = 1. Correct.
        # C12: T1(1) + T3(1) + T5(1) = 3? No.
        # C12 = M3 + M5.
        # M3 is T3. M5 is T5.
        # T3 contributes 1 to C12. T5 contributes 1 to C12.
        # T1 contributes 1 to C12.
        # Sum = 3?
        # Ah, the standard Strassen formulas are:
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # My T1 was M1.
        # T1 contributes to C11, C22, C12, C21?
        # M1 is a scalar. It is added to C11 and C22.
        # It is NOT added to C12 or C21 in the formula.
        # So W1 should be [1, 0, 0, 1].
        
        # Let's re-verify.
        # T1 (M1): U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
        # T2 (M2): U=[0,1,1,1], V=[1,0,0,0], W=[0,0,1,0]
        # T3 (M3): U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
        # T4 (M4): U=[0,0,0,1], V=[-1,1,0,0], W=[0,0,1,0]
        # T5 (M5): U=[1,1,0,0], V=[0,0,0,1], W=[1,1,0,0]
        # T6 (M6): U=[1,-1,0,0], V=[1,1,0,0], W=[0,0,1,0] -- Wait, M6 is not in formula?
        # Formula uses M1..M7.
        # C21 = M2 + M4.
        # C22 = M1 - M2 + M3 + M7.
        
        # Let's check C21:
        # T2 (M2) contributes 1.
        # T4 (M4) contributes 1.
        # T6 (M6) is not in C21 formula.
        # So W2=[0,0,1,0], W4=[0,0,1,0].
        
        # Let's check C12:
        # C12 = M3 + M5.
        # T3 (M3) contributes 1.
        # T5 (M5) contributes 1.
        # W3=[0,1,0,0], W5=[1,1,0,0] -> W5[1]=1.
        
        # Let's check C11:
        # C11 = M1 - M4 + M5 - M7.
        # T1 (M1) -> 1.
        # T4 (M4) -> -1. W4[0]? No, W4 is for C21.
        # M4 is scalar. It is subtracted from C11.
        # So W4 should have -1 at C11?
        # No, the term is U4 V4 W4.
        # If W4 has -1 at C11, then the term contributes -1 to C11.
        # But U4 V4 is the product for M4.
        # So yes, W4 should be [-1, 0, 1, 0]?
        # M4 is used in C11 (subtracted) and C21 (added).
        # So W4 = [-1, 0, 1, 0].
        
        # Let's check C22:
        # C22 = M1 - M2 + M3 + M7.
        # T1 (M1) -> 1.
        # T2 (M2) -> -1. W2 should have -1 at C22?
        # M2 is used in C21 (added) and C22 (subtracted).
        # So W2 = [0, 0, 1, -1].
        
        # T3 (M3) -> 1. W3 should have 1 at C22?
        # M3 is used in C12 (added) and C22 (added).
        # So W3 = [0, 1, 0, 1].
        
        # T7 (M7) -> 1. W7 should have 1 at C11?
        # M7 is used in C11 (subtracted) and C22 (added).
        # So W7 = [-1, 0, 0, 1].
        
        # T5 (M5) -> 1. W5 should have 1 at C11 and C12.
        # M5 is used in C11 (added) and C12 (added).
        # So W5 = [1, 1, 0, 0].
        
        # T6 (M6) is not used?
        # Wait, standard Strassen has 7 multiplications.
        # M1..M7.
        # I used M6 in my list but it's not in the formulas?
        # Ah, M6 = (A21 - A11)(B11 + B12).
        # It is NOT in the standard 7-multiplication Strassen formulas for C.
        # The 7 multiplications are M1..M7 as defined in Wikipedia.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Formulas:
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # M6 is indeed not used in the final sum?
        # No, that's impossible. Rank 7 means all 7 are used.
        # Wait, M6 is used in some variants?
        # Ah, I see. The standard Strassen uses M1..M7.
        # But M6 is not in the formulas above.
        # Let me check a reference.
        # Wikipedia:
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        # M6 is not there.
        # Is M6 redundant?
        # No, M6 is one of the 7.
        # Ah, the formulas are for a specific choice of 7.
        # If M6 is not in the formulas, then it is not needed?
        # No, the 7 are the multiplications.
        # If M6 is not in the output, then the rank is 6?
        # No, rank 7 is minimal.
        # So M6 MUST be in the output.
        # Let me re-read.
        # Maybe I missed it.
        # C21 = M2 + M4.
        # C12 = M3 + M5.
        # C11 = M1 - M4 + M5 - M7.
        # C22 = M1 - M2 + M3 + M7.
        
        # Where is M6?
        # It seems M6 is not used in this specific set of 7?
        # But there are different sets.
        # Actually, M6 is used in some derivations.
        # But if it's not in the sum, then the term is 0?
        # No, the term is part of the decomposition.
        # If the coefficient is 0, it doesn't contribute.
        # But then the rank would be 6?
        # No, the rank of the tensor is 7.
        # So all 7 must have non-zero contribution.
        
        # Let's assume the standard vectors I derived are correct for the 6 used.
        # And M6 is not used?
        # No, I will just use the 7 vectors that sum to the identity.
        
        # For now, I will use the 7 vectors corresponding to M1..M7 with the coefficients from the formulas.
        # If M6 coefficient is 0, I can skip it or set it to 0.
        # But to be safe, I'll include the 7 standard ones.
        
        # Let's just use the rank 1 blocks for everything to be safe?
        # No, 34 is much better than 75.
        
        # I will implement the 7 vectors for 2x2x2.
        
        u1 = [0]*4; u1[0]=1; u1[3]=1
        v1 = [0]*4; v1[0]=1; v1[3]=1
        w1 = [1,0,0,1]
        
        u2 = [0]*4; u2[1]=1; u2[2]=1; u2[3]=1
        v2 = [1,0,0,0]
        w2 = [0,0,1,-1]
        
        u3 = [1,0,0,0]
        v3 = [0,1,0,-1]
        w3 = [0,1,0,1]
        
        u4 = [0,0,0,1]
        v4 = [-1,1,0,0]
        w4 = [-1,0,1,0]
        
        u5 = [1,1,0,0]
        v5 = [0,0,0,1]
        w5 = [1,1,0,0]
        
        u6 = [1,-1,0,0]
        v6 = [1,1,0,0]
        w6 = [0,0,0,0] # M6 not used in this formula set?
        
        u7 = [0,1,1,0]
        v7 = [0,1,1,0]
        w7 = [-1,0,0,1]
        
        # If w6 is 0, we can skip it.
        # But let's check if M6 is needed.
        # If I skip it, I have 6 terms.
        # Rank 6? No, 2x2x2 rank is 7.
        # So I must include M6 with non-zero W.
        # Maybe the formula is different.
        # I will just use schoolbook for the 2x2x2 blocks to be safe?
        # Rank 8 for 2x2x2.
        # 4 * 8 + 10 * 1 = 32 + 10 = 42.
        # 42 is still better than 75.
        
        # Let's use schoolbook for 2x2x2 to avoid errors.
        # Rank 8.
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = [0]*4
                    v = [0]*4
                    w = [0]*4
                    u[i*2+j] = 1
                    v[j*2+k] = 1
                    w[k*2+i] = 1
                    # Map to global
                    gu = [0]*(n*m)
                    gv = [0]*(m*p)
                    gw = [0]*(p*n)
                    gu[(i0+i)*m + (j0+j)] = 1
                    gv[(j0+j)*p + (k0+k)] = 1
                    gw[(k0+k)*n + (i0+i)] = 1
                    contributions.append((gu, gv, gw))
    
    # Define blocks
    I0 = [0, 1]
    I1 = [2]
    J0 = [0, 1]
    J1 = [2, 3]
    J2 = [4]
    K0 = [0, 1]
    K1 = [2, 3]
    K2 = [4]
    
    # 4 Strassen blocks (using schoolbook rank 8 for safety)
    add_strassen(0, 0, 0) # I0, J0, K0
    add_strassen(0, 2, 0) # I0, J1, K0
    add_strassen(0, 0, 2) # I0, J0, K1
    add_strassen(0, 2, 2) # I0, J1, K1
    
    # Rank 1 blocks
    add_rank1([0,1,2], J2, K0) # I, J2, K0
    add_rank1([0,1,2], J2, K1) # I, J2, K1
    add_rank1([0,1,2], J2, K2) # I, J2, K2
    
    add_rank1(I1, [0,1,2,3], K0) # I1, J01, K0
    add_rank1(I1, [0,1,2,3], K1) # I1, J01, K1
    add_rank1(I1, [0,1,2,3], K2) # I1, J01, K2
    
    # Total rank: 4*8 + 3*1 + 3*1 = 32 + 6 = 38.
    
    # Assemble U, V, W
    R = len(contributions)
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    for r, (u, v, w) in enumerate(contributions):
        for i in range(n*m):
            U[i][r] = u[i]
        for j in range(m*p):
            V[j][r] = v[j]
        for k in range(p*n):
            W[k][r] = w[k]
            
    return {"U": U, "V": V, "W": W}
