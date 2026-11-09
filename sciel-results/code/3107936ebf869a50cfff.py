"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 8
    deadline = time.time() + 115
    
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify exact decomposition: sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] == 1 for all i,k,j"""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    coeff = np.sum(U_arr[i*m+k, :] * V_arr[k*p+j, :] * W_arr[j*n+i, :])
                    if abs(coeff - 1.0) > 1e-10:
                        return False
        return True

    # Strategy: Block Strassen construction
    # View 2x4 * 4x8 as block 2x2 where entries are matrices:
    # A = [[A00, A01], [A10, A11]] where Aij is 1x2 (row i, columns 2j..2j+1)
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where Bij is 1x2
    # Actually let's be more careful:
    # A is 2x4: rows i=0,1; columns k=0,1,2,3
    # B is 4x8: rows k=0,1,2,3; columns j=0..7
    # C is 2x8: rows i=0,1; columns j=0..7
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    
    # Block decomposition: split k into two groups of 2:
    # A = [A_L | A_R] where A_L is 2x2 (cols 0,1), A_R is 2x2 (cols 2,3)
    # B = [B_T; B_B] where B_T is 2x8 (rows 0,1), B_B is 2x8 (rows 2,3)
    # C = A_L @ B_T + A_R @ B_B
    
    # Now each term A_L @ B_T is 2x2 * 2x8, and A_R @ B_B is 2x2 * 2x8
    # For a 2x2 * 2x8 multiplication, we can use Strassen on the 2x2 part
    # 2x2 * 2x8: C[i,j] = sum_{k=0,1} A[i,k] * B[k,j] for i in {0,1}, j in {0..7}
    # This is equivalent to multiplying a 2x2 matrix by 8 independent 2x1 vectors
    # Using Strassen: rank 7 for each 2x2*2x1, so 7*4 = 28 for 2x2*2x8? No...
    # Actually, Strassen applies to the 2x2 core. For 2x2*2x8:
    # We have 8 output columns, each requiring a 2x2*2x1 computation
    # But we can share the Strassen factors across columns!
    # The 7 Strassen multiplications are: M1=A00*B00, M2=(A00+A01)*B10, etc.
    # For 2x2*2x8: M1 = A00 * B00 (2x1), M2 = (A00+A01) * B10 (2x1), etc.
    # Each Mk is a scalar * (2x1 vector) = 2x1 vector
    # So we need 7 "scalar times vector" operations, each producing a 2x1 result
    # Total rank contribution: 7 * 2 = 14? No, that's not right either.
    
    # Let me think about this differently using the tensor decomposition framework.
    # For 2x2 * 2x8, the tensor T has entries T[i,k,j] = 1 for i in {0,1}, k in {0,1}, j in {0..7}
    # That's 2*2*8 = 32 entries, all equal to 1.
    # We want: sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] = 1 for all i,k,j
    # where U is 2x2=4 entries per term, V is 2x8=16 entries per term, W is 8x2=16 entries per term
    
    # Using Strassen for 2x2*2x2 (rank 7), we can extend to 2x2*2x8:
    # The 2x2*2x8 tensor = sum over 4 column-pairs of 2x2*2x2 tensors
    # Each 2x2*2x2 has rank 7 (Strassen), so total rank = 4*7 = 28
    # But we can do better by sharing Strassen factors.
    
    # Actually, let's just use the direct Strassen extension:
    # For 2x2*2x8: C[i,j] = sum_k A[i,k]*B[k,j]
    # Strassen gives: C = f(A, B) using 7 products
    # For the extended case, each "product" A[i,k]*B[k,j] becomes a vector product
    # Specifically, define for each Strassen factor s in {1..7}:
    #   U_s[i,k] = coefficient of A[i,k] in factor s
    #   V_s[k,j] = coefficient of B[k,j] in factor s  
    #   W_s[j,i] = coefficient needed to reconstruct C[i,j]
    # The key insight: for 2x2*2x8, we can use the SAME Strassen U and V patterns
    # for all 8 columns, and the W pattern repeats.
    
    # Strassen's 7 factors for 2x2*2x2:
    # M1 = A00 * B00
    # M2 = (A00 + A01) * B10
    # M3 = A01 * (B01 - B11)
    # M4 = A10 * (B00 + B11)
    # M5 = (A10 + A00) * (B11 - B00)
    # M6 = (A11 - A10) * (B00 + B10)
    # M7 = (A00 - A01) * (B11 + B10)
    # C00 = M1 + M4 + M5 - M6 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 + M3 + M5 - M2 - M4
    
    # For 2x2*2x8, B is 2x8 instead of 2x2.
    # B has rows k=0,1 and columns j=0..7.
    # The Strassen factors become:
    # M1 = A00 * B[0, :]  (scalar * 1x8 vector = 1x8)
    # M2 = (A00+A01) * B[1, :]
    # M3 = A01 * (B[0,1:3] - B[1,1:3])  -- wait, this doesn't work directly
    # The issue is that Strassen's factors involve specific column combinations.
    
    # Let me reconsider. For 2x2*2x8:
    # C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
    # C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j]
    # for j = 0..7
    
    # This is just 8 independent 2x2*2x1 multiplications.
    # Each 2x2*2x1 has tensor rank at most 4 (schoolbook) or possibly 3.
    # So 2x2*2x8 has rank at most 8*4 = 32 (schoolbook) or 8*3 = 24 if rank-3 works.
    
    # But we can do better! Note that all 8 sub-problems share the same A matrix.
    # The Strassen trick for 2x2*2x2 uses 7 products. For 2x2*2x8:
    # We can define 7 "vector products":
    # P1 = A[0,0] * B[:, 0:8]  -- no, this mixes things up
    
    # Let me just directly construct using the Kronecker product approach.
    # 2x2*2x8 tensor = (2x2*2x1) ⊗ (1x1*1x8) in some sense
    # Actually, T[i,k,j] = 1 = T_2x2x2[i,k,j'] * T_2x1x8[k',j] where j' = j mod 2, k' = k
    # This isn't quite right either.
    
    # Simplest correct approach: use schoolbook for the inner 2x2*2x1 (rank 4 each)
    # and apply to all 8 columns: total rank = 4 * 8 = 32 for 2x2*2x8
    # Then for 2x4*4x8: two such blocks, total rank = 2 * 32 = 64 (same as schoolbook)
    
    # To improve, we need to use Strassen at the 2x2 level.
    # Key insight: For 2x2*2x8, we can use rank 7 * 4 = 28? No.
    # Actually, the correct way: 2x2*2x8 can be seen as applying a 2x2*2x2 Strassen
    # to each group of 2 columns. So 4 groups * rank 7 = 28.
    
    # Let me verify: 2x2*2x2 has Strassen rank 7.
    # 2x2*2x8 = 4 copies of 2x2*2x2 (columns 0-1, 2-3, 4-5, 6-7)
    # Each copy uses rank 7, total = 28.
    # Then 2x4*4x8 = 2x2*2x8 + 2x2*2x8 = 28 + 28 = 56.
    
    # Let's construct this!
    # For each block (which is 2x2*2x2), we use Strassen's 7 factors.
    
    # Strassen factors for 2x2*2x2 (standard):
    # Factor s: U_s[0,0], U_s[0,1], U_s[1,0], U_s[1,1], V_s[0,0], V_s[0,1], V_s[1,0], V_s[1,1], W_s[0,0], W_s[0,1], W_s[1,0], W_s[1,1]
    # where C[i,j] = sum_s U_s[i,0]*V_s[0,j]*W_s[j,i] + U_s[i,1]*V_s[1,j]*W_s[j,i]
    # Wait, I need to be careful with the indexing.
    
    # In our problem: U[i*m+k, r], V[k*p+j, r], W[j*n+i, r]
    # For a 2x2*2x2 block: i in {0,1}, k in {0,1}, j in {0,1}
    # U indices: i*2+k (0..3)
    # V indices: k*2+j (0..3) 
    # W indices: j*2+i (0..3)
    
    # Strassen's 7 factors for 2x2*2x2:
    # s=0: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]  -> A00*B00
    # s=1: U=[1,1,0,0], V=[0,0,1,0], W=[0,0,1,0]  -> (A00+A01)*B10
    # s=2: U=[0,1,0,0], V=[0,1,0,-1], W=[0,1,0,0] -> A01*(B01-B11)
    # s=3: U=[0,0,1,0], V=[1,0,0,1], W=[0,0,1,0] -> A10*(B00+B11)
    # s=4: U=[1,0,1,0], V=[0,0,0,1], W=[1,0,0,-1] -> (A10+A00)*(B11-B00)
    # s=5: U=[0,0,1,-1], V=[1,1,0,0], W=[0,0,1,0] -> (A11-A10)*(B00+B10)
    # s=6: U=[1,-1,0,0], V=[0,1,1,0], W=[0,1,0,1] -> (A00-A01)*(B11+B10)
    
    # Let me verify s=0: U[0,0]=1, others 0. V[0,0]=1, others 0. W[0,0]=1, others 0.
    # Contribution to C[i,j]: U[i*2+0, 0]*V[0*2+j, 0]*W[j*2+i, 0]
    # For i=0,j=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. Good.
    # For i=0,j=1: U[0,0]*V[1,0]*W[2,0] = 1*0*0 = 0. Good.
    # For i=1,j=0: U[2,0]*V[0,0]*W[0,1] = 0*1*0 = 0. Good.
    # For i=1,j=1: U[2,0]*V[1,0]*W[2,1] = 0*0*0 = 0. Good.
    # So s=0 contributes 1 to C[0,0] only. That's A00*B00. Correct.
    
    # s=1: U=[1,1,0,0], V=[0,0,1,0], W=[0,0,1,0]
    # U[0,0]=1, U[1,0]=1 (i=0,k=0 and i=0,k=1)
    # V[2,0]=1 (k=1,j=0)
    # W[2,0]=1 (j=1,i=0)
    # Contribution to C[i,j]: 
    # i=0,j=0: U[0,0]*V[0,0]*W[0,0] + U[1,0]*V[1,0]*W[0,0] = 1*0*0 + 1*0*0 = 0
    # Wait, V index is k*2+j. For k=1,j=0: V index = 1*2+0 = 2.
    # So V[2,1]=1 means V[k=1,j=0]=1.
    # W index is j*2+i. For j=1,i=0: W index = 1*2+0 = 2.
    # So W[2,1]=1 means W[j=1,i=0]=1.
    # Contribution to C[i,j] = sum_k U[i*2+k, s] * V[k*2+j, s] * W[j*2+i, s]
    # i=0,j=0: sum_k U[0*2+k,1]*V[k*2+0,1]*W[0*2+0,1] = U[0,1]*V[0,1]*W[0,1] + U[1,1]*V[2,1]*W[0,1]
    # = 1*0*0 + 1*1*0 = 0
    # i=0,j=1: sum_k U[k,1]*V[k*2+1,1]*W[1*2+0,1] = U[0,1]*V[1,1]*W[2,1] + U[1,1]*V[3,1]*W[2,1]
    # = 1*0*1 + 1*0*1 = 0
    # Hmm, that's not right. Let me reconsider.
    
    # I think I made an error. Let me redo Strassen properly.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M1 = A[0,0]*B[0,0]
    # M2 = (A[0,0]+A[0,1])*B[1,0]
    # M3 = A[0,1]*(B[0,1]-B[1,1])
    # M4 = A[1,0]*(B[0,0]+B[1,1])
    # M5 = (A[1,0]+A[0,0])*(B[1,1]-B[0,0])
    # M6 = (A[1,1]-A[1,0])*(B[0,0]+B[1,0])
    # M7 = (A[0,0]-A[0,1])*(B[1,1]+B[1,0])
    
    # C[0,0] = M1 + M4 + M5 - M6 - M7
    # C[0,1] = M3 + M7
    # C[1,0] = M2 + M6
    # C[1,1] = M1 + M3 + M5 - M2 - M4
    
    # Now in tensor form, each Ms is a product of a linear form in A and a linear form in B.
    # For Ms = (sum_ik a_ik * A[i,k]) * (sum_kj b_kj * B[k,j]):
    # The contribution to C[i',j'] is: a_ik * b_kj * W[j',i'] where we need to sum over the "output"
    # Actually, in the tensor decomposition:
    # C[i',j'] = sum_s U_s[i'*2+k, s] * V_s[k*2+j, s] * W_s[j'*2+i', s] summed over k,j,s
    # But that's not quite right. Let me use the correct formulation.
    
    # The tensor T[i,k,j] = 1 means C[i,j] = sum_k A[i,k]*B[k,j]
    # A decomposition of rank r means:
    # sum_{s=0}^{r-1} U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = 1 for all i,k,j
    # where the sum over s is implicit in the tensor equation.
    # Wait no. The correct formulation is:
    # C[i,j] = sum_k A[i,k] * B[k,j] = sum_s (sum_k U[i*m+k,s]*A[i,k]) * (sum_j V[k*p+j,s]*B[k,j]) * W[j*n+i,s]
    # That's not right either. Let me look at this more carefully.
    
    # The standard tensor decomposition for matrix multiplication:
    # T = sum_s u_s ⊗ v_s ⊗ w_s
    # where T[i,k,j] = 1, u_s is a vector in R^{n*m}, v_s in R^{m*p}, w_s in R^{p*n}
    # The meaning: C[i,j] = sum_k A[i,k]*B[k,j] = sum_s (u_s · A_vec) * (v_s · B_vec) * w_s[i,j]
    # where A_vec is A flattened, B_vec is B flattened, w_s[i,j] gives the coefficient.
    
    # So: sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] summed over k and j... no.
    # Actually: C[i,j] = sum_s (sum_k U[i*m+k,s] * A[i,k]) * (sum_k V[k*p+j,s] * B[k,j]) * W[j*n+i,s]
    # This is NOT a simple tensor product. The correct form is:
    # C[i,j] = sum_s W[j*n+i,s] * (sum_k U[i*m+k,s]*A[i,k]) * (sum_k V[k*p+j,s]*B[k,j])
    # For this to equal sum_k A[i,k]*B[k,j] for ALL A and B, we need:
    # sum_s W[j*n+i,s] * U[i*m+k,s] * V[k'*p+j,s] = delta_{k,k'} for all i,j,k,k'
    # Wait, that's the condition. Let me verify with schoolbook.
    # Schoolbook: for each (i,k,j): U[i*m+k, (i,k,j)] = 1, V[k*p+j, (i,k,j)] = 1, W[j*n+i, (i,k,j)] = 1
    # Then C[i,j] = sum_s W[j*n+i,s] * (sum_k U[i*m+k,s]*A[i,k]) * (sum_k V[k*p+j,s]*B[k,j])
    # = sum_{(i',k',j')} W[j*n+i,(i',k',j')] * (sum_k U[i*m+k,(i',k',j')]*A[i,k]) * (sum_k V[k*p+j,(i',k',j')]*B[k,j])
    # W[j*n+i,(i',k',j')] = 1 only if j'=j and i'=i.
    # U[i*m+k,(i,k',j')] = 1 only if i'=i and k'=k.
    # V[k'*p+j,(i,k',j')] = 1 only if k'=k and j'=j.
    # So C[i,j] = sum_k A[i,k]*B[k,j]. Correct!
    
    # So the verification condition is:
    # sum_s U[i*m+k,s] * V[k*p+j,s] * W[j*n+i,s] = 1 for all i,k,j
    # This is what the current verify function checks. Good.
    
    # Now, for Strassen on 2x2*2x2:
    # We need 7 terms (s=0..6) such that sum_s U[i*2+k,s]*V[k*2+j,s]*W[j*2+i,s] = 1 for all i,k,j in {0,1}
    
    # Let me construct these directly.
    # The 7 Strassen factors correspond to:
    # s=0: A[0,0]*B[0,0] -> U[0,0]=1, V[0,0]=1, W[0,0]=1
    #   Check: sum_s U[i*2+k,s]*V[k*2+j,s]*W[j*2+i,s] for s=0:
    #   i=0,k=0,j=0: U[0,0]*V[0,0]*W[0,0] = 1. 
    #   i=0,k=0,j=1: U[0,0]*V[1,0]*W[2,0] = 1*0*0 = 0.
    #   i=0,k=1,j=0: U[1,0]*V[2,0]*W[0,0] = 0*0*1 = 0.
    #   i=0,k=1,j=1: U[1,0]*V[3,0]*W[2,0] = 0*0*0 = 0.
    #   i=1,k=0,j=0: U[2,0]*V[0,0]*W[0,1] = 0*1*0 = 0.
    #   i=1,k=0,j=1: U[2,0]*V[1,0]*W[2,1] = 0*0*0 = 0.
    #   i=1,k=1,j=0: U[3,0]*V[2,0]*W[0,1] = 0*0*0 = 0.
    #   i=1,k=1,j=1: U[3,0]*V[3,0]*W[2,1] = 0*0*0 = 0.
    #   So s=0 contributes 1 to (i=0,k=0,j=0) and 0 elsewhere. 
    
    # This is getting complex. Let me just use a known correct Strassen decomposition.
    # I'll define the 7 factors explicitly and verify.
    
    # Based on standard Strassen, here are the 7 factors for 2x2*2x2:
    # Each factor s has U_s (4 entries: [U00, U01, U10, U11]), V_s (4 entries), W_s (4 entries)
    # where U_s[i,k], V_s[k,j], W_s[j,i]
    
    strassen_factors = [
        # s=0: A00*B00
        {'U': [1, 0, 0, 0], 'V': [1, 0, 0, 0], 'W': [1, 0, 0, 0]},
        # s=1: (A00+A01)*B10
        {'U': [1, 1, 0, 0], 'V': [0, 0, 1, 0], 'W': [0, 0, 1, 0]},
        # s=2: A01*(B01-B11)
        {'U': [0, 1, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 1, 0, 0]},
        # s=3: A10*(B00+B11)
        {'U': [0, 0, 1, 0], 'V': [1, 0, 0, 1], 'W': [0, 0, 1, 0]},
        # s=4: (A10+A00)*(B11-B00)
        {'U': [1, 0, 1, 0], 'V': [0, 0, 0, 1], 'W': [1, 0, 0, -1]},
        # s=5: (A11-A10)*(B00+B10)
        {'U': [0, 0, 1, -1], 'V': [1, 1, 0, 0], 'W': [0, 0, 1, 0]},
        # s=6: (A00-A01)*(B11+B10)
        {'U': [1, -1, 0, 0], 'V': [0, 1, 1, 0], 'W': [0, 1, 0, 1]},
    ]
    
    # Verify Strassen factors
    def verify_strassen_222(factors):
        n2, m2, p2 = 2, 2, 2
        for i in range(n2):
            for k in range(m2):
                for j in range(p2):
                    total = 0
                    for s, f in enumerate(factors):
                        total += f['U'][i*m2+k] * f['V'][k*p2+j] * f['W'][j*n2+i]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_strassen_222(strassen_factors):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now construct the full 2x4*4x8 decomposition using block Strassen.
    # A is 2x4, B is 4x8, C is 2x8.
    # Split: A = [A_L | A_R] (2x2 each), B = [B_T; B_B] (2x8 each)
    # C = A_L @ B_T + A_R @ B_B
    # A_L @ B_T: 2x2 * 2x8. Split B_T into 4 groups of 2 columns: B_T = [BT0 | BT1 | BT2 | BT3] (each 2x2)
    # A_L @ BTq: 2x2 * 2x2, use Strassen (rank 7)
    # So A_L @ B_T uses 4*7 = 28 terms
    # Similarly A_R @ B_B uses 4*7 = 28 terms
    # Total rank = 56
    
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 8*2 = 16 rows
    
    # For the left block (A_L @ B_T):
    # A_L[i,k] corresponds to A[i, k] where i in {0,1}, k in {0,1}
    # In our indexing: U index = i*m + k = i*4 + k
    # B_T[k,j] corresponds to B[k, j] where k in {0,1}, j in {0..7}
    # V index = k*p + j = k*8 + j
    # C[i,j] = sum_k A_L[i,k]*B_T[k,j]
    # W index = j*n + i = j*2 + i
    
    # For the right block (A_R @ B_B):
    # A_R[i,k] corresponds to A[i, k+2] where i in {0,1}, k in {0,1}
    # U index = i*4 + (k+2)
    # B_B[k,j] corresponds to B[k+2, j] where k in {0,1}, j in {0..7}
    # V index = (k+2)*8 + j
    # W index = j*2 + i (same)
    
    # For each block, split the 8 columns into 4 groups of 2.
    # Group q (q=0,1,2,3) covers columns j=2q, 2q+1.
    # Within each group, we have a 2x2*2x2 problem.
    
    for block in range(2):  # 0 = left (A_L@B_T), 1 = right (A_R@B_B)
        for q in range(4):  # 4 column groups
            base_col = 2 * q
            for s, f in enumerate(strassen_factors):
                rank_idx = block * 28 + q * 7 + s
                
                # U: for the 2x2*2x2 sub-problem
                # i in {0,1}, k in {0,1}
                # U index in full problem: i*4 + (k + block*2)
                for i in range(2):
                    for k in range(2):
                        u_idx = i * m + (k + block * 2)
                        U[u_idx][rank_idx] = f['U'][i * 2 + k]
                
                # V: for the 2x2*2x2 sub-problem
                # k in {0,1}, j in {0,1} (local), global j = base_col + j_local
                # V index in full problem: (k + block*2)*8 + (base_col + j_local)
                for k in range(2):
                    for j_local in range(2):
                        v_idx = (k + block * 2) * p + (base_col + j_local)
                        V[v_idx][rank_idx] = f['V'][k * 2 + j_local]
                
                # W: for the 2x2*2x2 sub-problem
                # j_local in {0,1}, i in {0,1}
                # W index in full problem: (base_col + j_local)*2 + i
                for j_local in range(2):
                    for i in range(2):
                        w_idx = (base_col + j_local) * n + i
                        W[w_idx][rank_idx] = f['W'][j_local * 2 + i]
    
    # Verify the full decomposition
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
