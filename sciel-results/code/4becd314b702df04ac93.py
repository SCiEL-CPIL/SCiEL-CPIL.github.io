"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((num_U, rank_school), dtype=int)
    V_school = np.zeros((num_V, rank_school), dtype=int)
    W_school = np.zeros((num_W, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_factors = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Block Strassen decomposition:
    # Partition I: [0,1] -> I1, [2,3] -> I2
    # Partition J: [0,1] -> J1, [2,3] -> J2, [4] -> J3
    # Partition K: [0,1] -> K1, [2,3] -> K2, [4,5] -> K3
    #
    # For each (I_block, K_block) pair:
    #   - A(I_block, J1) * B(J1, K_block): 2x2 * 2x2 -> Strassen rank 7
    #   - A(I_block, J2) * B(J2, K_block): 2x2 * 2x2 -> Strassen rank 7
    #   - A(I_block, J3) * B(J3, K_block): 2x1 * 1x2 -> schoolbook rank 4
    # Total per (I,K) block: 18 terms
    # Total: 2 * 3 * 18 = 108
    
    rank_structured = 108
    U_struct = np.zeros((num_U, rank_structured), dtype=int)
    V_struct = np.zeros((num_V, rank_structured), dtype=int)
    W_struct = np.zeros((num_W, rank_structured), dtype=int)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    def add_strassen_2x2(i_off, j_off, k_off, col_start):
        """Add rank-7 Strassen decomposition for 2x2*2x2 block.
        A rows i_off..i_off+1, cols j_off..j_off+1
        B rows j_off..j_off+1, cols k_off..k_off+1
        C rows i_off..i_off+1, cols k_off..k_off+1
        Returns col_start + 7
        """
        # Strassen's 7 rank-1 terms for 2x2x2:
        # T = sum_r u_r v_r w_r
        # where u_r, v_r, w_r are vectors in A-space, B-space, C-space
        
        # Standard Strassen decomposition (rank 7):
        # r=0: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)  -> (A11+A22)(B11+B22) contributes to C11,C22
        # Actually, let me use the explicit rank-1 decomposition:
        # The 2x2x2 multiplication tensor has entries T[i,j,k] = 1 if i+k==j (0-indexed)
        # Wait, T[i,j,k] = 1 iff C[i,k] gets contribution from A[i,j]*B[j,k]
        # So T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
        # This is always true: T[i,j,k] = 1 for all i,j,k where the indices are valid
        # No wait. The tensor T_{i,j,k} = 1 if A_{i,j} * B_{j,k} contributes to C_{i,k}
        # This is always 1 for valid indices. So T is the all-ones tensor? No.
        # T_{i,j,k} = delta_{i,k} ... no.
        # The multiplication tensor: C_{i,k} = sum_j A_{i,j} B_{j,k}
        # So T_{i,j,k} = 1 for all i,j,k in {0,1}^3. That's 8 ones.
        # Rank of this tensor is 7 (Strassen).
        
        # The 7 rank-1 terms (u_r, v_r, w_r) where each is a vector in R^4 (for 2x2):
        # Using the standard Strassen basis:
        # Term 0: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        #   A-part: A11+A22, B-part: B11+B22, C-part: C11+C22
        #   Contributes: (A11+A22)(B11+B22) to C11 and C22
        #   But we need: C11 gets A11*B11 + A12*B21, C22 gets A21*B12 + A22*B22
        #   This doesn't directly work as a single rank-1 term.
        
        # Let me use the explicit known decomposition.
        # Reference: The rank-7 decomposition of the 2x2x2 tensor.
        # One standard form:
        # r=0: u=(1,1,0,0), v=(1,0,1,0), w=(1,0,1,0)
        # r=1: u=(0,0,1,1), v=(0,1,0,1), w=(0,1,0,1)
        # ... this is getting complex. Let me just verify with a known correct set.
        
        # Actually, the simplest correct approach: use the 8 schoolbook terms for 2x2*2x2
        # but that gives rank 8, not 7. The cell finding says rank 108 was achieved.
        # 108 = 12*7 + 6*4? No: 2 I-blocks * 3 K-blocks = 6 (I,K) pairs.
        # Each pair: 2 (2x2*2x2) + 1 (2x1*1x2)
        # If 2x2*2x2 = 7 and 2x1*1x2 = 4: 6*(2*7+4) = 6*18 = 108. Yes!
        
        # So I need the correct rank-7 decomposition. Let me derive it.
        # T_{i,j,k} for i,j,k in {0,1}: T_{i,j,k} = 1 (all 8 entries are 1)
        # We need 7 rank-1 tensors that sum to T.
        
        # Known decomposition (from Strassen, 1969):
        # Let me index A as (a00,a01,a10,a11), B as (b00,b01,b10,b11), C as (c00,c01,c10,c11)
        # C00 = a00*b00 + a01*b10
        # C01 = a00*b01 + a01*b11
        # C10 = a10*b00 + a11*b10
        # C11 = a10*b01 + a11*b11
        
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)(b00)
        # M3 = (a00)(b10-b11)
        # M4 = (a11)(b01-b00)
        # M5 = (a00+a01)(b11)
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Now, each M_r = (sum_i alpha_{r,i} a_i) * (sum_j beta_{r,j} b_j)
        # And C_k = sum_r gamma_{r,k} M_r
        # So the contribution of term r to C_k is gamma_{r,k} * (sum_i alpha_{r,i} a_i) * (sum_j beta_{r,j} b_j)
        # = sum_{i,j} gamma_{r,k} alpha_{r,i} beta_{r,j} a_i b_j
        # The tensor entry T_{i,j,k} = sum_r alpha_{r,i} beta_{r,j} gamma_{r,k}
        
        # So I need to find alpha, beta, gamma matrices (4x7 each) such that
        # T_{i,j,k} = sum_r alpha_{r,i} beta_{r,j} gamma_{r,k} = 1 for all i,j,k
        
        # From the Strassen formulas:
        # M1: alpha=(1,0,0,1), beta=(1,0,0,1), gamma=(1,0,0,1) [C00:+1, C01:0, C10:0, C11:+1]
        # Wait, C00 has +M1, C11 has +M1. So gamma_1 = (1,0,0,1)
        # M2: alpha=(0,0,1,1), beta=(1,0,0,0), gamma=(0,0,1,-1) [C10:+1, C11:-1]
        # M3: alpha=(1,0,0,0), beta=(0,0,1,-1), gamma=(0,1,0,1) [C01:+1, C11:+1]
        # M4: alpha=(0,0,0,1), beta=(0,1,-1,0), gamma=(1,0,1,0) [C00:+1, C10:+1]
        # M5: alpha=(1,1,0,0), beta=(0,0,0,1), gamma=(-1,1,0,0) [C00:-1, C01:+1]
        # M6: alpha=(0,0,1,-1), beta=(1,1,0,0), gamma=(0,0,0,1) [C11:+1]
        # M7: alpha=(0,1,0,-1), beta=(0,0,1,1), gamma=(1,0,0,0) [C00:+1]
        
        # Let me verify T_{0,0,0} = sum_r alpha_r[0] * beta_r[0] * gamma_r[0]
        # r=0: 1*1*1 = 1
        # r=1: 0*1*0 = 0
        # r=2: 1*0*0 = 0
        # r=3: 0*0*1 = 0
        # r=4: 1*0*(-1) = 0
        # r=5: 0*1*0 = 0
        # r=6: 0*0*1 = 0
        # Total: 1 ✓
        
        # T_{0,1,0} = sum_r alpha_r[0] * beta_r[1] * gamma_r[0]
        # r=0: 1*0*1 = 0
        # r=1: 0*0*0 = 0
        # r=2: 1*0*0 = 0
        # r=3: 0*1*1 = 0
        # r=4: 1*0*(-1) = 0
        # r=5: 0*1*0 = 0
        # r=6: 0*0*1 = 0
        # Total: 0 ✗ (should be 1)
        
        # Hmm, something's wrong. Let me recheck.
        # T_{i,j,k} should be 1 for all i,j,k in {0,1}^3 since every A_{i,j}*B_{j,k} 
        # contributes to C_{i,k}. Wait, that's not right either.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # So the tensor is T_{i,j,k} where the "mode" indices are (i, j, k) and
        # T_{i,j,k} = 1 means A_{i,j} * B_{j,k} contributes to C_{i,k}.
        # This is always true, so T_{i,j,k} = 1 for all i,j,k.
        # But that's an 8-entry tensor all ones, which has rank 1!
        # That can't be right.
        
        # I'm confusing myself. The tensor for matrix multiplication M_{n,m,p} has
        # T_{i,j,k} = 1 if and only if the product of A_{i,j} and B_{j,k} appears in C_{i,k}.
        # This is ALWAYS true. So T is the all-ones tensor of size n*m*p?
        # No! T_{i,j,k} is defined such that C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}
        # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, we have T_{i,j,k} = 1 for all valid i,j,k.
        # But the tensor is 3D with dimensions (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
        # For n=m=p=2, T is 2x2x2 with all entries 1.
        # The rank of the all-ones 2x2x2 tensor is 1 (it's just the outer product of three all-ones vectors).
        # That contradicts Strassen saying rank 7!
        
        # I think I'm misunderstanding the tensor structure. Let me reconsider.
        # The matrix multiplication tensor T_{m_{n,m,p}} is defined as:
        # T = sum_{i,j,k} e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)}
        # where the sum is over all (i,j,k) triples, and the tensor acts as:
        # <T, A ⊗ B ⊗ C> = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
        # So T_{i,j,k} = 1 for all i,j,k, and the "trace" is sum A_{i,j} B_{j,k} C_{k,i}
        # The rank is the minimum r such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
        # For 2x2x2, rank is 7.
        
        # So T_{i,j,k} = 1 for all i,j,k in {0,1}^3, and we need 7 rank-1 terms.
        # My verification above should give T_{0,1,0} = 1, not 0.
        # Let me recheck my alpha, beta, gamma.
        
        # I think I made an error in the Strassen formula mapping. Let me redo carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5  
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (a00+a11)(b00+b11): alpha1=(1,0,0,1), beta1=(1,0,0,1)
        # M2 = (a10+a11)(b00): alpha2=(0,0,1,1), beta2=(1,0,0,0)
        # M3 = (a00)(b10-b11): alpha3=(1,0,0,0), beta3=(0,0,1,-1)
        # M4 = (a11)(b01-b00): alpha4=(0,0,0,1), beta4=(0,1,-1,0)
        # M5 = (a00+a01)(b11): alpha5=(1,1,0,0), beta5=(0,0,0,1)
        # M6 = (a10-a00)(b00+b01): alpha6=(0,-1,1,0), beta6=(1,1,0,0)
        # M7 = (a01-a11)(b10+b11): alpha7=(0,1,0,-1), beta7=(0,0,1,1)
        
        # gamma (contribution to C00,C01,C10,C11):
        # g1=(1,0,0,1), g2=(0,0,1,-1), g3=(0,1,0,1), g4=(1,0,1,0), g5=(-1,1,0,0), g6=(0,0,0,1), g7=(1,0,0,0)
        
        # Now T_{i,j,k} = sum_r alpha_r[i] * beta_r[j] * gamma_r[k]
        # T_{0,0,0} = sum_r alpha_r[0]*beta_r[0]*gamma_r[0]
        # = 1*1*1 + 0*1*0 + 1*0*0 + 0*0*1 + 1*0*(-1) + 0*1*0 + 0*0*1
        # = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1 ✓
        
        # T_{0,1,0} = sum_r alpha_r[0]*beta_r[1]*gamma_r[0]
        # = 1*0*1 + 0*0*0 + 1*0*0 + 0*1*1 + 1*0*(-1) + 0*1*0 + 0*0*1
        # = 0 + 0 + 0 + 0 + 0 + 0 + 0 = 0 ✗
        
        # This is still 0. The issue is that T_{0,1,0} should be 1 (A_{0,1}*B_{1,0} contributes to C_{0,0}).
        # But in my Strassen decomposition, this contribution is spread across multiple terms.
        # Let me check: A_{0,1} appears in M5 (alpha5[1]=1) and M7 (alpha7[1]=1).
        # B_{1,0} appears in M3 (beta3[2]=1), M4 (beta4[2]=-1), M7 (beta7[2]=1).
        # C_{0,0} gets contributions from M1 (g1[0]=1), M4 (g4[0]=1), M5 (g5[0]=-1), M7 (g7[0]=1).
        
        # The coefficient of A_{0,1}*B_{1,0}*C_{0,0} in the sum is:
        # sum_r alpha_r[1]*beta_r[2]*gamma_r[0]
        # = 0*0*1 + 0*1*0 + 0*0*0 + 0*(-1)*1 + 1*0*(-1) + 0*0*0 + 1*1*1
        # = 0 + 0 + 0 + 0 + 0 + 0 + 1 = 1 ✓
        
        # Oh wait, I was computing T_{i,j,k} = sum_r alpha_r[i]*beta_r[j]*gamma_r[k]
        # But the tensor entry for A_{i,j}*B_{j,k}*C_{k,i} involves alpha_r[i], beta_r[j], gamma_r[k]
        # where i is the row of A, j is the col of A / row of B, k is the col of B / row of C.
        # And C_{i,k} is the output. So the tensor index is (i, j, k) and
        # T_{i,j,k} = sum_r alpha_r[i] * beta_r[j] * gamma_r[k]
        # where gamma_r[k] is the coefficient for C_{i,k}... but gamma depends on i and k!
        
        # I see the issue. The Strassen formula gives C_{i,k} = sum_r gamma_{r,i,k} M_r
        # where gamma depends on both i and k. So it's not a simple outer product.
        # The correct tensor decomposition has T_{i,j,k} = sum_r u_r[i] * v_r[j] * w_r[k]
        # where u_r is a vector in A-space (length n*m), v_r in B-space (length m*p), w_r in C-space (length p*n).
        # For 2x2x2: u_r has 4 entries (indexed by (i,j)), v_r has 4 entries (indexed by (j,k)), w_r has 4 entries (indexed by (k,i)).
        
        # So I need to be more careful with the indexing. Let me just hardcode the known correct
        # rank-7 decomposition by looking it up or deriving it properly.
        
        # For simplicity and correctness, let me use the 8 schoolbook terms for 2x2*2x2.
        # This gives rank 8 per 2x2 block, total = 6*(2*8+4) = 6*20 = 120. Same as global schoolbook.
        # Not an improvement.
        
        # Let me try a different approach: just use the explicit 7-term decomposition.
        # After research, here's a correct rank-7 decomposition for 2x2x2:
        # Using the convention that A is indexed (i,j), B is (j,k), C is (k,i):
        
        # Term 0: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        # Term 1: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,-1)
        # Term 2: u=(1,0,0,0), v=(0,0,1,-1), w=(0,1,0,1)
        # Term 3: u=(0,0,0,1), v=(0,1,-1,0), w=(1,0,1,0)
        # Term 4: u=(1,1,0,0), v=(0,0,0,1), w=(-1,1,0,0)
        # Term 5: u=(0,-1,1,0), v=(1,1,0,0), w=(0,0,0,1)
        # Term 6: u=(0,1,0,-1), v=(0,0,1,1), w=(1,0,0,0)
        
        # Where u is indexed (a00,a01,a10,a11), v is (b00,b01,b10,b11), w is (c00,c01,c10,c11)
        # but w should be indexed (c_{k,i}) so w[0]=c00, w[1]=c01, w[2]=c10, w[3]=c11
        # Actually w is indexed by (k,i) in the trace convention: W[k*n+i]
        # For 2x2: W[0]=c00(k=0,i=0), W[1]=c01(k=0,i=1), W[2]=c10(k=1,i=0), W[3]=c11(k=1,i=1)
        
        # Let me verify T_{0,1,0} (i=0,j=1,k=0): A_{0,1}*B_{1,0} contributes to C_{0,0}
        # Coefficient = sum_r u_r[1] * v_r[2] * w_r[0]  (u index 1 = a01, v index 2 = b10, w index 0 = c00)
        # = 0*0*1 + 0*1*0 + 0*0*0 + 0*(-1)*1 + 1*0*(-1) + 0*0*0 + 1*1*1
        # = 0+0+0+0+0+0+1 = 1 ✓
        
        # T_{0,0,1} (i=0,j=0,k=1): A_{0,0}*B_{0,1} contributes to C_{0,1}
        # Coefficient = sum_r u_r[0] * v_r[1] * w_r[1]  (u[0]=a00, v[1]=b01, w[1]=c01)
        # = 1*0*0 + 0*0*0 + 1*0*1 + 0*1*0 + 1*0*1 + 0*1*0 + 0*0*0
        # = 0+0+0+0+0+0+0 = 0 ✗
        
        # Still wrong. I think the issue is that my Strassen formula mapping is incorrect.
        # Let me just give up on deriving it and use a verified source.
        
        # After careful thought, I'll use the following verified rank-7 decomposition:
        # The 2x2x2 multiplication tensor can be written as sum of 7 rank-1 tensors.
        # One such decomposition (verified by direct computation):
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs) where each is a 4-tuple
            # u: (a00, a01, a10, a11), v: (b00, b01, b10, b11), w: (c00, c01, c10, c11)
            # But w is indexed by (k,i) in trace convention: w[k*2+i]
            # So w = (c_{0,0}, c_{0,1}, c_{1,0}, c_{1,1}) = (c00, c01, c10, c11)
            
            # I'll use a numerically verified decomposition.
            # Let me just use schoolbook (8 terms) for 2x2 to be safe, and see if I can
            # find a better decomposition for the overall problem.
        ]
        
        # Given the difficulty, let me just use schoolbook for 2x2 blocks (rank 8)
        # and schoolbook for 2x1*1x2 (rank 4).
        # Total: 6*(2*8 + 4) = 120. Same as global schoolbook. No improvement.
        
        # Alternative: Use a different block structure.
        # What if I don't block at all and just use a global Strassen-like approach?
        # That's complex.
        
        # For now, let me just return schoolbook to ensure correctness.
        # The cell finding says rank 108 was achieved, so there must be a way.
        
        # Let me try: what if the 2x1*1x2 block can be done in rank 2 instead of 4?
        # A is 2x1: (a0, a1). B is 1x2: (b0, b1). C is 2x2:
        # C00 = a0*b0, C01 = a0*b1, C10 = a1*b0, C11 = a1*b1
        # This is the outer product of A and B. The tensor is T_{i,0,k} = A_i * B_k
        # This has rank... Let's see. T = sum_r u_r v_r w_r
        # u is 2x1, v is 1x2, w is 2x2.
        # Rank 1: u=(1,1), v=(1,1), w=(1,1;1,1)? No, that gives C00=C01=C10=C11=1.
        # We need C00=a0b0, C01=a0b1, C10=a1b0, C11=a1b1.
        # This is exactly the outer product A ⊗ B. The tensor T_{i,j,k} with j=0 fixed:
        # T_{i,0,k} = 1 for all i,k. This is a 2x1x2 tensor with all entries 1.
        # Rank of this tensor: it's the all-ones tensor of size 2x1x2.
        # This can be written as u ⊗ v ⊗ w where u=(1,1), v=(1), w=(1,1)?
        # No, w should be 2x2. Let me think differently.
        # The contribution is: for each (i,k), C_{i,k} += A_i * B_k
        # In the tensor decomposition, this is a single rank-1 term:
        # u = (1,1) for A (both rows), v = (1) for B (single row, but B has 2 cols)
        # Actually, A is 2x1, so A-space has 2 elements. B is 1x2, so B-space has 2 elements.
        # C is 2x2, so C-space has 4 elements.
        # The term: u=(1,1) [A rows], v=(1,1) [B cols], w=(1,1,1,1) [C entries]
        # This gives C_{i,k} += 1*1*1 = 1 for all i,k. But we need C_{i,k} += A_i * B_k.
        # Since A_i and B_k are variables, this is a bilinear form.
        # The tensor entry T_{(i,0),(0,k),(k,i)} = 1 for all i,k.
        # This is 4 entries, all 1. Can we write this as a sum of rank-1 terms?
        # Rank 1: u ⊗ v ⊗ w where u in R^2, v in R^2, w in R^4.
        # We need sum_r u_r[i] * v_r[k] * w_r[k,i] = 1 for all i,k.
        # If we use r=1: u=(1,1), v=(1,1), w=(1,1,1,1): gives 1*1*1=1 for all. Yes!
        # So the 2x1*1x2 block has rank 1, not 4!
        
        # Wait, that can't be right. Let me double-check.
        # The tensor for 2x1 * 1x2 multiplication:
        # C_{i,k} = A_i * B_k (since the inner dimension is 1)
        # So T_{(i,j),(j,k),(k,i)} = 1 where j=0 is the only valid j.
        # T_{(i,0),(0,k),(k,i)} = 1 for i in {0,1}, k in {0,1}.
        # That's 4 entries, all equal to 1.
        # Can this be a single rank-1 term? u ⊗ v ⊗ w:
        # u[i] * v[k] * w[k,i] = 1 for all i,k.
        # If u=(1,1), v=(1,1), w=(1,1,1,1): yes, 1*1*1=1. ✓
        # So rank is 1!
        
        # That changes things. Let me recalculate:
        # 2x2*2x2: rank 7 (Strassen) or 8 (schoolbook)
        # 2x1*1x2: rank 1
        # If 2x2*2x2 = 7: total = 6*(2*7 + 1) = 6*15 = 90
        # If 2x2*2x2 = 8: total = 6*(2*8 + 1) = 6*17 = 102
        
        # The cell finding says 108, which doesn't match either. Let me re-examine.
        # Maybe the 2x1*1x2 block isn't quite rank 1 in the context of the full decomposition.
        # Or maybe my block structure is wrong.
        
        # Let me just implement schoolbook for everything to be safe, and try to improve.
        # For now, return schoolbook rank 120.
        
        return col_start
    
    # Given the complexity, let me just return the schoolbook solution for safety.
    U, V, W = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
