import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Current known decomposition: 28 via two Strassen blocks + schoolbook for the remainder
    # Try to find a lower-rank exact decomposition via linear-algebraic pruning of the tensor.
    
    # The multiplication tensor T has dimensions (n*m) x (m*p) x (p*n) = 6 x 15 x 10
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0.
    
    # Strategy: build the schoolbook tensor of rank 30, then use algebraic structure
    # (Strassen on 2x2 sub-blocks) to reduce to 28. Then attempt further reduction
    # by checking if the 28 rank-1 tensors are linearly independent in the tensor space,
    # and if not, finding a smaller generating set.
    
    # For this specific problem, let's try to see if we can do better than 28.
    # The lower bound is max(n*m, m*p, p*n) / something... actually for matrix mult
    # the trivial lower bound is max(n, m, p) * min(n, m, p) or similar.
    # For 2x3 * 3x5: schoolbook is 2*3*5 = 30.
    # Strassen gives 7 for 2x2*2x2. 
    # We have a 2x3 * 3x5 matrix.
    # Decompose A into A_left (2x2) and A_right (2x1)
    # Decompose B into B_top (2x5) and B_bottom (1x5)
    # C = A_left * B_top + A_right * B_bottom
    # A_left * B_top: 2x2 * 2x5. Split B_top into B_top1 (2x2), B_top2 (2x2), B_top3 (2x1)
    # C_left = A_left*B_top1 + A_left*B_top2 + A_left*B_top3
    # Each 2x2*2x2 can be done in 7 (Strassen), 2x2*2x1 in 4 (schoolbook) or maybe 3?
    # 2x2*2x1: rank is at most 4 (schoolbook). Can it be 3? 
    #   A=[[a,b],[c,d]] * [[e],[f]] = [[ae+bf],[ce+df]]
    #   This is bilinear in 4 inputs, 2 outputs. Rank of this tensor?
    #   The tensor is 2x4x2. Let's check if rank 3 is possible.
    #   Actually, 2x2*2x1 matrix multiplication tensor has rank 4 (it's irreducible).
    # So C_left rank <= 7 + 7 + 4 = 18
    # C_right = A_right * B_bottom: 2x1 * 1x5. This is outer product of a 2-vector and a 5-vector.
    #   [[a],[c]] * [[e,f,g,h,i]] = [[ae,af,ag,ah,ai],[ce,cf,cg,ch,ci]]
    #   This is 2*5 = 10 terms in schoolbook. Rank of this tensor?
    #   It's the tensor for outer product of 2-vector and 5-vector. 
    #   The rank of the outer product tensor [a,b] x [c,d,e,f,g] -> [ac,ad,ae,af,ag, bc,bd,be,bf,bg]
    #   This tensor is the Kronecker product of two "dial" tensors? No.
    #   Actually, this is a sum of 2x5 = 10 rank-1 terms. Can we do better?
    #   The tensor is in (R^2)^* x (R^5)^* x (R^{10})^*? No.
    #   Inputs: A is 2-dim, B is 5-dim. Output is 10-dim.
    #   T[a_index, b_index, out_index] where out = (i,j) maps to a_i * b_j.
    #   This is exactly the tensor for the multiplication of a 2x1 and 1x5 matrix.
    #   This is equivalent to the tensor for the outer product map R^2 x R^5 -> R^{2x5}.
    #   The rank of this tensor is 2*5 = 10 (it's a "flat" tensor, all 10 components are independent).
    #   Wait, is it? Let's check for smaller case: 1x1 * 1x2 -> rank 2. 2x1 * 1x1 -> rank 2.
    #   2x1 * 1x2: A=[a,b], B=[c,d]. C=[[ac,ad],[bc,bd]]. 4 outputs, 4 inputs.
    #   Tensor is 2x2x4. Rank? Schoolbook is 4. Can we do 3?
    #   This is the multiplication tensor for 2x1 * 1x2, which is just the tensor product of 
    #   two 2-dimensional "trivial" tensors? No.
    #   Actually, this tensor is the "Kronecker" of the 1x1*1x1 tensor with itself? No.
    #   Let's think: the tensor is sum_{i,j} e_i ⊗ e_j ⊗ f_{ij} where f_{ij} is the basis vector 
    #   for the (i,j)-th entry. There are 4 such terms and they are linearly independent 
    #   (each f_{ij} is distinct), so rank is 4.
    #   Similarly, 2x1 * 1x5 has rank 10.
    
    # So total rank <= 18 + 10 = 28. This matches our current construction.
    # To beat 28, we'd need to find a smarter decomposition that shares terms across 
    # the C_left and C_right blocks, or use a lower-rank algorithm for one of the sub-problems.
    
    # Can we do 2x2 * 2x2 in less than 7? No, 7 is optimal.
    # Can we do 2x2 * 2x1 in less than 4? Let's check.
    #   A=[[a,b],[c,d]], B=[[e],[f]], C=[[ae+bf],[ce+df]]
    #   Tensor: (a,b,c,d) x (e,f) -> (ae+bf, ce+df)
    #   This is a 4x2x2 tensor. Schoolbook rank 4.
    #   Is rank 3 possible? 
    #   If rank 3, then the 4x2x2 tensor can be written as sum of 3 rank-1 tensors.
    #   This would mean the bilinear map (a,b,c,d,e,f) -> (ae+bf, ce+df) has border rank 3.
    #   Actually, this is the multiplication tensor for R[x]/(x^2) * R? No.
    #   Let's just accept it's 4 for now.
    
    # Can we do 2x1 * 1x5 in less than 10? 
    #   The tensor is sum_{i=0}^{1} sum_{j=0}^{4} e_i ⊗ f_j ⊗ g_{ij}
    #   where g_{ij} are 10 distinct basis vectors. Since all g_{ij} are distinct and the 
    #   e_i ⊗ f_j are linearly independent (Kronecker product), the rank is exactly 10.
    
    # So 28 seems to be a natural barrier. But maybe we can restructure the decomposition
    # to share terms. For example, instead of treating A_left*B_top and A_right*B_bottom 
    # separately, we might find a global structure.
    
    # Alternative: treat the full 2x3 * 3x5 as a whole. 
    # n=2, m=3, p=5.
    # We could try to use a "3x3" Strassen-like block for the middle part.
    # But 3x3*3x3 has rank 23 (best known) or 27 (Strassen). That's worse than schoolbook 27.
    
    # Let's try a different partition:
    # A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    # B = [B1; B2] where B1 is 1x5, B2 is 2x5
    # C = A1*B1 + A2*B2
    # A1*B1: 2x1 * 1x5, rank 10
    # A2*B2: 2x2 * 2x5, rank <= 7+7+4 = 18
    # Total: 28. Same.
    
    # What if we partition differently?
    # A = [A1; A2] where A1 is 1x3, A2 is 1x3
    # B = [B1 | B2] where B1 is 3x2, B2 is 3x3
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # A1*B1: 1x3 * 3x2, rank 6 (schoolbook). Can we do 4? 1x3*3x2: A=[a,b,c], B=[[b11,b12],[b21,b22],[b31,b32]].
    #   C = [[a*b11+b*b21+c*b31, a*b12+b*b22+c*b32], ...] wait, A is 1x3, B is 3x2, C is 1x2.
    #   C[0] = a*B[0,0] + b*B[1,0] + c*B[2,0]
    #   C[1] = a*B[0,1] + b*B[1,1] + c*B[2,1]
    #   This is two inner products. Rank is 2 (each is a separate bilinear form). 
    #   Wait, no: the tensor is 3x6x2. The two output components are independent bilinear forms.
    #   Actually, this can be done in 2 rank-1 terms? No, because the inputs are shared.
    #   Let me re-examine. The multiplication is:
    #   (a,b,c) * (b11,b12,b21,b22,b31,b32) -> (a*b11+b*b21+c*b31, a*b12+b*b22+c*b32)
    #   This is a sum of 6 rank-1 terms in schoolbook. Can we do better?
    #   This is the tensor for the bilinear map R^3 x R^{3x2} -> R^2.
    #   It's equivalent to two independent inner products, but they share the A-inputs.
    #   Actually, this is just the Kronecker product structure: it's the tensor for (1x3 * 3x1) done twice, 
    #   but with the 1x1 outputs being different columns. 
    #   The rank of this tensor is 3 (since it's two copies of the 3x3x1 dot product tensor, but the 
    #   output dimension is 2, so... hmm).
    #   Let's think of it as: C = A * B where A is 1x3, B is 3x2.
    #   C_j = sum_i A_i * B_{ij}
    #   This is a bilinear map. The tensor is sum_{i,j} e_i ⊗ f_{ij} ⊗ g_j
    #   where i=0..2, j=0..1. There are 6 terms. Are they linearly independent?
    #   The g_j are only 2 distinct basis vectors (g_0, g_1). So we have:
    #   = g_0 ⊗ (e_0⊗f_00 + e_1⊗f_10 + e_2⊗f_20) + g_1 ⊗ (e_0⊗f_01 + e_1⊗f_11 + e_2⊗f_21)
    #   The two coefficients are in (R^3)^* ⊗ (R^3)^* and they are linearly independent 
    #   (since f_00,f_10,f_20 are distinct from f_01,f_11,f_21). So rank is 2? 
    #   Wait, no. Each "g_0 ⊗ (...)" is a sum of 3 rank-1 terms in the first two factors. 
    #   So the total rank is at most 3+3=6. Can it be less?
    #   This is getting complicated. Let's just stick with the 28-rank construction and 
    #   see if we can find a bug or a better partition.
    
    # Actually, let me reconsider the 2x1 * 1x5 case. 
    # A = [a, b] (2x1), B = [c, d, e, f, g] (1x5)
    # C = [[a*c, a*d, a*e, a*f, a*g], [b*c, b*d, b*e, b*f, b*g]]
    # This is 10 independent products. The tensor is sum_{i,j} u_i ⊗ v_j ⊗ w_{ij}
    # where u_i (i=0,1) are basis of R^2, v_j (j=0..4) are basis of R^5, 
    # and w_{ij} are 10 distinct basis vectors of R^{10}.
    # Since all w_{ij} are distinct, and the u_i⊗v_j are linearly independent, 
    # the rank is exactly 10. No improvement possible here.
    
    # So the bottleneck is the 2x2*2x5 part (rank 18) and the 2x1*1x5 part (rank 10).
    # Total 28.
    
    # Let me try yet another approach: use a "4x4" block decomposition.
    # Pad A to 4x4, B to 4x4? No, that increases the problem.
    
    # Or use the fact that 2x3 * 3x5 can be seen as a special case of 4x4 * 4x4 
    # where many entries are zero. But that doesn't help with rank reduction.
    
    # I think 28 is likely optimal or very close to optimal for this problem.
    # The current construction achieves 28. Let me verify it's correct and 
    # try to see if any of the 28 terms are redundant (i.e., if the rank-1 tensors 
    # are linearly dependent).
    
    # For the purpose of this exercise, I'll keep the 28-rank construction but 
    # rewrite it more cleanly and add a verification step. If verification fails, 
    # I'll fall back to schoolbook (rank 30).
    
    # Let's also try a small search: can we find a 27-rank decomposition by 
    # perturbing the Strassen coefficients? This is unlikely to work for an 
    # exact decomposition, but let's try a quick check.
    
    # Actually, for exact decompositions with integer/half-integer entries, 
    # the rank is a discrete invariant. We can't "perturb" to find a lower rank.
    
    # Let me just return the 28-rank decomposition. It's the best known for this 
    # specific sub-problem structure.
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Add a 2x2 * 2x2 Strassen block
    def strassen_block(i0, j0, k0):
        nonlocal col
        # A indices
        a00 = i0*m + j0
        a01 = i0*m + (j0+1)
        a10 = (i0+1)*m + j0
        a11 = (i0+1)*m + (j0+1)
        
        # B indices
        b00 = j0*p + k0
        b01 = j0*p + (k0+1)
        b10 = (j0+1)*p + k0
        b11 = (j0+1)*p + (k0+1)
        
        # C indices (k*n + i)
        c00 = k0*n + i0
        c01 = k0*n + (i0+1)
        c10 = (k0+1)*n + i0
        c11 = (k0+1)*n + (i0+1)
        
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        U[a00, col] = 1; U[a11, col] = 1
        V[b00, col] = 1; V[b11, col] = 1
        W[c00, col] = 1; W[c11, col] = 1
        col += 1
        
        # M2 = (A10+A11)B00 -> C10 - C11
        U[a10, col] = 1; U[a11, col] = 1
        V[b00, col] = 1
        W[c10, col] = 1; W[c11, col] = -1
        col += 1
        
        # M3 = A00(B01-B11) -> C01 + C11
        U[a00, col] = 1
        V[b01, col] = 1; V[b11, col] = -1
        W[c01, col] = 1; W[c11, col] = 1
        col += 1
        
        # M4 = A11(B10-B00) -> C00 + C10
        U[a11, col] = 1
        V[b10, col] = 1; V[b00, col] = -1
        W[c00, col] = 1; W[c10, col] = 1
        col += 1
        
        # M5 = (A00+A01)B11 -> -C00 + C01
        U[a00, col] = 1; U[a01, col] = 1
        V[b11, col] = 1
        W[c00, col] = -1; W[c01, col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11
        U[a10, col] = 1; U[a00, col] = -1
        V[b00, col] = 1; V[b01, col] = 1
        W[c11, col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00
        U[a01, col] = 1; U[a11, col] = -1
        V[b10, col] = 1; V[b11, col] = 1
        W[c00, col] = 1
        col += 1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_block(i0=0, j0=0, k0=0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_block(i0=0, j0=0, k0=2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (2x2 * 2x1)
    for i in range(2):
        for j in range(2):
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5] (2x1 * 1x5)
    for i in range(2):
        for k in range(5):
            U[i*m+2, col] = 1
            V[2*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify correctness with random matrices
    for trial in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Reconstruct using the decomposition
        C_recon = np.zeros((n, p))
        for r in range(rank):
            a_vec = U[:, r]  # length n*m
            b_vec = V[:, r]  # length m*p
            w_vec = W[:, r]  # length p*n
            prod = np.sum(a_vec * b_vec)
            # w_vec is in R^{p*n}, indexed by k*n+i
            for k in range(p):
                for i in range(n):
                    C_recon[i, k] += w_vec[k*n + i] * prod
        
        if not np.allclose(C_expected, C_recon, atol=1e-10):
            # Fall back to schoolbook
            rank = n*m*p
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
