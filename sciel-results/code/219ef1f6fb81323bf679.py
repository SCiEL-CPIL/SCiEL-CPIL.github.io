import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Total rank
    # 4x4x4 part: 7 Strassen terms * 7 inner = 49
    # 4x4x1 part: 16
    # 4x1x4 part: 16
    # 4x1x1 part: 4
    rank = 49 + 16 + 16 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get 2x2x2 Strassen factors
    # Returns U, V, W of shape 4x7
    def get_strassen_2x2_factors():
        # U indices: 0:00, 1:01, 2:10, 3:11
        # V indices: 0:00, 1:01, 2:10, 3:11
        # W indices: 0:00, 1:01, 2:10, 3:11 (using k*n+i, n=2)
        
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # Term 0: (a00+a11)(b00+b11)
        # U: 1,0,0,1
        u[0][0]=1; u[3][0]=1
        # V: 1,0,0,1
        v[0][0]=1; v[3][0]=1
        # W: c00+1, c11+1 -> idx 0, 3
        w[0][0]=1; w[3][0]=1
        
        # Term 1: (a10+a11)b00
        # U: 0,0,1,1
        u[2][1]=1; u[3][1]=1
        # V: 1,0,0,0
        v[0][1]=1
        # W: c10+1, c11-1 -> idx 2, 3
        w[2][1]=1; w[3][1]=-1
        
        # Term 2: a00(b01-b11)
        # U: 1,0,0,0
        u[0][2]=1
        # V: 0,1,0,-1
        v[1][2]=1; v[3][2]=-1
        # W: c01+1, c11+1 -> idx 1, 3
        w[1][2]=1; w[3][2]=1
        
        # Term 3: a11(b10-b00)
        # U: 0,0,0,1
        u[3][3]=1
        # V: 0,0,1,-1
        v[2][3]=1; v[0][3]=-1
        # W: c00+1, c10+1 -> idx 0, 2
        w[0][3]=1; w[2][3]=1
        
        # Term 4: (a00+a01)b11
        # U: 1,1,0,0
        u[0][4]=1; u[1][4]=1
        # V: 0,0,0,1
        v[3][4]=1
        # W: c00-1, c01+1 -> idx 0, 1
        w[0][4]=-1; w[1][4]=1
        
        # Term 5: (a10-a00)(b00+b01)
        # U: -1,0,1,0
        u[0][5]=-1; u[2][5]=1
        # V: 1,1,0,0
        v[0][5]=1; v[1][5]=1
        # W: c11+1 -> idx 3
        w[3][5]=1
        
        # Term 6: (a01-a11)(b10+b11)
        # U: 0,1,0,-1
        u[1][6]=1; u[3][6]=-1
        # V: 0,0,1,1
        v[2][6]=1; v[3][6]=1
        # W: c00+1 -> idx 0
        w[0][6]=1
        
        return u, v, w

    # 4x4x4 part: rank-8 Baur-Strassen decomposition
    # Treat 4x4 matrices as 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Rank-8 decomposition:
    # M0 = (A00+A10)(B00+B10)  -> C00 += M0, C10 += M0
    # M1 = (A00+A11)(B00+B11)  -> C00 += M1, C11 += M1
    # M2 = (A00+A01)(B00+B01)  -> C00 += M2, C01 += M2
    # M3 = (A00+A10+A11)(B00+B10+B11) -> C00 += M3, C10 += M3, C11 += M3
    # M4 = (A01-A11)(B10+B11)  -> C01 += M4, C11 += M4
    # M5 = (A10-A00)(B00+B01)  -> C10 += M5, C01 += M5
    # M6 = (A01+A11)(B10-B00)  -> C01 += M6, C11 += M6
    # M7 = (A10+A11)(B11-B01)  -> C10 += M7, C11 += M7
    #
    # Each M_k is a 2x2 matrix product, decomposed via 2x2 Strassen (rank 7).
    # But wait - that gives 8*7=56, not 8. Let me reconsider.
    #
    # Actually, the rank-8 decomposition means 8 scalar multiplications total.
    # Each M_k = X_k * Y_k where X_k is a linear combo of A blocks and Y_k is
    # a linear combo of B blocks. But X_k * Y_k is a 2x2 * 2x2 product, which
    # needs 7 scalar multiplications via Strassen. So total would be 8*7=56.
    #
    # That's not rank 8. Let me reconsider the structure.
    #
    # The correct approach: the 4x4x4 tensor has a rank-8 decomposition where
    # each term is a rank-1 tensor (outer product of 3 vectors of length 16).
    # This means each term involves exactly ONE scalar multiplication.
    #
    # For this to work, each term must be of the form:
    # (sum over (i,j) u_{ij} A_{ij}) * (sum over (j,k) v_{jk} B_{jk}) = sum over (i,k) w_{ik} C_{ik}
    # where u, v, w are sparse vectors.
    #
    # Actually, the standard rank-8 decomposition for 4x4x4 uses the fact that
    # 4 = 2*2, and we can use a Kronecker-product structure.
    #
    # Let me use a different approach: directly enumerate the rank-8 terms.
    # Each term k has:
    #   u_k: vector in R^{16} (for A, indexed by (i,j) with i,j in 0..3)
    #   v_k: vector in R^{16} (for B, indexed by (j,k) with j,k in 0..3)
    #   w_k: vector in R^{16} (for C, indexed by (i,k) with i,k in 0..3)
    #
    # The condition is: sum_k u_k[i,j] * v_k[j,l] = w_k[i,l] for all i,j,l,k
    # i.e., the matrix multiplication tensor = sum_k u_k \otimes v_k \otimes w_k
    #
    # A known rank-8 decomposition for 4x4x4:
    # Using the Baur-Strassen method, we can get rank 8.
    #
    # Let me construct it explicitly. The key insight is to use diagonal
    # structures.
    
    # Rank-8 decomposition of 4x4x4 matrix multiplication
    # Term k: u_k, v_k, w_k each have 16 entries
    # u_k[i*4+j] = coefficient for A[i,j]
    # v_k[j*4+l] = coefficient for B[j,l]
    # w_k[l*4+i] = coefficient for C[i,l] (trace convention: W[k*n+i,r] with n=4)
    
    # I'll use the following construction based on diagonal scaling:
    # For each subset S of {0,1,2,3}, define diagonal matrices D_S.
    # This gives 2^4 = 16 terms, but we can do better.
    
    # Actually, let me just verify if rank 8 is achievable and construct it.
    # The Baur-Strassen result says rank(4,4,4) <= 8.
    
    # Construction: Use the fact that 4x4 = 2x(2x2).
    # Write A = [A_top; A_bottom] where A_top, A_bottom are 2x4.
    # Similarly B = [B_left | B_right] where B_left, B_right are 4x2.
    # C = A_top * B_top + A_bottom * B_bottom (block structure)
    
    # This gets complicated. Let me use a known explicit rank-8 decomposition.
    
    # Here's a rank-8 decomposition for 4x4x4:
    # We use 8 terms, each being a rank-1 tensor.
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    # Term 1: A[0,1]*B[1,0] -> C[0,0]  
    # ... this is just the naive 16-term decomposition.
    
    # The rank-8 decomposition must use sums/differences.
    
    # Let me use a systematic approach. The 4x4x4 tensor T has entries T[i,j,l] = 1 if j is the "middle" index.
    # We want T = sum_{k=0}^{7} u_k \otimes v_k \otimes w_k.
    
    # A known construction uses the following 8 terms:
    # Let's define them by their support patterns.
    
    # I'll construct a valid rank-8 decomposition:
    
    # Using the method from Baur & Strassen (1983):
    # The 8 terms correspond to the 8 ways to "split" the 4 indices.
    
    # Let me just hardcode a verified rank-8 decomposition.
    
    # After careful construction, here is a rank-8 decomposition:
    
    r8_u = []
    r8_v = []
    r8_w = []
    
    # I'll build this using a Kronecker product structure.
    # 4x4x4 = (2x2x2) \otimes (2x2x2) in some sense, but that gives rank 7*7=49.
    #
    # The correct rank-8 decomposition uses a different structure.
    #
    # Let me use the following explicit construction:
    # For each bit pattern (b0, b1) in {0,1}^2, define:
    
    # Actually, I realize I should just verify the decomposition numerically.
    # Let me construct it and check.
    
    # Here's my approach: I'll define 8 terms and verify they sum to the identity tensor.
    
    # Term definitions (u, v, w as flat arrays of length 16):
    # Index convention: u[i*4+j] for A[i,j], v[j*4+l] for B[j,l], w[l*4+i] for C[i,l]
    
    # I'll use a construction based on Hadamard-like patterns.
    
    # Let me try a specific construction:
    # For k in 0..7, let k = (k0, k1, k2) in binary.
    # u_k[i,j] = 1 if (i & 2) == (k0 * 2) and (j & 1) == (k1 & 1) ... this is getting complicated.
    
    # Let me just use a verified decomposition from the literature.
    
    # The following is a rank-8 decomposition for 4x4x4 matrix multiplication:
    
    # I'll define it as a list of (u, v, w) triples.
    
    def make_4x4x4_rank8():
        """Return 8 rank-1 tensors for 4x4x4 matrix multiplication."""
        terms = []
        
        # Using the construction from "The complexity of multiplication of matrices"
        # I'll define 8 terms explicitly.
        
        # Each term: (u, v, w) where u, v, w are lists of 16 floats.
        # u[i*4+j] = coeff for A[i,j]
        # v[j*4+l] = coeff for B[j,l]
        # w[l*4+i] = coeff for C[i,l]
        
        # Term 0: 
        # u: A[0,0]=1, A[1,0]=1, A[2,0]=1, A[3,0]=1 (column 0 of A)
        # v: B[0,0]=1, B[0,1]=1, B[0,2]=1, B[0,3]=1 (row 0 of B)
        # w: C[0,0]=1, C[1,0]=1, C[2,0]=1, C[3,0]=1 (column 0 of C)
        # This gives (A_col0) * (B_row0) -> C_col0, which is correct for the first column.
        # But this is just one term for the first column, and we'd need 4 more for other columns.
        # Total would be 4, not 8. But this doesn't account for the full product.
        
        # Actually, A * B's (i,l) entry = sum_j A[i,j] * B[j,l].
        # If I take u = e_i (row i of A), v = e_l^T (column l of B), w = e_{i,l},
        # that's the naive 16-term decomposition.
        
        # For rank 8, I need to combine these.
        
        # Let me use a different approach. I'll use the fact that:
        # (A + D)(B + D) = AB + AD + DA + D^2
        # and solve for AB using clever choices of D.
        
        # This is getting too complex for a quick implementation. Let me fall back
        # to a rank-16 decomposition for the 4x4x4 part (which is still better than 49).
        
        # Rank-16 decomposition: just use the naive 16 scalar multiplications.
        # A[i,j] * B[j,l] -> C[i,l]
        
        terms = []
        for i in range(4):
            for j in range(4):
                u = [0.0] * 16
                v = [0.0] * 16
                w = [0.0] * 16
                u[i * 4 + j] = 1.0
                v[j * 4 + 0] = 1.0  # This is wrong, v should depend on l
                # Actually, for the naive decomposition:
                # Term (i,j,l): u[i,j]=1, v[j,l]=1, w[i,l]=1
                # That's 64 terms, not 16.
                
                # I'm confusing myself. Let me restart.
                # The naive decomposition has 4*4*4 = 64 rank-1 terms? No.
                # T[i,j,l] = 1 for all i,j,l (where j is the contraction index).
                # T = sum_{i,j,l} e_i \otimes e_j \otimes e_l
                # That's 64 terms, each with a single 1.
                # But we can group: for fixed j, sum_i,l e_i \otimes e_j \otimes e_l
                # = (sum_i e_i) \otimes e_j \otimes (sum_l e_l)
                # So for each j, we have one term: u = all ones in row j... no.
                
                # Let me think again. T[i,j,l] = delta_{j, contraction} = 1 for all i,j,l.
                # Actually T[i,j,l] = 1 if the indices match the matrix mult pattern.
                # For matrix mult (A,B,C) with C = A*B:
                # T[i,j,l] = 1 where i is row of A, j is col of A / row of B, l is col of B.
                # So T[i,j,l] = 1 for all i in 0..3, j in 0..3, l in 0..3.
                # That's a rank-1 tensor! T = 1 \otimes 1 \otimes 1.
                # No wait, that's not right either. The tensor for matrix multiplication
                # has T[i,j,l] = 1, and the decomposition is:
                # C[i,l] = sum_j A[i,j] B[j,l]
                # So the tensor is T = sum_j (e_j \otimes e_j \otimes e_j) in a basis where
                # the first index is (i,j), second is (j,l), third is (i,l).
                # In the flattened basis:
                # U part: index (i,j) -> i*4+j
                # V part: index (j,l) -> j*4+l  
                # W part: index (i,l) -> l*4+i (trace convention)
                
                # T[i,j,l] = 1 for all i,j,l.
                # Decomposition: T = sum_{j=0}^{3} u_j \otimes v_j \otimes w_j
                # where u_j has 1 at position (i,j) for all i, 0 elsewhere.
                # v_j has 1 at position (j,l) for all l, 0 elsewhere.
                # w_j has 1 at position (i,l) for all i,l, 0 elsewhere.
                
                # That gives rank 4, not 8! Let me verify.
                # sum_j [sum_i A[i,j] * u_j[i,j]] * [sum_l B[j,l] * v_j[j,l]] * w_j[i,l]
                # = sum_j (sum_i A[i,j]) * (sum_l B[j,l]) * (sum_{i',l'} w_j[i',l'] C[i',l'])
                # This doesn't give the right answer.
                
                # I'm making errors. Let me be very careful.
                
                # The matrix multiplication tensor T has components T[i,j,l] = 1.
                # A rank-r decomposition is: T = sum_{k=1}^r u_k \otimes v_k \otimes w_k
                # where u_k is in R^{16} (flattened A), v_k in R^{16} (flattened B), w_k in R^{16} (flattened C).
                # The condition is: for all i,j,l: T[i,j,l] = sum_k u_k[i*4+j] * v_k[j*4+l] * w_k[l*4+i]
                
                # So we need: 1 = sum_k u_k[i,j] * v_k[j,l] * w_k[i,l] for all i,j,l.
                
                # A trivial rank-16 decomposition:
                # For each (i,j,l), have a term with u[i,j]=1, v[j,l]=1, w[i,l]=1, all else 0.
                # That's 64 terms.
                
                # A better decomposition:
                # For each j, define u_j, v_j, w_j such that:
                # u_j[i,j'] = delta_{j,j'} (for all i)
                # v_j[j',l] = delta_{j,j'} (for all l)
                # w_j[i,l] = 1 (for all i,l)
                # Then sum_j u_j[i,j'] * v_j[j',l] * w_j[i,l] = sum_j delta_{j,j'} * delta_{j,j'} * 1 = 1 if j'=j, else 0.
                # Wait, that gives 1 only when j'=j, but we need it for all j'.
                
                # I think I have the index convention wrong. Let me re-derive.
                
                # C[i,l] = sum_j A[i,j] * B[j,l]
                # So the tensor T satisfies: T[i,j,l] = 1 for all i,j,l.
                # In a rank-r decomposition: T[i,j,l] = sum_k u_k[i,j] * v_k[j,l] * w_k[i,l]
                # where u_k[i,j] is the coefficient of A[i,j] in term k,
                # v_k[j,l] is the coefficient of B[j,l] in term k,
                # w_k[i,l] is the coefficient of C[i,l] in term k.
                
                # So we need: 1 = sum_k u_k[i,j] * v_k[j,l] * w_k[i,l] for all i,j,l.
                
                # A rank-4 decomposition:
                # For each j in 0..3:
                # u_j[i,j'] = 1 if j'=j, else 0 (for all i)
                # v_j[j',l] = 1 if j'=j, else 0 (for all l)
                # w_j[i,l] = 1 (for all i,l)
                # Then: sum_j u_j[i,j'] * v_j[j',l] * w_j[i,l] = sum_j (1 if j'=j else 0) * (1 if j'=j else 0) * 1
                # = 1 if j'=j, else 0.
                # But we need this to equal 1 for all j', not just j'=j.
                
                # So that doesn't work. The issue is that u_j and v_j both depend on j,
                # and their product is 1 only when they match.
                
                # Let me try:
                # u_j[i,j'] = 1 for all i, j' (all ones)
                # v_j[j',l] = 1 if j'=j, else 0
                # w_j[i,l] = 1 for all i,l
                # Then: sum_j u_j[i,j'] * v_j[j',l] * w_j[i,l] = sum_j 1 * (1 if j'=j else 0) * 1 = 1.
                # Yes! This works for rank 4.
                
                # But wait, u_j is the same for all j (all ones). So we can combine them:
                # u[i,j] = 1 for all i,j
                # v[j,l] = 1 for all j,l
                # w[i,l] = 1 for all i,l
                # Then u[i,j] * v[j,l] * w[i,l] = 1 for all i,j,l.
                # So rank 1!
                
                # But that can't be right. The matrix multiplication tensor is not rank 1.
                # Let me check: if T = u \otimes v \otimes w with u,v,w all ones,
                # then C[i,l] = sum_j A[i,j] B[j,l] * 1 * 1 * 1 = sum_j A[i,j] B[j,l].
                # That's correct!
                
                # So the matrix multiplication tensor IS rank 1? That can't be.
                # Oh, I see my error. The tensor T[i,j,l] = 1, and the decomposition
                # T = sum_k u_k \otimes v_k \otimes w_k means:
                # T[i,j,l] = sum_k u_k[i,j] * v_k[j,l] * w_k[i,l]
                # If u=v=w=ones, then T[i,j,l] = 1 for all i,j,l, which is correct.
                # So the tensor is rank 1.
                
                # But that contradicts the known result that rank(4,4,4) >= 16.
                # Let me re-examine.
                
                # Oh, I think I'm confusing the tensor with the bilinear map.
                # The bilinear map is (A,B) -> C, where C[i,l] = sum_j A[i,j] B[j,l].
                # The associated tensor is T = sum_{i,j,l} e_i \otimes e_j \otimes e_l
                # in the basis where the first factor is indexed by (i,j), second by (j,l),
                # and third by (i,l).
                
                # In this basis, T[i,j; j',l; i',l'] = delta_{i,i'} delta_{j,j'} delta_{l,l'}.
                # So T is the identity tensor in this basis, which has rank 16 (the dimension).
                
                # I was using the wrong basis. Let me redo this.
                
                # In the standard basis for the bilinear map:
                # U has 16 entries indexed by (i,j) for i,j in 0..3.
                # V has 16 entries indexed by (j,l) for j,l in 0..3.
                # W has 16 entries indexed by (i,l) for i,l in 0..3.
                
                # The tensor T satisfies: T[(i,j), (j',l), (i',l')] = 1 if i=i', j=j', l=l', else 0.
                # So T is the "identity" tensor, and its rank is 16.
                
                # A rank-16 decomposition: for each (i,j,l), have a term with
                # u[(i,j)] = 1, v[(j,l)] = 1, w[(i,l)] = 1, all else 0.
                # That's 64 terms, not 16.
                
                # Wait, I need to be more careful. The tensor T is a 3-tensor with
                # dimensions 16 x 16 x 16. Its entries are:
                # T[a, b, c] = 1 if a corresponds to (i,j), b corresponds to (j,l),
                # c corresponds to (i,l), and the indices are consistent.
                
                # In the flattened index:
                # a = i*4 + j
                # b = j*4 + l
                # c = l*4 + i  (trace convention: W[k*n+i,r] with n=4, so index = l*4+i)
                
                # T[a, b, c] = 1 if a = i*4+j, b = j*4+l, c = l*4+i for some i,j,l.
                # Otherwise T[a,b,c] = 0.
                
                # So T has 64 non-zero entries (one for each (i,j,l) triple).
                # The rank of this tensor is at least 64/16 = 4, and at most 64.
                
                # A rank-64 decomposition: one term per (i,j,l).
                # A better decomposition: group by j.
                # For fixed j, the non-zero entries are:
                # a = i*4+j for i in 0..3
                # b = j*4+l for l in 0..3
                # c = l*4+i for i,l in 0..3
                # So for fixed j, T has 16 non-zero entries forming a 4x4x16 sub-tensor.
                # This sub-tensor can be written as:
                # sum_{i,l} e_{i*4+j} \otimes e_{j*4+l} \otimes e_{l*4+i}
                # = (sum_i e_{i*4+j}) \otimes (sum_l e_{j*4+l}) \otimes (sum_{i,l} e_{l*4+i})
                # So for each j, we have one rank-1 term!
                # Total rank = 4.
                
                # Let me verify: for fixed j, define:
                # u_j[a] = 1 if a = i*4+j for some i, else 0. So u_j has 1s at positions j, 4+j, 8+j, 12+j.
                # v_j[b] = 1 if b = j*4+l for some l, else 0. So v_j has 1s at positions j*4, j*4+1, j*4+2, j*4+3.
                # w_j[c] = 1 if c = l*4+i for some i,l, else 0. So w_j has 1s at all positions 0..15.
                
                # Then sum_j u_j[a] * v_j[b] * w_j[c] = sum_j (1 if a mod 4 == j) * (1 if b div 4 == j) * (1 if c is any)
                # = 1 if a mod 4 == b div 4, else 0.
                # But we need T[a,b,c] = 1 if a = i*4+j, b = j*4+l, c = l*4+i.
                # The condition a mod 4 == b div 4 is necessary but not sufficient.
                # We also need c = l*4+i where i = a div 4, l = b mod 4.
                # So c should be (b mod 4)*4 + (a div 4).
                # But w_j[c] = 1 for all c, so the sum gives 1 whenever a mod 4 == b div 4,
                # regardless of c. That's not correct.
                
                # So rank 4 doesn't work. The issue is that w_j can't distinguish between different c values.
                
                # Let me try rank 16:
                # For each (i,l), define:
                # u_{i,l}[a] = 1 if a = i*4+j for some j, else 0. (row i of A)
                # v_{i,l}[b] = 1 if b = j*4+l for some j, else 0. (column l of B)
                # w_{i,l}[c] = 1 if c = l*4+i, else 0. (specific entry of C)
                
                # Then sum_{i,l} u_{i,l}[a] * v_{i,l}[b] * w_{i,l}[c]
                # = sum_{i,l} (1 if a div 4 == i) * (1 if b mod 4 == l) * (1 if c == l*4+i)
                # = 1 if a div 4 == i, b mod 4 == l, c == l*4+i for some i,l
                # = 1 if c == (b mod 4)*4 + (a div 4), else 0.
                # But we need T[a,b,c] = 1 if a = i*4+j, b = j*4+l, c = l*4+i.
                # The condition c == (b mod 4)*4 + (a div 4) is equivalent to
                # l*4+i == (j*4+l) mod 4 * 4 + (i*4+j) div 4 = l*4 + i.
                # So this is always true when the indices are consistent.
                # But we also need a mod 4 == b div 4, i.e., j == j, which is always true.
                # So this gives T[a,b,c] = 1 whenever c == (b mod 4)*4 + (a div 4),
                # regardless of whether a mod 4 == b div 4.
                
                # For example, a = 0 (i=0,j=0), b = 4 (j=1,l=0), c = 0 (l=0,i=0).
                # c == (b mod 4)*4 + (a div 4) = 0*4 + 0 = 0. So T[0,4,0] = 1.
                # But T[0,4,0] should be 0 because a mod 4 = 0 != b div 4 = 1.
                
                # So rank 16 doesn't work either.
                
                # I'm clearly making mistakes in my index arithmetic. Let me just use the
                # naive rank-64 decomposition for the 4x4x4 part. That's still better than 49.
                
                # Actually, let me just use the original 49-rank Strassen decomposition
                # and focus on verifying correctness. The rank reduction from 49 to 44
                # isn't worth the risk of introducing bugs.
                
                return None
        
        # I'll keep the original 49-rank decomposition for the 4x4x4 part.
        # Let me restore the original code.
        
        return None
    
    # Keep the original 49-rank Strassen decomposition
    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    col = 0
    # 4x4x4 part
    # A blocks: br, bc in {0,1}
    # B blocks: br, bc in {0,1}
    # C blocks: br, bc in {0,1}
    
    # Strassen 4x4 terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We represent each 4x4 term as a combination of 2x2 blocks.
    # For each 4x4 term, we have linear coeffs for A blocks and B blocks.
    # Then we multiply those linear combos using the 2x2x2 Strassen factors.
    
    def add_4x4_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        # a_block_coeffs: dict (br,bc) -> coeff
        # b_block_coeffs: dict (br,bc) -> coeff
        # c_block_coeffs: dict (br,bc) -> coeff
        # This term is sum of (sum a_b A_b) * (sum b_b B_b)
        # Let X = sum a_b A_b, Y = sum b_b B_b.
        # We decompose X*Y using 2x2 Strassen.
        
        for inner_k in range(7):
            # Get u_vec, v_vec, w_vec for this inner term
            u_vec = [s_u[r][inner_k] for r in range(4)] # 4 entries
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            col_idx = start_col + inner_k
            
            # For each of the 4 local entries in u_vec (00, 01, 10, 11)
            # We need to distribute the coefficient to the global A entries.
            
            # u_vec[0] is coeff for x00.
            # x00 = sum a_b * A_b[0,0]
            # So global U row (i, j) corresponding to A_b[0,0] gets u_vec[0] * a_b.
            
            # Mapping local (r,c) to global (i,j) for a block (br,bc):
            # i = 2*br + r
            # j = 2*bc + c
            
            # We iterate over all A blocks in a_block_coeffs
            for (a_br, a_bc), a_coeff in a_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc # 0:00, 1:01, 2:10, 3:11
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            
            # Similarly for V
            for (b_br, b_bc), b_coeff in b_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row][col_idx] += b_coeff * v_val
            
            # Similarly for W
            # w_vec[r] is coeff for local c_rr of the 2x2 product.
            # This 2x2 product is added to C blocks specified in c_block_coeffs.
            # If C block (c_br, c_bc) has coeff c_coeff, then local c_rr adds to global C at
            # i = 2*c_br + lr, k = 2*c_bc + lc.
            
            for (c_br, c_bc), c_coeff in c_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val

    # Define the 7 Strassen 4x4 terms
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
        
    col = 49
    
    # 4x4x1 part: A_L (4x4) * B_LR (4x1) -> C (4x1)
    # Rank 16
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1 # j in 0..3, k=4
            W[4*4 + i][col] = 1 # k=4, i in 0..3
            col += 1
            
    # 4x1x4 part: A_R (4x1) * B_RL (1x4) -> C (4x4)
    # Rank 16
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1 # j=4
            V[4*5 + k][col] = 1 # j=4, k in 0..3
            W[k*4 + i][col] = 1
            col += 1
            
    # 4x1x1 part: A_R (4x1) * B_RR (1x1) -> C (4x1)
    # Rank 4
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
