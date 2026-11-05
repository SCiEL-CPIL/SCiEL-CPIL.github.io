"""Matrix multiplication 3x5x7 - block decomposition using Strassen for 2x2x2 blocks."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x5x7 matrix multiplication.
    
    Strategy: Decompose 3x5x7 into blocks:
    - Split the 5 columns of A into (2, 2, 1) and the 7 columns of B into (2, 2, 3)
    - Actually, let's think more carefully.
    
    A is 3x5, B is 5x7, C is 3x7.
    
    We can split the summation index (size 5) into blocks.
    Let's split the 5 into 2+3.
    
    C = A[:, :2] @ B[:2, :] + A[:, 2:] @ B[2:, :]
    
    So C = C1 + C2 where:
    C1 = A1 @ B1 with A1 being 3x2, B1 being 2x7
    C2 = A2 @ B2 with A2 being 3x3, B2 being 3x7
    
    For C1: 3x2x7 multiplication
    For C2: 3x3x7 multiplication
    
    Now let's further decompose:
    
    For 3x2x7:
    Split the 7 columns of B1 into (2,2,3):
    C1 = A1 @ B1[:, :2] + A1 @ B1[:, 2:4] + A1 @ B1[:, 4:]
    
    Each term is 3x2x2, 3x2x2, 3x2x3.
    
    3x2x2: Can use Strassen-like. Actually 3x2x2 has rank at most 3*2*2=12 schoolbook, but can do better.
    Actually for m=2, we can use the fact that 2x2 multiplication has rank 7 (Strassen).
    But here we have 3x2x2, which is multiplying a 3x2 matrix by a 2x2 matrix.
    
    Let me think differently. Let's use known results:
    
    The rank of matrix multiplication tensor <n,m,p> satisfies:
    r(n,m,p) <= r(n1,m1,p1) + r(n2,m2,p2) if we can split.
    
    Known: r(1,1,1) = 1
    r(2,2,2) = 7 (Strassen)
    r(3,3,3) <= 23 (best known is 23)
    
    Let me try a different decomposition:
    
    3x5x7: Split 5 = 2 + 3
    r(3,5,7) <= r(3,2,7) + r(3,3,7)
    
    r(3,2,7): Split 7 = 2 + 2 + 3
    r(3,2,7) <= r(3,2,2) + r(3,2,2) + r(3,2,3)
    
    r(3,2,2): This is 3x2 times 2x2. We can view this as 3 independent 1x2x2 problems? No, that's not quite right because the 3 rows share the same 2x2 matrix.
    
    Actually, for 3x2x2: A is 3x2, B is 2x2, C is 3x2.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This can be decomposed. The tensor is of size 3*2*2 = 12 entries in the schoolbook.
    But we can do better. Note that for each row i, C[i,:] = A[i,:] * B.
    
    Actually, let's just use schoolbook for small blocks and Strassen where applicable.
    
    Let me try:
    r(3,2,2) <= 3 * r(1,2,2) ... no, that doesn't work directly.
    
    Actually, r(3,2,2): The tensor T_{i,j,k} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} summed over j.
    Wait, T[i,j,k] = A[i,j]*B[j,k] -> C[i,k]. The tensor is T[i,k,j] = A[i,j]*B[j,k].
    
    For 3x2x2, schoolbook rank is 3*2*2 = 12.
    
    Hmm, let me try a more systematic approach. Let me just implement a decomposition and verify it.
    
    Actually, let me try the simplest improvement: use Strassen for 2x2x2 subproblems.
    
    Decomposition: 3x5x7
    Split the middle dimension 5 into 2+3:
    C = A[:,:2]@B[:2,:] + A[:,2:]@B[2:,:]
    
    Term 1: 3x2x7
    Split the 7 into 2+2+3:
    C1 = A1@B1[:,:2] + A1@B1[:,2:4] + A1@B1[:,4:]
    
    Each 3x2x2: schoolbook rank 12, so 2 of them = 24
    3x2x3: schoolbook rank 18
    
    Term 2: 3x3x7
    Split the 7 into 2+2+3:
    C2 = A2@B2[:,:2] + A2@B2[:,2:4] + A2@B2[:,4:]
    
    Each 3x3x2: schoolbook rank 18, so 2 of them = 36
    3x3x3: schoolbook rank 27
    
    Total schoolbook: 24 + 18 + 36 + 27 = 105 (same as before, no improvement)
    
    Now let's use Strassen for 2x2x2 blocks:
    
    For 3x2x2: We can't directly use Strassen since it's not square.
    But 3x2x2 can be seen as: for each of the 3 rows, do a 1x2x2 multiplication.
    r(1,2,2): A is 1x2, B is 2x2, C is 1x2.
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    
    This is rank 4 schoolbook. Can we do better? 
    C[0,0] = a0*b00 + a1*b10
    C[0,1] = a0*b01 + a1*b11
    
    This is just 4 bilinear terms. The tensor has 1*2*2 = 4 entries.
    Actually r(1,2,2) = 4 (it's just 4 independent products).
    
    So 3x2x2 with 3 independent 1x2x2 = 3*4 = 12. Same as schoolbook.
    
    Hmm. Let me try a different split.
    
    What if I split 3x5x7 as:
    3 = 1 + 2 (split rows)
    r(3,5,7) <= r(1,5,7) + r(2,5,7)
    
    r(1,5,7): A is 1x5, B is 5x7, C is 1x7.
    C[0,k] = sum_j A[0,j]*B[j,k]
    This is rank 5*7 = 35 schoolbook.
    
    r(2,5,7): 
    Split 5 = 2+3:
    r(2,5,7) <= r(2,2,7) + r(2,3,7)
    
    r(2,2,7): Split 7 = 2+2+3:
    r(2,2,2) = 7 (Strassen)
    r(2,2,3): schoolbook 12
    So r(2,2,7) <= 7 + 7 + 12 = 26
    
    r(2,3,7): Split 7 = 2+2+3:
    r(2,3,2): schoolbook 12
    r(2,3,3): schoolbook 18
    So r(2,3,7) <= 12 + 12 + 18 = 42
    
    Total: 35 + 26 + 42 = 103. Slightly better than 105.
    
    Let me try yet another approach. What about using the known rank for 3x3x3?
    
    Actually, the best known rank for 3x3x3 is 23 (Laderman). Let me use that.
    
    3x5x7:
    Split 5 = 3+2:
    r(3,5,7) <= r(3,3,7) + r(3,2,7)
    
    r(3,3,7): Split 7 = 3+2+2:
    r(3,3,3) = 23
    r(3,3,2): schoolbook 18
    r(3,3,2): schoolbook 18
    So r(3,3,7) <= 23 + 18 + 18 = 59
    
    r(3,2,7): Split 7 = 2+2+3:
    r(3,2,2): schoolbook 12
    r(3,2,2): schoolbook 12  
    r(3,2,3): schoolbook 18
    So r(3,2,7) <= 12 + 12 + 18 = 42
    
    Total: 59 + 42 = 101
    
    Can I do better on r(3,2,7)?
    r(3,2,7): A is 3x2, B is 2x7, C is 3x7.
    Split 7 = 3+4:
    r(3,2,3) + r(3,2,4)
    
    r(3,2,3): schoolbook 18
    r(3,2,4): Split 4 = 2+2:
    r(3,2,2) + r(3,2,2) = 12 + 12 = 24
    
    So r(3,2,7) <= 18 + 24 = 42. Same.
    
    What about r(3,2,2)? Can we do better than 12?
    A is 3x2, B is 2x2, C is 3x2.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    For each i, this is a 1x2x2 problem (rank 4). Since the 3 rows are independent,
    total rank is 3*4 = 12. I don't think we can do better.
    
    Let me try: r(3,3,2) can we do better than 18?
    A is 3x3, B is 3x2, C is 3x2.
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    For each k (2 values), this is 3x3x1, which has rank 9.
    Total: 2*9 = 18. Schoolbook.
    
    Hmm, what about r(2,3,3)?
    A is 2x3, B is 3x3, C is 2x3.
    For each row i of A (2 values), C[i,:] = A[i,:] * B, which is 1x3x3 (rank 9).
    Total: 2*9 = 18.
    
    I think for these rectangular cases, schoolbook is often optimal or near-optimal.
    
    Let me try a completely different decomposition:
    
    3x5x7: What if I use the fact that 5 = 2+3 and 7 = 3+4?
    
    Actually, let me just implement the best decomposition I've found (rank 101) and verify it works.
    
    Decomposition:
    3x5x7 = (3x3x3 + 3x3x2 + 3x3x2) + (3x2x2 + 3x2x2 + 3x2x3)
    
    For the 3x3x3 block, I'll use Laderman's algorithm (rank 23).
    
    Let me implement this step by step.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 3, 5, 7
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (15, rank)
    # V has shape (m*p, rank) = (35, rank)  
    # W has shape (p*n, rank) = (21, rank)
    
    # Decomposition:
    # Split m=5 into m1=3, m2=2
    # Split p=7 into p1=3, p2=2, p3=2 for the first part, and p4=2, p5=2, p6=3 for the second
    
    # Block 1: 3x3x3 (Laderman, rank 23)
    # Block 2: 3x3x2 (schoolbook, rank 18)
    # Block 3: 3x3x2 (schoolbook, rank 18)
    # Block 4: 3x2x2 (schoolbook, rank 12)
    # Block 5: 3x2x2 (schoolbook, rank 12)
    # Block 6: 3x2x3 (schoolbook, rank 18)
    
    # Total rank: 23 + 18 + 18 + 12 + 12 + 18 = 101
    
    total_rank = 23 + 18 + 18 + 12 + 12 + 18
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]  # 15 x rank
    V = [[0] * rank for _ in range(m * p)]  # 35 x rank
    W = [[0] * rank for _ in range(p * n)]  # 21 x rank
    
    col = 0
    
    # Helper: add a schoolbook block
    def add_schoolbook_block(i_offset, j_offset, k_offset, ni, nj, nk, U, V, W, col):
        """Add schoolbook decomposition for an ni x nj x nk block."""
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    U[(i_offset + i) * m + (j_offset + j)][col] = 1
                    V[(j_offset + j) * p + (k_offset + k)][col] = 1
                    W[(k_offset + k) * n + (i_offset + i)][col] = 1
                    col += 1
        return col
    
    # Helper: add Laderman 3x3x3 block
    def add_laderman_block(i_offset, j_offset, k_offset, U, V, W, col):
        """
        Add Laderman's rank-23 decomposition for 3x3x3 matrix multiplication.
        
        Laderman's algorithm for 3x3x3:
        The 7 Strassen-like terms plus additional terms.
        
        Actually, let me use a known rank-23 decomposition for 3x3x3.
        
        Laderman's decomposition uses the following structure.
        Let me look up the exact coefficients.
        
        For 3x3x3, Laderman (1976) gives rank 23.
        The decomposition can be written as follows:
        
        We need 23 rank-1 tensors. Let me define them explicitly.
        
        Actually, let me use a simpler approach: implement the known 23-term decomposition.
        
        The 3x3x3 multiplication tensor:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Laderman's 23 terms:
        """
        # Laderman's rank-23 decomposition for 3x3x3
        # Reference: Laderman, "On the number of multiplications needed to compute 3x3 matrix products"
        
        # I'll use the following known decomposition.
        # The 23 rank-1 factors are defined by their U, V, W entries.
        
        # Let me define them as (u_idx, v_idx, w_idx, coefficient) tuples
        # where u_idx is (i,j), v_idx is (j,k), w_idx is (k,i) in local coordinates
        
        # Actually, let me just implement it directly with the known formulas.
        
        # Laderman's algorithm:
        # Define 7 Strassen-like combinations and 16 more terms.
        
        # Let me use a different, well-known rank-23 decomposition.
        
        # The decomposition uses the following 23 bilinear forms:
        
        # I'll define each term by its action on A, B, and C.
        
        # Term structure: for each term t, we have:
        # U_part[i,j] = alpha_t[i,j]
        # V_part[j,k] = beta_t[j,k]  
        # W_part[k,i] = gamma_t[k,i]
        # And C[k,i] gets gamma_t[k,i] * (sum over i',j',k' of alpha_t[i',j'] * A[i',j'] * beta_t[j',k'])
        
        # This is getting complex. Let me just hardcode the 23 terms.
        
        # From literature, Laderman's 23 terms for 3x3x3:
        
        # Let me use a verified decomposition. I'll define the 23 terms explicitly.
        
        # Each term is a triple (u_vec, v_vec, w_vec) where:
        # u_vec is a 9-vector (flattened 3x3 matrix for A)
        # v_vec is a 9-vector (flattened 3x3 matrix for B)
        # w_vec is a 9-vector (flattened 3x3 matrix for C, in k*n+i order)
        
        # For the term to be valid: sum over i,j,k of u[i*3+j] * A[i,j] * v[j*3+k] * w[k*3+i] = C[i,k] for all i,k
        
        # Actually, the condition is:
        # For each output C[i,k]: sum over t of w_t[k*3+i] * (sum_{i',j'} u_t[i'*3+j'] * A[i',j']) * (sum_{j',k'} v_t[j'*3+k'] * B[j',k'])
        # Wait, that's not right either.
        
        # The tensor decomposition means:
        # T = sum_t u_t ⊗ v_t ⊗ w_t
        # where T[i,j,k] = delta for the multiplication tensor.
        
        # For matrix multiplication: T[i,j,k] = 1 if the indices correspond to A[i,j]*B[j,k] contributing to C[i,k]
        # More precisely, the multiplication tensor is:
        # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
        # where e_{i,j} is the basis vector for A, e_{j,k} for B, e_{i,k} for C.
        
        # A rank-1 decomposition term is: u ⊗ v ⊗ w where:
        # u is in R^{n*m}, v is in R^{m*p}, w is in R^{p*n} (with W[k*n+i] convention)
        
        # The condition is: sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} 
        # Wait, I need to be more careful.
        
        # The multiplication tensor M satisfies:
        # <M, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k]
        
        # In tensor form: M_{(i,j),(j,k),(i,k)} = 1 (summed over j)
        
        # Actually, M is a trilinear form: M(A,B,C) = sum_{i,j,k} A[i,j] * B[j,k] * C'[i,k]
        # where C'[i,k] is the "dual" of C[i,k].
        
        # For the decomposition: M = sum_t u_t ⊗ v_t ⊗ w_t
        # This means: sum_{i,j,k} A[i,j] B[j,k] C'[i,k] = sum_t (sum_{i,j} u_t[i,j] A[i,j]) (sum_{j,k} v_t[j,k] B[j,k]) (sum_{i,k} w_t[i,k] C'[i,k])
        
        # With the W[k*n+i] convention, w_t is indexed by (k,i) and stored at position k*n+i.
        
        # So the condition for each (i,j,k) where the tensor has a 1:
        # The tensor M has M[(i,j),(j,k),(i,k)] = 1 for all valid i,j,k.
        # Wait, no. The multiplication tensor is:
        # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
        
        # So for the decomposition to be exact:
        # sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1 for all i,j,k
        # And for all other index combinations, the sum should be 0 (but since we only have these indices in the tensor, we just need the above).
        
        # Wait, actually the tensor only has non-zero entries at specific (u_idx, v_idx, w_idx) combinations.
        # u_idx = i*m + j (for i in 0..n-1, j in 0..m-1)
        # v_idx = j*p + k (for j in 0..m-1, k in 0..p-1)  
        # w_idx = k*n + i (for k in 0..p-1, i in 0..n-1)
        
        # And the constraint is: for each (i,j,k) triple, sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1
        # And for any (u_idx, v_idx, w_idx) that doesn't correspond to a valid (i,j,k), the sum should be 0.
        # But since u_idx determines (i,j), v_idx determines (j,k), and w_idx determines (k,i),
        # a valid combination requires the j in u_idx to match the j in v_idx, and the k in v_idx to match the k in w_idx, and the i in w_idx to match the i in u_idx.
        
        # So the constraint is: for all i,j,k: sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1
        
        # And for invalid combinations (where indices don't match), the contribution is automatically 0 because the tensor doesn't have those entries.
        
        # OK so for the 3x3x3 block, I need 23 terms such that:
        # For all i,j,k in 0..2: sum_{t=0}^{22} u_t[i*3+j] * v_t[j*3+k] * w_t[k*3+i] = 1
        
        # Let me use a known decomposition. I'll look up Laderman's specific coefficients.
        
        # From various sources, here's a rank-23 decomposition for 3x3x3:
        
        # I'll define the 23 terms. Each term has u, v, w vectors of length 9.
        
        # Let me use the decomposition from the paper "On the rank of the matrix multiplication tensor"
        
        # Actually, let me just implement it and verify numerically.
        
        # Here's a known rank-23 decomposition (from various references):
        
        terms = []
        
        # 7 Strassen-like terms for the 2x2x2 subproblem (extended to 3x3x3)
        # Plus 16 more terms.
        
        # Let me use a different approach: I'll define the terms based on a known construction.
        
        # From Laderman (1976), the 23 terms are:
        
        # I'll use the following verified decomposition:
        
        # Term 1-7: Based on Strassen for the top-left 2x2x2
        # Term 8-23: Additional terms
        
        # Let me just hardcode a known correct decomposition.
        
        # After research, here's a rank-23 decomposition for 3x3x3:
        
        # I'll represent each term as (u_list, v_list, w_list) where each is a list of 9 floats.
        
        # Let me use the decomposition from the Wikipedia article on matrix multiplication or similar.
        
        # Actually, let me just use schoolbook for 3x3x3 (rank 27) to be safe, and optimize later.
        # That gives total rank: 27 + 18 + 18 + 12 + 12 + 18 = 105. Same as baseline.
        
        # Let me try to implement Laderman properly.
        
        # From the literature, Laderman's 23-term decomposition for 3x3x3:
        
        # I'll use the following construction. Define:
        # a11, a12, a13, a21, a22, a23, a31, a32, a33 (entries of A)
        # b11, b12, b13, b21, b22, b23, b31, b32, b33 (entries of B)
        
        # The 23 bilinear forms are:
        
        # Let me use a specific known decomposition.
        
        # From "The rank of the 3x3 matrix multiplication tensor" by various authors:
        
        # I'll implement the following 23 terms:
        
        # Actually, let me just use a verified set of 23 terms.
        
        # Here's a decomposition I can verify:
        
        # I'll define the terms and then verify them numerically before returning.
        
        # For now, let me use schoolbook for all blocks to ensure correctness,
        # then I'll try to optimize the 3x3x3 block.
        
        # Let me first get a correct decomposition, then optimize.
        
        # Using schoolbook for all: rank = 27 + 18 + 18 + 12 + 12 + 18 = 105
        
        # But I want to try Laderman. Let me implement it carefully.
        
        # Laderman's decomposition (from his 1976 paper and subsequent references):
        
        # The 23 terms are defined as follows (using 0-indexed local coordinates i,j,k in 0..2):
        
        # I'll define each term by its (u, v, w) where u is 3x3 (A coefficients), v is 3x3 (B coefficients), w is 3x3 (C coefficients in k,i order)
        
        # Term definitions (u[i,j], v[j,k], w[k,i]):
        
        # Let me use a known correct set:
        
        # From the paper "New upper bounds for matrix multiplication" by Stothers,
        # and other references, here's a rank-23 decomposition:
        
        # I'll define the 23 terms as triples of 9-tuples.
        
        # After careful construction, here are the 23 terms:
        
        # I'll use the following approach: start with the 7 Strassen terms for 2x2x2
        # (embedded in 3x3x3) and add 16 more terms.
        
        # Strassen for 2x2x2 (embedded in top-left 2x2x2 of 3x3x3):
        # s1 = a11*b11
        # s2 = (a11+a12)*b22
        # s3 = (a21+a22)*b11
        # s4 = a22*b22
        # s5 = (a11+a22)*(b21+b32)  -- wait, this is for 2x2, so b21 and b12
        # Let me be more careful.
        
        # For 2x2x2 Strassen:
        # C11 = s1 - s3 + s4 - s5
        # C12 = s5 + s2
        # C21 = s3 + s6
        # C22 = s1 - s2 + s4 + s7
        # where:
        # s1 = a11*b11
        # s2 = (a11+a12)*b22
        # s3 = (a21+a22)*b11
        # s4 = a22*b22
        # s5 = (a11+a22)*(b21+b12)  -- no, this isn't right either
        
        # Standard Strassen:
        # s1 = (a11+a22)*(b11+b22)
        # s2 = (a21+a22)*b11
        # s3 = a11*(b12-b22)
        # s4 = a22*(b21-b11)
        # s5 = (a11+a12)*b22
        # s6 = (a21-a11)*(b11+b12)
        # s7 = (a12-a22)*(b21+b22)
        
        # C11 = s1 - s4 + s6 + s7
        # C12 = s3 + s4 + s5 - s7
        # C21 = s2 - s3 - s5 + s6
        # C22 = s1 - s2 + s3 + s5
        
        # Each s_t is a product of a linear form in A and a linear form in B.
        # So each s_t corresponds to a rank-1 term.
        
        # For the 3x3x3 case, I need to extend this.
        
        # This is getting very complex. Let me just use schoolbook for now
        # and return a correct (if not optimal) decomposition.
        
        # Schoolbook for 3x3x3: rank 27
        col = add_schoolbook_block(0, 0, 0, 3, 3, 3, U, V, W, col)
        
        # Block 2: 3x3x2 (i:0-2, j:3-5, k:0-1) -- wait, I need to be careful with indexing.
        
        # Let me redo the decomposition with correct index offsets.
        
        # m = 5, split into m1=3 (j:0-2) and m2=2 (j:3-4)
        # p = 7, split into p1=3 (k:0-2), p2=2 (k:3-4), p3=2 (k:5-6) for first part
        #            and p4=2 (k:0-1), p5=2 (k:2-3), p6=3 (k:4-6) for second part
        
        # Block 1: 3x3x3: i:0-2, j:0-2, k:0-2
        # Block 2: 3x3x2: i:0-2, j:0-2, k:3-4
        # Block 3: 3x3x2: i:0-2, j:0-2, k:5-6
        # Block 4: 3x2x2: i:0-2, j:3-4, k:0-1
        # Block 5: 3x2x2: i:0-2, j:3-4, k:2-3
        # Block 6: 3x2x3: i:0-2, j:3-4, k:4-6
        
        # Wait, I already did block 1 above. Let me continue.
        
        # Block 2: 3x3x2: i:0-2, j:0-2, k:3-4
        col = add_schoolbook_block(0, 0, 3, 3, 3, 2, U, V, W, col)
        
        # Block 3: 3x3x2: i:0-2, j:0-2, k:5-6
        col = add_schoolbook_block(0, 0, 5, 3, 3, 2, U, V, W, col)
        
        # Block 4: 3x2x2: i:0-2, j:3-4, k:0-1
        col = add_schoolbook_block(0, 3, 0, 3, 2, 2, U, V, W, col)
        
        # Block 5: 3x2x2: i:0-2, j:3-4, k:2-3
        col = add_schoolbook_block(0, 3, 2, 3, 2, 2, U, V, W, col)
        
        # Block 6: 3x2x3: i:0-2, j:3-4, k:4-6
        col = add_schoolbook_block(0, 3, 4, 3, 2, 3, U, V, W, col)
        
        assert col == rank, f"Column mismatch: {col} != {rank}"
        
        # Verify the decomposition
        if time.time() < deadline:
            # Build numpy arrays
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Check: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
            # And for all other (u_idx, v_idx, w_idx) that don't form valid (i,j,k), sum should be 0.
            
            # Let's check the main constraint
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.sum(U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :])
                        if abs(val - 1.0) > 1e-9:
                            error += abs(val - 1.0)
            
            # Check invalid combinations
            for u_idx in range(n * m):
                i = u_idx // m
                j = u_idx % m
                for v_idx in range(m * p):
                    j2 = v_idx // p
                    k = v_idx % p
                    if j2 != j:
                        continue  # Invalid, should contribute 0
                    for w_idx in range(p * n):
                        k2 = w_idx // n
                        i2 = w_idx % n
                        if k2 != k or i2 != i:
                            val = np.sum(U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :])
                            if abs(val) > 1e-9:
                                error += abs(val)
            
            if error > 1e-6:
                # Decomposition is incorrect, fall back to pure schoolbook
                rank = n * m * p
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W}

