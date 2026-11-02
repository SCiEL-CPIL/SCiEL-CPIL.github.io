import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    # Target dimensions for factors
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    
    # We will construct a decomposition of the matrix multiplication tensor.
    # The tensor T has indices (u, v, w) where:
    # u corresponds to A[i, j] with u = i*m + j
    # v corresponds to B[j, k] with v = j*p + k
    # w corresponds to C[i, k] with w = k*n + i  (Note: task says W[k*n+i, r])
    # Let's verify the W index convention: "W[k*n+i,r]".
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i).
    # Specifically, for a given term r, U[u,r] V[v,r] W[w,r] should sum to T[u,v,w].
    
    # Strategy: Use the "schoolbook" algorithm which has rank n*m*p.
    # Rank = 4 * 4 * 5 = 80.
    # This is a valid, exact decomposition.
    # We can try to optimize this, but 80 is a safe baseline.
    # Given the complexity of implementing Strassen for rectangular matrices correctly within the time limit,
    # and the previous failure due to empty lists, providing a valid schoolbook decomposition is the priority.
    
    # Can we do better?
    # For 4x4x4, Strassen gives rank 49.
    # For 4x4x5, we can view it as two 4x4x4 products? No, dimensions don't match directly.
    # We can embed 4x4x5 into 4x4x8? No.
    # We can use the 4x4x4 Strassen decomposition for the first 4 columns of B and output,
    # and then handle the 5th column?
    # Let's stick to the schoolbook rank 80 first to ensure correctness and a non-zero score.
    # Then, if time permits, we might attempt a refinement or a known lower bound.
    # However, the prompt implies finding a "rank-minimal" or at least improved one.
    # The "island best" mentioned in the knowledge suggests a score of 1.111 was achieved previously with a different strategy.
    # Score = seed/value. If seed is 1.0, value 1.111 means rank was lower than seed?
    # Wait, "score is the rank". "score is seed/value".
    # Usually in these tasks, lower rank is better.
    # If the score is seed/value, and higher score is better, then value must be lower?
    # "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    # This implies: Score = Seed_Rank / Current_Rank.
    # If Current_Rank < Seed_Rank, Score > 1.
    # What is the Seed_Rank?
    # The previous program failed. The "seed program" likely refers to a baseline rank, often the schoolbook rank or a specific known rank.
    # If the seed program is the schoolbook (rank 80), then a rank of 72 would give 80/72 = 1.111.
    # So the goal is to find a rank < 80.
    
    # Let's try to construct a rank < 80.
    # We can use the Strassen algorithm for the 4x4 part.
    # A is 4x4, B is 4x5.
    # We can split B into B1 (4x4) and B2 (4x1).
    # C1 = A * B1 (4x4). Rank 49 using Strassen.
    # C2 = A * B2 (4x1). This is a matrix-vector product.
    # Matrix-vector product 4x4x1.
    # Rank of 4x4x1 multiplication tensor?
    # It's just summing 4 terms per output.
    # C2[i] = sum_j A[i,j] B2[j].
    # This is a bilinear form.
    # The rank of the tensor for multiplying a 4x4 matrix by a 4x1 vector is 4*4 = 16?
    # Let's check.
    # T[u, v, w] where u=(i,j), v=j, w=i.
    # Sum over j.
    # Each output i requires 4 multiplications. Total 16 multiplications.
    # Can we do better?
    # The tensor is a sum of 4 rank-1 tensors?
    # C[i] = A[i,0]B[0] + A[i,1]B[1] + A[i,2]B[2] + A[i,3]B[3].
    # This is exactly 16 rank-1 terms if we treat each product A[i,j]B[j] as a term contributing to C[i].
    # Is it possible to share multiplications?
    # No, because the A entries are distinct and B entries are distinct.
    # So rank for 4x4x1 is 16.
    
    # Total rank = 49 (for 4x4x4) + 16 (for 4x4x1) = 65.
    # 65 < 80.
    # Score would be 80/65 = 1.23.
    
    # Let's implement this.
    # 1. Generate Strassen terms for 4x4x4.
    # 2. Generate schoolbook terms for 4x4x1.
    # 3. Combine them.
    
    def strassen_4x4_terms():
        # Returns list of (u_vec, v_vec, w_vec)
        # u_vec length 16 (4x4)
        # v_vec length 16 (4x4)
        # w_vec length 16 (4x4)
        # Indices:
        # u: i*4+j
        # v: j*4+k
        # w: k*4+i  (Wait, for 4x4x4, W index is k*4+i? 
        # The task says W[k*n+i]. For n=4, W[k*4+i].
        # In the 4x4x4 sub-problem, the "global" n is 4.
        # But we are treating the 4x4 part as a separate block.
        # The w index for the 4x4 part must map to the global w index.
        # Global w index = k_global * n + i_global.
        # For the 4x4 block (rows 0-3, cols 0-3), k is 0-3, i is 0-3.
        # So w_local = k * 4 + i.
        # This matches the global indexing if we just use the same formula.
        
        terms = []
        
        # Strassen 2x2x2 local terms
        # We need to map 2x2 blocks to 4x4 indices.
        
        # Block (0,0): A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
        # Block (0,1): A[0:2, 2:4], B[2:4, 0:2], C[0:2, 2:4] -- Wait, B is 4x4.
        # Standard Strassen splits A into 4 2x2 blocks, B into 4 2x2 blocks.
        # A11 A12
        # A21 A22
        # B11 B12
        # B21 B22
        # C11 C12
        # C21 C22
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)(B11)
        # M3 = (A11)(B22-B21)
        # M4 = (A22)(B11-B12)
        # M5 = (A11+A12)(B21+B22)
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M2 + M5
        # C21 = M3 + M6
        # C22 = M1 - M2 + M3 + M4
        
        # We need to generate 7 terms for the 2x2x2 multiplication inside each "super-term"?
        # No, Strassen replaces 8 scalar multiplications with 7.
        # The 7 products are of 2x2 matrices.
        # Each 2x2 matrix product can be computed with 4 scalar multiplications (schoolbook) or recursively.
        # If we use schoolbook for the 2x2 products, we get 7 * 4 = 28 terms?
        # No, that's not right.
        # Strassen's algorithm reduces the number of recursive calls.
        # If we go down to scalar (1x1), the rank is 7^log2(4) = 49.
        # This means we need to recursively apply Strassen to the 2x2 blocks.
        
        # Let's implement a recursive function that generates terms for an N x N x N multiplication.
        # Base case N=1: 1 term.
        # Case N=2: 7 terms, each involving linear combinations of the 4 entries.
        # But the "terms" in the tensor decomposition are rank-1 tensors.
        # A "term" in Strassen corresponds to a product of two linear forms in A and B,
        # which is then added to the output C.
        # However, the product of two linear forms is not a single rank-1 tensor in the A,B,C space?
        # Actually, $M = ( \sum a_i A_i ) ( \sum b_j B_j ) = \sum_{i,j} a_i b_j A_i B_j$.
        # This is a sum of rank-1 tensors $A_i B_j$ (which correspond to specific u,v) mapped to C.
        # So each Strassen "multiplication" step expands into multiple rank-1 terms in the final decomposition.
        
        # Let's define the function to generate rank-1 terms for N x N x N.
        # Input: N
        # Output: List of (u_vec, v_vec, w_vec) where vecs are length N*N.
        
        def gen_terms(N):
            if N == 1:
                # u: [1], v: [1], w: [1]
                return [([1.0], [1.0], [1.0])]
            
            # For N=2, we use Strassen.
            # But we can recursively call gen_terms(N/2) for the 2x2 blocks?
            # No, Strassen combines the blocks.
            
            # Let's hardcode the 7 linear forms for N=2.
            # A indices: 0,1,2,3 (row-major: 00, 01, 10, 11)
            # B indices: 0,1,2,3
            # C indices: 0,1,2,3 (col-major for W? W[k*N+i]. 
            # For N=2, W[k*2+i].
            # C[0,0] is k=0,i=0 -> idx 0.
            # C[0,1] is k=1,i=0 -> idx 2.
            # C[1,0] is k=0,i=1 -> idx 1.
            # C[1,1] is k=1,i=1 -> idx 3.
            
            # The 7 products M1..M7 are products of 2x2 matrices.
            # Each M is a matrix.
            # We need to decompose the contribution of M to C into rank-1 terms.
            # But M is computed as a product of two linear combinations of A and B.
            # Let $U_M = \sum c_i A_i$ and $V_M = \sum d_j B_j$.
            # Then $M = U_M V_M$.
            # This M contributes to C.
            # But $U_M V_M$ is a matrix.
            # The tensor decomposition requires expressing the final C as a sum of rank-1 tensors.
            # $C = \sum_r U_r \otimes V_r \otimes W_r$.
            # The Strassen algorithm expresses C as a linear combination of 7 products $M_k$.
            # Each $M_k$ is a 2x2 matrix product.
            # We can recursively decompose each $M_k$ into rank-1 terms?
            # Yes! The rank of N x N x N is $T(N) = 7 T(N/2)$.
            # So for N=2, $T(2) = 7 T(1) = 7$.
            # For N=4, $T(4) = 7 T(2) = 49$.
            
            # So, to generate terms for N=4:
            # 1. Define the 7 linear forms for the 2x2 blocks of A and B.
            # 2. For each of the 7 forms, we have a "sub-problem" of multiplying two 2x2 matrices.
            # 3. We recursively generate the 7 terms for that 2x2 sub-problem.
            # 4. We map the indices of the sub-problem to the global indices.
            
            # Let's implement this recursion.
            
            if N == 2:
                # The 7 terms for 2x2x2 are the Strassen products.
                # But wait, if we call gen_terms(1) for each of the 7 blocks,
                # we get 7 * 1 = 7 terms.
                # Each term from gen_terms(1) is a scalar product.
                # We need to map the scalar indices to the 2x2 block indices.
                
                # Let's define the linear forms explicitly for N=2.
                # A blocks:
                # A11: indices 0,1,4,5 (in 4x4? No, in 2x2 local)
                # In local 2x2, indices are 0,1,2,3.
                # A11 (top-left): 0, 1? No, A is 2x2.
                # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3.
                # A11 is just A[0,0]? No, A11 is the top-left 1x1 block if we split 2x2 into 1x1?
                # Yes, for N=2, we split into four N/2=1 blocks.
                
                # So for N=2:
                # A11 = A[0,0] (idx 0)
                # A12 = A[0,1] (idx 1)
                # A21 = A[1,0] (idx 2)
                # A22 = A[1,1] (idx 3)
                
                # Same for B.
                
                # M1 = (A11+A22)(B11+B22)
                # u1 = A[0] + A[3]
                # v1 = B[0] + B[3]
                # This product contributes to C.
                # How does it map to C?
                # C11 = M1 + ...
                # C11 is the top-left 1x1 block of C.
                # In local 2x2 C, C11 is C[0,0] (idx 0).
                # So the term is:
                # u_vec: 1 at 0, 1 at 3.
                # v_vec: 1 at 0, 1 at 3.
                # w_vec: 1 at 0. (Contributes to C[0,0])
                
                # But wait, M1 is a 1x1 matrix product here (since N/2=1).
                # So it's just a scalar.
                # The contribution to C11 is M1.
                # So w_vec has 1 at C11 index.
                
                # Let's define the 7 terms for N=2 directly.
                
                terms_n2 = []
                
                # M1: (A00+A11)(B00+B11) -> C00, C11?
                # No, in the standard Strassen for 2x2:
                # C11 = M1 + M4 - M5 + M7
                # C12 = M2 + M5
                # C21 = M3 + M6
                # C22 = M1 - M2 + M3 + M4
                
                # M1 = (A11+A22)(B11+B22)
                # u: A11, A22
                # v: B11, B22
                # w: C11, C22 (with coeffs 1, 1)
                
                # Let's build the vectors.
                # Size 4.
                
                # M1
                u = [0]*4; v = [0]*4; w = [0]*4
                u[0] = 1; u[3] = 1
                v[0] = 1; v[3] = 1
                w[0] = 1; w[3] = 1 # C11 (idx 0), C22 (idx 3)
                terms_n2.append((u, v, w))
                
                # M2 = (A21+A22)(B11)
                u = [0]*4; v = [0]*4; w = [0]*4
                u[2] = 1; u[3] = 1
                v[0] = 1
                w[1] = 1 # C12 (idx 1? C[0,1] is k=1,i=0 -> 1*2+0=2? No.)
                # Let's re-verify W index for N=2.
                # W[k*2+i].
                # C[0,0]: k=0, i=0 -> 0.
                # C[0,1]: k=1, i=0 -> 2.
                # C[1,0]: k=0, i=1 -> 1.
                # C[1,1]: k=1, i=1 -> 3.
                
                # So:
                # C11 (top-left) is C[0,0] -> idx 0.
                # C12 (top-right) is C[0,1] -> idx 2.
                # C21 (bottom-left) is C[1,0] -> idx 1.
                # C22 (bottom-right) is C[1,1] -> idx 3.
                
                # Redo M1:
                # w[0] = 1 (C11)
                # w[3] = 1 (C22)
                
                # M2:
                # w[2] = 1 (C12)
                
                # M3 = (A11)(B22-B21)
                # w[1] = 1 (C21)
                
                # M4 = (A22)(B11-B12)
                # w[0] = 1 (C11)
                # w[3] = 1 (C22)
                
                # M5 = (A11+A12)(B21+B22)
                # w[0] = -1 (C11)
                # w[2] = 1 (C12)
                
                # M6 = (A21-A11)(B11+B12)
                # w[1] = 1 (C21)
                
                # M7 = (A12-A22)(B21+B22)
                # w[0] = 1 (C11)
                # w[3] = -1 (C22)
                
                # Let's implement this carefully.
                
                def make_term(u_dict, v_dict, w_dict):
                    u = [0]*4; v = [0]*4; w = [0]*4
                    for idx, val in u_dict.items(): u[idx] = val
                    for idx, val in v_dict.items(): v[idx] = val
                    for idx, val in w_dict.items(): w[idx] = val
                    return (u, v, w)
                
                t1 = make_term({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
                t2 = make_term({2:1, 3:1}, {0:1}, {2:1})
                t3 = make_term({0:1}, {3:1, 2:-1}, {1:1}) # B22-B21 -> B[3]-B[2]
                t4 = make_term({3:1}, {0:1, 1:-1}, {0:1, 3:1}) # B11-B12 -> B[0]-B[1]
                t5 = make_term({0:1, 1:1}, {2:1, 3:1}, {0:-1, 2:1}) # B21+B22 -> B[2]+B[3]
                t6 = make_term({2:1, 0:-1}, {0:1, 1:1}, {1:1}) # A21-A11, B11+B12
                t7 = make_term({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:-1}) # A12-A22, B21+B22
                
                return [t1, t2, t3, t4, t5, t6, t7]
            
            # If N > 2, we recurse.
            # For N=4:
            # Split A, B, C into 2x2 blocks of size 2x2.
            # A11, A12, A21, A22 are 2x2 matrices.
            # The 7 Strassen products are 2x2 matrix products.
            # We call gen_terms(2) to get the 7 terms for a 2x2x2 multiplication.
            # Each of these terms has vectors of length 4.
            # We need to map these length-4 vectors to the global length-16 vectors.
            
            sub_terms = gen_terms(N // 2)
            
            # Define the linear forms for the 7 Strassen products for N.
            # This part is complex to generalize recursively without a symbolic algebra system.
            # Given the time constraint and the specific N=4, I will hardcode the mapping for N=4.
            
            # For N=4, the 7 products are:
            # M1 = (A11+A22)(B11+B22)
            # ...
            
            # Let's just return the N=2 terms if N==2.
            # For N=4, we need to expand.
            
            # Since I can't easily write the recursive expansion for arbitrary N in this short time,
            # and I only need N=4, I will write a specific function for N=4 that uses the N=2 logic.
            
            if N == 4:
                # Get the 7 terms for 2x2x2.
                # These terms represent the structure of a 2x2 matrix multiplication.
                # But we are multiplying 2x2 blocks.
                # So each "scalar" in the 2x2 Strassen formula is actually a 2x2 matrix.
                # And the "multiplication" is a 2x2 matrix multiplication.
                # So we need to substitute the 2x2 matrix multiplication (which is 4 terms)
                # into the 7 Strassen formulas?
                # No, Strassen says we need 7 *multiplications* of size N/2.
                # So we need 7 * (rank of N/2 x N/2 x N/2) terms.
                # Rank of 2x2x2 is 4 (schoolbook) or 7 (Strassen)?
                # If we use Strassen for the sub-problems, Rank(4) = 7 * Rank(2).
                # Rank(2) = 7 * Rank(1) = 7.
                # So Rank(4) = 49.
                
                # So we need to take the 7 "meta-terms" of Strassen for N=4,
                # and for each one, expand the "matrix multiplication" into the 7 terms of Strassen for N=2.
                
                # Let's define the linear forms for the 7 meta-terms.
                # A11 is A[0:2, 0:2]. Indices 0,1,4,5.
                # A12 is A[0:2, 2:4]. Indices 2,3,6,7.
                # A21 is A[2:4, 0:2]. Indices 8,9,12,13.
                # A22 is A[2:4, 2:4]. Indices 10,11,14,15.
                
                # We need to create vectors of length 16 for U, V, W.
                
                terms_n4 = []
                
                # Helper to create a vector from a dict
                def vec16(d):
                    v = [0.0]*16
                    for k, val in d.items():
                        v[k] = val
                    return v
                
                # Define the 7 meta-linear forms for A and B.
                # M1: A11+A22, B11+B22
                # M2: A21+A22, B11
                # M3: A11, B22-B21
                # M4: A22, B11-B12
                # M5: A11+A12, B21+B22
                # M6: A21-A11, B11+B12
                # M7: A12-A22, B21+B22
                
                # For each meta-term, we have a linear form in A (length 16) and B (length 16).
                # But the "multiplication" is a 2x2 matrix product.
                # This means the "u" vector for the term is not just a single vector,
                # but a combination of vectors corresponding to the 4 elements of the 2x2 matrix?
                
                # Wait, the tensor decomposition is linear.
                # $T = \sum_r U_r \otimes V_r \otimes W_r$.
                # The Strassen algorithm expresses $C$ as a sum of 7 terms $M_k$.
                # $M_k = U_k(A) V_k(B)$.
                # $U_k(A)$ is a 2x2 matrix. $V_k(B)$ is a 2x2 matrix.
                # The product $M_k$ is a 2x2 matrix.
                # This $M_k$ contributes to the blocks of $C$.
                
                # To decompose this into rank-1 tensors, we need to expand the 2x2 matrix product.
                # A 2x2 matrix product $P Q$ can be written as a sum of 4 rank-1 tensors?
                # $P Q = \sum_{j} P_{:,j} Q_{j,:}$.
                # Each term $P_{:,j} Q_{j,:}$ is a rank-1 matrix.
                # But in the tensor decomposition, we need rank-1 tensors in A, B, C.
                
                # Let's look at one term: $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$.
                # Let $X = A_{11}+A_{22}$ and $Y = B_{11}+B_{22}$.
                # $M_1 = X Y$.
                # $X$ is a 2x2 matrix formed by linear combinations of A's blocks.
                # $Y$ is a 2x2 matrix formed by linear combinations of B's blocks.
                # $X Y$ is a 2x2 matrix.
                # We can write $X Y = \sum_{j=0}^1 X_{:,j} Y_{j,:}$.
                # $X_{:,j}$ is a column vector (2x1) of linear forms in A.
                # $Y_{j,:}$ is a row vector (1x2) of linear forms in B.
                # The product $X_{:,j} Y_{j,:}$ is a 2x2 matrix.
                # This matrix contributes to $C$.
                
                # So for each of the 7 Strassen products, we generate 2 terms (for j=0,1).
                # Total 14 terms?
                # No, $X_{:,j}$ is a vector of 2 linear forms.
                # $Y_{j,:}$ is a vector of 2 linear forms.
                # The product is a sum of 4 scalar products?
                # No, the tensor decomposition allows coefficients.
                # $U_r$ is a vector in the A-space.
                # $V_r$ is a vector in the B-space.
                # $W_r$ is a vector in the C-space.
                
                # If $X_{:,j} = [x_1, x_2]^T$ and $Y_{j,:} = [y_1, y_2]$,
                # then $X_{:,j} Y_{j,:} = \begin{pmatrix} x_1 y_1 & x_1 y_2 \\ x_2 y_1 & x_2 y_2 \end{pmatrix}$.
                # This is a sum of 4 rank-1 matrices in the C-space?
                # No, it's a single matrix.
                # But in the tensor decomposition, we can have a single term $r$ where:
                # $U_r$ encodes $x_1, x_2$?
                # $V_r$ encodes $y_1, y_2$?
                # $W_r$ encodes the mapping to C?
                
                # Actually, $x_1, x_2$ are linear forms in A.
                # So $U_r$ can be a vector that has non-zero entries for the A components involved in $x_1$ and $x_2$.
                # But $U_r$ is a single vector.
                # If $x_1 = A_{00}$ and $x_2 = A_{01}$, then $U_r$ has 1 at A00 and 1 at A01?
                # And $V_r$ has 1 at B00 and 1 at B01?
                # Then $U_r \otimes V_r$ creates a 4x4 matrix in the A,B space?
                # No, $U_r$ is length 16.
                # The product $U_r V_r$ is not defined as a matrix product in the decomposition.
                # The decomposition is $T[u,v,w] = \sum_r U[u,r] V[v,r] W[w,r]$.
                
                # So if we have a term corresponding to $x_1 y_1$ contributing to $C_{00}$,
                # we need $U_r$ to pick out $x_1$ from A, $V_r$ to pick out $y_1$ from B,
                # and $W_r$ to pick out $C_{00}$ from C.
                
                # So, for $M_1$, we have 4 scalar products contributing to the 4 elements of the 2x2 result.
                # Each scalar product is a rank-1 term.
                # So $M_1$ expands to 4 terms.
                # 7 meta-terms * 4 terms/meta = 28 terms?
                # But Rank(4) is 49.
                # Where do the other 21 come from?
                # Ah, the sub-problems are 2x2x2.
                # Rank(2) is 7? No, Rank(2) for 2x2x2 is 7 (Strassen).
                # But if we use schoolbook for the sub-problems, Rank(2)=4.
                # If we use Strassen for the sub-problems, Rank(2)=7.
                # So Rank(4) = 7 * 7 = 49.
                
                # So we must recursively use Strassen for the 2x2 blocks.
                
                # This is getting too complex to implement from scratch in this format.
                # I will fall back to the schoolbook rank 80 to ensure a valid submission.
                # 80 is a valid score (1.0 if seed is 80).
                
                return None # Signal to use schoolbook

        # If Strassen generation fails or is too complex, use schoolbook.
        terms = []
        
        # Schoolbook 4x4x5
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    u_vec = [0.0] * 16
                    v_vec = [0.0] * 20
                    w_vec = [0.0] * 20 # p*n = 5*4 = 20
                    
                    u_idx = i * 4 + j
                    v_idx = j * 5 + k
                    w_idx = k * 4 + i
                    
                    u_vec[u_idx] = 1.0
                    v_vec[v_idx] = 1.0
                    w_vec[w_idx] = 1.0
                    
                    terms.append((u_vec, v_vec, w_vec))
        
        # Convert to matrices
        rank = len(terms)
        U = np.zeros((u_dim, rank))
        V = np.zeros((v_dim, rank))
        W = np.zeros((w_dim, rank))
        
        for r, (u, v, w) in enumerate(terms):
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
            
        return {"U": U, "V": V, "W": W}

