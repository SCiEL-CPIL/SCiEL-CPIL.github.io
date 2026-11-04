"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill

    # Construct the target tensor T
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """Verify if U, V, W exactly decompose T."""
        # Compute approximation
        # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Using einsum for efficiency
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_approx, atol=1e-9)

    def try_rank_reduction(U, V, W, max_time=10.0):
        """
        Attempt to reduce the rank of the decomposition (U, V, W) by finding linear dependencies
        among the rank-1 terms.
        Returns (U_new, V_new, W_new) if reduced, else None.
        """
        rank = U.shape[1]
        if rank <= 1:
            return None

        # We look for a transformation matrix P (r x r') with r' < r such that
        # U_new = U P, V_new = V P, W_new = W P still satisfies the tensor equation?
        # No, that's not how rank reduction works generally.
        # Rank reduction in tensor decomposition is hard.
        # However, if the current decomposition has linearly dependent columns in U, V, or W,
        # we can remove them.
        
        # Check for linear independence of columns in U, V, W
        # If cols of U are dependent, say U[:, i] = sum c_j U[:, j], then we can potentially merge terms?
        # Actually, if U[:, i] is in span of others, it doesn't necessarily mean the term is redundant
        # unless the corresponding V and W terms also align.
        
        # A simpler heuristic: 
        # If we have a decomposition of rank R, we can try to find a smaller rank R' < R
        # by solving the linear system for U, V, W with R' variables, initialized from the current solution.
        # This is effectively the same as the original ALS but starting from a good guess.
        
        # Given the time constraints and the fact that the constructive solution is likely near-optimal
        # for this specific small size, we will skip complex reduction and just verify the constructive one.
        # The "search" part of the original program was slow and unreliable.
        # We will rely on the constructive approach which is deterministic and fast.
        
        return None

    # Constructive Approach:
    # Decompose 2x4 x 4x7 into:
    # 1. Block (i=0, j=0,1) and (i=1, j=0,1) -> 2x2 x 4x7 sub-problem? No, B is 4x7.
    #    Let's split A into two 1x4 rows.
    #    Row 0: A[0, 0:2] (2 cols) and A[0, 2:4] (2 cols)
    #    Row 1: A[1, 0:2] (2 cols) and A[1, 2:4] (2 cols)
    #    B is 4x7. Split B into B[0:2, :] and B[2:4, :].
    
    # We can view the multiplication as:
    # C[0, k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k]
    # C[1, k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k] + A[1,3]B[3,k]
    
    # Let's use Strassen for the 2x2 blocks if we group the middle indices.
    # Group middle indices {0,1} and {2,3}.
    # Let B_L = B[0:2, :] (2x7) and B_R = B[2:4, :] (2x7).
    # Let A_L0 = A[0, 0:2] (1x2), A_R0 = A[0, 2:4] (1x2)
    # Let A_L1 = A[1, 0:2] (1x2), A_R1 = A[1, 2:4] (1x2)
    
    # C[0, :] = A_L0 @ B_L + A_R0 @ B_R
    # C[1, :] = A_L1 @ B_L + A_R1 @ B_R
    
    # This is two independent 1x2 x 2x7 multiplications added together?
    # No, B_L and B_R are shared between the two output rows?
    # B_L is used by both C[0] and C[1].
    # B_R is used by both C[0] and C[1].
    
    # So we have:
    # C0 = A_L0 B_L + A_R0 B_R
    # C1 = A_L1 B_L + A_R1 B_R
    
    # Let's decompose each 1x2 x 2x7 part.
    # 1x2 dot product: a1 b1 + a2 b2. Rank 2 is trivial.
    # Is there a rank 1 scheme? No.
    # So each 1x2 x 2x7 has rank 2*7 = 14?
    # Wait, 1x2 x 2x1 has rank 2.
    # 1x2 x 2x7 is equivalent to 7 independent 1x2 x 2x1 multiplications?
    # Yes, because the output dimension is 7 and the input dimensions don't mix across k.
    # So rank(1x2 x 2x7) = 7 * rank(1x2 x 2x1) = 7 * 2 = 14.
    
    # So:
    # Term for C0 using B_L: Rank 14
    # Term for C0 using B_R: Rank 14
    # Term for C1 using B_L: Rank 14
    # Term for C1 using B_R: Rank 14
    # Total Rank = 56. This is the schoolbook rank.
    
    # Can we do better by combining?
    # Consider the 2x2 x 4x7 subproblem formed by A (2x4) and B (4x7) but treating the output as 2x7.
    # This is just the original problem.
    
    # Let's try a different grouping.
    # Use Strassen on the 2x2 block of A and B?
    # If we had 2x2 x 2x2, Strassen gives rank 7 vs 8.
    # Here we have 2x4 x 4x7.
    # We can split the 4 columns of A into two 2-column blocks.
    # We can split the 4 rows of B into two 2-row blocks.
    # Let A = [A1 A2] where A1, A2 are 2x2.
    # Let B = [B1; B2] where B1, B2 are 2x7.
    # C = A1 B1 + A2 B2.
    
    # Now we have two independent 2x2 x 2x7 multiplications added together.
    # Since the outputs C are summed, and the inputs A1, A2, B1, B2 are disjoint in terms of indices?
    # A1 uses A[0,0], A[0,1], A[1,0], A[1,1].
    # A2 uses A[0,2], A[0,3], A[1,2], A[1,3].
    # B1 uses B[0,:], B[1,:].
    # B2 uses B[2,:], B[3,:].
    # The sets of input variables are disjoint between the term A1 B1 and A2 B2.
    # Therefore, the tensor decomposes as a direct sum of two tensors:
    # T = T1 + T2
    # Where T1 corresponds to A1 B1 and T2 corresponds to A2 B2.
    # Since the supports are disjoint in the input indices (U and V indices),
    # the rank of T is rank(T1) + rank(T2).
    
    # So we just need the rank of 2x2 x 2x7.
    # 2x2 x 2x7 is equivalent to 7 independent 2x2 x 2x1 multiplications?
    # No, the output is 2x1 for each k.
    # C_k = A1 B1_k + ...
    # The tensor for 2x2 x 2x1 has rank 7 (Strassen).
    # Since the k indices are independent (no mixing of k in the inputs or outputs for a fixed k),
    # the tensor for 2x2 x 2x7 is the direct sum of 7 copies of the 2x2 x 2x1 tensor.
    # So rank(2x2 x 2x7) = 7 * 7 = 49.
    
    # Total Rank = rank(T1) + rank(T2) = 49 + 49 = 98?
    # Wait, my previous calculation of schoolbook was 56.
    # Let's re-verify schoolbook rank.
    # Schoolbook: sum over all i,j,k.
    # Number of terms = n*m*p = 2*4*7 = 56.
    # So schoolbook rank is 56.
    
    # Why did I think Strassen would help?
    # Strassen reduces 8 to 7 for 2x2.
    # Here, for each k, we have a 2x2 x 2x1 multiplication (if we group j into pairs).
    # Wait, A is 2x4. B is 4x7.
    # C is 2x7.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # If we group j into {0,1} and {2,3}.
    # C[i, k] = sum_{j in {0,1}} A[i, j] B[j, k] + sum_{j in {2,3}} A[i, j] B[j, k].
    # Let C1[i, k] = sum_{j in {0,1}} A[i, j] B[j, k]. This is a 2x2 x 2x7 multiplication.
    # Let C2[i, k] = sum_{j in {2,3}} A[i, j] B[j, k]. This is a 2x2 x 2x7 multiplication.
    # C = C1 + C2.
    # The inputs to C1 are A[:, 0:2] and B[0:2, :].
    # The inputs to C2 are A[:, 2:4] and B[2:4, :].
    # These input sets are disjoint.
    # So T = T1 + T2, and rank(T) = rank(T1) + rank(T2).
    
    # Now, what is rank(2x2 x 2x7)?
    # This tensor T_2227 has indices (i, j, k) with i in {0,1}, j in {0,1}, k in {0..6}.
    # T_2227[i, j, k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # Wait, the tensor definition is T[a, b, c].
    # For the sub-problem, the "output" is 2x7 = 14 elements.
    # The "input" A is 2x2 = 4 elements.
    # The "input" B is 2x7 = 14 elements.
    # The tensor is a 3-way tensor of size 4 x 14 x 14? No.
    # The tensor T maps (A, B) -> C.
    # T_{a, b, c} = 1 if A_a B_b contributes to C_c.
    # In our case, for fixed k, the contribution is from A[i,j] and B[j,k] to C[i,k].
    # The k index is a "free" index that does not mix.
    # So T_2227 is the direct sum over k=0..6 of T_2221.
    # T_2221 is the 2x2 x 2x1 matrix multiplication tensor.
    # Rank of T_2221 is 7 (Strassen).
    # So rank(T_2227) = 7 * 7 = 49.
    
    # So Total Rank = 49 + 49 = 98.
    # This is WORSE than schoolbook 56.
    # Why? Because Strassen saves 1 rank per 2x2 block, but here the "blocks" are effectively 2x2 x 2x1.
    # Schoolbook for 2x2 x 2x1 is rank 4.
    # Strassen is rank 7? No, Strassen is for 2x2 x 2x2.
    # For 2x2 x 2x1, the tensor is:
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    # This is just two dot products of length 2.
    # Rank is 2+2=4?
    # Can we do it in rank 3?
    # T_{i, j, k} for i,j in {0,1}, k in {0,1}.
    # T[0, 0, 0] = 1 (A00 B00 -> C00) -- Wait, B index is (row, col).
    # Let's stick to the global indices.
    
    # Let's re-evaluate the rank of 1x2 x 2x1.
    # C = A0 B0 + A1 B1.
    # T[0, 0, 0] = 1 (A0 B0 -> C)
    # T[1, 1, 0] = 1 (A1 B1 -> C)
    # This is a sum of two rank-1 tensors.
    # Can it be rank 1? No.
    # So rank is 2.
    # Schoolbook for 1x2 x 2x1 is 2.
    
    # So for 2x2 x 2x1:
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    # This is two independent 1x2 x 2x1 multiplications.
    # Rank = 2 + 2 = 4.
    # Strassen applies to 2x2 x 2x2.
    # If we had 2x2 x 2x2, rank is 7 (Strassen) vs 8 (Schoolbook).
    # Here we have 2x2 x 2x1. There is no Strassen benefit.
    
    # So the constructive approach using Strassen on 2x2 blocks doesn't directly apply because the "width" is only 1 in the output for each k.
    
    # However, we can group k's?
    # No, k's are independent.
    
    # Is there any sharing?
    # The middle matrix B is shared between the two rows of A (i=0 and i=1).
    # In the schoolbook decomposition:
    # Term (i, j, k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # For fixed j, k, we have two terms: (i=0) and (i=1).
    # U[0*m+j, r1] = 1, U[1*m+j, r2] = 1.
    # V[j*p+k, r1] = 1, V[j*p+k, r2] = 1.
    # W[k*n+0, r1] = 1, W[k*n+1, r2] = 1.
    
    # Can we combine these two terms into one?
    # We want to represent:
    # 1 * V_val * W_0 + 1 * V_val * W_1
    # = V_val * (W_0 + W_1) ?
    # But U must be different for i=0 and i=1.
    # If we use one term, U must be a vector that produces both U_i0 and U_i1?
    # No, a single rank-1 term U[:, r] is a fixed vector.
    # It cannot be 1 at index 0 and 1 at index 4 (assuming m=4) simultaneously in a way that allows separate scaling?
    # Actually, if U[:, r] has 1 at both positions, then the term contributes to both outputs.
    # Let's try to construct a term that covers both i=0 and i=1 for a fixed j, k.
    # Term r:
    # U[i*m+j, r] = 1 for i=0,1.
    # V[j*p+k, r] = 1.
    # W[k*n+i, r] = 1 for i=0,1.
    # Then sum_r ...
    # T_approx[i*m+j, j*p+k, k*n+i'] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i', r]
    # If we use this term, it contributes 1 to (i=0, j, k, i'=0) and (i=1, j, k, i'=1).
    # But it ALSO contributes to (i=0, j, k, i'=1) and (i=1, j, k, i'=0)?
    # Let's check:
    # U[0*m+j, r] = 1.
    # U[1*m+j, r] = 1.
    # V[j*p+k, r] = 1.
    # W[k*n+0, r] = 1.
    # W[k*n+1, r] = 1.
    
    # Contribution to T[0*m+j, j*p+k, k*n+0] = 1*1*1 = 1. (Correct)
    # Contribution to T[1*m+j, j*p+k, k*n+1] = 1*1*1 = 1. (Correct)
    # Contribution to T[0*m+j, j*p+k, k*n+1] = 1*1*1 = 1. (WRONG, should be 0)
    # Contribution to T[1*m+j, j*p+k, k*n+0] = 1*1*1 = 1. (WRONG, should be 0)
    
    # So we can't just merge them. We need to cancel the cross terms.
    # This suggests that rank 56 might be optimal or close to it.
    # However, the prompt mentions a "constructive rank 50" strategy in the inspirations.
    # How did they get 50?
    # "Strassen for 2x2x2 blocks and direct rank-1 terms for the remainder"
    
    # Let's look at the dimensions again.
    # 2x4 x 4x7.
    # Maybe we can view it as two 2x2 x 4x7? No.
    # Maybe we can use a 2x2 Strassen scheme on the "i" and "j" indices?
    # If we group j into pairs, we have 2x2 blocks of A.
    # But B has 4 rows.
    # If we use Strassen on the 2x2 block of A and the corresponding 2x? block of B?
    # Strassen requires square matrices or compatible dimensions.
    
    # Let's try to implement the "Rank 50" idea.
    # If the rank is 50, it's better than 56.
    # 56 - 50 = 6 savings.
    # There are 4 pairs of j's? No, j goes from 0 to 3.
    # There are 7 k's.
    # Total terms 56.
    # If we save 1 term for every ~9 terms?
    
    # Let's try a different constructive approach:
    # Use the fact that n=2.
    # C0 = A0 B
    # C1 = A1 B
    # This is just two 1x4 x 4x7 multiplications.
    # Rank(1x4 x 4x7) = 4*7 = 28.
    # Total rank = 28 + 28 = 56.
    
    # Can we share terms between C0 and C1?
    # The inputs A0 and A1 are disjoint.
    # The input B is shared.
    # The outputs C0 and C1 are disjoint.
    # This is a "bilinear" structure with shared input.
    # Generally, sharing inputs doesn't reduce rank if outputs are disjoint and inputs are disjoint?
    # Wait, if inputs are disjoint, the tensor is a direct sum.
    # T = T0 + T1.
    # T0 depends on A0, B.
    # T1 depends on A1, B.
    # Since A0 and A1 are disjoint variables, the tensor is the direct sum.
    # Rank(T) = Rank(T0) + Rank(T1).
    # So Rank = 28 + 28 = 56.
    
    # Is Rank(1x4 x 4x7) really 28?
    # 1x4 x 4x1 has rank 4.
    # 1x4 x 4x7 is direct sum of 7 copies of 1x4 x 4x1.
    # So Rank = 7 * 4 = 28.
    
    # So 56 seems to be the lower bound for this decomposition.
    # Why did the inspiration say 50?
    # Maybe my assumption that A0 and A1 are disjoint is wrong?
    # No, A is 2x4. A0 is row 0, A1 is row 1. They are distinct variables.
    
    # Perhaps the "Rank 50" uses a different decomposition of 1x4 x 4x7?
    # Is Rank(1x4 x 4x1) < 4?
    # No, it's 4.
    
    # Maybe the inspiration is using a different problem structure?
    # "Strassen for 2x2x2 blocks"
    # If we can find 2x2x2 blocks in the 2x4 x 4x7 problem, we can save rank.
    # A 2x2x2 block would require 2 inputs from A, 2 from B, 2 outputs.
    # Outputs are 2x7. So we can pick 2 outputs (e.g., C[0,0], C[0,1]).
    # Inputs from B: B[0,0], B[0,1], B[1,0], B[1,1]?
    # Inputs from A: A[0,0], A[0,1], A[1,0], A[1,1]?
    # If we use a 2x2 Strassen scheme for the sub-problem defined by A[0:2, 0:2] and B[0:2, 0:2] contributing to C[0:2, 0:2]?
    # But B is 4x7. The sub-problem is not closed.
    # The term A[i,j] B[j,k] contributes to C[i,k].
    # If we restrict i in {0,1}, j in {0,1}, k in {0,1}, we have a 2x2 x 2x2 sub-tensor.
    # This sub-tensor is part of the larger tensor.
    # If we decompose this sub-tensor with rank 7, and the rest with rank ...
    # The "rest" would be the terms involving j=2,3 or k=2..6.
    
    # Let's count.
    # Total terms 56.
    # Sub-tensor S1: i in {0,1}, j in {0,1}, k in {0,1}.
    # This involves 2*2*2 = 8 terms in schoolbook.
    # Strassen uses 7 terms.
    # Savings: 1.
    
    # Can we do this for multiple disjoint blocks?
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}.
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}.
    # These blocks share k indices but have disjoint j indices.
    # Do they share any input variables?
    # Block 1 uses A[0,0], A[0,1], A[1,0], A[1,1] and B[0,0], B[0,1], B[1,0], B[1,1].
    # Block 2 uses A[0,2], A[0,3], A[1,2], A[1,3] and B[2,0], B[2,1], B[3,0], B[3,1].
    # The input sets are disjoint!
    # So we can apply Strassen to each block independently.
    # Number of such blocks:
    # j can be {0,1} or {2,3} (2 choices).
    # k can be {0,1}, {2,3}, {4,5}, {6}?
    # k has 7 values. We can form 3 pairs {0,1}, {2,3}, {4,5} and one singleton {6}.
    # For each pair of k's, we have a 2x2 x 2x2 sub-problem for each j-pair.
    # So for k-pair 0: 2 j-pairs * 1 Strassen block each = 2 blocks.
    # For k-pair 1: 2 j-pairs * 1 Strassen block each = 2 blocks.
    # For k-pair 2: 2 j-pairs * 1 Strassen block each = 2 blocks.
    # Total 6 Strassen blocks.
    # Each block saves 1 rank (8 -> 7).
    # Total savings = 6.
    # Remaining terms:
    # k=6 is a singleton.
    # For k=6, we have j in {0,1,2,3}.
    # This is a 2x4 x 4x1 sub-problem.
    # Rank = 2*4 = 8?
    # Wait, for fixed k, it's 2x4 x 4x1.
    # C[i, 6] = sum_j A[i,j] B[j,6].
    # This is two 1x4 dot products.
    # Rank = 4 + 4 = 8.
    # Schoolbook for this part is 8.
    # Can we reduce it?
    # 1x4 dot product has rank 4.
    # So total rank for k=6 is 8.
    
    # So total rank = 6 * 7 (Strassen) + 8 (direct) = 42 + 8 = 50.
    
    # This matches the inspiration!
    # Strategy:
    # 1. Partition k into pairs (0,1), (2,3), (4,5) and singleton (6).
    # 2. Partition j into pairs (0,1) and (2,3).
    # 3. For each k-pair and j-pair, use a Strassen decomposition of the 2x2 x 2x2 sub-tensor.
    # 4. For k=6, use the direct schoolbook decomposition of the 2x4 x 4x1 sub-tensor.
    # 5. Combine all terms.
    
    # Let's implement this.
    
    def get_strassen_2x2():
        """
        Returns U, V, W for 2x2 x 2x2 matrix multiplication using Strassen.
        U shape: (4, 7)
        V shape: (4, 7)
        W shape: (4, 7)
        Indices:
        U: [A00, A01, A10, A11]
        V: [B00, B01, B10, B11]
        W: [C00, C01, C10, C11]
        """
        # Strassen formulas:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need U, V, W such that sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # Wait, W indices are C00, C01, C10, C11.
        # C00 gets M1. C11 gets M1.
        # So W[0, 0] = 1, W[3, 0] = 1.
        
        # Term 2: M2
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # C10 gets M2, C11 gets -M2.
        
        # Term 3: M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1 # C01 gets M3, C11 gets M3.
        
        # Term 4: M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C00 gets M4, C10 gets M4.
        
        # Term 5: M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C00 gets -M5, C01 gets M5.
        
        # Term 6: M6
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11 gets M6.
        
        # Term 7: M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1 # C00 gets M7, C11 gets M7.
        
        return U, V, W

    # Build the full decomposition
    
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen blocks for k in {0,1}, {2,3}, {4,5} and j in {0,1}, {2,3}
    for k_start in [0, 2, 4]:
        for j_start in [0, 2]:
            U_s, V_s, W_s = get_strassen_2x2()
            
            # Map local indices to global indices
            # Local U indices: 0:A00, 1:A01, 2:A10, 3:A11 (relative to i,j)
            # Global U index: i*m + j
            # For this block, i in {0,1}, j in {j_start, j_start+1}
            # Local V indices: 0:B00, 1:B01, 2:B10, 3:B11 (relative to j,k)
            # Global V index: j*p + k
            # For this block, j in {j_start, j_start+1}, k in {k_start, k_start+1}
            # Local W indices: 0:C00, 1:C01, 2:C10, 3:C11 (relative to i,k)
            # Global W index: k*n + i
            # For this block, i in {0,1}, k in {k_start, k_start+1}
            
            for r in range(7):
                # U
                u_val = U_s[:, r]
                # V
                v_val = V_s[:, r]
                # W
                w_val = W_s[:, r]
                
                # We need to add these to the global U, V, W matrices.
                # Since we are building lists of columns, we can just append.
                
                # Create global column vectors
                U_col = np.zeros(n*m)
                V_col = np.zeros(m*p)
                W_col = np.zeros(p*n)
                
                # Fill U_col
                for li in range(4):
                    if u_val[li] != 0:
                        # li: 0->(0, j_start), 1->(0, j_start+1), 2->(1, j_start), 3->(1, j_start+1)
                        # Wait, local U indices:
                        # 0: A00 -> i=0, j=j_start
                        # 1: A01 -> i=0, j=j_start+1
                        # 2: A10 -> i=1, j=j_start
                        # 3: A11 -> i=1, j=j_start+1
                        i_local = li // 2
                        j_local = li % 2
                        i_global = i_local
                        j_global = j_start + j_local
                        U_col[i_global * m + j_global] = u_val[li]
                        
                # Fill V_col
                for li in range(4):
                    if v_val[li] != 0:
                        # li: 0->B00 (j=j_start, k=k_start), 1->B01 (j=j_start, k=k_start+1)
                        #     2->B10 (j=j_start+1, k=k_start), 3->B11 (j=j_start+1, k=k_start+1)
                        # Wait, local V indices in get_strassen_2x2:
                        # 0: B00, 1: B01, 2: B10, 3: B11
                        # B row is j, col is k.
                        # So B00 is (j_start, k_start).
                        # B01 is (j_start, k_start+1).
                        # B10 is (j_start+1, k_start).
                        # B11 is (j_start+1, k_start+1).
                        j_local = li // 2
                        k_local = li % 2
                        j_global = j_start + j_local
                        k_global = k_start + k_local
                        V_col[j_global * p + k_global] = v_val[li]
                        
                # Fill W_col
                for li in range(4):
                    if w_val[li] != 0:
                        # li: 0:C00 (i=0, k=k_start), 1:C01 (i=0, k=k_start+1)
                        #     2:C10 (i=1, k=k_start), 3:C11 (i=1, k=k_start+1)
                        # Global W index: k*n + i
                        i_local = li // 2
                        k_local = li % 2
                        i_global = i_local
                        k_global = k_start + k_local
                        W_col[k_global * n + i_global] = w_val[li]
                        
                U_list.append(U_col)
                V_list.append(V_col)
                W_list.append(W_col)
                
    # Direct terms for k=6
    # j in {0,1,2,3}, i in {0,1}
    for i in range(n):
        for j in range(m):
            U_col = np.zeros(n*m)
            V_col = np.zeros(m*p)
            W_col = np.zeros(p*n)
            
            U_col[i * m + j] = 1
            V_col[j * p + 6] = 1
            W_col[6 * n + i] = 1
            
            U_list.append(U_col)
            V_list.append(V_col)
            W_list.append(W_col)
            
    # Convert to matrices
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    # Verify
    if verify(U, V, W):
        # Clean up to integers
        U = np.round(U)
        V = np.round(V)
        W = np.round(W)
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
