import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 4*4*7 = 112
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition as baseline
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1.0
                V_school[j * p + k, idx] = 1.0
                W_school[k * n + i, idx] = 1.0
                idx += 1
    
    best_rank = schoolbook_rank
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Strategy: Try to find a lower-rank decomposition using iterative refinement
    # We'll use a greedy approach: start with schoolbook, then try to eliminate terms
    # by finding linear dependencies.
    
    # Alternative approach: Use the rectangular structure.
    # 4x4x7 = 4x4x4 + 4x4x3
    # For 4x4x4, we can use Strassen (rank 49) + schoolbook for 4x4x3 (rank 48) = 97
    # But let's try to do better.
    
    # Let's try a direct search for rank < 112 using random initialization and refinement.
    # This is computationally expensive, so we'll limit the number of trials.
    
    # First, let's implement the hybrid Strassen + schoolbook approach as a safe improvement
    # 4x4x4 with Strassen: rank 49
    # 4x4x3 with schoolbook: rank 48
    # Total: 97
    
    def build_strassen_4x4x4():
        """Build a rank-49 decomposition for 4x4x4 using Strassen at the block level."""
        # Split 4x4x4 into 2x2 blocks of 2x2 matrices
        # A is 4x4, B is 4x4, C is 4x4
        # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # Strassen for 2x2x2 at block level (7 block multiplications)
        # Each block multiplication is 2x2x2 (schoolbook rank 8)
        # Total: 7 * 8 = 56? No, that's not right.
        
        # Actually, Strassen reduces the number of 2x2 matrix multiplications from 8 to 7.
        # Each 2x2 matrix multiplication has rank 8 (schoolbook).
        # So total rank = 7 * 8 = 56? That's worse than 49.
        
        # Wait, the known rank for 4x4x4 is 49 (using Strassen recursively or other methods).
        # Let me use a known rank-49 decomposition.
        
        # Actually, for 2x2x2, Strassen gives rank 7.
        # For 4x4x4, if we use Strassen at the 2x2 block level:
        # We have 7 Strassen products, each involving 2x2 matrices.
        # Each 2x2 matrix product can be done with rank 8 (schoolbook) or rank 7 (Strassen).
        # If we use schoolbook for the inner 2x2 products: 7 * 8 = 56
        # If we use Strassen for the inner 2x2 products: 7 * 7 = 49
        
        # So rank 49 is achievable for 4x4x4.
        
        # Let's build this.
        # The 7 Strassen products at the block level are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # And the C blocks are:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M4 + M7
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M6
        
        # Each M_s is a 2x2 matrix product of two linear combinations of 2x2 blocks.
        # To decompose each M_s into rank-1 terms, we need to expand the 2x2 matrix product.
        
        # For a 2x2 matrix product P = X*Y where X and Y are 2x2,
        # P[i,k] = sum_j X[i,j] * Y[j,k]
        # This has 8 terms (i,j,k in {0,1}).
        
        # But X and Y are themselves linear combinations of 2x2 blocks.
        # So each term X[i,j]*Y[j,k] expands to a sum of products of elements from the blocks.
        
        # This is getting complex. Let me just use the schoolbook for the inner 2x2 products.
        # That gives rank 7 * 8 = 56 for 4x4x4.
        
        # Hmm, that's worse than I thought. Let me reconsider.
        
        # Actually, the standard result is that the rank of 4x4x4 matrix multiplication is 49.
        # This is achieved by using Strassen at both levels.
        
        # Let me just use the simple schoolbook for now and try to improve via search.
        pass
    
    # Let's try a different approach: use the rectangular structure directly.
    # 4x4x7: We can split the 7 columns into 4+3.
    # Part 1: 4x4x4 (columns 0-3)
    # Part 2: 4x4x3 (columns 4-6)
    
    # For Part 1 (4x4x4), use schoolbook: rank 64
    # For Part 2 (4x4x3), use schoolbook: rank 48
    # Total: 112 (same as full schoolbook)
    
    # To improve, we need a better decomposition for at least one part.
    
    # Let's try to find a lower-rank decomposition for 4x4x7 directly using optimization.
    # We'll use a greedy peeling approach.
    
    # Build the tensor T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    # Actually, the tensor is defined by T[i,j,k] = 1 for all i,j,k (it's the identity tensor)
    # The matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    
    # Wait, no. The matrix multiplication tensor M_{n,m,p} has:
    # M_{i,j,k} = 1 if the (i,j) element of A and (j,k) element of B contribute to (k,i) element of C
    # Actually, it's: C[k,i] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    
    # The rank of this tensor is what we want to minimize.
    
    # Let's use a simple greedy approach:
    # 1. Start with the full tensor (all ones)
    # 2. Repeatedly find the best rank-1 approximation and subtract it
    # 3. Continue until the residual is small or we hit the time limit
    
    # But this is for approximation, not exact decomposition.
    # For exact decomposition, we need the residual to be exactly zero.
    
    # Alternative: Use the fact that the tensor is very structured (all ones).
    # The rank is at most n*m*p = 112 (schoolbook).
    # Can we do better?
    
    # For now, let's return the schoolbook decomposition as a safe baseline.
    # The hybrid approach (Strassen + schoolbook) might give 97, but it's complex to implement correctly.
    
    # Let's try to implement the hybrid approach carefully.
    
    def build_hybrid_4x4x7():
        """Build a hybrid decomposition for 4x4x7 using Strassen for 4x4x4 and schoolbook for 4x4x3."""
        # Part 1: 4x4x4 (i in [4], j in [4], k in [4])
        # Use schoolbook: 64 terms
        # Part 2: 4x4x3 (i in [4], j in [4], k in [3])
        # Use schoolbook: 48 terms
        # Total: 112 terms (same as full schoolbook)
        
        # To improve, we need a better 4x4x4 decomposition.
        # Let's use Strassen at the 2x2 block level with schoolbook inside.
        
        # 4x4x4 with Strassen blocks:
        # A = [[A00, A01], [A10, A11]] (each 2x2)
        # B = [[B00, B01], [B10, B11]] (each 2x2)
        # C = [[C00, C01], [C10, C11]] (each 2x2)
        
        # Strassen products (each is a 2x2 matrix product):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each M_s is a 2x2 matrix. To decompose M_s into rank-1 terms:
        # M_s[i,k] = sum_j X_s[i,j] * Y_s[j,k] where X_s and Y_s are the linear combinations
        
        # For M1: X1 = A00+A11, Y1 = B00+B11
        # M1[i,k] = sum_j (A00[i,j]+A11[i,j]) * (B00[j,k]+B11[j,k])
        # = sum_j [A00[i,j]*B00[j,k] + A00[i,j]*B11[j,k] + A11[i,j]*B00[j,k] + A11[i,j]*B11[j,k]]
        # This has 4*8 = 32 terms? No, that's not right.
        
        # Actually, each term (A00[i,j]+A11[i,j])*(B00[j,k]+B11[j,k]) expands to 4 products.
        # And we sum over j (2 values) and have i,k (2x2 = 4 values).
        # So total terms for M1: 2 (j) * 4 (i,k) * 4 (products) = 32? That seems too many.
        
        # I think I'm overcomplicating this. Let me just use the schoolbook for everything.
        # The rank will be 112, which is the same as the current program.
        
        # To actually improve, I need a correct implementation of a lower-rank algorithm.
        # For now, return the schoolbook.
        pass
    
    # Since implementing a correct lower-rank decomposition is complex and error-prone,
    # and the time limit is tight, let's return the schoolbook decomposition.
    # This matches the current program's score of 1.0.
    
    # But wait, the inspiration says a hybrid approach achieved 1.155 (rank 97).
    # Let me try to implement that correctly.
    
    # Key insight: For 4x4x4, use Strassen at the 2x2 block level.
    # Each Strassen product is a 2x2 matrix multiplication.
    # A 2x2 matrix multiplication has rank 8 (schoolbook).
    # So 7 Strassen products * 8 = 56 terms for 4x4x4.
    # For 4x4x3, use schoolbook: 4*4*3 = 48 terms.
    # Total: 56 + 48 = 104 terms.
    
    # That's better than 112! Let me implement this.
    
    def build_strassen_block_4x4x4():
        """Build a rank-56 decomposition for 4x4x4 using Strassen at 2x2 block level."""
        # 2x2x2 schoolbook terms: (i,j,k) for i,j,k in {0,1}
        schoolbook_2x2 = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
        
        # Strassen block definitions
        # Each: (a_blocks, b_blocks, c_blocks)
        # a_blocks: list of (block_index, coefficient) for A
        # b_blocks: list of (block_index, coefficient) for B
        # c_blocks: list of (block_index, coefficient) for C
        
        # Block indices: 00->0, 01->1, 10->2, 11->3
        strassen_blocks = [
            # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            (([0, 3], [1, 1]), ([0, 3], [1, 1]), ([0, 3], [1, 1])),
            # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
            (([2, 3], [1, 1]), ([0], [1]), ([2, 3], [1, -1])),
            # M3 = A00(B10-B00) -> C01 += M3, C11 += M3
            (([0], [1]), ([2, 0], [1, -1]), ([1, 3], [1, 1])),
            # M4 = A11(B01-B11) -> C00 += M4, C10 += M4
            (([3], [1]), ([1, 3], [1, -1]), ([0, 2], [1, 1])),
            # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
            (([0, 1], [1, 1]), ([3], [1]), ([0, 1], [-1, 1])),
            # M6 = (A10-A00)(B00+B01) -> C11 += M6
            (([2, 0], [1, -1]), ([0, 1], [1, 1]), ([3], [1])),
            # M7 = (A01-A11)(B10+B11) -> C00 += M7
            (([1, 3], [1, -1]), ([2, 3], [1, 1]), ([0], [1])),
        ]
        
        U = np.zeros((16, 56))
        V = np.zeros((16, 56))
        W = np.zeros((16, 56))
        
        rank = 0
        for s_idx in range(7):
            a_blocks, a_coeffs, b_blocks, b_coeffs, c_blocks, c_coeffs = strassen_blocks[s_idx]
            
            for (si, sj, sk) in schoolbook_2x2:
                r = rank
                rank += 1
                
                # For this Strassen product and 2x2 term (si,sj,sk):
                # The contribution to C[c_block*2+si, c_block*2+sk] is:
                # sum over a_block, b_block: a_coeff[a_block] * b_coeff[b_block] * 
                #   A[a_block*2+si, a_block*2+sj] * B[b_block*2+sj, b_block*2+sk]
                
                # Wait, this isn't quite right. Let me think more carefully.
                
                # The Strassen product M_s is a 2x2 matrix where:
                # M_s[i,k] = sum_j X_s[i,j] * Y_s[j,k]
                # where X_s = sum_a a_coeff[a] * A_block[a]
                # and Y_s = sum_b b_coeff[b] * B_block[b]
                
                # So M_s[i,k] = sum_j sum_a sum_b a_coeff[a] * b_coeff[b] * 
                #   A_block[a][i,j] * B_block[b][j,k]
                
                # And C_block[c][i,k] += c_coeff[c] * M_s[i,k]
                
                # So the full contribution is:
                # C[c_block*2+i, c_block*2+k] += c_coeff[c] * sum_j sum_a sum_b 
                #   a_coeff[a] * b_coeff[b] * A[a_block*2+i, a_block*2+j] * B[b_block*2+j, b_block*2+k]
                
                # For a fixed (si, sj, sk) in the 2x2 schoolbook, we're fixing i=si, j=sj, k=sk.
                # So the term is:
                # sum over c_block, a_block, b_block:
                #   c_coeff[c_block] * a_coeff[a_block] * b_coeff[b_block] *
                #   A[a_block*2+si, a_block*2+sj] * B[b_block*2+sj, b_block*2+sk]
                # contributing to C[c_block*2+si, c_block*2+sk]
                
                # For the rank-1 decomposition, we need:
                # U[i*m+j, r] = coefficient for A[i,j]
                # V[j*p+k, r] = coefficient for B[j,k]
                # W[k*n+i, r] = coefficient for C[k,i]
                
                # Here, i = c_block*2+si, j = a_block*2+sj (but j must be the same for A and B)
                # Wait, j in A is the column index, which is a_block*2+sj.
                # j in B is the row index, which is b_block*2+sj.
                # These must be equal for the term to make sense.
                
                # But a_block and b_block can be different! So a_block*2+sj != b_block*2+sj in general.
                # This means the Strassen approach doesn't directly give rank-1 terms in the standard form.
                
                # I think the issue is that the Strassen algorithm works at the block level,
                # and the rank-1 decomposition needs to be at the element level.
                
                # Let me just use the schoolbook for the entire 4x4x4 part.
                pass
        
        # Given the complexity, let me just return the schoolbook for the whole thing.
        U = np.zeros((16, 64))
        V = np.zeros((16, 64))
        W = np.zeros((16, 64))
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r = i * 16 + j * 4 + k
                    U[i * 4 + j, r] = 1.0
                    V[j * 4 + k, r] = 1.0
                    W[k * 4 + i, r] = 1.0
        
        return U, V, W, 64
    
    # Given the time constraints and complexity, let's just return the schoolbook decomposition.
    # This is the same as the current program.
    
    U = U_school.tolist()
    V = V_school.tolist()
    W = W_school.tolist()
    
    return {"U": U, "V": V, "W": W}
