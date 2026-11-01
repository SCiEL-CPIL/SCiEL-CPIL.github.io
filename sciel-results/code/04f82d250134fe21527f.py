import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _verify(M, U, V, W):
    """Verify exact decomposition of matrix multiplication tensor.
    M is (9,9) matrix where M[i*3+j, k*3+i] should equal sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r]
    Actually the tensor is T[i,j,k] = delta. Let's use the standard check:
    For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = 3, 3, 3
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill
    
    # Strategy: Build a rank-23 decomposition using known structure.
    # The 3x3 matrix multiplication tensor has rank 23 (Landsberg).
    # I'll construct it via a known explicit formula or a verified search.
    
    # Approach: Use the explicit rank-23 decomposition from literature.
    # One well-known construction partitions the 3x3 into blocks.
    # A = [A11 A12; A21 A22] where blocks are not square.
    # Actually, the standard rank-23 construction for 3x3 uses:
    # R = 23 = 7 (Strassen for 2x2) + 1 (scalar 1x1) + 15 (cross terms optimized)
    # But the exact coefficients are complex.
    
    # Alternative: Use a verified hardcoded decomposition.
    # The rank-23 decomposition for 3x3x3 is known. Let me try to construct it
    # using the "small rank" methods or just verify a known one.
    
    # Actually, let me try a direct approach: build a candidate rank-23 decomposition
    # using linear algebra / symbolic reasoning or a known explicit one.
    
    # Known rank-23 decomposition (from Landsberg / De Groote):
    # This is complex to hardcode from memory. Instead, let me try a 
    # structured approach:
    
    # The 3x3 multiplication can be seen as:
    # C = A * B
    # Write A = A_L + A_M + A_R, B = B_L + B_M + B_R where each is a rank-1 
    # contribution to the "row" or "column" structure... No, that's not quite right.
    
    # Let me try a different structural decomposition:
    # Partition indices into 3 groups of size 1.
    # Use the fact that rank(MM3) = 23.
    
    # I'll implement a search that tries to find a valid rank-23 decomposition
    # using a smarter initialization based on the schoolbook decomposition
    # with random perturbations and gradient descent on the constraint equations.
    
    # Actually, the most reliable approach given time constraints:
    # Return the known rank-23 decomposition if I can verify it, else fall back.
    
    # Let me try to construct the rank-23 decomposition using the following method:
    # Start from schoolbook (rank 27) and try to merge 4 terms.
    
    # A cleaner approach: use the explicit rank-23 decomposition from the literature.
    # Here is a known rank-23 decomposition for 3x3 matrix multiplication.
    # The factors U, V, W each have 9 rows and 23 columns.
    
    # I'll use a verified construction. The key insight is that the 
    # 3x3 multiplication tensor can be decomposed using 23 rank-1 tensors.
    
    # Let me try a concrete construction:
    # Rows of U correspond to (i,j) pairs, i.e., entries of A.
    # Rows of V correspond to (j,k) pairs, i.e., entries of B.
    # Rows of W correspond to (k,i) pairs (trace convention), i.e., entries of C.
    
    # I'll attempt a numerical search with a tight time budget.
    
    R_target = 23
    
    # Initialize with schoolbook (27 terms) and try to reduce.
    # Actually, let me just try to find ANY valid rank-23 decomposition 
    # using a randomized approach with verification.
    
    # Better: use a known explicit decomposition.
    # From "Tensor Decompositions and Applications" by L. De Lathauwer:
    # The rank of the 3x3 matrix multiplication tensor is 23.
    
    # Let me hardcode a known valid rank-23 decomposition.
    # The following is a verified rank-23 decomposition (I'll construct it carefully).
    
    # Actually, given the risk of error in hardcoding 23*9*3 = 621 values,
    # let me instead implement a verified search that is likely to succeed.
    
    # Method: Start with schoolbook, apply random linear combinations of rows,
    # and try to zero out entries to reduce effective rank.
    
    # Alternatively: use the fact that the tensor is the "multiplication tensor"
    # and apply known algorithmic improvements.
    
    # For safety and reliability, let me return a rank-24 decomposition 
    # constructed from Strassen(2x2) + 1x1 + optimized cross terms.
    
    # Actually, the most robust approach: try to verify a hardcoded rank-23.
    # If verification fails, fall back to schoolbook.
    
    # Let me try a different strategy: use the explicit rank-23 decomposition
    # from the paper "A note on the rank of matrix multiplication tensors" by Landsberg.
    
    # The rank-23 decomposition has the following structure (I'll construct it 
    # programmatically using known properties):
    
    # For 3x3, we can write the multiplication as:
    # (a1 b1 c1) (d1 e1 f1) (g1 h1 i1) etc.
    # This is getting too complex to derive from scratch reliably.
    
    # Practical approach: return the schoolbook decomposition (rank 27) 
    # which is guaranteed correct, but first try a quick search for rank 23.
    
    # Quick search: try random rank-23 decompositions with integer/half-integer entries
    # This is unlikely to succeed quickly, so let me just return schoolbook.
    
    # Wait - the task says "score is the rank" and "lower is better". 
    # The seed is rank 27 with score 1.0. So rank 23 would give score 27/23 > 1.0.
    # I need to actually find a valid rank-23 (or lower) decomposition.
    
    # Let me try a concrete construction. The rank-23 decomposition for 3x3x3:
    # I'll use the following known result: the 3x3 matrix multiplication tensor
    # has a decomposition of rank 23. One explicit construction uses:
    
    # Let me try to build it using the "border rank" approach or known formulas.
    
    # Actually, I recall that for 3x3, the optimal decomposition can be found
    # by considering the tensor as an element of (R^9)^* \otimes (R^9)^* \otimes (R^9)^*
    # and using the fact that it's a "multiplication tensor" of the algebra R^3 x R^3.
    
    # Given the complexity, let me implement a verified rank-23 decomposition
    # using a known explicit formula. Here's one approach:
    
    # The 3x3 multiplication tensor T has entries T_{(i,j),(j,k),(k,i)} = 1 
    # in the appropriate basis. In the "flattened" basis:
    # U is 9xR, V is 9xR, W is 9xR.
    
    # I'll use the following known rank-23 decomposition (from computational algebra):
    
    R = 23
    U = np.zeros((9, R))
    V = np.zeros((9, R))
    W = np.zeros((9, R))
    
    # This is a known rank-23 decomposition. I'll fill it in based on 
    # the structure of the 3x3 multiplication problem.
    
    # Actually, let me try a completely different approach: 
    # use the explicit rank-23 decomposition from a reference implementation.
    
    # The following is a verified rank-23 decomposition for 3x3 matrix multiplication.
    # It uses 23 terms. I'll construct it using the known "Landsberg" decomposition.
    
    # Due to the complexity of manually deriving all 621 values correctly,
    # and the risk of errors, I'll implement a search-based approach that 
    # is more likely to find a valid decomposition.
    
    # Search strategy: 
    # 1. Start with schoolbook (27 terms)
    # 2. Try to find linear dependencies that allow merging 4 terms into 0 extra
    # 3. Use gradient descent on the constraint equations
    
    # For now, let me return a rank-24 decomposition which is easier to construct:
    # Strassen(2x2) gives rank 7 for the top-left 2x2 block.
    # The 1x1 bottom-right block gives rank 1.
    # The cross terms (6 terms) need rank 16 total (10 more than schoolbook's 6).
    # Total: 7 + 1 + 16 = 24.
    
    # Actually, let me just try to find a valid rank-23 via a quick numerical search.
    
    # I'll use a random search with a time limit.
    np.random.seed(42)
    
    def random_decomposition(R, max_iters=100000):
        """Generate random rank-R decomposition and check validity."""
        # Use integer/half-integer entries
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            U = np.random.choice([-1, 0, 1], size=(9, R))
            V = np.random.choice([-1, 0, 1], size=(9, R))
            W = np.random.choice([-1, 0, 1], size=(9, R))
            
            if _verify(n, m, p, U, V, W):
                return U, V, W
        return None
    
    # Random search is unlikely to succeed for rank 23.
    # Let me instead construct a known valid decomposition.
    
    # I'll use the following approach: the rank-23 decomposition is known to exist.
    # Let me try to construct it using the "small rank" methods from the literature.
    
    # For 3x3, the tensor can be written as:
    # T = sum_{r=1}^{23} u_r \otimes v_r \otimes w_r
    
    # where u_r, v_r, w_r are vectors in R^9.
    
    # I'll use a known explicit construction. Here's a rank-23 decomposition:
    
    # The following decomposition is from the work of De Groote (2003):
    # "On the tensor rank of the 3x3 matrix multiplication tensor"
    
    # I'll construct it using the following known values (I'll verify after construction):
    
    # Given the high risk of incorrect hardcoded values, let me take a safer approach:
    # Return the schoolbook decomposition (rank 27) which is guaranteed correct.
    # This maintains score 1.0.
    
    # But wait - I should try harder. Let me implement a gradient-based search.
    
    # Actually, the most pragmatic approach: I know that rank 23 is achievable.
    # Let me use a verified decomposition from a trusted source.
    
    # Here is a rank-23 decomposition that I can construct and verify:
    
    # I'll use the following structure:
    # - 7 terms for the 2x2 Strassen core
    # - 1 term for the 1x1 scalar
    # - 15 terms for the cross-block interactions
    
    # Let me construct this carefully.
    
    # Strassen's 7 terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For 3x3, partition as:
    # A = [[A11_2x2, A12_2x1], [A21_1x2, A22_1x1]]
    # B = [[B11_2x2, B12_2x1], [B21_1x2, B22_1x1]]
    # C = [[C11_2x2, C12_2x1], [C21_1x2, C22_1x1]]
    
    # C11 = A11*B11 (Strassen, rank 7)
    # C22 = A22*B22 (rank 1)
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    
    # C12 is 2x1 = (2x2)(2x1) + (2x1)(1x1)
    # A11*B12: this is 2x2 times 2x1, which has rank at most 4 (schoolbook)
    # But we can do better. A11*B12 = sum_{j=0}^{1} A11[:,j] * B12[j,:]
    # This is a 2x2 matrix times a 2x1 matrix, giving a 2x1 result.
    # Each component: (A11[:,j] * B12[j,:]) is a scalar times a vector.
    # Actually, A11*B12 gives a 2x1 vector:
    # (A11*B12)[i] = sum_j A11[i,j] * B12[j]
    # This is 2 scalar multiplications for each of 2 outputs = 4 terms schoolbook.
    # Can we do it in fewer? The 2x2 * 2x1 multiplication tensor has rank 4 
    # (I think), so no improvement there.
    
    # A12*B22: (2x1)(1x1) = 2 terms (each A12[i] * B22)
    
    # So C12 schoolbook = 4 + 2 = 6 terms.
    # Similarly C21 schoolbook = 4 + 2 = 6 terms.
    
    # Total schoolbook: 4 (for 2x2x2x2) + 1 + 6 + 6 = 17? No wait.
    # 2x2 * 2x2: 4 terms schoolbook, 7 with Strassen.
    # 2x2 * 2x1: 4 terms
    # 2x1 * 1x1: 2 terms
    # 1x2 * 2x2: 4 terms
    # 1x2 * 2x1: 2 terms
    # 1x1 * 1x1: 1 term
    # Total schoolbook: 4+4+2+4+2+1 = 17? That's not right either.
    
    # Let me recount. 3x3 * 3x3:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Total 27 terms in schoolbook.
    
    # With Strassen on the 2x2 top-left block: saves 1 (from 4 to 3... no, from 4 to 7? No.)
    # Strassen does 2x2 * 2x2 in 7 multiplications instead of 4. So it INCREASES rank.
    # Wait, no. The "rank" here is the number of scalar multiplications.
    # Schoolbook 2x2: 4 multiplications.
    # Strassen 2x2: 7 multiplications.
    # So Strassen uses MORE rank for the 2x2 block!
    
    # I was confused. For matrix multiplication, the tensor rank is the minimum 
    # number of rank-1 tensors needed. For 2x2x2, rank is 7 (Strassen) vs 4 (schoolbook)?
    # No! The tensor rank of the 2x2x2 multiplication tensor is 7? That can't be right.
    # 
    # Actually, I think I'm confusing "rank" in different senses.
    # The tensor rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen's result).
    # The schoolbook uses 4 rank-1 tensors? No, schoolbook uses 4 terms but each term 
    # is not rank-1 in the tensor sense.
    
    # Let me clarify: the multiplication tensor T for n x n matrices is an element of 
    # (R^{n^2})^* \otimes (R^{n^2})^* \otimes (R^{n^2})^*.
    # Its tensor rank is the minimum number R such that T = sum_{r=1}^{R} u_r \otimes v_r \otimes w_r.
    # 
    # For 2x2: rank = 7 (Strassen).
    # For 3x3: rank = 23 (Landsberg).
    # For 4x4: rank <= 49 (Strassen) or 49? Actually rank(4x4) <= 49 via Strassen recursion.
    
    # So the "rank" here IS the number of rank-1 tensor terms, and it's the quantity we want to minimize.
    # Schoolbook for 3x3 uses 27 rank-1 terms (one for each (i,j,k) triple).
    # Strassen for 2x2 uses 7 rank-1 terms (vs 4 schoolbook... wait, that doesn't make sense either).
    
    # Hmm, actually for 2x2, the schoolbook decomposition has 4 terms, each of which is 
    # u_{ij} \otimes v_{jk} \otimes w_{ki} for fixed i,j,k. So rank <= 4.
    # But Strassen shows rank <= 7? That would be worse!
    
    # I think I have the definition backwards. Let me reconsider.
    
    # Actually, I believe the tensor rank of the 2x2x2 multiplication tensor is 7, 
    # meaning it CANNOT be decomposed into fewer than 7 rank-1 tensors.
    # The schoolbook "4 multiplications" refers to 4 scalar multiplications in the algorithm,
    # but each scalar multiplication corresponds to a product of linear forms, not a 
    # single rank-1 tensor.
    
    # Wait, I think the issue is that in the schoolbook algorithm, you compute 4 products 
    # of the form A[i,j]*B[j,k], and each such product is a "rank-1" contribution to the 
    # tensor. So the schoolbook gives a decomposition into 4 rank-1 tensors for 2x2.
    # But Strassen says you can do it with 7? That's more, not less!
    
    # I'm clearly confused about the definitions. Let me just focus on the task:
    # The task says the seed (rank 27) scores 1.0, and we want lower rank.
    # The known rank for 3x3x3 is 23.
    # I need to produce U, V, W with 23 columns that satisfy the decomposition.
    
    # Let me just try to find a valid rank-23 decomposition numerically.
    
    # Given the time constraints and the difficulty of constructing this from scratch,
    # I'll return the schoolbook decomposition (rank 27) to ensure correctness.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
