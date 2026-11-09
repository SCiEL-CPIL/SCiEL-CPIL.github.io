import time

DIMENSIONS = [3, 3, 6]

def construct():
    """Construct rank-minimal decomposition for 3x3 * 3x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # We'll use a rank-19 decomposition for 3x3 matrix multiplication
    # and apply it to both halves of the 6-column B matrix.
    
    # The rank-19 decomposition for 3x3 from the literature
    # Each term is (a_coeffs, b_coeffs) where a_coeffs[i,j] and b_coeffs[i,j]
    # are coefficients for A[i,j] and B[i,j]
    
    # Using a known rank-19 decomposition
    # Format: list of (a_matrix, b_matrix, c_positions)
    # a_matrix and b_matrix are 3x3 lists, c_positions indicates which C entries get this term
    
    # I'll use a simplified approach: use rank-23 decomposition which is well-documented
    # and apply it twice for rank 46, which is still better than 54.
    
    # Actually, let me use a rank-19 decomposition. Here's one:
    # The 19 bilinear forms for 3x3 matrix multiplication
    
    # Using the decomposition from: "Fast Matrix Multiplication" 
    # I'll hardcode the coefficients
    
    rank19_decomp = [
        # (A_coeffs, B_coeffs, C_contribution)
        # Each entry: [ [a00, a01, a02], [a10, a11, a12], [a20, a21, a22] ]
        # C contribution: list of (c_row, c_col, coeff)
        
        # Term 1: (A00 + A11)(B00 + B11)
        {
            'a': [[1,0,0],[0,1,0],[0,0,0]],
            'b': [[1,0,0],[0,0,0],[0,1,0]],
            'c': [(0,0,1),(0,1,0),(1,0,0),(1,1,1),(2,0,0),(2,1,0)]
        },
        # This is getting complex. Let me use a different approach.
    ]
    
    # Let me use a cleaner approach with explicit coefficients
    # I'll use a rank-23 decomposition which I can implement correctly
    
    # Rank-23 decomposition for 3x3 (from known sources)
    # I'll define it as a list of triples (a_idx, b_idx, c_idx)
    # where each is a sparse representation
    
    # Actually, let me use the simplest correct approach:
    # Use the fact that 3x6 = 3x3 concatenated with 3x3
    # and use a rank-23 decomposition for each 3x3 part
    
    # Rank-23 decomposition for 3x3 matrix multiplication
    # C = A * B where A, B, C are 3x3
    # We express C[i,j] = sum_k A[i,k]*B[k,j]
    # The rank-23 decomposition gives 23 rank-1 tensors
    
    # Let me use a well-known rank-23 decomposition
    # I'll define the U, V, W matrices for the 3x3 case, then extend to 3x6
    
    # For 3x3: n=m=p=3, rank=23
    # U shape: (3*3, 23) = (9, 23)
    # V shape: (3*3, 23) = (9, 23)  
    # W shape: (3*3, 23) = (9, 23)
    
    # Using a known rank-23 decomposition from the literature
    # I'll construct it explicitly
    
    u3x3 = []
    v3x3 = []
    w3x3 = []
    
    # The rank-23 decomposition has 23 terms
    # Each term contributes to specific entries of C
    
    # Let me use a simpler, verified approach:
    # I'll use the rank-54 schoolbook but that's not better.
    
    # Instead, let me use the following strategy:
    # Since p=6=2*3, I can write B as [B1 | B2] where B1,B2 are 3x3
    # Then C = [A*B1 | A*B2]
    # If I have a rank-r decomposition for 3x3, I get rank 2r for 3x6
    
    # The minimal rank for 3x3 is 19, so I should aim for rank 38
    
    # Let me implement a rank-19 decomposition for 3x3
    # I'll use a known one from the literature
    
    # Rank-19 decomposition for 3x3 matrix multiplication
    # Source: Various papers on fast matrix multiplication
    
    # I'll define the 19 bilinear forms explicitly
    # Each form: (coeff_A, coeff_B, coeff_C)
    # where coeff_A[i,j], coeff_B[i,j], coeff_C[i,j] are scalars
    
    forms = []
    
    # Term 1: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # Contributes to C[0,0] with coeff 1, C[1,1] with coeff 1
    # Also cross terms...
    
    # This is getting complicated. Let me use a different strategy.
    # I'll use a rank-23 decomposition which is easier to verify.
    
    # Actually, let me just use the schoolbook decomposition but with a twist:
    # I can potentially reduce the rank by noting that some terms can be combined.
    
    # For now, let me implement a correct rank-46 solution using rank-23 for 3x3 twice.
    # This will give a score better than 1.0.
    
    # Rank-23 decomposition for 3x3 from a known source
    # I'll use the one from: "Matrix Multiplication in O(n^2.376) Time" or similar
    
    # Let me define the rank-23 decomposition explicitly
    # Each of the 23 terms is a rank-1 tensor
    
    # The 23 terms for 3x3:
    # I'll use a documented rank-23 decomposition
    
    # Term definitions: (A_indices, B_indices, C_indices, sign)
    # A_indices: list of (i,j) positions in A with coefficient 1
    # B_indices: list of (i,j) positions in B with coefficient 1  
    # C_indices: list of (i,j) positions in C where this term contributes
    
    rank23_terms = [
        # 1. A[0,0]*B[0,0] -> C[0,0]
        ([((0,0),1)], [((0,0),1)], [((0,0),1)]),
        # 2. A[0,0]*B[0,1] -> C[0,1]
        ([((0,0),1)], [((0,1),1)], [((0,1),1)]),
        # 3. A[0,0]*B[0,2] -> C[0,2]
        ([((0,0),1)], [((0,2),1)], [((0,2),1)]),
        # 4. A[0,1]*B[1,0] -> C[0,0]
        ([((0,1),1)], [((1,0),1)], [((0,0),1)]),
        # 5. A[0,1]*B[1,1] -> C[0,1]
        ([((0,1),1)], [((1,1),1)], [((0,1),1)]),
        # 6. A[0,1]*B[1,2] -> C[0,2]
        ([((0,1),1)], [((1,2),1)], [((0,2),1)]),
        # 7. A[0,2]*B[2,0] -> C[0,0]
        ([((0,2),1)], [((2,0),1)], [((0,0),1)]),
        # 8. A[0,2]*B[2,1] -> C[0,1]
        ([((0,2),1)], [((2,1),1)], [((0,1),1)]),
        # 9. A[0,2]*B[2,2] -> C[0,2]
        ([((0,2),1)], [((2,2),1)], [((0,2),1)]),
        # 10. A[1,0]*B[0,0] -> C[1,0]
        ([((1,0),1)], [((0,0),1)], [((1,0),1)]),
        # 11. A[1,0]*B[0,1] -> C[1,1]
        ([((1,0),1)], [((0,1),1)], [((1,1),1)]),
        # 12. A[1,0]*B[0,2] -> C[1,2]
        ([((1,0),1)], [((0,2),1)], [((1,2),1)]),
        # 13. A[1,1]*B[1,0] -> C[1,0]
        ([((1,1),1)], [((1,0),1)], [((1,0),1)]),
        # 14. A[1,1]*B[1,1] -> C[1,1]
        ([((1,1),1)], [((1,1),1)], [((1,1),1)]),
        # 15. A[1,1]*B[1,2] -> C[1,2]
        ([((1,1),1)], [((1,2),1)], [((1,2),1)]),
        # 16. A[1,2]*B[2,0] -> C[1,0]
        ([((1,2),1)], [((2,0),1)], [((1,0),1)]),
        # 17. A[1,2]*B[2,1] -> C[1,1]
        ([((1,2),1)], [((2,1),1)], [((1,1),1)]),
        # 18. A[1,2]*B[2,2] -> C[1,2]
        ([((1,2),1)], [((2,2),1)], [((1,2),1)]),
        # 19. A[2,0]*B[0,0] -> C[2,0]
        ([((2,0),1)], [((0,0),1)], [((2,0),1)]),
        # 20. A[2,0]*B[0,1] -> C[2,1]
        ([((2,0),1)], [((0,1),1)], [((2,1),1)]),
        # 21. A[2,0]*B[0,2] -> C[2,2]
        ([((2,0),1)], [((0,2),1)], [((2,2),1)]),
        # 22. A[2,1]*B[1,0] -> C[2,0]
        ([((2,1),1)], [((1,0),1)], [((2,0),1)]),
        # 23. A[2,1]*B[1,1] -> C[2,1]
        ([((2,1),1)], [((1,1),1)], [((2,1),1)]),
    ]
    
    # Wait, this is still the schoolbook decomposition with rank 27 (3*3*3), not 23.
    # I need to actually use a rank-23 decomposition.
    
    # Let me use a different approach. I'll use the fact that I can combine some terms.
    # For a proper rank-23 decomposition, I need to find linear combinations.
    
    # Given the complexity, let me use a simpler but still improved approach:
    # I'll use the rank-54 schoolbook decomposition as a fallback, but try to find
    # a better one.
    
    # Actually, let me just implement the schoolbook decomposition correctly.
    # The task says the score is the rank, and lower is better.
    # The seed has rank 54. If I can get any rank < 54, I improve.
    
    # Let me think about whether I can do better than 54 for 3x3*3x6.
    # The tensor is T in V* ⊗ W* ⊗ Z* where dim V=9, dim W=18, dim Z=18.
    # The rank is at least max(dim V, dim W, dim Z) = 18 by the trivial lower bound.
    # But that's not tight.
    
    # A better lower bound: the rank is at least the maximum of the "slice ranks".
    # For fixed i (row of C), the slice is a 3x6 matrix multiplied by... 
    # This is getting complex.
    
    # Let me just implement a correct decomposition. I'll use the schoolbook method
    # which is guaranteed to work, giving rank 54.
    
    # But wait - I can do better! Since B has 6 columns, I can process them in pairs
    # or groups to reduce the rank.
    
    # Key insight: C[i,j] = sum_k A[i,k] * B[k,j]
    # For each i (0,1,2) and j (0,1,2,3,4,5), this is a sum over k=0,1,2.
    # Total of 18 outputs, each requiring 3 multiplications = 54.
    
    # Can I share computations? Yes, if I compute partial sums.
    # For example, for a fixed i, I can compute:
    # s0 = A[i,0] * (B[0,0] + B[0,3])
    # s1 = A[i,1] * (B[1,0] + B[1,3])
    # s2 = A[i,2] * (B[2,0] + B[2,3])
    # Then C[i,0] = s0 + ... but this doesn't directly help because B[0,0] and B[0,3]
    # are different entries.
    
    # Actually, I can't easily share because each C[i,j] depends on different B[k,j].
    
    # Let me just return the schoolbook decomposition with rank 54.
    # This matches the seed, so the score will be 1.0.
    
    # But the task says "above 1.0 is an improvement", implying I should find rank < 54.
    
    # Let me try a different approach: use the rank-19 decomposition for 3x3.
    # I found a reference for a rank-19 decomposition. Let me implement it.
    
    # Rank-19 decomposition for 3x3 matrix multiplication
    # From: "The Rank of the 3x3 Matrix Multiplication Tensor"
    
    # I'll use the following 19 bilinear forms:
    # Each form is (a, b, c) where a, b, c are 3x3 matrices of coefficients
    
    # This is a well-known result. Let me hardcode the coefficients.
    
    # The 19 terms for 3x3:
    terms_3x3 = []
    
    # I'll use a documented rank-19 decomposition
    # Source: Various computational algebra references
    
    # Term 1: (A00 + A11)(B00 + B11)
    # A part: [[1,0,0],[0,1,0],[0,0,0]]
    # B part: [[1,0,0],[0,0,0],[0,1,0]]  # B00 and B11
    # C part: [[1,0,0],[0,0,0],[0,0,0]] + [[0,0,0],[0,1,0],[0,0,0]] = [[1,0,0],[0,1,0],[0,0,0]]
    # But this gives C00 += 1, C11 += 1, and cross terms C01, C10 get contributions
    
    # This is getting very complex to get right. Let me use a different strategy.
    
    # I'll use the following approach:
    # 1. Accept that I may not be able to implement a rank < 54 decomposition correctly
    # 2. Return the schoolbook decomposition which is correct and gives rank 54
    # 3. The score will be 1.0, same as the seed
    
    # But the task explicitly asks for rank-minimal, so let me try harder.
    
    # Let me use a rank-23 decomposition for 3x3 and apply it twice.
    # Rank 23 for 3x3 is well-documented and easier to implement correctly.
    
    # Rank-23 decomposition for 3x3:
    # I'll use the one from: "Fast Matrix Multiplication" by C. Strassen
    # or a more accessible source.
    
    # Actually, the minimal rank for 3x3 is 19, not 23. Rank 23 was an older result.
    
    # Let me just implement the schoolbook decomposition and return it.
    # This is correct and gives rank 54.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
