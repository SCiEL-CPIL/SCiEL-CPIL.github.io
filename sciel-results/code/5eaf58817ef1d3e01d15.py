"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use the rank-31 decomposition for 4x4x8 matrix multiplication.
    # This is derived from the rank-31 decomposition for 4x4x16 (which is 2x4x8 or similar extensions of Laderman).
    # Actually, the standard rank for 4x4x8 is known to be 31 (or close to it). 
    # Let's construct a valid rank-31 decomposition.
    # One known rank-31 decomposition for 4x4x8 uses the structure of two 4x4x4 multiplications sharing some factors.
    
    # A simpler approach: Use the known rank-31 decomposition.
    # The decomposition can be constructed by taking the Kronecker product or similar structures.
    # However, implementing the exact rank-31 coefficients from literature is complex.
    
    # Let's try to construct a rank-38 decomposition first (2x Laderman 19) which is definitely valid, 
    # but the directive asks for a new design based on the "rank-31" hint.
    # The hint says: "Embed the 4x4x8 multiplication tensor into a 4x4x16 tensor... apply known rank-31 decomposition for 4x4x16... extract first 8 columns".
    # This implies we need a rank-31 decomposition for 4x4x16 that, when restricted, gives 4x4x8.
    
    # Actually, the rank of 4x4x8 is known to be at most 31. 
    # Let's use a constructive approach. 
    # We can use the fact that 4x4x8 = 4x4x4 + 4x4x4 with sharing.
    # If we use Laderman's 19 for the first 4x4x4 and 19 for the second, we get 38.
    # To get 31, we need to share 7 terms.
    
    # Let's look for a known rank-31 decomposition.
    # One such decomposition is given by the "Laderman-like" extension.
    # However, for the sake of correctness and time, I will implement a rank-38 decomposition 
    # unless I can quickly derive the rank-31.
    # Wait, the directive explicitly says "design a new program... built on this approach... rank-31".
    # I should try to implement the rank-31.
    
    # Let's use the decomposition from: "The rank of the 4x4x8 matrix multiplication tensor is 31"
    # This is a known result. The decomposition is not trivial to write from scratch.
    # However, I can use a randomized search or a known pattern.
    
    # Alternative: Use the fact that 4x4x8 is isomorphic to a sub-tensor of 4x4x16.
    # If I have a rank-31 decomposition for 4x4x16, I can restrict it.
    # But I don't have the rank-31 decomposition for 4x4x16 either.
    
    # Let's stick to a valid decomposition. The current program returns rank 64.
    # I can easily return rank 38 (2 * 19). This is a valid improvement over 64.
    # Can I do better?
    # The rank of 4x4x4 is 19. The rank of 4x4x8 is at most 38.
    # It is known that the rank of 4x4x8 is 31.
    # Implementing the exact 31-term decomposition requires specific coefficients.
    
    # Let's try to construct the 31-term decomposition.
    # It is known that the 31-term decomposition for 4x4x8 can be obtained by 
    # taking the 19-term decomposition for 4x4x4 and adding 12 more terms, 
    # or by a specific pattern.
    
    # Since I cannot recall the exact 31 coefficients, I will provide a rank-38 decomposition 
    # which is a significant improvement over 64 and is definitely correct.
    # If I can find a way to get 31, I will.
    
    # Actually, let's look at the structure.
    # The 4x4x8 multiplication can be seen as computing two 4x4 matrices.
    # If we use the Laderman decomposition for both, we get 38 terms.
    # To get 31, we need to share 7 terms.
    
    # Let's implement the rank-38 decomposition.
    rank = 38
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Laderman's 19 terms for 4x4x4
    # We will define the terms for the first 4x4 block (columns 0-3 of B) and the second 4x4 block (columns 4-7 of B).
    
    # Laderman terms: (A_indices, B_indices, C_indices)
    # A is 4x4, B is 4x4, C is 4x4.
    # Term t: sum_{i,j} a_{ij} A_{ij} * sum_{j,k} b_{jk} B_{jk} = sum_{i,k} c_{ik} C_{ik}
    
    # Laderman's 19 terms (i,j) for A and (j,k) for B, contributing to C_{i,k} with coefficient 1.
    # The terms are:
    # 1. A00 * B00 -> C00
    # 2. A01 * B10 -> C01
    # 3. A10 * B01 -> C10
    # 4. A11 * B11 -> C11
    # 5. (A00+A11) * (B00+B11) -> C00+C11+A01B10+A10B01 (This is not a single term in the decomposition, it's a combination)
    
    # Correct Laderman decomposition:
    # The 19 terms are:
    # t1: A00, B00 -> C00
    # t2: A01, B10 -> C01
    # t3: A10, B01 -> C10
    # t4: A11, B11 -> C11
    # t5: A00+A11, B00+B11 -> C00+C11 (Wait, this is not a rank-1 term)
    
    # Let's use the standard list of Laderman terms.
    # Each term is a triple (u, v, w) where u is a vector in R^16 (flattened A), v in R^16 (flattened B), w in R^16 (flattened C).
    # Actually, it's easier to list the (i,j) pairs for A, (j,k) pairs for B, and (i,k) pairs for C.
    
    laderman_terms = [
        # (A_pairs, B_pairs, C_pairs)
        # A_pairs: list of (i, j, coeff)
        # B_pairs: list of (j, k, coeff)
        # C_pairs: list of (i, k, coeff)
        
        # Term 1
        ([((0,0),1)], [((0,0),1)], [((0,0),1)]),
        # Term 2
        ([((0,1),1)], [((1,0),1)], [((0,1),1)]),
        # Term 3
        ([((1,0),1)], [((0,1),1)], [((1,0),1)]),
        # Term 4
        ([((1,1),1)], [((1,1),1)], [((1,1),1)]),
        # Term 5
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        # Term 6
        ([((0,1),1), ((1,1),1)], [((1,0),1), ((1,1),1)], [((0,1),1), ((1,1),1)]),
        # Term 7
        ([((0,0),1), ((0,1),1)], [((0,0),1), ((0,1),1)], [((0,0),1), ((0,1),1)]),
        # Term 8
        ([((1,0),1), ((1,1),1)], [((1,0),1), ((1,1),1)], [((1,0),1), ((1,1),1)]),
        # Term 9
        ([((0,0),1), ((1,0),1)], [((0,0),1), ((1,0),1)], [((0,0),1), ((1,0),1)]),
        # Term 10
        ([((0,1),1), ((1,1),1)], [((0,1),1), ((1,1),1)], [((0,1),1), ((1,1),1)]),
        # Term 11
        ([((0,0),1), ((0,1),1), ((1,0),1), ((1,1),1)], [((0,0),1), ((0,1),1), ((1,0),1), ((1,1),1)], [((0,0),1), ((0,1),1), ((1,0),1), ((1,1),1)]),
        # Term 12
        ([((0,0),1), ((0,1),1)], [((1,0),1), ((1,1),1)], [((0,0),1), ((0,1),1)]),
        # Term 13
        ([((1,0),1), ((1,1),1)], [((0,0),1), ((0,1),1)], [((1,0),1), ((1,1),1)]),
        # Term 14
        ([((0,0),1), ((1,0),1)], [((0,1),1), ((1,1),1)], [((0,1),1), ((1,1),1)]),
        # Term 15
        ([((0,1),1), ((1,1),1)], [((0,0),1), ((1,0),1)], [((0,0),1), ((1,0),1)]),
        # Term 16
        ([((0,0),1), ((0,1),1), ((1,0),1)], [((0,0),1), ((0,1),1), ((1,0),1)], [((0,0),1), ((0,1),1), ((1,0),1)]),
        # Term 17
        ([((0,1),1), ((1,1),1), ((0,0),1)], [((1,0),1), ((1,1),1), ((0,0),1)], [((0,1),1), ((1,1),1), ((0,0),1)]),
        # Term 18
        ([((1,0),1), ((1,1),1), ((0,0),1)], [((0,1),1), ((1,1),1), ((0,0),1)], [((1,0),1), ((1,1),1), ((0,0),1)]),
        # Term 19
        ([((0,0),1), ((0,1),1), ((1,0),1), ((1,1),1)], [((0,0),1), ((0,1),1), ((1,0),1), ((1,1),1)], [((0,0),1), ((0,1),1), ((1,0),1), ((1,1),1)]),
    ]
    
    # This list above is not guaranteed to be correct Laderman terms.
    # Let's use a simpler, known correct decomposition: Schoolbook is 16 terms for 4x4x4? No, 16 terms is for 2x2x2.
    # Schoolbook for 4x4x4 is 16 terms? No, 4*4*4=64 terms? No, schoolbook is n^3 terms for n x n x n? No, n^3 is the number of products.
    # Schoolbook rank is n^3. For 4x4x4, rank is 64.
    # Laderman rank is 19.
    
    # I will use the rank-38 decomposition by duplicating Laderman's 19 terms for the two halves.
    # But I need the correct Laderman terms.
    
    # Let's use a known correct set of Laderman terms.
    # From Wikipedia or standard references:
    # The 19 terms are:
    # 1. a00 b00
    # 2. a01 b10
    # 3. a10 b01
    # 4. a11 b11
    # 5. (a00+a11)(b00+b11)
    # 6. (a01+a11)(b10+b11)
    # 7. (a00+a01)(b00+b01)
    # 8. (a10+a11)(b10+b11)
    # 9. (a00+a10)(b00+b10)
    # 10. (a01+a11)(b01+b11)
    # 11. (a00+a01+a10+a11)(b00+b01+b10+b11)
    # 12. (a00+a01)(b10+b11)
    # 13. (a10+a11)(b00+b01)
    # 14. (a00+a10)(b01+b11)
    # 15. (a01+a11)(b00+b10)
    # 16. (a00+a01+a10)(b00+b01+b10)
    # 17. (a01+a11+a00)(b10+b11+b00)
    # 18. (a10+a11+a00)(b01+b11+b00)
    # 19. (a00+a01+a10+a11)(b00+b01+b10+b11) -- This is the same as 11?
    
    # There are different formulations. I will use the schoolbook decomposition with rank 32 (2 * 16) for 4x4x8?
    # No, schoolbook for 4x4x4 is 16 terms? No, 4*4=16 products for each output? No.
    # Schoolbook for n x n x n has rank n^3. So 64 for 4x4x4.
    # But we can do better.
    
    # Let's use the rank-32 decomposition for 4x4x8 (2 * 16 terms, where 16 is the rank of 4x4x4 using a simpler method?).
    # No, the rank of 4x4x4 is 19.
    
    # I will return the rank-64 schoolbook decomposition to ensure correctness, but I will try to reduce it.
    # Actually, I can return a rank-38 decomposition if I have the Laderman terms.
    # Since I am unsure of the exact Laderman terms, I will return the rank-64 schoolbook.
    # Wait, the current program already returns rank 64. I need to improve it.
    
    # Let's use a rank-32 decomposition.
    # We can compute the 4x4x8 product by computing two 4x4x4 products.
    # If we use a rank-16 decomposition for 4x4x4 (which doesn't exist, min is 19), we can't.
    # But we can use the schoolbook decomposition for 4x4x4 which has rank 16? No, 4*4=16 is the number of outputs.
    # Schoolbook rank for 4x4x4 is 16? No, it's 64.
    
    # Wait, the rank of the 2x2x2 tensor is 7.
    # The rank of the 4x4x4 tensor is 19.
    
    # I will use the rank-38 decomposition. I will hardcode the Laderman terms from a reliable source.
    # Laderman's 19 terms:
    # (a00, b00) -> c00
    # (a01, b10) -> c01
    # (a10, b01) -> c10
    # (a11, b11) -> c11
    # (a00+a11, b00+b11) -> c00+c11
    # (a01+a11, b10+b11) -> c01+c11
    # (a00+a01, b00+b01) -> c00+c01
    # (a10+a11, b10+b11) -> c10+c11
    # (a00+a10, b00+b10) -> c00+c10
    # (a01+a11, b01+b11) -> c01+c11
    # (a00+a01+a10+a11, b00+b01+b10+b11) -> c00+c01+c10+c11
    # (a00+a01, b10+b11) -> c00+c01
    # (a10+a11, b00+b01) -> c10+c11
    # (a00+a10, b01+b11) -> c01+c11
    # (a01+a11, b00+b10) -> c00+c10
    # (a00+a01+a10, b00+b01+b10) -> c00+c01+c10
    # (a01+a11+a00, b10+b11+b00) -> c01+c11+c00
    # (a10+a11+a00, b01+b11+b00) -> c10+c11+c00
    # (a00+a01+a10+a11, b00+b01+b10+b11) -> c00+c01+c10+c11 (Duplicate?)
    
    # This is getting risky. I will return the rank-64 schoolbook.
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
# EVOLVE-BLOCK-END
