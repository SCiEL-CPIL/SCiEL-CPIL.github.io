import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: Use Strassen for the 4x4x4 part + schoolbook for the last column
    # 4x4x4 via Strassen: partition into 2x2 blocks, each 2x2x2 uses 7 terms
    # 4 blocks * 7 = 28 terms for 4x4x4
    # Plus 4x4x1 schoolbook: 4*4*1 = 16 terms
    # Total: 28 + 16 = 44 terms
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (16, 44)
    V = [[0] * rank for _ in range(m * p)]  # (20, 44)
    W = [[0] * rank for _ in range(p * n)]  # (20, 44)
    
    col = 0
    
    # Strassen 2x2x2: 7 rank-1 terms
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_2x2(r0, c0, k0):
        """Add 7 Strassen terms for 2x2x2 block.
        A block: rows r0,r0+1; cols c0,c0+1
        B block: rows c0,c0+1; cols k0,k0+1
        C block: rows k0,k0+1; cols r0,r0+1
        """
        nonlocal col
        # M1 = (A[r0,c0] + A[r0+1,c0+1]) * (B[c0,k0] + B[c0+1,k0+1])
        # Expand: 4 rank-1 terms
        terms = [
            # (i,j,k, coeff_in_A, coeff_in_B, coeff_in_C)
            # M1 expansion
            (r0, c0, k0, 1, 1, 1),    # A00*B00 -> C00
            (r0, c0, k0+1, 1, 1, 1),  # A00*B11 -> C01... wait, need to be careful
        ]
        # Let me think more carefully. Each Strassen Mi is a product of two linear forms.
        # To express as rank-1 tensors, I need to expand each product.
        # But that gives more than 7 terms. The Strassen algorithm reduces the number
        # of multiplications, but in tensor decomposition terms, each "multiplication"
        # of linear forms still needs to be expanded.
        
        # Actually, the tensor rank of 2x2x2 matrix multiplication is 7 (Strassen).
        # The 7 rank-1 tensors are:
        # T1: A00*B00 -> contributes to C with specific pattern
        # But Strassen's 7 terms are not simply rank-1; they are products of sums.
        
        # Let me look up the exact 7 rank-1 tensors for 2x2x2 matmul.
        # The 7 terms of Strassen correspond to 7 rank-1 decompositions.
        
        # Actually, I recall that the standard way is:
        # Each of the 7 Mi terms is itself a "tensor" that can be written as a sum
        # of rank-1 tensors. But that defeats the purpose.
        
        # Let me reconsider. The tensor for 2x2x2 matrix multiplication has rank 7.
        # The 7 rank-1 tensors are:
        # T1 = e00 ⊗ e00 ⊗ e00 + e00 ⊗ e01 ⊗ e01 + e10 ⊗ e00 ⊗ e10 + e11 ⊗ e01 ⊗ e11
        # ... this is getting complicated.
        
        # Let me just use the known result: 2x2x2 has rank 7, and the 7 terms
        # can be represented as follows (using the standard Strassen decomposition):
        
        # I'll define the 7 rank-1 tensors explicitly.
        # For 2x2x2: A(i,j), B(j,k), C(i,k) where i,j,k in {0,1}
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # This contributes: C00 += M1, C01 += M1, C10 += M1, C11 -= M1... no
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mi is a product of a linear form in A and a linear form in B.
        # To write as rank-1 tensors, I need to expand each product into individual
        # (i,j,k) contributions.
        
        # Let me just do it systematically. For each Mi, I'll add the rank-1 terms.
        
        # M1 = (A00 + A11) * (B00 + B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Contributions to C: C00 += M1, C01 += M1, C10 += M1, C11 -= M1 (from C11 = M1 - M2 + M3 + M6)
        # Wait, that's not right either. Let me re-derive.
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1 appears in: C00 (+1), C11 (+1)
        # M2 appears in: C10 (+1), C11 (-1)
        # M3 appears in: C01 (+1), C11 (+1)
        # M4 appears in: C00 (-1), C10 (+1)
        # M5 appears in: C00 (+1), C01 (+1)
        # M6 appears in: C11 (+1)
        # M7 appears in: C00 (-1)
        
        # Now expand each Mi:
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M2 = (A10+A11)B00 = A10B00 + A11B00
        # M3 = A00(B01-B11) = A00B01 - A00B11
        # M4 = A11(B10-B00) = A11B10 - A11B00
        # M5 = (A00+A01)B11 = A00B11 + A01B11
        # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
        # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
        
        # This is getting very complex. Let me just use schoolbook for the 4x4x4 part
        # and see if I can find a better approach.
        
        # Actually, let me just implement schoolbook for now and return rank 80.
        # The key issue is that implementing Strassen correctly in tensor decomposition
        # form is non-trivial.
        
        pass
    
    # Implement Strassen for 4x4x4 part + schoolbook for 4x4x1 part
    # Total rank: 4*7 + 4*4*1 = 28 + 16 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (16, 44)
    V = [[0] * rank for _ in range(m * p)]  # (20, 44)
    W = [[0] * rank for _ in range(p * n)]  # (20, 44)
    
    col = 0
    
    # The 7 rank-1 tensors for 2x2x2 matrix multiplication
    # Each tensor is defined by (a_indices, b_indices, c_indices)
    # where a_indices maps to A[i,j], b_indices to B[j,k], c_indices to C[i,k]
    # Format: list of (i, j, k, coeff) for each rank-1 term
    
    # Strassen's 7 terms for 2x2x2:
    # We need to express the 7 Mi products as sums of rank-1 tensors
    # and account for how they contribute to C.
    
    # Let me define the 7 rank-1 tensors directly.
    # After careful derivation, the 7 rank-1 tensors for 2x2x2 matmul are:
    
    # T1: contributes to C00, C01, C10, C11
    # The standard 7 tensors are:
    
    strassen_tensors = [
        # Tensor 1: (A00+A11)(B00+B11)
        # Expansion and C contributions:
        # C00 += M1, C11 += M1
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # So: (0,0,0,+1), (0,1,0,+1), (1,0,0,+1), (1,1,0,+1) for C00
        #     (0,0,1,+1), (0,1,1,+1), (1,0,1,+1), (1,1,1,+1) for C11
        # But each rank-1 tensor has a fixed (i,j,k) pattern.
        
        # I think I need to be more careful. Let me just use the known fact
        # that rank(2x2x2) = 7 and implement it correctly.
    ]
    
    # Use Strassen for 4x4x4 + schoolbook for 4x4x1
    # 4x4x4: partition into four 2x2x2 blocks, each with rank 7 -> 28 terms
    # 4x4x1: schoolbook -> 16 terms
    # Total: 44 terms
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 7 rank-1 tensors for 2x2x2 matmul
    # For local indices (i,j,k) in {0,1}^3:
    # The 7 tensors are (using Strassen's algorithm):
    
    # After careful derivation, here are the 7 rank-1 tensors:
    # Each is a list of (i, j, k, coeff) tuples
    
    # T1: from M1 = (A00+A11)(B00+B11), contributes +1 to C00 and C11
    # T2: from M2 = (A10+A11)B00, contributes +1 to C10 and -1 to C11
    # T3: from M3 = A00(B01-B11), contributes +1 to C01 and C11
    # T4: from M4 = A11(B10-B00), contributes -1 to C00 and +1 to C10
    # T5: from M5 = (A00+A01)B11, contributes +1 to C00 and C01
    # T6: from M6 = (A10-A00)(B00+B01), contributes +1 to C11
    # T7: from M7 = (A01-A11)(B10+B11), contributes -1 to C00
    
    # Expanding each Mi into rank-1 terms and combining:
    
    # I'll directly specify the 7 rank-1 tensors for 2x2x2:
    # Based on Strassen's original paper and standard references:
    
    strassen_7 = [
        # (list of (i,j,k,coeff) for this rank-1 tensor)
        # Tensor 1
        [(0,0,0,1), (0,1,0,1), (1,0,0,1), (1,1,0,1),
         (0,0,1,1), (0,1,1,1), (1,0,1,1), (1,1,1,1)],
        # This doesn't look right. A rank-1 tensor should have a single (i,j,k) pattern.
        
        # Let me restart. A rank-1 tensor in the 2x2x2 matmul is of the form:
        # (sum of a_i * e_i) ⊗ (sum of b_j * e_j) ⊗ (sum of c_k * e_k)
        # which expands to many (i,j,k) terms.
        
        # The 7 rank-1 tensors for Strassen are:
    ]
    
    # Proper Strassen implementation for 4x4x4 + schoolbook for 4x4x1
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # The 7 rank-1 tensors for 2x2x2 matmul, specified as (u, v, w) vectors
    # where u is in R^4 (for A), v is in R^4 (for B), w is in R^4 (for C)
    # A is indexed as (i,j) -> i*2+j, B as (j,k) -> j*2+k, C as (i,k) -> i*2+k
    
    # Strassen's 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00  
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a product of two linear forms. To express as rank-1 tensors,
    # I need to expand each product. But that gives more than 7 terms.
    
    # The resolution: Strassen's algorithm reduces the number of scalar multiplications,
    # but in the tensor decomposition sense, each "multiplication" of linear forms
    # still corresponds to multiple rank-1 tensors when expanded.
    
    # However, the tensor rank of 2x2x2 matmul IS 7, meaning there exist 7 rank-1
    # tensors that sum to the matmul tensor. These are not simply the expanded Mi.
    
    # Let me just use the known 7 rank-1 tensors from literature:
    
    # After careful study, the 7 rank-1 tensors for 2x2x2 matmul are:
    # T1 = (e00+e11) ⊗ (e00+e11) ⊗ (e00+e11)  -- NO, this is not a rank-1 tensor
    
    # A rank-1 tensor is u ⊗ v ⊗ w where u, v, w are vectors.
    # The 7 such tensors for Strassen are:
    
    # I give up on deriving this from scratch. Let me just use schoolbook.
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
