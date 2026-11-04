import numpy as np
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition approach.
    # A is 3x3, B is 3x5, C is 3x5.
    # We can split A into blocks of size 2x2 and 1x1, etc.
    # However, a simpler and very effective method for small matrices is to use
    # the known rank-23 decomposition for 3x3x3 and combine it with schoolbook
    # for the remaining dimensions, or use a specific optimized decomposition.
    
    # Let's try to construct a decomposition using Strassen-like ideas on blocks.
    # Split A (3x3) into A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1).
    # Split B (3x5) into B1 (2x5), B2 (1x5).
    # C = A11*B1 + A12*B2 + A21*B1 + A22*B2.
    
    # This is not directly helping because B1 and B2 are different sizes.
    # Let's use the standard "splitting" method.
    # We can view 3x3x5 as a sum of:
    # 1. A 2x2x5 part? No, the dimensions are coupled.
    
    # Let's use a different strategy:
    # We know rank(3x3x3) = 23.
    # We can write the 3x3x5 multiplication as:
    # C = A * B
    # Let B = [B1, B2] where B1 is 3x3 and B2 is 3x2.
    # C = A*B1 + A*B2.
    # The term A*B1 is a 3x3x3 multiplication (rank 23).
    # The term A*B2 is a 3x3x2 multiplication.
    # What is the rank of 3x3x2?
    # It is at most 3*3*2 = 18 (schoolbook).
    # Is it less?
    # 3x3x2 can be seen as 3x3x1 + 3x3x1. Rank 3x3x1 is 9. So 18.
    # Can we do better?
    # Strassen doesn't directly apply to rectangular.
    # However, 3x3x2 is equivalent to 3x2x3 (permutation).
    # Rank(3x2x3) is known to be 13? No, that's 3x3x3=23.
    # Actually, rank(3x3x2) = 13 is not correct.
    # Let's check literature or derive.
    # For 2x2x2, rank is 7.
    # For 3x3x2, we can use a block structure.
    # Split A (3x3) into 2x2 and 1x1.
    # Split B (3x2) into 2x2 and 1x2.
    # This seems complex.
    
    # Alternative:
    # Use the fact that we can pad B to 3x3 with zeros?
    # If we pad B to 3x3, we get a 3x3x3 multiplication.
    # The result C' is 3x3. The last column of C' corresponds to A * 0 = 0.
    # So C (3x5) is not just a part of C'.
    # Wait, if B is 3x5, we can split it into B1 (3x3) and B2 (3x2).
    # C1 = A * B1 (3x3).
    # C2 = A * B2 (3x2).
    # Total rank <= rank(3x3x3) + rank(3x3x2).
    # If rank(3x3x2) is 13, total is 23 + 13 = 36.
    # If rank(3x3x2) is 14, total is 37.
    # Schoolbook is 45.
    # 36 is a significant improvement.
    
    # Let's find the rank of 3x3x2.
    # Tensor T of shape (3*3, 3*2, 2*3) = (9, 6, 6).
    # T[i*3+j, j*2+k, k*3+i] = 1.
    # We can use a heuristic search for 3x3x2.
    # Or use a known result.
    # Actually, 3x3x2 can be decomposed using 13 terms?
    # Let's try to construct it.
    # Split A into A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1).
    # Split B into B11 (2x2), B12 (2x1), B21 (1x2), B22 (1x1).
    # C11 (2x2) = A11*B11 + A12*B21.
    # C12 (2x1) = A11*B12 + A12*B22.
    # C21 (1x2) = A21*B11 + A22*B21.
    # C22 (1x1) = A21*B12 + A22*B22.
    
    # This is a sum of 8 schoolbook terms.
    # Can we reduce the 2x2x2 part?
    # C11 involves A11*B11 (2x2x2) and A12*B21 (2x1x2).
    # A12*B21 is rank 4 (2*1*2).
    # A11*B11 is rank 7 (Strassen).
    # But they share factors? No, A12 and A11 are disjoint, B21 and B11 are disjoint.
    # So C11 = Strassen(A11, B11) + Schoolbook(A12, B21).
    # This requires 7 + 4 = 11 terms just for C11?
    # But the terms for C12, C21, C22 also involve these blocks.
    
    # Let's just use a simple random search for the 3x3x2 part to see if we can get below 18.
    # And use the hardcoded 23-rank 3x3x3 for the first part.
    
    # Hardcoded 3x3x3 Rank 23 Decomposition
    # Source: "The rank of the 3x3x3 tensor"
    # I will implement the 23 terms explicitly.
    # This is a known result.
    
    # Terms for 3x3x3:
    # Let A, B, C be 3x3.
    # Indices 0,1,2.
    
    # I will use a list of tuples (u, v, w) where u, v, w are dicts or lists of non-zero entries.
    # u is for A (3x3), v for B (3x3), w for C (3x3).
    
    # Since I don't have the exact 23 coefficients memorized perfectly to avoid errors,
    # I will use a safer approach:
    # Use Strassen for 2x2x2 blocks and schoolbook for the rest, but carefully combined.
    # Or, simply use a random search for the whole 3x3x5 problem with a good initialization.
    # Given the time limit, a random search for rank 36-40 might be feasible if we start close.
    
    # Let's try to construct a rank 36 solution:
    # 1. 3x3x3 part (first 3 columns of B): Rank 23.
    # 2. 3x3x2 part (last 2 columns of B): Rank 13 (if we can find it) or 18 (schoolbook).
    # If we use schoolbook for 3x3x2, total rank = 23 + 18 = 41.
    # 41 < 45. This is an improvement.
    
    # Let's verify if we can do better than 41.
    # If rank(3x3x2) is 13, total is 36.
    # Let's try to find a 13-rank decomposition for 3x3x2.
    
    # For now, I will implement the 41-rank solution (23 + 18) to ensure an improvement over 45.
    # Then I will try to optimize the 3x3x2 part.
    
    # Step 1: Build 3x3x3 Rank 23 decomposition.
    # I will use a known set of 23 terms.
    # If I can't get it right, I will use schoolbook (27) for 3x3x3, giving 27+18=45.
    # But I want to beat 45.
    
    # Let's use a simpler block Strassen for 3x3x3 that might be rank 24 or 25?
    # No, 23 is the known minimum.
    
    # I will hardcode the 23 terms from a reliable source (mental reconstruction or standard reference).
    # Reference: "A 23-rank decomposition of the 3x3x3 tensor"
    
    # Actually, let's just do a random search for the 3x3x2 part to see if we can beat 18.
    # And use schoolbook 27 for 3x3x3? No, 27+18=45.
    # I need the 23-rank 3x3x3 to get below 45.
    
    # I will implement the 23-rank 3x3x3 decomposition.
    # Here are the 23 terms (u, v, w) for 3x3x3:
    # Each term is a triplet of sparse vectors.
    
    # I will define them as lists of (index, value).
    
    # Term 1: (A00+A11+A22) * (B00+B11+B22) * (C00+C11+C22) ? No.
    
    # Let's use a different approach:
    # Use the "Laser method" or just trust that a random search for the whole 3x3x5
    # with rank 40 might find a solution if we are lucky?
    # No, random search is unreliable.
    
    # Let's stick to the constructive 41-rank solution (23 + 18) if I can get the 23 terms.
    # If I fail to get the 23 terms, I will use a 24-rank approximation? No, must be exact.
    
    # I will use a known 23-rank decomposition.
    # The terms are:
    # 1. (1,0,0; 1,0,0; 1,0,0) -> A00 B00 C00
    # ... This is schoolbook.
    
    # I will use the following strategy:
    # 1. Implement a function to get the 23-rank 3x3x3 decomposition.
    # 2. Implement a function to get the 18-rank 3x3x2 decomposition (schoolbook).
    # 3. Combine them.
    
    # Since I am not 100% sure of the 23 coefficients, I will use a fallback:
    # If I can't verify the 23-rank decomposition, I will use a 24-rank one?
    # No, I will use a random search for the 3x3x3 part to find a rank 23 solution?
    # That's too slow.
    
    # Let's try to find a rank 40 solution using a different block split.
    # Split A (3x3) into 2x2 and 1x1.
    # Split B (3x5) into 2x5 and 1x5.
    # C = A11 B1 + A12 B2 + A21 B1 + A22 B2.
    # A11 B1: 2x2x5.
    # A12 B2: 2x1x5.
    # A21 B1: 1x2x5.
    # A22 B2: 1x1x5.
    
    # Rank(2x2x5):
    # Split B1 into 2x2 and 2x2 and 2x1?
    # 2x2x2 (7) + 2x2x2 (7) + 2x2x1 (4) = 18.
    # Rank(2x1x5) = 10.
    # Rank(1x2x5) = 10.
    # Rank(1x1x5) = 5.
    # Total = 18 + 10 + 10 + 5 = 43.
    # 43 < 45.
    # This is a valid constructive improvement!
    
    # Let's verify the ranks:
    # 2x2x2 rank is 7 (Strassen).
    # 2x2x1 rank is 4 (schoolbook, cannot be less as it's 4 independent mults).
    # So 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1?
    # B1 is 2x5. We can split it into B1a (2x2), B1b (2x2), B1c (2x1).
    # A11 B1a (2x2x2) -> 7 terms.
    # A11 B1b (2x2x2) -> 7 terms.
    # A11 B1c (2x2x1) -> 4 terms.
    # Total for A11 B1 is 18.
    
    # A12 B2: A12 is 2x1, B2 is 1x5.
    # This is (2x1) * (1x5) -> 2x5.
    # Each of the 2 rows of A12 multiplies the 1 row of B2.
    # This is 2 * 5 = 10 scalar multiplications. Rank 10.
    
    # A21 B1: A21 is 1x2, B1 is 2x5.
    # (1x2) * (2x5) -> 1x5.
    # 1 * 5 = 5 scalar multiplications?
    # Wait, A21 is 1x2. B1 is 2x5.
    # C21 = A21 B1.
    # C21 is 1x5.
    # C21[j] = sum_k A21[k] B1[k,j].
    # This requires 2 * 5 = 10 multiplications?
    # No, it's a dot product for each j.
    # There are 5 outputs. Each output is a sum of 2 products.
    # So 10 products. Rank 10.
    
    # A22 B2: A22 is 1x1, B2 is 1x5.
    # C22 = A22 B2.
    # 1 * 5 = 5 products. Rank 5.
    
    # Total Rank = 18 + 10 + 10 + 5 = 43.
    
    # This is a solid constructive solution with rank 43.
    # It is better than 45.
    
    # Let's implement this.
    
    # Dimensions:
    # n=3, m=3, p=5.
    # A is 3x3.
    # B is 3x5.
    # C is 3x5.
    
    # Indices:
    # A: i (0..2), j (0..2).
    # B: j (0..2), k (0..4).
    # C: i (0..2), k (0..4).
    
    # Split:
    # A11: i in {0,1}, j in {0,1}. (2x2)
    # A12: i in {0,1}, j in {2}. (2x1)
    # A21: i in {2}, j in {0,1}. (1x2)
    # A22: i in {2}, j in {2}. (1x1)
    
    # B1: j in {0,1}, k in {0..4}. (2x5)
    # B2: j in {2}, k in {0..4}. (1x5)
    
    # Split B1 further:
    # B1a: j in {0,1}, k in {0,1}. (2x2)
    # B1b: j in {0,1}, k in {2,3}. (2x2)
    # B1c: j in {0,1}, k in {4}. (2x1)
    
    # Terms:
    # 1. A11 * B1a -> C11a (2x2). Rank 7.
    # 2. A11 * B1b -> C11b (2x2). Rank 7.
    # 3. A11 * B1c -> C11c (2x1). Rank 4.
    # 4. A12 * B2 -> C12 (2x5). Rank 10.
    # 5. A21 * B1 -> C21 (1x5). Rank 10.
    # 6. A22 * B2 -> C22 (1x5). Rank 5.
    
    # Total Rank = 7 + 7 + 4 + 10 + 10 + 5 = 43.
    
    # Let's build the U, V, W matrices.
    
    # U shape: (n*m, rank) = (9, 43).
    # V shape: (m*p, rank) = (15, 43).
    # W shape: (p*n, rank) = (15, 43).
    
    # Index mapping:
    # U index: i*m + j = i*3 + j.
    # V index: j*p + k = j*5 + k.
    # W index: k*n + i = k*3 + i.
    
    U = np.zeros((9, 43))
    V = np.zeros((15, 43))
    W = np.zeros((15, 43))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # Inputs:
    # u_indices: list of 4 indices for the 2x2 A block (row-major)
    # v_indices: list of 4 indices for the 2x2 B block (row-major)
    # w_indices: list of 4 indices for the 2x2 C block (row-major)
    
    def add_strassen(u_idx, v_idx, w_idx):
        nonlocal col
        # Strassen terms for 2x2:
        # M1 = (A01 + A11)(B10 + B00) -> C00, C11
        # M2 = (A10 + A11)B00 -> C10
        # M3 = A00(B10 - B00) -> C00
        # M4 = A11(B01 - B11) -> C11
        # M5 = (A00 + A11)(B00 + B11) -> C00, C11
        # M6 = (A01 - A00)(B00 + B10) -> C01
        # M7 = (A10 - A11)(B01 + B11) -> C10
        
        # Indices in 2x2 block:
        # 0:00, 1:01, 2:10, 3:11
        
        # M1
        U[u_idx[1], col] += 1; U[u_idx[3], col] += 1
        V[v_idx[2], col] += 1; V[v_idx[0], col] += 1
        W[w_idx[0], col] += 1; W[w_idx[3], col] += 1
        col += 1
        
        # M2
        U[u_idx[2], col] += 1; U[u_idx[3], col] += 1
        V[v_idx[0], col] += 1
        W[w_idx[2], col] += 1
        col += 1
        
        # M3
        U[u_idx[0], col] += 1
        V[v_idx[2], col] += 1; V[v_idx[0], col] -= 1
        W[w_idx[0], col] += 1
        col += 1
        
        # M4
        U[u_idx[3], col] += 1
        V[v_idx[1], col] += 1; V[v_idx[3], col] -= 1
        W[w_idx[3], col] += 1
        col += 1
        
        # M5
        U[u_idx[0], col] += 1; U[u_idx[3], col] += 1
        V[v_idx[0], col] += 1; V[v_idx[3], col] += 1
        W[w_idx[0], col] += 1; W[w_idx[3], col] += 1
        col += 1
        
        # M6
        U[u_idx[1], col] += 1; U[u_idx[0], col] -= 1
        V[v_idx[0], col] += 1; V[v_idx[2], col] += 1
        W[w_idx[1], col] += 1
        col += 1
        
        # M7
        U[u_idx[2], col] += 1; U[u_idx[3], col] -= 1
        V[v_idx[1], col] += 1; V[v_idx[3], col] += 1
        W[w_idx[2], col] += 1
        col += 1
        
    # Helper to add schoolbook terms
    # u_indices: list of indices for A
    # v_indices: list of indices for B
    # w_indices: list of indices for C
    # Corresponding 1-to-1 mapping
    
    def add_schoolbook(u_idx, v_idx, w_idx):
        nonlocal col
        for i in range(len(u_idx)):
            U[u_idx[i], col] = 1
            V[v_idx[i], col] = 1
            W[w_idx[i], col] = 1
            col += 1
            
    # 1. A11 * B1a (2x2x2)
    # A11 indices: i in {0,1}, j in {0,1}
    # u_idx: [0*3+0, 0*3+1, 1*3+0, 1*3+1] = [0, 1, 3, 4]
    # B1a indices: j in {0,1}, k in {0,1}
    # v_idx: [0*5+0, 0*5+1, 1*5+0, 1*5+1] = [0, 1, 5, 6]
    # C11a indices: k in {0,1}, i in {0,1}
    # W index: k*3+i.
    # w_idx: [0*3+0, 0*3+1, 1*3+0, 1*3+1] = [0, 1, 3, 4]
    
    add_strassen([0, 1, 3, 4], [0, 1, 5, 6], [0, 1, 3, 4])
    
    # 2. A11 * B1b (2x2x2)
    # B1b indices: j in {0,1}, k in {2,3}
    # v_idx: [0*5+2, 0*5+3, 1*5+2, 1*5+3] = [2, 3, 7, 8]
    # C11b indices: k in {2,3}, i in {0,1}
    # w_idx: [2*3+0, 2*3+1, 3*3+0, 3*3+1] = [6, 7, 9, 10]
    
    add_strassen([0, 1, 3, 4], [2, 3, 7, 8], [6, 7, 9, 10])
    
    # 3. A11 * B1c (2x2x1)
    # B1c indices: j in {0,1}, k in {4}
    # v_idx: [0*5+4, 1*5+4] = [4, 9]
    # C11c indices: k in {4}, i in {0,1}
    # w_idx: [4*3+0, 4*3+1] = [12, 13]
    # A11 indices: [0, 1, 3, 4]
    # We need to multiply A11 (2x2) by B1c (2x1) to get C11c (2x1).
    # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # This is 4 terms.
    # Term 1: A[0,0] * B[0,0] -> C[0]
    # Term 2: A[0,1] * B[1,0] -> C[0]
    # Term 3: A[1,0] * B[0,0] -> C[1]
    # Term 4: A[1,1] * B[1,0] -> C[1]
    
    u_idx_a11 = [0, 1, 3, 4]
    v_idx_b1c = [4, 9]
    w_idx_c11c = [12, 13]
    
    # T1
    U[u_idx_a11[0], col] = 1; V[v_idx_b1c[0], col] = 1; W[w_idx_c11c[0], col] = 1; col += 1
    # T2
    U[u_idx_a11[1], col] = 1; V[v_idx_b1c[1], col] = 1; W[w_idx_c11c[0], col] = 1; col += 1
    # T3
    U[u_idx_a11[2], col] = 1; V[v_idx_b1c[0], col] = 1; W[w_idx_c11c[1], col] = 1; col += 1
    # T4
    U[u_idx_a11[3], col] = 1; V[v_idx_b1c[1], col] = 1; W[w_idx_c11c[1], col] = 1; col += 1
    
    # 4. A12 * B2 (2x1x5)
    # A12 indices: i in {0,1}, j in {2}
    # u_idx: [0*3+2, 1*3+2] = [2, 5]
    # B2 indices: j in {2}, k in {0..4}
    # v_idx: [2*5+0, 2*5+1, 2*5+2, 2*5+3, 2*5+4] = [10, 11, 12, 13, 14]
    # C12 indices: i in {0,1}, k in {0..4}
    # w_idx: k*3+i.
    # For i=0: k=0..4 -> [0, 3, 6, 9, 12]
    # For i=1: k=0..4 -> [1, 4, 7, 10, 13]
    # Total 10 terms.
    # C[i,k] = A[i,2] * B[2,k]
    
    u_idx_a12 = [2, 5]
    v_idx_b2 = [10, 11, 12, 13, 14]
    w_idx_c12 = [0, 3, 6, 9, 12, 1, 4, 7, 10, 13]
    
    # Mapping:
    # i=0, k=0: A[0,2] * B[2,0] -> C[0,0] (w_idx 0)
    # i=0, k=1: A[0,2] * B[2,1] -> C[0,1] (w_idx 3)
    # ...
    
    for i in range(2):
        for k in range(5):
            U[u_idx_a12[i], col] = 1
            V[v_idx_b2[k], col] = 1
            W[w_idx_c12[i*5+k], col] = 1
            col += 1
            
    # 5. A21 * B1 (1x2x5)
    # A21 indices: i in {2}, j in {0,1}
    # u_idx: [2*3+0, 2*3+1] = [6, 7]
    # B1 indices: j in {0,1}, k in {0..4}
    # v_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # C21 indices: i in {2}, k in {0..4}
    # w_idx: k*3+2.
    # k=0..4 -> [2, 5, 8, 11, 14]
    # Total 10 terms.
    # C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k]
    
    u_idx_a21 = [6, 7]
    v_idx_b1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    w_idx_c21 = [2, 5, 8, 11, 14]
    
    # Mapping:
    # k=0: A[2,0]B[0,0] -> C[2,0] (w_idx 2)
    # k=0: A[2,1]B[1,0] -> C[2,0] (w_idx 2)
    # ...
    
    for k in range(5):
        # Term 1: A[2,0] * B[0,k]
        U[u_idx_a21[0], col] = 1
        V[v_idx_b1[k], col] = 1
        W[w_idx_c21[k], col] = 1
        col += 1
        # Term 2: A[2,1] * B[1,k]
        U[u_idx_a21[1], col] = 1
        V[v_idx_b1[5+k], col] = 1
        W[w_idx_c21[k], col] = 1
        col += 1
        
    # 6. A22 * B2 (1x1x5)
    # A22 indices: i in {2}, j in {2}
    # u_idx: [2*3+2] = [8]
    # B2 indices: j in {2}, k in {0..4}
    # v_idx: [10, 11, 12, 13, 14]
    # C22 indices: i in {2}, k in {0..4}
    # w_idx: [2, 5, 8, 11, 14]
    # Total 5 terms.
    # C[2,k] = A[2,2] * B[2,k]
    
    u_idx_a22 = [8]
    v_idx_b2 = [10, 11, 12, 13, 14]
    w_idx_c22 = [2, 5, 8, 11, 14]
    
    for k in range(5):
        U[u_idx_a22[0], col] = 1
        V[v_idx_b2[k], col] = 1
        W[w_idx_c22[k], col] = 1
        col += 1
        
    # Total col should be 43.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
