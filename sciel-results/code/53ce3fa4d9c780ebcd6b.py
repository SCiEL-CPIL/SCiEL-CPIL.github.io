import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 3, 6
    
    # We construct a Rank 22 decomposition for M_{2,3,6}.
    # Strategy:
    # M_{2,3,6} can be viewed as two M_{2,3,3} multiplications in parallel.
    # C = A * B, where A is 2x3, B is 3x6.
    # Split B into B1 (3x3) and B2 (3x3).
    # C = A*B1 + A*B2.
    # Let C1 = A*B1 (2x3) and C2 = A*B2 (2x3).
    # C = [C1 C2].
    # If we have a rank R decomposition for M_{2,3,3}, we can use it for C1 and C2.
    # Since the output indices for C1 and C2 are disjoint (columns 0-2 and 3-5),
    # the rank of the combined decomposition is R + R = 2R.
    
    # We need a rank 11 decomposition for M_{2,3,3}.
    # Is Rank 11 possible for 2x3x3?
    # Schoolbook is 18.
    # Strassen for 2x2 is 7.
    # 2x3x3 can be embedded in 3x3x3?
    # Actually, there is a known result that R(2,3,3) <= 11 is NOT standard.
    # Let's check known bounds.
    # R(2,3,3) is known to be at most 14? Or 13?
    # Wait, the "inspiration" mentioned in the prompt context (from a previous successful run) 
    # stated: "Uses two copies of a Rank 11 decomposition for 2x3x3 to construct a Rank 22 decomposition for 2x3x6".
    # This implies that a Rank 11 decomposition for 2x3x3 EXISTS or was found in that context.
    # However, I must be careful. If I don't have the exact coefficients for Rank 11, I cannot use it.
    # 
    # Let's look at the current program's fallback. It uses a Rank 17 decomposition for 2x3x3?
    # The current program's `get_rank34_solution` uses `U233_seed` of shape (6, 17).
    # So it uses Rank 17 for 2x3x3.
    # 17 * 2 = 34.
    # The score is 36/34 = 1.0588.
    #
    # If I can find a Rank 14 decomposition for 2x3x3, I get Rank 28 for 2x3x6.
    # Score: 36/28 = 1.2857.
    # If I can find a Rank 12 decomposition for 2x3x3, I get Rank 24 for 2x3x6.
    # Score: 36/24 = 1.5.
    # If I can find a Rank 11 decomposition for 2x3x3, I get Rank 22 for 2x3x6.
    # Score: 36/22 = 1.636.
    
    # Let's try to construct a lower rank decomposition for 2x3x3.
    # Or, better yet, let's try to directly construct a lower rank decomposition for 2x3x6 
    # that is NOT simply 2 * R(2,3,3).
    # 
    # Alternative Strategy: **Strassen-like for 2x3x6 directly**.
    # Or **Kronecker Product**:
    # M_{2,3,6} = M_{2,3,2} \otimes M_{1,1,3}? No.
    # M_{2,3,6} = M_{2,1,2} \otimes M_{1,3,3}? No.
    # 
    # Let's analyze M_{2,3,6} structure.
    # A (2x3), B (3x6).
    # We can view this as M_{2,3,2} with a 3x3 block in the output?
    # No.
    #
    # Let's try to find a Rank 24 solution using M_{2,3,2} with Rank 8.
    # If R(2,3,2) = 8, then R(2,3,6) <= 8 * 3 = 24.
    # Is R(2,3,2) = 8?
    # M_{2,3,2}: A(2x3), B(3x2).
    # Schoolbook: 12.
    # Can we do better?
    # A = [A1 A2] (2x2, 2x1).
    # B = [B1; B2] (2x2, 1x2).
    # C = A1 B1 + A2 B2.
    # This is a sum of two products.
    # A1 B1 is 2x2x2 (Rank 7 via Strassen).
    # A2 B2 is 2x1x2 (Rank 2? Schoolbook 2).
    # Total Rank <= 7 + 2 = 9.
    # Can we do 8?
    # Maybe.
    # If R(2,3,2) = 9, then R(2,3,6) <= 27. Score 36/27 = 1.33.
    # If R(2,3,2) = 8, then R(2,3,6) <= 24. Score 36/24 = 1.5.
    
    # Let's try to construct a Rank 9 decomposition for M_{2,3,2}.
    # Then duplicate it 3 times for M_{2,3,6}.
    
    # M_{2,3,2} Decomposition:
    # Indices:
    # U: (2*3, R) = (6, R)
    # V: (3*2, R) = (6, R)
    # W: (2*2, R) = (4, R)
    # 
    # Let's use Strassen for the 2x2 part (A1 B1) and simple multiplication for the 2x1 * 1x2 part (A2 B2).
    # A1 = A[:, 0:2], A2 = A[:, 2].
    # B1 = B[0:2, :], B2 = B[2, :].
    # 
    # Strassen for A1 B1 (2x2 * 2x2):
    # M1 = (A1[0,0]+A1[1,1]) * (B1[0,0]+B1[1,1])
    # M2 = (A1[1,0]+A1[1,1]) * B1[0,0]
    # M3 = A1[0,0] * (B1[0,1]-B1[1,1])
    # M4 = A1[1,1] * (B1[1,0]-B1[0,0])
    # M5 = (A1[0,0]+A1[0,1]) * B1[1,1]
    # M6 = (A1[1,0]-A1[0,0]) * (B1[0,0]+B1[0,1])
    # M7 = (A1[0,1]-A1[1,1]) * (B1[1,0]+B1[1,1])
    # 
    # C1 = A1 B1 (2x2).
    # C1[0,0] = M1 + M4 - M5 + M7
    # C1[0,1] = M3 + M5
    # C1[1,0] = M2 + M4
    # C1[1,1] = M1 - M2 + M3 + M6
    # 
    # Now A2 B2 (2x1 * 1x2):
    # A2 is 2x1 vector [a0, a1]^T.
    # B2 is 1x2 vector [b0, b1].
    # C2 = [a0*b0, a0*b1; a1*b0, a1*b1].
    # This requires 2 multiplications:
    # M8 = a0 * (b0 + b1)
    # M9 = a1 * (b0 + b1)
    # Then C2[0,0] = a0*b0. We need a0*b0.
    # Wait, simple multiplication for 2x1 * 1x2 requires 4 mults?
    # a0*b0, a0*b1, a1*b0, a1*b1.
    # Can we do it in 2?
    # (a0+a1)(b0+b1) = a0b0 + a0b1 + a1b0 + a1b1.
    # This gives the sum.
    # We need individual terms.
    # So Rank 4 for this part?
    # Total Rank = 7 + 4 = 11.
    # This is worse than schoolbook 12? No, 11 < 12.
    # So R(2,3,2) <= 11.
    # Then R(2,3,6) <= 33. Score 36/33 = 1.09.
    # Current score is 1.0588 (Rank 34).
    # So Rank 33 is an improvement!
    
    # Let's verify if we can do better for A2 B2.
    # A2 B2 is a rank-1 matrix? No, A2 is 2x1, B2 is 1x2. Product is 2x2 rank 1.
    # But we need to compute the entries.
    # The tensor for 2x1 * 1x2 multiplication has rank 4?
    # T[a, i, j] = A[a] * B[j] where i is dummy?
    # Actually, the multiplication of a column vector (2x1) and a row vector (1x2) to give a 2x2 matrix.
    # C[a, j] = A[a] * B[j].
    # This is a rank-1 tensor in the (A, B, C) space?
    # No, C has 4 entries.
    # The bilinear map is (A, B) -> C.
    # C[a, j] = sum_k A[a, k] B[k, j].
    # Here k is fixed to 0 (since it's 1x1 interaction).
    # So C[a, j] = A[a] B[j].
    # This is 4 separate products?
    # No, it's a single outer product.
    # But in bilinear complexity, we count multiplications of scalars.
    # To compute A[0]B[0], A[0]B[1], A[1]B[0], A[1]B[1].
    # We can compute:
    # M1 = A[0] * B[0]
    # M2 = A[0] * B[1]
    # M3 = A[1] * B[0]
    # M4 = A[1] * B[1]
    # Is there a way to do it in fewer?
    # (A0+A1)(B0+B1) = A0B0 + A0B1 + A1B0 + A1B1.
    # (A0-A1)(B0-B1) = A0B0 - A0B1 - A1B0 + A1B1.
    # Sum = 2 A0B0 + 2 A1B1.
    # Diff = 2 A0B1 + 2 A1B0.
    # So we get sums and diffs.
    # We can recover all 4 with 2 multiplications?
    # Let M1 = (A0+A1)(B0+B1)
    # Let M2 = (A0-A1)(B0-B1)
    # M1+M2 = 2(A0B0 + A1B1)
    # M1-M2 = 2(A0B1 + A1B0)
    # We still don't have A0B0 and A1B1 separately.
    # So we need at least 4?
    # Actually, the rank of the tensor for 2x1 * 1x2 is 4.
    # Proof: The matrix of the bilinear form is block diagonal?
    # Anyway, let's assume Rank 4.
    
    # So R(2,3,2) <= 7 + 4 = 11.
    # R(2,3,6) <= 11 * 3 = 33.
    
    # Can we do better for M_{2,3,2}?
    # Maybe the Strassen part and the outer product part can be merged?
    # The inputs are disjoint (A1, A2 and B1, B2).
    # So the rank is the sum.
    
    # Let's try to find a Rank 10 decomposition for M_{2,3,2}.
    # If R(2,3,2) = 10, then R(2,3,6) = 30. Score 1.2.
    
    # What if we use a different split?
    # Split A into 1x3 and 1x3?
    # A1 (1x3), A2 (1x3).
    # B1 (3x2), B2 (3x2).
    # C1 = A1 B1 (1x2). Rank 3? (1x3 * 3x2).
    # C2 = A2 B2 (1x2). Rank 3?
    # Total Rank 6?
    # No, C = C1 + C2.
    # C1 is 1x2, C2 is 1x2.
    # C is 2x2? No, A is 2x3.
    # If we split A by rows:
    # A = [R1; R2] where R1, R2 are 1x3.
    # C = R1 B + R2 B.
    # R1 B is 1x6.
    # R2 B is 1x6.
    # This doesn't help with the 2x2 structure of C.
    
    # Let's stick with the Rank 33 solution (2x3x2 Rank 11 x 3).
    # It's a structural improvement over Rank 34.
    
    # Let's construct the Rank 11 decomposition for M_{2,3,2}.
    
    def get_rank11_232():
        # Returns U (6, 11), V (6, 11), W (4, 11) for M_{2,3,2}
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        
        # Part 1: Strassen for A1 (2x2) * B1 (2x2)
        # A indices: 0,1 (row 0), 2,3 (row 1).
        # A1 uses A[0,0], A[0,1], A[1,0], A[1,1].
        # In U (size 6), indices are:
        # 0: A[0,0]
        # 1: A[0,1]
        # 2: A[0,2]
        # 3: A[1,0]
        # 4: A[1,1]
        # 5: A[1,2]
        
        # B indices: 0,1 (col 0), 2,3 (col 1).
        # B1 uses B[0,0], B[0,1], B[1,0], B[1,1].
        # In V (size 6), indices are:
        # 0: B[0,0]
        # 1: B[0,1]
        # 2: B[1,0]
        # 3: B[1,1]
        # 4: B[2,0]
        # 5: B[2,1]
        
        # C indices: 0,1 (row 0), 2,3 (row 1).
        # C1 uses C[0,0], C[0,1], C[1,0], C[1,1].
        # In W (size 4), indices are:
        # 0: C[0,0]
        # 1: C[0,1]
        # 2: C[1,0]
        # 3: C[1,1]
        
        # Strassen Multiplications M1..M7
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C00 += M1
        W[0, 0] = 1
        # C11 += M1
        W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        # C10 += M2
        W[2, 1] = 1
        # C11 -= M2
        W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C01 += M3
        W[1, 2] = 1
        # C11 += M3
        W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[4, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C00 += M4
        W[0, 3] = 1
        # C10 += M4
        W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C00 -= M5
        W[0, 4] = -1
        # C01 += M5
        W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C11 += M6
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[4, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C00 += M7
        W[0, 6] = 1
        
        # Part 2: Outer Product A2 (2x1) * B2 (1x2)
        # A2 is A[:, 2] -> indices 2, 5 in U.
        # B2 is B[2, :] -> indices 4, 5 in V.
        # C2 is C[:, 1]? No.
        # Wait, B is 3x2.
        # B1 is B[0:2, :] (2x2).
        # B2 is B[2, :] (1x2).
        # A1 is A[:, 0:2] (2x2).
        # A2 is A[:, 2] (2x1).
        # C = A1 B1 + A2 B2.
        # A2 B2 is 2x2.
        # C2[a, j] = A[a, 2] * B[2, j].
        # This adds to ALL entries of C?
        # Yes, C is 2x2.
        # So C2[0,0] = A[0,2]B[2,0]
        # C2[0,1] = A[0,2]B[2,1]
        # C2[1,0] = A[1,2]B[2,0]
        # C2[1,1] = A[1,2]B[2,1]
        
        # We need 4 multiplications for this.
        # M8 = A[0,2] * B[2,0]
        U[2, 7] = 1
        V[4, 7] = 1
        W[0, 7] = 1
        
        # M9 = A[0,2] * B[2,1]
        U[2, 8] = 1
        V[5, 8] = 1
        W[1, 8] = 1
        
        # M10 = A[1,2] * B[2,0]
        U[5, 9] = 1
        V[4, 9] = 1
        W[2, 9] = 1
        
        # M11 = A[1,2] * B[2,1]
        U[5, 10] = 1
        V[5, 10] = 1
        W[3, 10] = 1
        
        return U, V, W

    U232, V232, W232 = get_rank11_232()
    
    # Now construct M_{2,3,6} from 3 copies of M_{2,3,2}.
    # B is 3x6. Split into B1 (3x2), B2 (3x2), B3 (3x2).
    # C is 2x6. Split into C1 (2x2), C2 (2x2), C3 (2x2).
    # C = A B1 + A B2 + A B3.
    # Each term is a M_{2,3,2} multiplication.
    # Since output indices are disjoint, we can concatenate the decompositions.
    
    rank_232 = U232.shape[1]
    total_rank = rank_232 * 3
    
    U236 = np.zeros((6, total_rank))
    V236 = np.zeros((18, total_rank))
    W236 = np.zeros((12, total_rank))
    
    for k in range(3):
        start = k * rank_232
        end = start + rank_232
        
        # U is same for all
        U236[:, start:end] = U232
        
        # V: B is 3x6.
        # Block k of B is B[:, k*2 : (k+1)*2].
        # In V232, indices are 0..5 corresponding to B[0,0]..B[2,1].
        # In V236, indices are 0..17 corresponding to B[0,0]..B[2,5].
        # For block k, B[0, k*2] is V236 index k*2.
        # B[0, k*2+1] is V236 index k*2+1.
        # B[1, k*2] is V236 index 6 + k*2.
        # B[1, k*2+1] is V236 index 6 + k*2+1.
        # B[2, k*2] is V236 index 12 + k*2.
        # B[2, k*2+1] is V236 index 12 + k*2+1.
        
        # V232 indices:
        # 0: B[0,0]
        # 1: B[0,1]
        # 2: B[1,0]
        # 3: B[1,1]
        # 4: B[2,0]
        # 5: B[2,1]
        
        for i in range(6):
            # Map V232 index i to V236 index for block k
            # i = row * 2 + col
            row = i // 2
            col = i % 2
            v236_idx = row * 6 + k * 2 + col
            V236[v236_idx, start:end] = V232[i]
            
        # W: C is 2x6.
        # Block k of C is C[:, k*2 : (k+1)*2].
        # In W232, indices are 0..3 corresponding to C[0,0]..C[1,1].
        # In W236, indices are 0..11 corresponding to C[0,0]..C[1,5].
        # W index is k*n + i? No, task says W[k*n+i, r].
        # Wait, task says "output factor follows the official trace convention W[k*n+i,r]".
        # k is column index? i is row index?
        # Usually C[i, k].
        # If W index is k*n + i, then for n=2, p=6.
        # Index 0: k=0, i=0 -> C[0,0]
        # Index 1: k=0, i=1 -> C[1,0]
        # Index 2: k=1, i=0 -> C[0,1]
        # Index 3: k=1, i=1 -> C[1,1]
        # ...
        # Index 2k + i.
        
        # In W232, indices are 0..3.
        # 0: C[0,0]
        # 1: C[0,1] ?
        # Let's check my W232 construction.
        # I used:
        # 0: C[0,0]
        # 1: C[0,1]
        # 2: C[1,0]
        # 3: C[1,1]
        # This is row-major (i, k).
        # But the task requires column-major (k, i) for the index?
        # "W[k*n+i, r]"
        # So Index = k*n + i.
        # For C[0,0]: k=0, i=0 -> 0.
        # For C[0,1]: k=1, i=0 -> 2.
        # For C[1,0]: k=0, i=1 -> 1.
        # For C[1,1]: k=1, i=1 -> 3.
        
        # My W232 was:
        # W[0] = C[0,0]
        # W[1] = C[0,1]
        # W[2] = C[1,0]
        # W[3] = C[1,1]
        
        # So I need to map my W232 indices to the task's W236 indices.
        # W232[0] (C[0,0]) -> Task Index 0 (k=0, i=0)
        # W232[1] (C[0,1]) -> Task Index 2 (k=1, i=0)
        # W232[2] (C[1,0]) -> Task Index 1 (k=0, i=1)
        # W232[3] (C[1,1]) -> Task Index 3 (k=1, i=1)
        
        # For block k (cols k*2, k*2+1):
        # C[0, k*2] -> Task Index (k*2)*2 + 0 = 4k
        # C[0, k*2+1] -> Task Index (k*2+1)*2 + 0 = 4k + 2
        # C[1, k*2] -> Task Index (k*2)*2 + 1 = 4k + 1
        # C[1, k*2+1] -> Task Index (k*2+1)*2 + 1 = 4k + 3
        
        # Map W232 indices:
        # 0 (C[0,0] local) -> 4k
        # 1 (C[0,1] local) -> 4k + 2
        # 2 (C[1,0] local) -> 4k + 1
        # 3 (C[1,1] local) -> 4k + 3
        
        for i in range(4):
            if i == 0:
                w236_idx = 4 * k
            elif i == 1:
                w236_idx = 4 * k + 2
            elif i == 2:
                w236_idx = 4 * k + 1
            else:
                w236_idx = 4 * k + 3
            W236[w236_idx, start:end] = W232[i]
            
    # Verify
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                target = 1.0 if (a == i and c == k and b == d) else 0.0
                                s = np.dot(U[a * m + b, :], np.multiply(V[c * p + d, :], W[k * n + i, :]))
                                if not np.isclose(s, target, atol=1e-9):
                                    return False
        return True

    if verify(U236, V236, W236, n, m, p):
        return {
            "U": U236.tolist(),
            "V": V236.tolist(),
            "W": W236.tolist()
        }
    else:
        # Fallback to Rank 34 if something went wrong
        # Reuse the logic from the current program's fallback
        U0, V0, W0 = get_rank34_solution()
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }

def get_rank34_solution():
    U233_seed = np.zeros((6, 17))
    V233_seed = np.zeros((9, 17))
    W233_seed = np.zeros((6, 17))
    
    U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
    V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
    W233_seed[0, 0] = 1
    W233_seed[3, 0] = 1
    
    U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
    V233_seed[0, 1] = 1
    W233_seed[1, 1] = 1
    W233_seed[3, 1] = -1
    
    U233_seed[0, 2] = 1
    V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
    W233_seed[2, 2] = 1
    W233_seed[3, 2] = 1
    
    U233_seed[4, 3] = 1
    V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
    W233_seed[0, 3] = 1
    W233_seed[1, 3] = 1
    
    U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
    V233_seed[4, 4] = 1
    W233_seed[0, 4] = -1
    W233_seed[2, 4] = 1
    
    U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
    V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
    W233_seed[3, 5] = 1
    
    U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
    V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
    W233_seed[0, 6] = 1
    
    U233_seed[0, 7] = 1
    V233_seed[2, 7] = 1
    W233_seed[4, 7] = 1
    
    U233_seed[1, 8] = 1
    V233_seed[5, 8] = 1
    W233_seed[4, 8] = 1
    
    U233_seed[3, 9] = 1
    V233_seed[2, 9] = 1
    W233_seed[5, 9] = 1
    
    U233_seed[4, 10] = 1
    V233_seed[5, 10] = 1
    W233_seed[5, 10] = 1
    
    U233_seed[2, 11] = 1
    V233_seed[6, 11] = 1
    W233_seed[0, 11] = 1
    
    U233_seed[2, 12] = 1
    V233_seed[7, 12] = 1
    W233_seed[2, 12] = 1
    
    U233_seed[2, 13] = 1
    V233_seed[8, 13] = 1
    W233_seed[4, 13] = 1
    
    U233_seed[5, 14] = 1
    V233_seed[6, 14] = 1
    W233_seed[1, 14] = 1
    
    U233_seed[5, 15] = 1
    V233_seed[7, 15] = 1
    W233_seed[3, 15] = 1
    
    U233_seed[5, 16] = 1
    V233_seed[8, 16] = 1
    W233_seed[5, 16] = 1
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U_comb[:, :17] = U233_seed
    U_comb[:, 17:] = U233_seed
    
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
            
    return U_comb, V_comb, W_comb

if __name__ == "__main__":
    construct()
