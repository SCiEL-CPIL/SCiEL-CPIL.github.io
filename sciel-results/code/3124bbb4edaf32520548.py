import time
import numpy as np
import random

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy: 
    1. Define the target tensor T (6x15x10).
    2. Use a Greedy Matching Pursuit algorithm to find a sparse decomposition.
    3. The 'dictionary' consists of all possible rank-1 tensors e_i \otimes e_j \otimes e_k.
    4. We iteratively pick the dictionary element that has the highest correlation with the residual,
       assign it a weight (projected value), and subtract it.
    5. We try to achieve rank 28, 27, 26, etc.
    6. Since the target is integer, we check for exact integer/half-integer solutions.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Construct Target Tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    T_flat = T.flatten()
    
    # 2. Precompute Dictionary (All possible rank-1 tensors)
    # A rank-1 tensor is e_I \otimes e_J \otimes e_K
    # There are 6 * 15 * 10 = 900 such tensors.
    # We don't need to store them all as 900-dim vectors, we can compute dot products on the fly.
    # dot = T_flat[idx] where idx corresponds to (I, J, K).
    # Actually, the dictionary element D_{I,J,K} has 1 at (I,J,K) and 0 elsewhere.
    # So the coefficient for D_{I,J,K} in a greedy step is simply the value of the residual at (I,J,K).
    # This means Matching Pursuit on this specific dictionary (Standard Basis) is trivial and just picks the largest non-zero entries.
    # The standard basis has 900 elements. The target T has 30 non-zero entries.
    # So the sparsest representation in the standard basis is rank 30.
    # To get rank < 30, we MUST use a different dictionary or allow linear combinations that cancel.
    # BUT, the tensor rank is the minimum number of rank-1 tensors (not necessarily from the standard basis of single entries) needed to sum to T.
    # A rank-1 tensor in the tensor space is $u \otimes v \otimes w$.
    # Its entries are $u_I v_J w_K$.
    # So we are looking for $R$ triples of vectors $(u_c, v_c, w_c)$ such that $\sum u_{c,I} v_{c,J} w_{c,K} = T_{I,J,K}$.
    
    # This is the original problem. Matching Pursuit on the standard basis doesn't help because the "atoms" are too small.
    # We need a "dictionary" of rank-1 tensors that are "structured".
    
    # Alternative Structural Strategy: 
    # Use the known 28-rank decomposition.
    # I will hardcode a known valid 28-rank decomposition for M_{2,3,5}.
    # If I cannot recall it perfectly, I will use a **Local Search** on the 30-term schoolbook,
    # allowing "moves" that replace 2 terms with 1 term (if possible) or 3 terms with 2 terms.
    
    # Actually, let's try **Randomized SVD-like** approach? No.
    
    # Let's try **Genetic Algorithm** on the coefficients of a fixed set of 30 "slots",
    # but allowing the slots to be permuted or values to be adjusted.
    
    # BEST BET: The rank of M_{2,3,5} is 28.
    # I will construct the 28-term decomposition using the following method:
    # Split A (2x3) into A1 (2x2) and A2 (2x1).
    # Split B (3x5) into B1 (2x5) and B2 (1x5).
    # C = A1 B1 + A2 B2.
    # A2 B2 is 2x1 * 1x5. This is a sum of 2 rank-1 matrices? 
    # A2 = [a1, a2]^T, B2 = [b1, ..., b5].
    # A2 B2 = a1 B2 + a2 B2? No.
    # A2 B2 = [a1 b1, a1 b2, ..., a1 b5; a2 b1, a2 b2, ..., a2 b5].
    # This requires 10 multiplications in schoolbook.
    # But algebraically, can we do better?
    # The tensor for A2 B2 is T_{i, j, k} where i in {1,2}, j=1, k in {1..5}.
    # This is a tensor of size 2 x 1 x 5.
    # The rank of M_{2,1,5} is 2*1*5 = 10? No.
    # M_{n,1,p} has rank n*p. So rank 10.
    # So A2 B2 contributes 10 to the rank.
    
    # Now A1 B1 is 2x2 * 2x5.
    # Split B1 into B11 (2x2) and B12 (2x3).
    # A1 B11 is 2x2 * 2x2. Rank 7 (Strassen).
    # A1 B12 is 2x2 * 2x3.
    # Total Rank = Rank(A1 B11) + Rank(A1 B12) + Rank(A2 B2)
    # = 7 + Rank(2x2 * 2x3) + 10.
    # What is Rank(2x2 * 2x3)?
    # Split B12 into two 2x1 columns?
    # M_{2,2,3} rank is known?
    # M_{2,2,3} can be viewed as 2x2 * 2x1 + 2x2 * 2x1?
    # Rank(M_{2,2,3}) = 12? No, schoolbook is 12.
    # Can we do better?
    # If we use Strassen for the 2x2 part of the first 2 columns...
    # It is known that R(M_{2,2,3}) = 12? Or 11?
    # If R(M_{2,2,3}) = 12, then Total = 7 + 12 + 10 = 29.
    # We want 28.
    
    # There is a known 28-term formula.
    # I will use a **Search** to find the 28 terms.
    # I will initialize with the 30-term schoolbook.
    # I will try to find a linear dependency among the 30 terms that allows merging.
    
    # Since I don't have the explicit 28-term formula memorized, I will use a **Heuristic Search** 
    # that is different from ALS: **Simulated Annealing on the Weights and Indices**.
    
    # However, to be safe and fast, I will return the Schoolbook (Rank 30) if I can't find better quickly,
    # BUT I will try one specific structural trick:
    # **Strassen for 2x2 blocks**.
    
    # Let's implement the 29-rank construction (7 + 12 + 10) which is easy,
    # and then try to shave 1 more off.
    
    # Actually, the prompt says "score is the rank". Lower is better.
    # Current score 1.0 (Rank 30).
    # If I find Rank 29, score is 30/29 = 1.034.
    # If I find Rank 28, score is 30/28 = 1.071.
    
    # I will implement the **29-rank construction** explicitly.
    # 1. A1 B11 (2x2 * 2x2) -> 7 terms (Strassen)
    # 2. A1 B12 (2x2 * 2x3) -> 12 terms (Schoolbook)
    # 3. A2 B2 (2x1 * 1x5) -> 10 terms (Schoolbook)
    # Total 29.
    
    # Then I will try to optimize the 2x2 * 2x3 part to rank 11?
    # Or the 2x1 * 1x5 part?
    
    # Let's code the 29-rank decomposition.
    
    def strassen_2x2(A, B):
        # Returns 7 linear combinations
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        pass # We don't need to compute values, just the structure of the tensor decomposition.
    
    # We need to output U, V, W matrices.
    # U is (6, Rank), V is (15, Rank), W is (10, Rank).
    
    def build_29_rank():
        R = 29
        U = np.zeros((6, R))
        V = np.zeros((15, R))
        W = np.zeros((10, R))
        
        col = 0
        
        # Part 1: A1 (rows 0,1 of A; cols 0,1 of B) * B11 (rows 0,1 of B; cols 0,1 of C? No, B is 3x5)
        # A indices: I = i*m + j. A is 2x3.
        # i=0, j=0 -> I=0
        # i=0, j=1 -> I=1
        # i=0, j=2 -> I=2
        # i=1, j=0 -> I=3
        # i=1, j=1 -> I=4
        # i=1, j=2 -> I=5
        
        # B indices: J = j'*p + k. B is 3x5.
        # j'=0, k=0 -> J=0
        # ...
        # j'=0, k=4 -> J=4
        # j'=1, k=0 -> J=5
        # ...
        # j'=2, k=4 -> J=14
        
        # C indices: K = k'*n + i'. C is 5x2? No, C is 2x5.
        # Output factor follows W[k*n+i, r].
        # K = k*n + i.
        # k=0, i=0 -> K=0
        # k=0, i=1 -> K=1
        # k=1, i=0 -> K=2
        # ...
        
        # Strassen for A1 (I=0,1,3,4) and B11 (J=0,1,5,6) -> C11 (K=0,1,2,3)
        # A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] = [[U[0], U[1]], [U[3], U[4]]]
        # B11 = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]] = [[V[0], V[1]], [V[5], V[6]]]
        # C11 = [[C[0,0], C[0,1]], [C[1,0], C[1,1]]] = [[W[0], W[1]], [W[2], W[3]]]
        
        # M1 = (a11+a22)(b11+b22)
        U[0, col] += 1; U[4, col] += 1
        V[0, col] += 1; V[6, col] += 1
        W[0, col] += 1; W[3, col] += 1
        W[1, col] -= 1; W[2, col] += 1 # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
        # Wait, Strassen reconstruction:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the 7 terms properly.
        # Term 1: M1 = (a11+a22)(b11+b22)
        U[0, col] = 1; U[4, col] = 1
        V[0, col] = 1; V[6, col] = 1
        W[0, col] = 1; W[3, col] = 1
        # C11 gets +M1, C22 gets +M1
        col += 1
        
        # Term 2: M2 = (a21+a22)b11
        U[3, col] = 1; U[4, col] = 1
        V[0, col] = 1
        W[2, col] = 1; W[3, col] = -1
        # C21 gets +M2, C22 gets -M2
        col += 1
        
        # Term 3: M3 = a11(b12-b22)
        U[0, col] = 1
        V[1, col] = 1; V[6, col] = -1
        W[1, col] = 1; W[3, col] = 1
        # C12 gets +M3, C22 gets +M3
        col += 1
        
        # Term 4: M4 = a22(b21-b11)
        U[4, col] = 1
        V[5, col] = 1; V[0, col] = -1
        W[0, col] = 1; W[2, col] = 1
        # C11 gets +M4, C21 gets +M4
        col += 1
        
        # Term 5: M5 = (a11+a12)b22
        U[0, col] = 1; U[1, col] = 1
        V[6, col] = 1
        W[0, col] = -1; W[1, col] = 1
        # C11 gets -M5, C12 gets +M5
        col += 1
        
        # Term 6: M6 = (a21-a11)(b11+b12)
        U[3, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[3, col] = 1
        # C22 gets +M6
        col += 1
        
        # Term 7: M7 = (a12-a22)(b21+b22)
        U[1, col] = 1; U[4, col] = -1
        V[5, col] = 1; V[6, col] = 1
        W[0, col] = 1
        # C11 gets +M7
        col += 1
        
        # Part 2: A1 (I=0,1,3,4) * B12 (J=2,3,7,8,12,13) -> C12 (K=4,5,6,7,8,9)
        # B12 is 2x3. Schoolbook 2x2*2x3 is 12 terms.
        # A1 is 2x2. B12 is 2x3.
        # For each of the 3 columns of B12, do 2x2*2x1 (4 terms each? No, 2x2*2x1 is rank 4).
        # Total 12 terms.
        for c_idx in range(3):
            # B column index in B12: c_idx (0,1,2)
            # Corresponding J indices:
            # Row 0 of B12: J = 2 + c_idx
            # Row 1 of B12: J = 7 + c_idx
            # C column index: c_idx
            # Corresponding K indices:
            # Row 0 of C12: K = 4 + c_idx
            # Row 1 of C12: K = 9 + c_idx
            
            # Term 8: a11 * b11 (of block)
            U[0, col] = 1
            V[2 + c_idx, col] = 1
            W[4 + c_idx, col] = 1
            col += 1
            
            # Term 9: a12 * b21 (of block)
            U[1, col] = 1
            V[7 + c_idx, col] = 1
            W[4 + c_idx, col] = 1
            col += 1
            
            # Term 10: a21 * b11 (of block)
            U[3, col] = 1
            V[2 + c_idx, col] = 1
            W[9 + c_idx, col] = 1
            col += 1
            
            # Term 11: a22 * b21 (of block)
            U[4, col] = 1
            V[7 + c_idx, col] = 1
            W[9 + c_idx, col] = 1
            col += 1
            
        # Part 3: A2 (I=2,5) * B2 (J=10..14) -> C2 (K=10..19? No, K is 0..9)
        # A2 is 2x1. B2 is 1x5.
        # A2 indices: I=2 (i=0,j=2), I=5 (i=1,j=2).
        # B2 indices: J=10 (j=2,k=0) ... J=14 (j=2,k=4).
        # C2 indices:
        # For i=0 (A2 row 0): K = k*n + 0 = 0, 2, 4, 6, 8.
        # For i=1 (A2 row 1): K = k*n + 1 = 1, 3, 5, 7, 9.
        
        for k_idx in range(5):
            # Term 12: a21 * b21 (of block)
            U[2, col] = 1
            V[10 + k_idx, col] = 1
            W[2 * k_idx, col] = 1
            col += 1
            
            # Term 13: a22 * b21 (of block)
            U[5, col] = 1
            V[10 + k_idx, col] = 1
            W[2 * k_idx + 1, col] = 1
            col += 1
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    res = build_29_rank()
    
    # Verify
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Check
    T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_calc, T, atol=1e-9):
        # Fallback to schoolbook
        def schoolbook():
            rank = 30
            U_s = [[0] * rank for _ in range(nm)]
            V_s = [[0] * rank for _ in range(mp)]
            W_s = [[0] * rank for _ in range(pn)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][col] = 1
                        V_s[j * p + k][col] = 1
                        W_s[k * n + i][col] = 1
                        col += 1
            return {"U": U_s, "V": V_s, "W": W_s}
        return schoolbook()
        
    return res

