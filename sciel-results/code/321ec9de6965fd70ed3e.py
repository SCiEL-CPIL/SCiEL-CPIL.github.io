import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 10, 25, 10
    
    start_time = time.time()
    deadline = 140.0
    
    # We construct a decomposition using Strassen for the 4x4 core and schoolbook for the rest.
    # A is 2x5, B is 5x5.
    # Split indices j, k into {0,1,2,3} and {4}.
    # A = [A_core (2x4) | A_tail (2x1)]
    # B = [B_core (4x4) ; B_tail (1x4)]  (top part)
    #     [B_tail_row (1x4) ; B_tail_scalar (1x1)] (bottom part)
    # Actually B is 5x5.
    # B = [ B00 (4x4) B01 (4x1) ]
    #     [ B10 (1x4) B11 (1x1) ]
    
    # C = A B
    # C = [ A_core B00 + A_tail B10 , A_core B01 + A_tail B11 ]
    # C is 2x5.
    # C_core (2x4) = A_core B00 + A_tail B10
    # C_tail (2x1) = A_core B01 + A_tail B11
    
    # We can compute A_core B00 using Strassen 4x4 (rank 15) if we can embed 2x4 into 4x4.
    # Or we can just use schoolbook for 2x4x4 which is rank 8.
    # 8 < 15, so schoolbook is better for the core?
    # Wait, Strassen 4x4 is rank 15 for 4x4x4.
    # For 2x4x4, the rank is at most 8 (schoolbook).
    # So using Strassen doesn't help for the 2x4 part directly unless we share with another 2x4 part?
    # But we only have one 2x4 part.
    
    # Let's check the rank of 2x4x4.
    # Schoolbook: 2*4*4 = 32? No, 2*4*4 terms.
    # Rank is 2*4 = 8?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For fixed i, it's 1x4x4. Rank 4.
    # For 2 rows, rank 8.
    
    # So:
    # C_core (2x4): Rank 8.
    # C_tail (2x1):
    #   A_core B01: 2x4 x 4x1. Rank 4.
    #   A_tail B11: 2x1 x 1x1. Rank 1.
    #   A_tail B10: 2x1 x 1x4. Rank 2. (Wait, B10 is 1x4, so A_tail (2x1) * B10 (1x4) -> 2x4. This contributes to C_core? No.)
    
    # Let's re-evaluate the blocks.
    # A = [ A_c (2x4) | A_t (2x1) ]
    # B = [ B_cc (4x4) | B_ct (4x1) ]
    #     [ B_tc (1x4) | B_tt (1x1) ]
    
    # C = A B
    # C = [ A_c B_cc + A_t B_tc , A_c B_ct + A_t B_tt ]
    # C_left (2x4) = A_c B_cc + A_t B_tc
    # C_right (2x1) = A_c B_ct + A_t B_tt
    
    # Term 1: A_c B_cc. Rank 8 (schoolbook).
    # Term 2: A_t B_tc. A_t is 2x1, B_tc is 1x4. Rank 2 (each row of A_t multiplies B_tc).
    # Term 3: A_c B_ct. A_c is 2x4, B_ct is 4x1. Rank 4.
    # Term 4: A_t B_tt. A_t is 2x1, B_tt is 1x1. Rank 1.
    
    # Total rank = 8 + 2 + 4 + 1 = 15.
    
    # This is a rank-15 decomposition!
    # Let's verify.
    # We need to construct U, V, W.
    
    rank = 15
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx, col] = val
        V[v_idx, col] = val
        W[w_idx, col] = val
        col += 1

    # Indices:
    # U: i*m + j. i in 0..1, j in 0..4.
    # V: j*p + k. j in 0..4, k in 0..4.
    # W: k*n + i. k in 0..4, i in 0..1.
    
    # Term 1: A_c B_cc (2x4 x 4x4)
    # i in 0..1, j in 0..3, k in 0..3.
    for i in range(2):
        for j in range(4):
            for k in range(4):
                add_term(i * 5 + j, j * 5 + k, k * 2 + i)
                
    # Term 2: A_t B_tc (2x1 x 1x4)
    # A_t is j=4. B_tc is j=4, k in 0..3.
    # i in 0..1, j=4, k in 0..3.
    for i in range(2):
        for k in range(4):
            add_term(i * 5 + 4, 4 * 5 + k, k * 2 + i)
            
    # Term 3: A_c B_ct (2x4 x 4x1)
    # A_c is j in 0..3. B_ct is k=4, j in 0..3.
    # i in 0..1, j in 0..3, k=4.
    for i in range(2):
        for j in range(4):
            add_term(i * 5 + j, j * 5 + 4, 4 * 2 + i)
            
    # Term 4: A_t B_tt (2x1 x 1x1)
    # A_t is j=4. B_tt is j=4, k=4.
    # i in 0..1, j=4, k=4.
    for i in range(2):
        add_term(i * 5 + 4, 4 * 5 + 4, 4 * 2 + i)
        
    # Total terms: 2*4*4 + 2*4 + 2*4 + 2 = 32 + 8 + 8 + 2 = 50?
    # Wait, my rank calculation was 15.
    # Where is the error?
    # Rank of A_c B_cc (2x4x4) is 8.
    # My loop: i(2) * j(4) * k(4) = 32 terms.
    # Ah, the tensor is T[i,j,k].
    # For A_c B_cc, we have i in 0..1, j in 0..3, k in 0..3.
    # The product is C[i,k] = sum_j A[i,j] B[j,k].
    # This requires 1 term for each (i,j,k) in schoolbook?
    # No, schoolbook is sum over j.
    # The term is A[i,j] B[j,k].
    # So for each i, k, we sum over j.
    # In tensor decomposition, each (i,j,k) triple is a rank-1 term if we use schoolbook.
    # So 2 * 4 * 4 = 32 terms for the core.
    # My previous rank calculation "Rank 8" was for the matrix multiplication map, but the tensor rank is higher.
    # The rank of the tensor $T_{i,j,k}$ for $2 \times 4 \times 4$ is $2 \times 4 \times 4 = 32$?
    # No, the rank of $n \times m \times p$ matrix multiplication tensor is $nmp$ for schoolbook.
    # Is it reducible?
    # For $2 \times 2 \times 2$, rank is 7 (Strassen).
    # For $2 \times 4 \times 4$, rank is likely less than 32.
    # But without a specific algorithm, I can't easily reduce it.
    
    # So the simple block decomposition gives rank 50 (schoolbook).
    # The rank-40 construction using $5 \times 5 \times 5$ is safer if I have the factors.
    # Since I don't have the rank-20 $5 \times 5 \times 5$ factors hardcoded, I will return the schoolbook rank 50.
    # But I can try to find a rank-40 solution by using two disjoint $4 \times 4 \times 4$ Strassen? No.
    
    # Let's just return the schoolbook solution which is valid.
    # To ensure it's valid, I will use the simple construction.
    
    rank = 50
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

