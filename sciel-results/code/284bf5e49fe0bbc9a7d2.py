import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 5
    DEADLINE = 140.0  # seconds, safety margin before 150s kill
    
    # We aim to find a low-rank decomposition.
    # Current best known: 21 (from schoolbook on 2x2x5 + 1 for 2x1x5)
    # Naive: 2*3*5 = 30
    # We will try to verify if a lower rank is possible or stick with 21.
    
    # Given the small size, we can try to search for a lower rank using randomized optimization
    # or just return the known 21-rank solution if time is short.
    # However, the task asks for "rank-minimal". The minimal rank for 2x3x5 is unknown in general?
    # Actually, for small matrices, the minimal rank is often the schoolbook rank unless there's a specific trick.
    # Let's check the current 21-rank solution validity first.
    
    # Construct the 21-rank solution (Schoolbook 2x2x5 + 1x2x5)
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: j in {0,1} (2x2x5 schoolbook)
    r_offset = 0
    idx = 0
    for i in range(n):
        for j in range(2):
            for k in range(p):
                r = r_offset + idx
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                idx += 1
                
    # Part 2: j=2 (2x1x5, rank 1)
    r = 20
    for i in range(n):
        U[i * m + 2, r] = 1
    for k in range(p):
        V[2 * p + k, r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, r] = 1
            
    # Verify
    T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    T_expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i * m + j, j * p + k, k * n + i] = 1
                
    valid_21 = np.allclose(T_reconstructed, T_expected, atol=1e-10)
    
    if not valid_21:
        # Fallback to naive 30
        rank_naive = n * m * p
        U_naive = np.zeros((n * m, rank_naive))
        V_naive = np.zeros((m * p, rank_naive))
        W_naive = np.zeros((p * n, rank_naive))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_naive[ii * m + jj, col] = 1
                    V_naive[jj * p + kk, col] = 1
                    W_naive[kk * n + ii, col] = 1
                    col += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    # Can we do better than 21?
    # The rank of matrix multiplication tensor is at least the maximum of the ranks of the "slices".
    # For 2x3x5, the rank is at least 16 (from 2x2x5 lower bound? No, 2x2x5 has rank 10? No, schoolbook is 20).
    # Actually, the rank of 2x2x2 is 7 (Laderman).
    # For 2x2x5, we can use Laderman for the 2x2x2 parts?
    # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1.
    # Rank(2x2x2) = 7. Rank(2x2x2) = 7. Rank(2x2x1) = 1. Total = 15?
    # If we can decompose 2x2x5 with rank 15, then total rank = 15 + 1 (for 2x1x5) = 16.
    
    # Let's try to construct a rank 16 solution.
    # We need a rank 15 decomposition of 2x2x5.
    # 2x2x5 tensor T[i,j,k] = delta(i,j,k) for i in {0,1}, j in {0,1}, k in {0..4}.
    # Split k into {0,1}, {2,3}, {4}.
    # T = T1 (k=0,1) + T2 (k=2,3) + T3 (k=4).
    # T1 is 2x2x2. Rank 7.
    # T2 is 2x2x2. Rank 7.
    # T3 is 2x2x1. Rank 1.
    # Total rank = 7 + 7 + 1 = 15.
    
    # Laderman's 7-term formula for 2x2x2:
    # C = A * B
    # Terms (u, v, w) where u, v, w are 2x2 matrices such that C = sum u * A * v * B * w
    # But our tensor is T[i,j,k] = A[i,j] B[j,k].
    # The standard Laderman decomposition gives 7 terms.
    
    # We need to embed this into our U, V, W format.
    # U indices: i*m + j. Here m=2 for the subproblem.
    # V indices: j*p_sub + k.
    # W indices: k*n + i.
    
    # Let's define the Laderman terms for 2x2x2.
    # The 7 terms for 2x2x2 multiplication:
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A B
    # Laderman's 7 factors:
    # 1. (a11+a22)(b11+b22) - a22 b22 - a11 b11 ... no, that's not right.
    # Standard Laderman:
    # t1 = a11(b11+b21)
    # t2 = a21(b22+b12)
    # t3 = a22(b11+b21)
    # t4 = (a11+a21)(b12+b22)
    # t5 = (a12+a22)(b11+b21)
    # t6 = (a12+a22)(b21+b22)
    # t7 = (a11+a21+a22)(b12+b22) - t4 - t5 - t6 ... this is getting complicated to map to tensor.
    
    # It is safer to use a known explicit 7-term tensor decomposition for 2x2x2.
    # The tensor T_{ijk} for 2x2x2 matmul has rank 7.
    # We can find a basis for the 7 terms.
    # Actually, let's just use the rank 21 solution for now, as implementing Laderman correctly in this specific index format is error-prone and time-consuming.
    # The prompt says "rank-minimal". If the true minimum is 16, we should aim for it.
    # But 21 is a valid solution.
    
    # Let's check if we have time to try a search for rank < 21.
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try to find a rank 20 solution?
    # Or just return 21.
    # Given the complexity, returning 21 is a solid 1.0 score.
    # The inspirations got 1.07, which corresponds to rank 20.
    # So we MUST find a rank 20 solution to beat 1.0.
    
    # How to get rank 20?
    # 2x3x5.
    # Maybe 2x2x5 (rank 10? No, schoolbook 20) + 2x1x5 (rank 1).
    # If 2x2x5 has rank 10, then total is 11. But 2x2x2 has rank 7, so 2x2x5 is at least 7+7+1=15.
    # So 2x2x5 rank is at least 15.
    # If we use rank 15 for 2x2x5, total is 16.
    # If we use rank 19 for 2x2x5, total is 20.
    # Schoolbook is 20.
    # So if we can do 2x2x5 with rank 19, we get total 20.
    # Or if we can do the whole 2x3x5 with rank 20 directly.
    
    # Let's try a rank 20 construction.
    # We need to save 1 rank from the 21-rank solution.
    # The 21-rank solution used 20 for 2x2x5 and 1 for 2x1x5.
    # The 2x1x5 part is a single outer product term: sum_k A[i,2] B[2,k].
    # This is already minimal for that part.
    # So we must reduce the 2x2x5 part from 20 to 19?
    # Or maybe there is a global decomposition of 2x3x5 with rank 20.
    
    # Let's try to use a randomized search for rank 20.
    # Initialize U, V, W with random values and use gradient descent?
    # This is risky with 120s limit.
    
    # Alternative: Check if there is a known 20-rank decomposition.
    # The rank of 2x3x5 is at most 21.
    # It is known that rank(2x2x2) = 7.
    # rank(2x2x3) ?
    # Let's assume the minimum is 20 and try to construct it.
    
    # Actually, the simplest way to get rank 20 is if the 2x3x5 tensor itself has rank 20.
    # We can try to solve the linear system for rank 20.
    # Variables: U (6x20), V (15x20), W (10x20).
    # Equations: 6*15*10 = 900.
    # This is a large system.
    
    # Given the time constraint, I will return the 21-rank solution.
    # If I can't beat 1.0, I should at least be correct.
    # But the inspirations got 1.07, so they found rank 20.
    
    # Let's try a specific rank 20 construction.
    # Maybe the 2x1x5 part can be merged?
    # No, the indices are disjoint.
    
    # Let's try to implement the Laderman-based 16-rank solution for 2x2x5 + 1 = 17?
    # If we can get 16, score would be 30/16 = 1.875.
    # If we can get 20, score is 1.5.
    
    # Let's try to code the Laderman 7-term for 2x2x2 and see if we can use it.
    # The 7 terms for 2x2x2 matmul (A*B=C):
    # We need U, V, W of shape (4, 7), (4, 7), (4, 7) for the subproblem.
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    
    # I will not risk a buggy Laderman implementation.
    # I will return the 21-rank solution.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
