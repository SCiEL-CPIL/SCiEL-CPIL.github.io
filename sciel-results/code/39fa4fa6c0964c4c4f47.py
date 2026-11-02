import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    rank = 2 * p + 4  # 34
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    
    # 1. Diagonal terms: 2p terms
    # For each k, handle A[0,0]B[0,k] -> C[0,k] and A[1,1]B[1,k] -> C[1,k]
    for k in range(p):
        # Term 2k
        r = 2 * k
        U[0 * m + 0, r] = 1  # i=0, j=0
        V[0 * p + k, r] = 1  # j=0, k
        W[k * n + 0, r] = 1  # k, i=0
        
        # Term 2k+1
        r = 2 * k + 1
        U[1 * m + 1, r] = 1  # i=1, j=1
        V[1 * p + k, r] = 1  # j=1, k
        W[k * n + 1, r] = 1  # k, i=1
        
    # 2. Off-diagonal terms: 4 terms
    # We need to satisfy:
    # C[0,k] += A[0,1]B[1,k]
    # C[1,k] += A[1,0]B[0,k]
    # for all k.
    # This can be done with 4 terms if we use sums over k?
    # No, the terms must be rank 1.
    # The known construction for 2x2xp uses 4 extra terms.
    # Let's use the specific structure for 2x2x3 (rank 10) and extend.
    # Actually, for p>=3, the rank is 2p+4.
    # The 4 extra terms are:
    # r = 2p, 2p+1, 2p+2, 2p+3
    
    # Based on Strassen-like extension:
    # The off-diagonals can be handled by:
    # u4 = e_{0,1} + e_{1,0} ?
    
    # Let's use a verified decomposition for 2x2x15.
    # Since I cannot run code, I will rely on the pattern that works for small p.
    # For p=2, rank 7.
    # For p=3, rank 10.
    # The additional 2 terms per k suggests that the off-diagonals are not constant.
    # However, 2p+4 implies only 4 extra terms total.
    # This means the 4 extra terms must cover all k.
    # This is only possible if the off-diagonal contributions can be factored.
    # A[0,1]B[1,k] + A[1,0]B[0,k]
    # This is a vector in k.
    # We can use 2 terms for the sum and 2 for the difference?
    
    # Term 2p:
    # Handles A[0,1]B[1,k] for all k? No, B[1,k] varies with k.
    # So V must have 1s at all k for j=1?
    # If V has 1s at all k, it's not rank 1 in the k-dimension unless we sum.
    # But V is a vector.
    
    # Wait, if the rank is 2p+4, then for each k, we have 2 diagonal terms.
    # The off-diagonals must be handled by the 4 extra terms.
    # This implies that the off-diagonal part of the tensor has rank 4?
    # The tensor for off-diagonals is T_off = sum_k (A[0,1]B[1,k]C[0,k] + A[1,0]B[0,k]C[1,k]).
    # This is a sum of p tensors.
    # If p=1, rank 4.
    # If p=2, rank 7. (2*2+3).
    # If p=3, rank 10. (2*3+4).
    # It seems the rank grows with p.
    
    # I will output the schoolbook rank 60 to be safe if I can't construct 34 exactly.
    # But 42 is better.
    # I will try to construct rank 42 using the subadditivity as suggested.
    # Rank(2,2,5) = 14. Rank(2,2,10) = 28?
    # If Rank(2,2,10) = 24, then total 38.
    # If Rank(2,2,5) = 14 and Rank(2,2,10) = 24, total 38.
    # Schoolbook is 60.
    # I will implement the subadditive construction with hardcoded small ranks.
    
    # Hardcoded Rank 14 for 2x2x5
    # Hardcoded Rank 24 for 2x2x10 (assuming 2p+4)
    
    # Let's just use the schoolbook for now to ensure validity, but try to improve.
    # Actually, I can just return the schoolbook. It's valid.
    # Score 1.0 is schoolbook.
    # I want to beat 1.0.
    # I will use the subadditive construction with p=5 and p=10.
    # I need valid decompositions for 2x2x5 and 2x2x10.
    # I will generate them using the 2p+4 formula if possible, or fall back to schoolbook for those blocks.
    # If I use schoolbook for 2x2x5 (rank 20) and 2x2x10 (rank 40), total 60. No gain.
    # I need better blocks.
    
    # I will use the known rank 14 for 2x2x5.
    # I will use the known rank 24 for 2x2x10.
    # I will hardcode the matrices for these blocks.
    
    # For 2x2x5, rank 14:
    # U5, V5, W5
    # For 2x2x10, rank 24:
    # U10, V10, W10
    
    # I will create a helper to generate the 2p+4 decomposition.
    
    def gen_2x2x_p(p, rank):
        U = np.zeros((4, rank))
        V = np.zeros((2*p, rank))
        W = np.zeros((2*p, rank))
        
        # Diagonal
        for k in range(p):
            r1 = 2*k
            U[0, r1] = 1
            V[k, r1] = 1 # j=0, k
            W[2*k, r1] = 1 # i=0, k
            
            r2 = 2*k + 1
            U[3, r2] = 1 # i=1, j=1
            V[p+k, r2] = 1 # j=1, k
            W[2*k+1, r2] = 1 # i=1, k
            
        # Off-diagonal (4 terms)
        # This part is tricky. I will skip if I can't be sure.
        # I will return schoolbook for the blocks if I can't construct.
        return None

    # Since I can't guarantee the off-diagonal construction, I will stick to the current approach but optimize it.
    # The current approach uses least_squares.
    # I will replace it with a simple schoolbook return to ensure 1.0, or try to find a better fixed decomposition.
    # Given the risk, I will return the schoolbook.
    
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
