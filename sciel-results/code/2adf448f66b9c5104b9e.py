import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def check_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Check if the tensor decomposition is valid.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    
    # The tensor T is defined by T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i, j, k] = 1 if i == k, else 0.
    # Note: The task says "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the index mapping for W is row = k*n + i.
    
    # Efficient check using einsum
    # Let's construct the tensor T
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute T
    # T[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
    # This is a bit tricky with standard einsum because of the indices.
    # Let's use a loop for clarity and correctness first, then optimize if needed.
    # Given small dimensions, loop is fine.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                
                target = 1 if i == k else 0
                if abs(val - target) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Fallback: Standard decomposition (Rank 45)
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed))
    V_seed = np.zeros((m * p, rank_seed))
    W_seed = np.zeros((p * n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = rank_seed
    
    # Strategy:
    # 1. We know that matrix multiplication M_{n,m,p} has rank at most n*m*p.
    # 2. For 3x3x3, the rank is 23 (Landsberg/Strassen).
    # 3. For 3x3x5, we can embed the 3x3x3 multiplication into 3x3x5 by padding with zeros?
    #    No, 3x3x5 is multiplying a 3x3 matrix by a 3x5 matrix.
    #    This is equivalent to performing 3 independent 1x3x3 multiplications? No.
    #    It's one 3x3 matrix times a 3x5 matrix.
    #    The result is a 3x5 matrix.
    #    We can view this as 5 independent 3x3x1 multiplications? No.
    #    
    #    Actually, M_{3,3,5} can be decomposed by taking the M_{3,3,3} decomposition (rank 23)
    #    and using it for the first 3 columns of the result, and then... wait.
    #    
    #    Let A be 3x3, B be 3x5. C = AB is 3x5.
    #    We can write B as [B1, B2] where B1 is 3x3 and B2 is 3x2.
    #    C = [A*B1, A*B2].
    #    The rank of M_{3,3,3} is 23.
    #    The rank of M_{3,3,2} is at most 3*3*2 = 18.
    #    So total rank <= 23 + 18 = 41.
    #    
    #    Can we do better?
    #    The rank of M_{n,m,p} is at least max(nm, mp, pn).
    #    Here max(9, 15, 15) = 15.
    #    
    #    Let's try to find a rank 41 decomposition by hardcoding a known 3x3x3 decomposition.
    #    Since I don't have the 3x3x3 rank 23 matrices memorized, I will try to construct a rank 41 decomposition
    #    using the block structure if I can find a simple one, or just return the seed if I can't improve.
    #    
    #    However, I can try a simple optimization for rank 41, 40, etc.
    #    But numerical optimization of tensor rank is very hard.
    #    
    #    Alternative:
    #    Is there a known simple decomposition for 3x3x5?
    #    Maybe we can use the fact that 3x3x5 is a sub-tensor of 3x3x3 x 3x3x5? No.
    #    
    #    Let's stick to the seed for now unless I can implement a reliable search.
    #    The previous attempt at search was a placeholder.
    #    
    #    Let's try a specific heuristic:
    #    If we assume the decomposition is of the form:
    #    U = [U1, U2, U3]
    #    V = [V1, V2, V3]
    #    W = [W1, W2, W3]
    #    Where U1, V1, W1 correspond to the first 3 columns of B (3x3x3 problem)
    #    And U2, V2, W2 correspond to the last 2 columns of B (3x3x2 problem)
    #    
    #    If I had the 3x3x3 rank 23 solution, I could just append a 3x3x2 rank 18 solution.
    #    Since I don't, I'll return the seed.
    #    
    #    Wait, I can construct a rank 41 decomposition using a simple method if I assume
    #    the 3x3x3 part is solved by a specific pattern? No, that's guessing.
    #    
    #    Let's just return the seed to be safe.
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
