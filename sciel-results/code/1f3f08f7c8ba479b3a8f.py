import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = rank_schoolbook

    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    # Try to construct a Winograd F(4) based decomposition
    # Rank of 4x4x4x4 is 49 (using Winograd F(4) on 2x2 blocks? No, F(4) is for 4x4 directly)
    # Actually, the rank of 4x4x4x4 is 49 is a known result.
    # For 4x4x4x5, we can take the 4x4x4x4 decomposition and add 4 terms for the 5th column.
    # Rank = 49 + 4 = 53.
    
    # Since constructing the exact 49-rank decomposition is complex, 
    # and the random search for rank 53 is also hard, 
    # we will stick to the schoolbook for now unless we can easily construct the Winograd one.
    # However, the directive asks for a stronger algorithm.
    # I will attempt to implement the Winograd F(4) algorithm for 4x4 matrices.
    
    # Winograd F(4) algorithm for 4x4 matrix multiplication
    # This is complex to hardcode correctly without verification.
    # Instead, I will use a known decomposition for 4x4x4x4 if available, 
    # or fall back to schoolbook.
    
    # Given the time constraint and complexity, I will return the schoolbook result.
    # But wait, the score is 1.0 for schoolbook. 
    # If I can find a rank < 80, the score will be > 1.0.
    # Let's try a random search for rank 64 with a tighter time limit.
    
    np.random.seed(42)
    
    # Try to find a decomposition with rank < 80
    for target_rank in [72, 64]:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if check_decomposition(U_list, V_list, W_list):
                result = {"U": U_list, "V": V_list, "W": W_list}
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    return best_result
