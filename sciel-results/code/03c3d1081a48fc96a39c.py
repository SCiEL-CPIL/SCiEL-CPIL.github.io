import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 6
    
    # Schoolbook decomposition as guaranteed valid fallback
    rank_schoolbook = n * m * p  # 96
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    def verify(U_mat, V_mat, W_mat, rank):
        """Check if decomposition is exact.
        U_mat: (n*m, rank)
        V_mat: (m*p, rank)
        W_mat: (p*n, rank)
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using random search / optimization
    # Strategy: Use a tensor train / SVD-based approach or random initialization with gradient descent
    # But given time constraints, let's try a structured approach first.
    
    # Approach: Split the 4x4x6 multiplication into blocks
    # We can view it as multiplying a 4x4 matrix A by a 4x6 matrix B
    # C = A * B where C is 4x6
    
    # One structural idea: Use the fact that we can decompose based on row/column groups
    # For 4x4x4, Strassen gives rank 49. For 4x4x6, we need something different.
    
    # Let's try a random search approach with verification
    # Initialize with random low-rank tensors and try to optimize
    
    def try_random_decomposition(rank, num_trials=50):
        """Try to find a valid decomposition of given rank using random initialization."""
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                return None
            
            # Random initialization with values in {-1, 0, 1} or {-1, 0.5, 0, 0.5, 1}
            U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            # Quick check using numpy for speed
            # The condition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
            # Let's vectorize this check
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.sum(U_rand[i*m+j] * V_rand[j*p+k] * W_rand[k*n+i])
                        if abs(total - 1.0) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U_rand.tolist(), V_rand.tolist(), W_rand.tolist()
        return None
    
    # Try ranks from 70 down to 96 (schoolbook)
    # Start with some promising ranks
    # Known results: 4x4x4 can be done in 49, so 4x4x6 might be doable in less than 96
    
    # Let's try a more systematic approach: use the structure of matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Another approach: try to find a decomposition using integer coefficients
    # by solving a system of equations
    
    # For now, let's try random search for ranks 80, 84, 88, 92
    for rank in [80, 84, 88, 92]:
        if time.time() > deadline - 10.0:
            break
        
        result = try_random_decomposition(rank, num_trials=100)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list, rank):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break
    
    # If random search didn't find anything, return schoolbook
    return best_result
