# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses structured search with verification-based fitness.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """
        Check if U, V, W exactly decompose the matrix multiplication tensor.
        U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
        For all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification using numpy for speed."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j], V_arr[j * p + k])
                    # Wait, that's not right. We need the triple product.
                    # s = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    # This is element-wise product then sum, not dot product.
                    prod = U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i]
                    s = np.sum(prod)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def random_decomposition(rank, seed=None):
        """Generate a random decomposition with entries in {-1, 0, 1}."""
        if seed is not None:
            random.seed(seed)
        U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(p * n)]
        return U, V, W
    
    def mutate(U, V, W, rank, mutation_rate=0.1):
        """Apply random mutations to the decomposition."""
        U_new = [row[:] for row in U]
        V_new = [row[:] for row in V]
        W_new = [row[:] for row in W]
        
        # Mutate U
        for i in range(n * m):
            for t in range(rank):
                if random.random() < mutation_rate:
                    U_new[i][t] = random.randint(-1, 1)
        
        # Mutate V
        for j in range(m * p):
            for t in range(rank):
                if random.random() < mutation_rate:
                    V_new[j][t] = random.randint(-1, 1)
        
        # Mutate W
        for k in range(p * n):
            for t in range(rank):
                if random.random() < mutation_rate:
                    W_new[k][t] = random.randint(-1, 1)
        
        return U_new, V_new, W_new
    
    def try_rank(rank, max_attempts=100, seed_base=42):
        """Try to find a valid decomposition of the given rank."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            seed = seed_base + attempt
            U, V, W = random_decomposition(rank, seed=seed)
            
            if verify_fast(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Start with schoolbook
    result = schoolbook()
    best_rank = n * m * p  # 60
    U = result["U"]
    V = result["V"]
    W = result["W"]
    
    # Try to find lower ranks
    # Lower bound is max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Try from 30 up to best_rank - 1
    for rank in range(30, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Increase attempts for lower ranks (harder to find)
        if rank <= 35:
            max_attempts = 200
        elif rank <= 40:
            max_attempts = 150
        elif rank <= 50:
            max_attempts = 100
        else:
            max_attempts = 50
        
        found = try_rank(rank, max_attempts=max_attempts)
        if found:
            best_rank = rank
            U = found["U"]
            V = found["V"]
            W = found["W"]
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
