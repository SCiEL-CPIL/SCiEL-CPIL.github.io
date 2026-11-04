"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin before 150s limit
    
    n, m, p = 3, 4, 7
    nmp = n * m * p  # 84 (schoolbook rank)
    
    # The schoolbook decomposition has rank 84
    # We want to find something better if possible
    
    # First, let's create the schoolbook decomposition as a fallback
    def make_schoolbook():
        rank = nmp
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
        return U, V, W
    
    # Fast verify using numpy broadcasting
    def verify(U, V, W):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # For each (i,j,k), check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = int(np.dot(
                        U_np[i * m + j, :] * V_np[j * p + k, :],
                        W_np[k * n + i, :]
                    ))
                    if total != 1:
                        return False
        return True
    
    best_rank = nmp
    best_U, best_V, best_W = make_schoolbook()
    
    random.seed(12345)
    np.random.seed(12345)
    
    # Try structured initializations first, then random
    # Strategy: use patterns inspired by Strassen-like methods
    
    # Try ranks from lower to upper
    # Start with a reasonable range
    target_ranks = list(range(50, 84))
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        # Try multiple strategies
        strategies = ['random', 'structured', 'mixed']
        
        for strategy in strategies:
            if time.time() > deadline - 15:
                break
            
            max_attempts = 30 if target_rank < 70 else 10
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                U = np.zeros((n * m, target_rank), dtype=int)
                V = np.zeros((m * p, target_rank), dtype=int)
                W = np.zeros((p * n, target_rank), dtype=int)
                
                if strategy == 'random':
                    # Random sparse initialization
                    for c in range(target_rank):
                        U[:, c] = np.random.choice([-1, 0, 1], size=n * m, p=[0.2, 0.6, 0.2])
                        V[:, c] = np.random.choice([-1, 0, 1], size=m * p, p=[0.2, 0.6, 0.2])
                        W[:, c] = np.random.choice([-1, 0, 1], size=p * n, p=[0.2, 0.6, 0.2])
                        
                elif strategy == 'structured':
                    # Structured: try to exploit the matrix multiplication structure
                    # Each rank-1 term should correspond to a "product" pattern
                    for c in range(target_rank):
                        # Choose which i, j, k indices to involve
                        i_idx = random.randint(0, n-1)
                        j_idx = random.randint(0, m-1)
                        k_idx = random.randint(0, p-1)
                        
                        # Set values with some randomness
                        U[i_idx * m + j_idx, c] = random.choice([-1, 1])
                        V[j_idx * p + k_idx, c] = random.choice([-1, 1])
                        W[k_idx * n + i_idx, c] = random.choice([-1, 1])
                        
                        # Sometimes add extra terms
                        if random.random() < 0.3:
                            i2 = random.randint(0, n-1)
                            j2 = random.randint(0, m-1)
                            k2 = random.randint(0, p-1)
                            U[i2 * m + j2, c] += random.choice([-1, 1])
                            V[j2 * p + k2, c] += random.choice([-1, 1])
                            W[k2 * n + i2, c] += random.choice([-1, 1])
                            
                else:  # mixed
                    # Mix of random and structured
                    for c in range(target_rank):
                        if random.random() < 0.5:
                            # Random
                            U[:, c] = np.random.choice([-1, 0, 1], size=n * m, p=[0.2, 0.6, 0.2])
                            V[:, c] = np.random.choice([-1, 0, 1], size=m * p, p=[0.2, 0.6, 0.2])
                            W[:, c] = np.random.choice([-1, 0, 1], size=p * n, p=[0.2, 0.6, 0.2])
                        else:
                            # Structured
                            i_idx = random.randint(0, n-1)
                            j_idx = random.randint(0, m-1)
                            k_idx = random.randint(0, p-1)
                            U[i_idx * m + j_idx, c] = random.choice([-1, 1])
                            V[j_idx * p + k_idx, c] = random.choice([-1, 1])
                            W[k_idx * n + i_idx, c] = random.choice([-1, 1])
                
                # Verify
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
            
            if best_rank < nmp:
                break
        
        if best_rank < nmp:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
