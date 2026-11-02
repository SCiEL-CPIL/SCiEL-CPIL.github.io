import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    n, m, p = DIMENSIONS
    
    # Trivial decomposition (Rank 45)
    rank_trivial = n * m * p
    U_triv = [[0] * rank_trivial for _ in range(n * m)]
    V_triv = [[0] * rank_trivial for _ in range(m * p)]
    W_triv = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    best_rank = rank_trivial
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reproduces the tensor."""
        # Check all specific points (i,j,k) -> (a,b,c)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if abs(val - 1) > 1e-10:
                        return False
        # Check all off-diagonal points
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    is_consistent = (i == i2 and j == j2 and k == k2)
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if is_consistent:
                        if abs(val - 1) > 1e-10:
                            return False
                    else:
                        if abs(val) > 1e-10:
                            return False
        return True
    
    def try_rank(rank, max_iter=5000):
        """Try to find a decomposition of given rank using random initialization + refinement."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Convert to list format for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list, rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 17 up to trivial rank
    # Based on known results, matrix multiplication tensors often have rank less than n*m*p
    # For 3x3x3, the minimum rank is 23 (Strassen)
    # For 3x3x5, we'll try a range of ranks
    for rank in range(17, min(40, rank_trivial + 1)):
        if time.time() > deadline:
            break
        
        result = try_rank(rank, max_iter=2000)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
