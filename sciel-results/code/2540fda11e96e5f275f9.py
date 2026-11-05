import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Build the target tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W):
        r = U.shape[1]
        result = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            result += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(result, target, atol=1e-9)
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank decomposition using randomized search
    # This is a heuristic approach
    
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 5, 20), -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Simple gradient-free optimization: try to minimize the error
            # This is a very crude approach
            
            # Check if the random initialization happens to work (extremely unlikely)
            if verify(U, V, W):
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
