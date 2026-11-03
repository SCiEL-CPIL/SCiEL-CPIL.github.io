import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a better decomposition
    # Target rank: 32 (23 for 3x3x3 + 9 for 3x3x2)
    # We will try to find a decomposition of rank R
    # using a heuristic search
    
    def validate(U, V, W):
        # Check if U, V, W exactly decompose the tensor
        # T[i,j,k] = 1 for all i,j,k
        # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        n, m, p = DIMENSIONS
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # Compute the tensor
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can use einsum
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
        # Use einsum: 'ijr,jkr,kr->ijk' ? No, W is (p, n, rank) -> k, n, r
        # W_r[k, i, r]
        # So 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # Check if T is all ones
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def try_rank(R):
        # Random initialization
        # We want integer/half-integer entries
        # Start with random integers in [-2, 2]
        U = np.random.randint(-2, 3, size=(n * m, R))
        V = np.random.randint(-2, 3, size=(m * p, R))
        W = np.random.randint(-2, 3, size=(p * n, R))
        
        # Validate
        if validate(U, V, W):
            return U, V, W
        return None

    # Try ranks from 30 to 44
    for R in range(30, 45):
        if time.time() > deadline:
            break
        # Try a few random initializations
        for _ in range(100):
            if time.time() > deadline:
                break
            res = try_rank(R)
            if res is not None:
                U, V, W = res
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break
        if best_rank < rank_sb:
            break
            
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
