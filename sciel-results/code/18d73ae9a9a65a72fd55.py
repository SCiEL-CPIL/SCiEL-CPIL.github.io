"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nU = n * m  # 6
    nV = m * p  # 21
    nW = p * n  # 14
    
    # Build the target tensor (flattened)
    target = np.zeros((nU, nV, nW))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    def compute_T(U, V, W, rank):
        """Compute the reconstructed tensor."""
        T = np.zeros((nU, nV, nW))
        for t in range(rank):
            T += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        return T
    
    def check_valid(U, V, W, rank, tol=1e-8):
        """Check if the decomposition is valid."""
        T = compute_T(U, V, W, rank)
        return np.allclose(T, target, atol=tol)
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nU, rank_school))
    V_school = np.zeros((nV, rank_school))
    W_school = np.zeros((nW, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    start_time = time.time()
    deadline = start_time + 125  # Safety margin
    
    # Try to find a lower rank decomposition
    for rank in range(21, rank_school):
        if time.time() > deadline - 3:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 1:
                break
            
            np.random.seed(rank * 1000 + attempt * 100)
            U = np.random.randn(nU, rank) * 0.3
            V = np.random.randn(nV, rank) * 0.3
            W = np.random.randn(nW, rank) * 0.3
            
            lr = 0.02
            for iteration in range(100):
                if time.time() > deadline - 0.5:
                    break
                
                T_rec = compute_T(U, V, W, rank)
                error = T_rec - target
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-7:
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for t in range(rank):
                    dU[:,t] = np.einsum('abc,bc->a', error, V[:,t] * W[:,t])
                    dV[:,t] = np.einsum('abc,ac->b', error, U[:,t] * W[:,t])
                    dW[:,t] = np.einsum('abc,ab->c', error, U[:,t] * V[:,t])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if check_valid(U, V, W, rank):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
    
    # Convert to list format
    U_list = [[best_U[i, j] for j in range(best_rank)] for i in range(nU)]
    V_list = [[best_V[i, j] for j in range(best_rank)] for i in range(nV)]
    W_list = [[best_W[i, j] for j in range(best_rank)] for i in range(nW)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
