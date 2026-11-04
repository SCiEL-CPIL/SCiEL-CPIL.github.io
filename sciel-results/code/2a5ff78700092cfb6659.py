"""Matrix multiplication 3,3,5 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 15, 15
    
    # Build the target tensor T (flattened)
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    def verify(U, V, W, tol=1e-10):
        """Check if U,V,W exactly decompose T using vectorized einsum."""
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.max(np.abs(P - T)) < tol
    
    def score(U, V, W):
        """Compute the max error."""
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.max(np.abs(P - T))
    
    # Try to find a lower-rank decomposition
    best_result = {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
    best_rank = rank_school
    
    # Attempt optimization for rank < 45
    # We try ranks from 44 down to 30, with a few trials each
    for target_rank in range(rank_school - 1, max(rank_school - 15, 28), -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(4):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Gradient descent with vectorized operations
            lr = 0.02
            for step in range(300):
                if time.time() > deadline:
                    break
                
                # Compute reconstruction and error
                P = np.einsum('ak,bk,ck->abc', U, V, W)
                err = P - T
                loss = np.sum(err ** 2)
                
                if loss < 1e-14:
                    break
                
                # Compute gradients using vectorized einsum
                # dU[a,k] = sum_{b,c} err[a,b,c] * V[b,k] * W[c,k]
                dU = np.einsum('abc,bk,ck->ak', err, V, W)
                # dV[b,k] = sum_{a,c} err[a,b,c] * U[a,k] * W[c,k]
                dV = np.einsum('abc,ak,ck->bk', err, U, W)
                # dW[c,k] = sum_{a,b} err[a,b,c] * U[a,k] * V[b,k]
                dW = np.einsum('abc,ak,bk->ck', err, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if time.time() > deadline:
                break
            
            # Check if we found an exact solution
            if verify(U, V, W):
                best_result = {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
