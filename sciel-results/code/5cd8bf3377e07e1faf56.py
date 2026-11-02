import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_U, best_V, best_W, best_r = U_sb, V_sb, W_sb, r_sb
    
    # Function to verify if a decomposition is valid (integer/half-integer and exact)
    def verify(U, V, W):
        # Check if entries are half-integers
        # Multiply by 2 and check if integers
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        
        # Check exactness
        # T_decomp = sum_k u_k v_k w_k
        # Using einsum: 'ak,bk,ck->abc'
        T_decomp = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_decomp, T, atol=1e-9)

    # Function to try to find a decomposition of rank r
    def try_rank(r, time_limit):
        if time.time() > time_limit:
            return None
        
        # Try a few random restarts
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > time_limit:
                break
            
            np.random.seed(restart * 1000 + r)
            # Initialize with small random values
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Gradient Descent
            lr = 0.01
            for it in range(500):
                if time.time() > time_limit:
                    break
                
                # Forward pass
                T_decomp = np.einsum('ak,bk,ck->abc', U, V, W)
                E = T_decomp - T
                loss = np.sum(E**2)
                
                if loss < 1e-8:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    break
                
                # Backward pass (Gradients)
                # dL/dU = 2 * E * V * W (contracted over b, c)
                # grad_U[a,k] = 2 * sum_{b,c} E[a,b,c] V[b,k] W[c,k]
                grad_U = 2 * np.einsum('abc,bk,ck->ak', E, V, W)
                grad_V = 2 * np.einsum('abc,ak,ck->bk', E, U, W)
                grad_W = 2 * np.einsum('abc,ak,bk->ck', E, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Simple adaptive learning rate: if loss increases, reduce lr
                # (Simplified: just keep lr fixed or reduce slowly)
                if it % 50 == 0:
                    lr *= 0.9
                    
        return None

    # Search for lower ranks
    # We start from r_sb - 1 down to some lower bound
    # Lower bound is at least max(nm, mp, pn) = 35
    # But we don't want to search too many if it's hard.
    # Let's try 104 down to 90.
    
    current_deadline = start_time + 110
    
    for r in range(r_sb - 1, 89, -1):
        if time.time() > current_deadline - 20: # 20s margin
            break
        
        # Allocate less time for higher ranks (likely to fail) and more for lower?
        # Actually, if a lower rank exists, it might be hard to find.
        # Let's give each rank 5 seconds.
        res = try_rank(r, time.time() + 5)
        if res is not None:
            best_U, best_V, best_W, best_r = res
            break # Found the lowest rank in this sequence
            
    # If we didn't find anything lower, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
