import time
import numpy as np

def construct():
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Dimensions
    dim_u = n * m  # 9
    dim_v = m * p  # 15
    dim_w = p * n  # 15
    
    # Build target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_solution(U, V, W, T):
        # Compute P = sum_k U[:,k] outer V[:,k] outer W[:,k]
        # P[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        # Check exact match
        if not np.allclose(P, T, atol=1e-9):
            return False
        # Check if entries are half-integers
        # Multiply by 2 and check if close to integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if not np.allclose(U2, np.round(U2), atol=1e-6): return False
        if not np.allclose(V2, np.round(V2), atol=1e-6): return False
        if not np.allclose(W2, np.round(W2), atol=1e-6): return False
        return True

    def snap_to_half_integers(mat):
        return np.round(mat * 2) / 2.0

    # Schoolbook solution as fallback
    def get_schoolbook():
        r = n * m * p
        U = np.zeros((dim_u, r))
        V = np.zeros((dim_v, r))
        W = np.zeros((dim_w, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W

    best_U, best_V, best_W = get_schoolbook()
    best_rank = n * m * p

    # Try to find lower rank solutions
    # We'll try ranks from 25 to 40
    # For each rank, we do a few random restarts
    
    for target_rank in range(25, 41):
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + target_rank + attempt)
            # Initialize with small random values
            U = np.random.randn(dim_u, target_rank) * 0.5
            V = np.random.randn(dim_v, target_rank) * 0.5
            W = np.random.randn(dim_w, target_rank) * 0.5
            
            # Optimization
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 10:
                    break
                    
                # Forward
                P = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = P - T
                loss = np.sum(diff**2)
                
                if loss < 1e-10:
                    break
                
                # Gradients
                grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
                grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
                grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if step % 100 == 0 and step > 0:
                    if loss < 1e-6:
                        lr *= 0.5
                    else:
                        lr *= 0.8

            # Snap and verify
            U_snap = snap_to_half_integers(U)
            V_snap = snap_to_half_integers(V)
            W_snap = snap_to_half_integers(W)
            
            if verify_solution(U_snap, V_snap, W_snap, T):
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                best_rank = target_rank
                found = True
                break
        
        if found:
            break

    # If we found a better solution, use it. Otherwise, schoolbook.
    # Note: The schoolbook solution is already valid half-integers.
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
