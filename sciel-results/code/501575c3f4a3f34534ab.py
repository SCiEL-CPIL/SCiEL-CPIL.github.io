import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Target tensor indices
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    # Try to find a lower rank decomposition
    # We know rank is likely 15. Try 15, then 16.
    # If we find one, great.
    
    def check_decomposition(U, V, W, rank):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Check if sum_c U[a,c] V[b,c] W[c',c] == T[a,b,c']
        # T is 1 at target_indices, 0 elsewhere.
        # Since T is sparse, we only need to check:
        # 1. At target indices, the sum is 1.
        # 2. At all other indices, the sum is 0.
        # Checking all other indices is expensive if we do it naively.
        # But the tensor is small: 6 * 9 * 6 = 324 entries.
        # We can just compute the full tensor product and compare.
        
        # Compute T_decomp
        # T_decomp[a, b, c'] = sum_c U[a, c] V[b, c] W[c', c]
        # This is a tensor of size 6x9x6.
        
        T_decomp = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            T_decomp += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
            
        # Check target indices
        for a, b, c_prime in target_indices:
            if abs(T_decomp[a, b, c_prime] - 1.0) > 1e-9:
                return False
        
        # Check non-target indices
        # Create a mask of target indices
        T_target = np.zeros((n*m, m*p, p*n))
        for a, b, c_prime in target_indices:
            T_target[a, b, c_prime] = 1.0
            
        # Check if T_decomp matches T_target everywhere
        if not np.allclose(T_decomp, T_target, atol=1e-9):
            return False
            
        return True

    def round_to_half_integer(M):
        # Round to nearest 0.5
        return np.round(2 * M) / 2.0

    # Search for rank 15
    start_time = time.time()
    deadline = start_time + 120  # Leave 30s margin
    
    for target_rank in [15, 16]:
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Simple gradient descent or just check if random works? No.
            # We need to optimize.
            # Let's use a few steps of gradient descent.
            
            # Loss: || T_decomp - T_target ||_F^2
            # Gradient w.r.t U[a, c]: sum_{b, c'} (T_decomp[a, b, c'] - T_target[a, b, c']) V[b, c] W[c', c]
            
            T_target = np.zeros((n*m, m*p, p*n))
            for a, b, c_prime in target_indices:
                T_target[a, b, c_prime] = 1.0
                
            for step in range(100):
                if time.time() > deadline:
                    break
                    
                T_decomp = np.zeros((n*m, m*p, p*n))
                for c in range(target_rank):
                    T_decomp += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
                    
                err = T_decomp - T_target
                loss = np.sum(err**2)
                
                if loss < 1e-10:
                    break
                    
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for c in range(target_rank):
                    # dU[a, c] = sum_{b, c'} err[a, b, c'] V[b, c] W[c', c]
                    # dV[b, c] = sum_{a, c'} err[a, b, c'] U[a, c] W[c', c]
                    # dW[c', c] = sum_{a, b} err[a, b, c'] U[a, c] V[b, c]
                    
                    # Efficiently using einsum
                    dU[:, c] = np.einsum('abc,bc->a', err, V[:, c] * W[:, c]) # Wait, V[:, c] is vector b, W[:, c] is vector d.
                    # err is (a, b, d). V[:, c] is (b). W[:, c] is (d).
                    # sum_{b, d} err[a, b, d] V[b, c] W[d, c]
                    dU[:, c] = np.einsum('abd,b,d->a', err, V[:, c], W[:, c])
                    
                    dV[:, c] = np.einsum('abd,a,d->b', err, U[:, c], W[:, c])
                    
                    dW[:, c] = np.einsum('abd,a,b->d', err, U[:, c], V[:, c])
                    
                # Update
                lr = 0.01
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
            # Round and check
            U_r = round_to_half_integer(U)
            V_r = round_to_half_integer(V)
            W_r = round_to_half_integer(W)
            
            if check_decomposition(U_r, V_r, W_r, target_rank):
                # Convert to lists
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
