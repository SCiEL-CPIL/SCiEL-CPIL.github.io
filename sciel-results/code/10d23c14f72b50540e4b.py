import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook initialization
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Try to find a lower rank
    # This is a heuristic search
    # We try ranks from best_rank - 1 down to a limit
    # Due to time constraints, we only try a few
    start_time = time.time()
    limit_time = 10.0
    
    for rank in range(best_rank - 1, max(50, best_rank - 10), -1):
        if time.time() - start_time > limit_time:
            break
        
        # Try random initialization
        for attempt in range(10):
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Check if this decomposition is valid?
            # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{j,j'} delta_{i,i''} delta_{k',k''}
            # This is a tensor equality.
            # We can check a few random points.
            
            # Construct the tensor from U,V,W
            # T_reconstructed[i, j, j', k', i'', k''] = sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i'', r]
            # We want T_reconstructed == T_target
            # T_target[i, j, j', k', i'', k''] = 1 if j==j' and i==i'' and k==k' else 0
            
            # This is expensive to check fully.
            # We can check if the rank of the tensor is low?
            # Actually, we can just check if the decomposition satisfies the equations for a random subset of indices.
            
            valid = True
            for _ in range(100):
                i = np.random.randint(n)
                j = np.random.randint(m)
                jp = np.random.randint(m)
                k = np.random.randint(p)
                ip = np.random.randint(n)
                kp = np.random.randint(p)
                
                # Target value
                if j == jp and i == ip and k == kp:
                    target = 1.0
                else:
                    target = 0.0
                
                # Reconstructed value
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                
                if abs(val - target) > 1e-6:
                    valid = False
                    break
            
            if valid:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break

    # Convert to list of lists of floats (or ints if close)
    def to_list(arr):
        return arr.tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
