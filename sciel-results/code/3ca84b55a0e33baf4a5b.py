import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor is all ones
    # Shape of U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Try ranks from 16 to 30
    for r in range(16, 31):
        # Time budget per rank
        start_time = time.time()
        max_time = 10.0
        
        for attempt in range(100):
            if time.time() - start_time > max_time:
                break
                
            # Initialize with random integers or half-integers
            # Using small integers to keep numbers manageable
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            W = np.random.randint(-2, 3, size=(p*n, r))
            
            # Check if this random guess is already good (unlikely)
            # Compute the tensor sum
            # This is expensive to do naively, but n,m,p are small
            # We can use einsum
            # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Reshape for einsum
            # U is (n*m, r) -> (n, m, r)
            # V is (m*p, r) -> (m, p, r)
            # W is (p*n, r) -> (p, n, r) -> (n, p, r) is better for output C[i, k]?
            # W index is k*n+i. So W[k, i] in a (p, n) matrix.
            # We want C[i, k] = sum_j sum_c ...
            # Actually, the decomposition is for the tensor T[i, j, k] = 1.
            # The condition is sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1?
            # Wait, W index is k*n+i. So W[k, i].
            # The term is U[i, j, c] * V[j, k, c] * W[k, i, c].
            
            U_3d = U.reshape(n, m, r)
            V_3d = V.reshape(m, p, r)
            W_3d = W.reshape(p, n, r) # W[k, i]
            
            # Sum over c
            # Result[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
            # Note: W is indexed by (k, i).
            
            # We can compute this using einsum
            # 'ijc,jkc,kic->ijk'
            try:
                T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            except:
                continue
                
            # Check if T is all ones
            if np.allclose(T, 1.0, atol=1e-6):
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
            # If not, we could try a local search, but for simplicity, we just do more random restarts
            # A simple refinement: adjust one entry at a time?
            # Given time constraints, just more random restarts is safer.
            
    # Fallback: Schoolbook rank 40
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
