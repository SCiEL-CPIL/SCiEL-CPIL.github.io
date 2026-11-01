import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a better decomposition for 2x3x3 and duplicate it
    # Target rank for 2x3x3 is 11, so total rank 22
    # If we find rank 12, total rank 24
    # We will try to find a rank 11 decomposition for 2x3x3
    
    n2, m2, p2 = 2, 3, 3
    target_rank = 11
    
    # Create the tensor for 2x3x3
    # Indices: i in 0..1, j in 0..2, k in 0..2
    # T[i, j, k] is not directly the tensor, the tensor is T[(i,j), (j',k), (k',i')]
    # Let's define the tensor T of shape (n2*m2, m2*p2, p2*n2)
    # T[a, b, c] = 1 if a corresponds to (i,j), b to (j',k), c to (k',i') and j==j' and k==k'
    # a = i*m2 + j
    # b = j'*p2 + k
    # c = k'*n2 + i'
    
    T = np.zeros((n2 * m2, m2 * p2, p2 * n2), dtype=int)
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                a = i * m2 + j
                b = j * p2 + k
                c = k * n2 + i
                T[a, b, c] = 1
                
    # Try to find a decomposition of rank target_rank
    # We will use a simple random search with rounding
    # This is a heuristic and may not find the exact solution
    # Given the time constraint, we will limit the search
    
    best_U = None
    best_V = None
    best_W = None
    best_error = np.inf
    
    start_time = time.time()
    max_time = 10 # seconds
    
    # Random search
    for _ in range(1000):
        if time.time() - start_time > max_time:
            break
            
        U_rand = np.random.randint(-2, 3, size=(n2 * m2, target_rank))
        V_rand = np.random.randint(-2, 3, size=(m2 * p2, target_rank))
        W_rand = np.random.randint(-2, 3, size=(p2 * n2, target_rank))
        
        # Compute the tensor
        T_recon = np.zeros_like(T)
        for r in range(target_rank):
            T_recon += np.einsum('i,r,j,r,k,r->ijk', U_rand[:, r], V_rand[:, r], W_rand[:, r])
            
        error = np.sum((T - T_recon) ** 2)
        if error < best_error:
            best_error = error
            best_U = U_rand.copy()
            best_V = V_rand.copy()
            best_W = W_rand.copy()
            
            if error == 0:
                break
                
    if best_error == 0:
        # Construct the 2x3x6 decomposition
        # U is 6x11, V is 9x11, W is 6x11 for 2x3x3
        # For 2x3x6, we duplicate this for the two halves of p
        # U_236 is 6x22, V_236 is 18x22, W_236 is 12x22
        rank_236 = target_rank * 2
        U_236 = np.zeros((n * m, rank_236), dtype=int)
        V_236 = np.zeros((m * p, rank_236), dtype=int)
        W_236 = np.zeros((p * n, rank_236), dtype=int)
        
        # First half
        U_236[:, 0:target_rank] = best_U
        V_236[0:m2*p2, 0:target_rank] = best_V
        W_236[0:p2*n2, 0:target_rank] = best_W
        
        # Second half
        U_236[:, target_rank:2*target_rank] = best_U
        V_236[m2*p2:m2*p2+m2*p2, target_rank:2*target_rank] = best_V
        W_236[p2*n2:p2*n2+p2*n2, target_rank:2*target_rank] = best_W
        
        return {"U": U_236.tolist(), "V": V_236.tolist(), "W": W_236.tolist()}
    else:
        # Return schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
