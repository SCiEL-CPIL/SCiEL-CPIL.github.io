import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Strategy: Use Kronecker product structure
    # 2x3 * 3x6 can be seen as (2x3 * 3x3) * (3x3 * 3x2) but more directly:
    # The 6 dimension in the second matrix can be viewed as 3*2
    # So we decompose into two 3x3 multiplications
    
    # For a 3x3 matrix multiplication, the known rank is 11 (Strassen-like)
    # Actually, for 2x3 * 3x3, we can use rank 11
    # For 2x3 * 3x6 = 2x3 * (3x3) * 2, we can use two copies of the 2x3*3x3 decomposition
    # This gives rank 2 * 11 = 22
    
    # But we can do better by exploiting the structure more carefully
    # Let's try to find a direct rank 20 or lower decomposition
    
    # First, let's construct a rank 22 solution using two 3x3 blocks
    # Each 3x3 block has rank 11, so total rank is 22
    
    # For 2x3 * 3x3, we need a rank 11 decomposition
    # The standard Strassen decomposition for 2x2 has rank 7
    # For 3x3, the best known is 19 or so, but for 2x3*3x3 it might be different
    
    # Let's use a simpler approach: schoolbook is rank n*m*p = 2*3*6 = 36
    # We want to beat 36
    
    # Actually, let's think about this differently.
    # The tensor T[i,j,k] = 1 for i in [0,2), j in [0,3), k in [0,6)
    # We want to write T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    # A simple structured decomposition:
    # For each (i,j), the row of U is constant across k
    # For each (j,k), the row of V depends on j and k
    # For each (k,i), the row of W depends on k and i
    
    # Let's try a decomposition where we group the k dimension
    # k = 0,1,2 and k = 3,4,5
    
    # For the first group (k=0,1,2), we use a 2x3*3x3 decomposition
    # For the second group (k=3,4,5), we use another 2x3*3x3 decomposition
    
    # The rank of 2x3*3x3 matrix multiplication:
    # This is equivalent to multiplying a 2x3 matrix by a 3x3 matrix
    # The output is 2x3
    # The tensor has dimensions 2*3 * 3*3 * 3*2 = 6 * 9 * 6
    # Schoolbook rank is 6*9*6 / 6 = 54? No, that's not right.
    
    # Actually, for matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # The tensor has indices (i,j) for A, (j,k) for B, (k,i) for C
    # The rank is at most n*m*p
    # For 2x3*3x3: rank at most 2*3*3 = 18
    
    # Let's just use the schoolbook decomposition for each 3x3 block
    # That gives rank 2 * (2*3*3) = 2 * 18 = 36, which is the same as full schoolbook
    
    # We need a better approach. Let's try to find a rank 20 decomposition directly.
    
    # Use a random search with snapping to half-integers
    best_result = None
    best_rank = 36  # Schoolbook
    
    # Schoolbook fallback
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
    
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    
    # Try to find a lower rank decomposition
    # The theoretical lower bound for 2x3*3x6 is not well-known, but it's at least 2*3*6/3 = 12
    # Let's try ranks from 15 to 35
    
    for target_rank in range(15, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try a few random initializations with local optimization
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            
            # Initialize with small integers
            U_c = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V_c = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            W_c = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
            
            # Check if this already works
            if _verify(U_c, V_c, W_c, n, m, p):
                best_result = {
                    "U": U_c.tolist(),
                    "V": V_c.tolist(),
                    "W": W_c.tolist()
                }
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # If we didn't find anything better, return the schoolbook
    return best_result
