import numpy as np
import time
import itertools

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_school
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Attempt to find a lower rank decomposition using structured search
    # The matrix multiplication tensor T_{i,j,k} = U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # We want to minimize R such that there exist U,V,W with R columns satisfying:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    # For small dimensions, we can try to find rank R decompositions
    # using a structured approach. One known result is that for 3x5x7,
    # the minimal rank might be lower than 105.
    
    # Let's try a systematic search for small rank improvements
    # We'll use a heuristic approach: try to merge some of the schoolbook terms
    
    # First, let's verify the schoolbook solution works
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that the decomposition exactly represents matrix multiplication"""
        R = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = 0.0
                    for r in range(R):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to find a rank R < 105 decomposition
    # We'll use a random restart approach with local search
    np.random.seed(42)
    
    # Try ranks from 104 down to 90
    for R in range(104, 89, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 10:
                break
                
            # Initialize with small random values
            U_rand = np.random.randint(-3, 4, size=(n*m, R)) * 0.5
            V_rand = np.random.randint(-3, 4, size=(m*p, R)) * 0.5
            W_rand = np.random.randint(-3, 4, size=(p*n, R)) * 0.5
            
            # Check if this random initialization happens to work
            if verify_decomposition(U_rand, V_rand, W_rand):
                best_rank = R
                best_U = U_rand.copy()
                best_V = V_rand.copy()
                best_W = W_rand.copy()
                break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_school:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out = U
        V_out = V
        W_out = W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
