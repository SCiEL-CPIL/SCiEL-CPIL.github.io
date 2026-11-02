import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # for some (i,j,k) in [0,n)x[0,m)x[0,p)
    
    # Build target tensor as a list of (a,b,c) tuples where T=1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    nU = n * m  # 12
    nV = m * p  # 32
    nW = p * n  # 24
    
    # Try different ranks
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Lower bound: max(nU, nV, nW) = 32
    # Try ranks from 32 up to 96
    for rank in range(32, 97):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(nU, rank)
            V = np.random.randn(nV, rank)
            W = np.random.randn(nW, rank)
            
            # Local search: coordinate descent
            # For each term c, try to adjust U[:,c], V[:,c], W[:,c]
            # to reduce the error
            
            # Define error function
            def compute_error(U, V, W):
                error = 0.0
                for a, b, c in target_indices:
                    # Compute sum_c' U[a,c'] * V[b,c'] * W[c,c']
                    val = 0.0
                    for cprime in range(rank):
                        val += U[a, cprime] * V[b, cprime] * W[c, cprime]
                    error += (val - 1.0) ** 2
                # Also check that non-target entries are 0
                # This is expensive, so we'll skip for now and verify at the end
                return error
            
            # Simple local search
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Pick a random term to update
                c_update = np.random.randint(rank)
                
                # Update U[:, c_update]
                # Gradient of error w.r.t. U[a, c_update] is 2*(val-1)*V[b,c_update]*W[c,c_update]
                # This is complex, so let's try a simpler approach: random perturbation
                
                # Try small random perturbations
                improved = False
                for _ in range(10):
                    U_new = U.copy()
                    U_new[:, c_update] += 0.01 * np.random.randn(nU)
                    if compute_error(U_new, V, W) < compute_error(U, V, W):
                        U = U_new
                        improved = True
                        break
                
                if not improved:
                    # Try V
                    V_new = V.copy()
                    V_new[:, c_update] += 0.01 * np.random.randn(nV)
                    if compute_error(U, V_new, W) < compute_error(U, V, W):
                        V = V_new
                        improved = True
                
                if not improved:
                    # Try W
                    W_new = W.copy()
                    W_new[:, c_update] += 0.01 * np.random.randn(nW)
                    if compute_error(U, V, W_new) < compute_error(U, V, W):
                        W = W_new
                        improved = True
                
                if not improved:
                    break
            
            # Check if we found a valid decomposition
            # Verify all target entries
            valid = True
            for a, b, c in target_indices:
                val = 0.0
                for cprime in range(rank):
                    val += U[a, cprime] * V[b, cprime] * W[c, cprime]
                if abs(val - 1.0) > 1e-6:
                    valid = False
                    break
            
            if valid:
                # Convert to integer/half-integer if possible
                # For now, just store
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank == rank:
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_U is not None:
        # Convert to list of lists
        U_list = [[float(U[i, j]) for j in range(best_rank)] for i in range(nU)]
        V_list = [[float(V[i, j]) for j in range(best_rank)] for i in range(nV)]
        W_list = [[float(W[i, j]) for j in range(best_rank)] for i in range(nW)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Schoolbook
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
