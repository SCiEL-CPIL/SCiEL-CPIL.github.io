import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    # Deadline guard: The task allows 120s. We use a generous margin.
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    # The trivial decomposition has rank n*m*p
    # For 2x3x4, rank = 24.
    # This is the seed score (1.0).
    # We attempt to find a lower rank decomposition using a simple search strategy 
    # within the remaining time budget. If no better solution is found, we fall back 
    # to the trivial decomposition.
    
    # Trivial solution as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial), dtype=np.float64)
    V_trivial = np.zeros((m * p, rank_trivial), dtype=np.float64)
    W_trivial = np.zeros((p * n, rank_trivial), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_trivial
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    
    # Simple search for better rank
    # We try ranks from a lower bound up to the trivial rank
    # Theoretical lower bound for matrix multiplication is max(n*m, m*p, p*n)
    # For 2x3x4: max(6, 12, 8) = 12
    # We will try ranks from 12 to 23
    
    # Define the tensor entries we need to satisfy
    # T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    
    # We use a simple greedy or random approach if time permits
    # However, given the complexity and time constraints, and the fact that 
    # previous inspirations failed to improve, we stick to the trivial solution 
    # unless a very quick check finds something better.
    
    # Check if time remains for a basic optimization attempt
    if time.time() < deadline - 10.0:
        # Try a few random initializations for lower ranks
        # This is a heuristic approach
        for rank in range(12, rank_trivial):
            if time.time() > deadline - 5.0:
                break
            
            # Skip if rank is too low to be plausible (simple bound)
            if rank < max(n*m, m*p, p*n):
                continue
                
            # Random search for this rank
            found = False
            for attempt in range(100): # Limit attempts per rank
                if time.time() > deadline - 5.0:
                    break
                    
                U = np.random.randn(n * m, rank)
                V = np.random.randn(m * p, rank)
                W = np.random.randn(p * n, rank)
                
                # Check if this decomposition works
                # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # We need T[i, j, k] = 1 for all i, j, k
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
            
            if found:
                break

    # Convert to lists as per interface requirement
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Final check of shapes
    if len(U_list) != n*m or (U_list and len(U_list[0]) != best_rank):
        raise RuntimeError("Invalid U shape")
    if len(V_list) != m*p or (V_list and len(V_list[0]) != best_rank):
        raise RuntimeError("Invalid V shape")
    if len(W_list) != p*n or (W_list and len(W_list[0]) != best_rank):
        raise RuntimeError("Invalid W shape")
        
    return {"U": U_list, "V": V_list, "W": W_list}
