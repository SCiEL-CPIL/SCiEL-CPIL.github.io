import time
import numpy as np
import random

N, M, P = 2, 3, 10

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # 1. Initial Construction: Kronecker Product / Structural Decomposition
    # We decompose the multiplication T_{2,3,10} into the tensor product of:
    # T_{2,1,1} (Rank 2: Identity on the first dimension)
    # T_{1,3,10} (Rank 30: Scalar multiplication, effectively the 3x10 part)
    # This yields a rank of 2 * 30 = 60.
    
    # U shape: (N*M, R) = (6, 60)
    # V shape: (M*P, R) = (30, 60)
    # W shape: (P*N, R) = (20, 60)
    
    R_init = 60
    U = np.zeros((N*M, R_init))
    V = np.zeros((M*P, R_init))
    W = np.zeros((P*N, R_init))
    
    # Indices:
    # U: (i, j) -> i*M + j
    # V: (j, k) -> j*P + k
    # W: (k, i) -> k*N + i
    
    # Kronecker structure:
    # The term t corresponds to (i, j, k) where:
    # i in 0..1 (from 2x1x1 part)
    # j in 0..2 (from 1x3x10 part)
    # k in 0..9 (from 1x3x10 part)
    
    # We map t to (i, j, k)
    # Let's iterate i, then j, then k.
    t = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                # U[i, j] gets 1
                U[i*M + j, t] = 1.0
                # V[j, k] gets 1
                V[j*P + k, t] = 1.0
                # W[k, i] gets 1
                W[k*N + i, t] = 1.0
                t += 1
                
    # Verify initial solution (should be exact)
    # T_recon = np.einsum('at,bt,ct->abc', U, V, W)
    # This is the schoolbook multiplication.
    
    # 2. Optimization: Local Search to Reduce Rank
    # We want to find a lower rank R < 60.
    # Strategy:
    # a) Try to remove terms one by one (set them to 0) if the solution remains exact.
    #    This is unlikely to work for schoolbook, but maybe if there's redundancy?
    #    Actually, schoolbook is minimal for this structure? No, Strassen reduces it.
    #    But we started with a specific structure.
    # b) Use a randomized local search to find a lower rank decomposition.
    #    We will try ranks from 55 down to 50.
    #    For each rank R, we initialize with the top R terms of the Kronecker solution?
    #    No, that breaks the structure.
    #    
    # Better approach:
    # We know Strassen for 2x2x2 gives rank 7.
    # The current program uses block decomposition.
    # Let's try a different search:
    # Start with the 60-term solution.
    # Randomly perturb entries.
    # Try to "compress" by setting some columns to zero and seeing if we can fix the rest.
    # This is hard.
    
    # Alternative: Direct Search for Low Rank
    # We will try to find a solution of rank R for R in [55, 56, 57, 58, 59].
    # We will use a simple hill-climbing with random restarts.
    # Objective: Minimize the error of the decomposition.
    # Error = sum_{a,b,c} (T[a,b,c] - sum_t U[a,t]V[b,t]W[c,t])^2
    
    # Precompute target tensor T
    T = np.zeros((N*M, M*P, P*N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i*M+j, j*P+k, k*N+i] = 1.0
                
    def calculate_error(U, V, W):
        # Efficient error calculation
        # T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        # Error = np.sum((T - T_recon)**2)
        # To save time, we can just check a subset or use a faster method?
        # For R=55, U(6,55), V(30,55), W(20,55).
        # Einsum is fast enough.
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.sum((T - T_recon)**2)
    
    def is_valid(U, V, W, tol=1e-9):
        # Check if exact
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol, rtol=0)
    
    best_rank = R_init
    best_U, best_V, best_W = U, V, W
    
    # Try to improve
    # We will try ranks 55 to 59.
    # For each rank, we do a few random restarts.
    
    candidates = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    
    for R_try in range(55, 60):
        if time.time() > deadline - 5:
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W randomly
            U_try = np.random.choice(candidates, (N*M, R_try))
            V_try = np.random.choice(candidates, (M*P, R_try))
            W_try = np.random.choice(candidates, (P*N, R_try))
            
            # Scale initialization?
            # Maybe better to initialize with the Kronecker solution truncated?
            # No, random is better for finding new structures.
            
            current_error = calculate_error(U_try, V_try, W_try)
            
            # Hill Climbing
            max_iters = 2000
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                # Randomly perturb one entry
                var_idx = np.random.randint(3)
                if var_idx == 0:
                    a, t = np.random.randint(N*M), np.random.randint(R_try)
                    old_val = U_try[a, t]
                    U_try[a, t] = np.random.choice(candidates)
                    new_error = calculate_error(U_try, V_try, W_try)
                    if new_error < current_error:
                        current_error = new_error
                    else:
                        U_try[a, t] = old_val
                elif var_idx == 1:
                    b, t = np.random.randint(M*P), np.random.randint(R_try)
                    old_val = V_try[b, t]
                    V_try[b, t] = np.random.choice(candidates)
                    new_error = calculate_error(U_try, V_try, W_try)
                    if new_error < current_error:
                        current_error = new_error
                    else:
                        V_try[b, t] = old_val
                else:
                    c, t = np.random.randint(P*N), np.random.randint(R_try)
                    old_val = W_try[c, t]
                    W_try[c, t] = np.random.choice(candidates)
                    new_error = calculate_error(U_try, V_try, W_try)
                    if new_error < current_error:
                        current_error = new_error
                    else:
                        W_try[c, t] = old_val
                        
                if current_error < 1e-9:
                    if is_valid(U_try, V_try, W_try):
                        best_rank = R_try
                        best_U, best_V, best_W = U_try, V_try, W_try
                        found = True
                        break
                        
            if found:
                break
                
        if found:
            break # Found a lower rank, stop searching lower ranks? 
                   # Actually, we can continue to see if we can go lower, but 
                   # the loop is 55 to 59. If we found 55, we can stop.
                   # If we found 56, we should try 55? No, the loop goes 55, 56...
                   # So if we find 55, we break. If we find 56, we break? 
                   # No, we should try to find the MINIMAL rank.
                   # So we should iterate from low to high? 
                   # But low ranks are harder to find.
                   # Let's stick to the current loop: try 55 first. If found, stop.
                   # If not, try 56. If found, stop.
                   # This finds the first valid rank in the sequence 55, 56, 57, 58, 59.
                   # Since 55 is lower, it's better.
                   
    # If no improvement found, return the initial rank 60 solution
    # But we can also try to return the best found in the search, even if not exact?
    # No, the task requires EXACT decomposition.
    # So we must return a valid one.
    # If the search didn't find a valid one with R < 60, we return the R=60 one.
    
    # Ensure the returned solution is valid
    # The initial solution is valid.
    # The found solution is validated by is_valid.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
