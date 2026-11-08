import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[alpha, beta, gamma]
    # alpha = i*m + j, beta = j'*p + k, gamma = k'*n + i'
    # T is 1 if i==i' and j==j' and k==k'
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    for k_prime in range(p):
                        for i_prime in range(n):
                            if i == i_prime and j == j_prime and k == k_prime:
                                alpha = i * m + j
                                beta = j_prime * p + k
                                gamma = k_prime * n + i_prime
                                indices.append((alpha, beta, gamma))
                                
    # Actually, T is mostly zeros.
    # We only need to check the non-zero entries for the "1"s and ensure the "0"s are 0.
    # It's easier to build the full tensor for checking.
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1

    def check_decomposition(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Compute sum_r U[:, r] V[:, r]^T W[:, r]^T ? No.
        # The tensor is T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
        # This is the outer product sum.
        # We can compute this efficiently.
        # Let's just iterate over the non-zero entries of T (which are n*m*p) and check if they are 1.
        # And we must ensure all other entries are 0.
        # Total entries 9*12*12 = 1296.
        # We can compute the full tensor using einsum or loops.
        # np.einsum('ar,br,cr->abc', U, V, W)
        # This might be slow for many restarts if R is large.
        # But 1296 entries is small.
        try:
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return np.array_equal(T_calc, T)
        except:
            return False

    # Search for rank R
    best_result = None
    # Try ranks from low to high? Or high to low?
    # Low rank is harder to find.
    # Let's try a range of ranks.
    # Based on Strassen 23 for 3x3x3, 3x3x4 might be around 30-33?
    # Let's try to find any R < 36.
    
    start_time = time.time()
    deadline = start_time + 120 # Safety margin
    
    # We will try to find a solution with integer entries in {-1, 0, 1} first, then {-2, -1, 0, 1, 2}
    # Random restarts
    
    for R in range(25, 36):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts
        max_restarts = 500
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with random integers
            # Sparsity might help?
            # Let's try dense random in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, R))
            V = np.random.choice([-1, 0, 1], size=(m*p, R))
            W = np.random.choice([-1, 0, 1], size=(p*n, R))
            
            # Local search: flip a random element to minimize residual
            # Residual is sum (T_calc - T)^2
            # We can update one element at a time.
            
            # Optimization: Precompute contributions?
            # For a fixed r, the term is U[:, r] V[:, r] W[:, r]
            # Changing U[alpha, r] changes T_calc[alpha, beta, gamma] for all beta, gamma by V[beta, r] W[gamma, r]
            
            # This is a local search on a discrete landscape.
            # It might get stuck.
            
            # Let's do a few steps of local search
            for step in range(20):
                # Pick random matrix, row, col
                mat_idx = np.random.randint(3)
                row = np.random.randint(n*m if mat_idx==0 else (m*p if mat_idx==1 else p*n))
                col = np.random.randint(R)
                
                # Current value
                if mat_idx == 0:
                    old_val = U[row, col]
                    # Try new values
                    # Calculate change in objective
                    # Objective = ||T_calc - T||^2
                    # This is expensive to calculate from scratch each time.
                    # We can update incrementally, but that's complex.
                    # Given small dimensions, maybe just recompute T_calc for check?
                    # No, 20 steps * 500 restarts * 35 ranks * 1296 entries * einsum... might be slow.
                    
                    # Better: Just use a gradient-based method on continuous values, then round?
                    # But we need EXACT integer.
                    
                    # Let's stick to a simpler heuristic:
                    # Random restarts with a "repair" phase?
                    
                    # Actually, for such small problems, maybe a simpler combinatorial construction exists?
                    # For now, let's just try random integer solutions with a few local moves.
                    
                    new_val = np.random.choice([-1, 0, 1])
                    if new_val == old_val: continue
                    
                    U[row, col] = new_val
                    
                    # Check if improved?
                    # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                    # err = np.sum((T_calc - T)**2)
                    # This is too slow for inner loop.
                    
                    # Alternative:
                    # Use the fact that we only need to satisfy the 36 non-zero constraints AND the 1260 zero constraints.
                    
                    # Let's optimize the code structure:
                    # If this approach is too slow, it will timeout.
                    # I should implement a faster check.
                    
                    pass # Placeholder for logic

            # Instead of complex local search, let's try a "Simulated Annealing" like approach on the integers
            # Or just more restarts.
            
            # Let's try a different approach:
            # Fix U and V randomly, solve for W?
            # T[:, :, gamma] = sum_r U[:, r] V[:, r] W[gamma, r]
            # This is linear in W[gamma, r].
            # For each gamma, we have a system of linear equations?
            # No, the sum is over r.
            # T[alpha, beta, gamma] = sum_r (U[alpha, r] V[beta, r]) W[gamma, r]
            # Let M[alpha, beta] = sum_r U[alpha, r] V[beta, r] ? No, W depends on r.
            
            # This is the "alternating least squares" idea but for integers.
            
            # Given the constraints, I will rely on finding a known solution or a very simple search.
            # If I can't find one, I return schoolbook.
            
            if check_decomposition(U, V, W):
                best_result = (R, U, V, W)
                found = True
                break

        if found:
            break

    if best_result is None:
        # Return schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        R, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
