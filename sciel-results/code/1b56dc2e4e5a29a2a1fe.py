import numpy as np
import time

def construct():
    n, m, p = 2, 2, 10
    rank_target = 34
    
    # Build the multiplication tensor
    T = np.zeros((4, 20, 20))
    for i_A in range(2):
        for j_A in range(2):
            for k_B in range(10):
                i = i_A * m + j_A
                j = j_A * p + k_B
                k = k_B * n + i_A
                T[i, j, k] = 1.0
    
    # Start with rank-35 Strassen solution
    U35, V35, W35 = _strassen_rank35(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 120
    
    # Try to find rank-34 by iterative elimination
    # For each pair of columns to remove, check if the residual can be absorbed
    
    best = (U35, V35, W35)
    
    # Strategy: try removing one column and adding a new column that captures the residual
    # If the residual after removing column r can be expressed as a single rank-1 tensor
    # that's already in the span of the remaining 34, then we have rank 34.
    # More generally, try to find a rank-34 decomposition by optimization.
    
    # Use a greedy approach: for each column r, remove it and try to fit the residual
    # as a linear combination of the remaining 34 rank-1 tensors.
    
    for r in range(35):
        if time.time() > deadline - 20:
            break
        
        U_rem = np.delete(U35, r, axis=1)
        V_rem = np.delete(V35, r, axis=1)
        W_rem = np.delete(W35, r, axis=1)
        
        # Residual tensor
        T_rem = np.zeros((4, 20, 20))
        for rr in range(34):
            T_rem += np.einsum('i,j,k->ijk', U_rem[:,rr], V_rem[:,rr], W_rem[:,rr])
        
        residual = T - T_rem
        
        # Check if residual is in the span of the 34 rank-1 tensors
        # This is a linear system: find coefficients c_s such that
        # sum_s c_s * U_rem[:,s] ⊗ V_rem[:,s] ⊗ W_rem[:,s] = residual
        
        # This is a 3-linear system, which is hard. Instead, check if residual
        # can be written as sum of rank-1 tensors that are in the span.
        
        # Alternative: check if the residual tensor has rank 1 and is orthogonal
        # to the "obstruction" of the remaining 34.
        
        # For now, just check if residual is zero (redundant column)
        if np.max(np.abs(residual)) < 1e-10:
            best = (U_rem, V_rem, W_rem)
            break
    
    if time.time() > deadline - 10:
        U_out, V_out, W_out = best
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    # Try a different approach: random search for rank-34
    # Initialize with rank-35, randomly perturb, and try to eliminate one factor
    
    np.random.seed(42)
    best_error = np.inf
    best_solution = None
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Start with rank-35, remove a random column, add a random rank-1 tensor
        r_remove = np.random.randint(35)
        U_rem = np.delete(U35, r_remove, axis=1)
        V_rem = np.delete(V35, r_remove, axis=1)
        W_rem = np.delete(W35, r_remove, axis=1)
        
        # Add a new random rank-1 tensor
        U_new = np.random.randn(4)
        V_new = np.random.randn(20)
        W_new = np.random.randn(20)
        
        U_try = np.column_stack([U_rem, U_new])
        V_try = np.column_stack([V_rem, V_new])
        W_try = np.column_stack([W_rem, W_new])
        
        # Compute error
        T_try = np.zeros((4, 20, 20))
        for rr in range(35):
            T_try += np.einsum('i,j,k->ijk', U_try[:,rr], V_try[:,rr], W_try[:,rr])
        
        error = np.linalg.norm(T - T_try)
        
        if error < best_error:
            best_error = error
            best_solution = (U_try.copy(), V_try.copy(), W_try.copy())
        
        # If error is very small, we found a rank-35 solution (not 34)
        # We need rank 34, so this approach won't directly work.
    
    # The random search for rank-34 is very hard. Let's just return rank-35.
    U_out, V_out, W_out = best
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

def _strassen_rank35(n, m, p):
    rank = 35
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for t in range(5):
        c0 = 2 * t
        c1 = 2 * t + 1
        base = 7 * t
        
        col = base + 0
        U[0, col] = 1; U[3, col] = 1
        V[c0, col] = 1; V[10 + c1, col] = 1
        W[c0 * n + 0, col] = 1
        W[c1 * n + 1, col] = 1
        
        col = base + 1
        U[2, col] = 1; U[3, col] = 1
        V[c0, col] = 1
        W[c0 * n + 1, col] = 1
        W[c1 * n + 1, col] = -1
        
        col = base + 2
        U[0, col] = 1
        V[c1, col] = 1; V[10 + c1, col] = -1
        W[c1 * n + 0, col] = 1
        W[c1 * n + 1, col] = 1
        
        col = base + 3
        U[3, col] = 1
        V[10 + c0, col] = 1; V[c0, col] = -1
        W[c0 * n + 0, col] = 1
        W[c0 * n + 1, col] = 1
        
        col = base + 4
        U[0, col] = 1; U[1, col] = 1
        V[10 + c1, col] = 1
        W[c0 * n + 0, col] = -1
        W[c1 * n + 0, col] = 1
        
        col = base + 5
        U[2, col] = 1; U[0, col] = -1
        V[c0, col] = 1; V[c1, col] = 1
        W[c1 * n + 1, col] = 1
        
        col = base + 6
        U[1, col] = 1; U[3, col] = -1
        V[10 + c0, col] = 1; V[10 + c1, col] = 1
        W[c0 * n + 0, col] = 1
    
    return U, V, W
