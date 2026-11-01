import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute valid indices and target values
    # T[u, v, w] = 1 if valid, 0 otherwise
    # u = i*m + j, v = j*p + k, w = k*n + i
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_indices.append((u, v, w))
                
    # We will search for a decomposition with rank r < 50
    # If not found, return schoolbook rank 50
    
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    random.seed(42)
    np.random.seed(42)
    
    def verify(U, V, W):
        """
        Verify if U, V, W exactly decompose the tensor.
        U, V, W are lists of lists (or arrays) with integer/half-integer entries.
        """
        r = len(U[0]) if U else 0
        # Convert to numpy arrays for efficient computation
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check dimensions
        if U_arr.shape != (nm, r) or V_arr.shape != (mp, r) or W_arr.shape != (pn, r):
            return False
            
        # Compute the tensor product sum
        # We need to check if sum_t U[u,t] V[v,t] W[w,t] == T[u,v,w]
        
        # Efficient check:
        # For each valid (u,v,w), sum must be 1.
        # For each invalid (u,v,w), sum must be 0.
        
        # Instead of iterating all 2500, we can compute the full tensor 
        # and check it, or just check the constraints.
        # Given r is small (< 50), computing the full tensor is feasible.
        # Tensor shape (nm, mp, pn) = (10, 25, 10) = 2500 elements.
        
        # res[u, v, w] = sum_t U[u,t] * V[v,t] * W[w,t]
        # We can do this by iterating t and adding outer products.
        res = np.zeros((nm, mp, pn), dtype=float)
        for t in range(r):
            # Outer product of U[:,t], V[:,t], W[:,t]
            # This is a 3D tensor
            # np.einsum('i,j,k->ijk', U_arr[:,t], V_arr[:,t], W_arr[:,t])
            # Or use broadcasting
            term = np.einsum('i,j,k->ijk', U_arr[:, t], V_arr[:, t], W_arr[:, t])
            res += term
            
        # Check valid indices
        for u, v, w in valid_indices:
            if abs(res[u, v, w] - 1.0) > 1e-9:
                return False
                
        # Check invalid indices
        # Create a mask of valid indices
        valid_mask = np.zeros((nm, mp, pn), dtype=bool)
        for u, v, w in valid_indices:
            valid_mask[u, v, w] = True
            
        # Check if any invalid element is non-zero
        # Note: due to floating point, we check for magnitude
        invalid_vals = res[~valid_mask]
        if np.any(np.abs(invalid_vals) > 1e-9):
            return False
            
        return True

    def solve_for_W(U, V, r):
        """
        Solve for W given U, V and rank r.
        Returns W as a list of lists.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        
        W_sol = np.zeros((pn, r), dtype=float)
        
        for w in range(pn):
            # Construct A_w and b_w
            # Equations: sum_t U[u,t] V[v,t] W[w,t] = T[u,v,w] for all u,v
            # Variables: W[w, 0..r-1]
            
            A_w = np.zeros((nm * mp, r), dtype=float)
            b_w = np.zeros(nm * mp, dtype=float)
            
            idx = 0
            for u in range(nm):
                for v in range(mp):
                    # Determine T[u, v, w]
                    # Decode u, v, w
                    i1, j1 = divmod(u, m)
                    j2, k1 = divmod(v, p)
                    k2, i2 = divmod(w, n)
                    
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        target = 1.0
                    else:
                        target = 0.0
                    
                    b_w[idx] = target
                    for t in range(r):
                        A_w[idx, t] = U_arr[u, t] * V_arr[v, t]
                    idx += 1
            
            # Solve least squares
            sol, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
            W_sol[w, :] = sol
            
        return W_sol.tolist()

    def round_to_half_integer(W):
        """
        Round each element of W to the nearest half-integer.
        """
        W_r = []
        for row in W:
            new_row = []
            for val in row:
                # Multiply by 2, round, divide by 2
                new_val = round(val * 2) / 2.0
                new_row.append(new_val)
            W_r.append(new_row)
        return W_r

    # Search Loop
    # Try ranks from 49 down to 40
    # If we find a solution, return it.
    # If we run out of time, return schoolbook.
    
    best_result = None
    
    for r in range(49, 39, -1):
        if time.time() > deadline:
            break
            
        # Number of trials depends on remaining time
        # Let's try 200 trials per rank if time permits
        trials_done = 0
        max_trials = 200
        
        while trials_done < max_trials:
            if time.time() > deadline:
                break
                
            # Initialize U, V
            # Use small integers to keep numbers small
            # Entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(nm, r)).tolist()
            V = np.random.randint(-1, 2, size=(mp, r)).tolist()
            
            # Solve for W
            W_float = solve_for_W(U, V, r)
            
            # Round W to half-integers
            W_r = round_to_half_integer(W_float)
            
            # Verify
            if verify(U, V, W_r):
                return {
                    "U": U,
                    "V": V,
                    "W": W_r
                }
                
            trials_done += 1
            
            # Occasionally try half-integer initialization for U, V
            if trials_done % 5 == 0:
                U = (np.random.randint(-1, 2, size=(nm, r)) / 2.0).tolist()
                V = (np.random.randint(-1, 2, size=(mp, r)) / 2.0).tolist()
                W_float = solve_for_W(U, V, r)
                W_r = round_to_half_integer(W_float)
                if verify(U, V, W_r):
                    return {
                        "U": U,
                        "V": V,
                        "W": W_r
                    }

    # Fallback: Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }

# EVOLVE-BLOCK-END
