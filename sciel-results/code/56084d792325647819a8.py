import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # 115s safety margin

    n, m, p = DIMENSIONS

    # Build the target tensor T
    # T[i, j, j2, k, k2, i2] = 1 if j==j2 and i==i2 and k==k2 else 0
    # Dimensions: (n, m, m, p, p, n)
    T = np.zeros((n, m, m, p, p, n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, k, i] = 1.0

    T_flat = T.flatten()

    def compute_residual(U, V, W):
        """
        Compute the flattened residual vector.
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        """
        R = U.shape[1]
        # Reshape for einsum
        # U_r[i, j]
        U_r = U.reshape(R, n, m)
        # V_r[j, k]
        V_r = V.reshape(R, m, p)
        # W_r[k, i]  (Note: W index is k*n + i, so W_r[k, i])
        W_r = W.reshape(R, p, n)

        # recon[i, j, j2, k, k2, i2] = sum_r U_r[i, j] * V_r[j2, k] * W_r[k2, i2]
        # Wait, the standard matmul tensor is:
        # C_{i, k} = sum_j A_{i, j} B_{j, k}
        # Decomposition: sum_r U_r[i, j] V_r[j, k] W_r[k, i] ? No.
        # The tensor T is defined such that:
        # sum_r U_r[i, j] V_r[j2, k] W_r[k2, i2] = T[i, j, j2, k, k2, i2]
        # Where T[i, j, j2, k, k2, i2] = 1 if i==i2, j==j2, k==k2.
        
        # Let's verify the einsum indices.
        # U_r: (R, n, m) -> indices r, i, j
        # V_r: (R, m, p) -> indices r, j2, k
        # W_r: (R, p, n) -> indices r, k2, i2
        
        # We want to sum over r.
        # Result shape: (n, m, m, p, p, n) -> indices i, j, j2, k, k2, i2
        # Term: U_r[i, j] * V_r[j2, k] * W_r[k2, i2]
        
        recon = np.einsum('rij,rjk,rki->ijkki', U_r, V_r, W_r) # This is wrong shape
        
        # Correct einsum:
        # 'rij' (U), 'rjk' (V), 'rki' (W) -> result 'ijk' ? No.
        # We need result 'i j j2 k k2 i2'
        # U_r[i, j] -> 'rij'
        # V_r[j2, k] -> 'rjk' (here j in V is j2, k in V is k)
        # W_r[k2, i2] -> 'rki' (here k in W is k2, i in W is i2)
        
        # Let's map carefully:
        # U: r, i, j
        # V: r, j2, k
        # W: r, k2, i2
        # Result: i, j, j2, k, k2, i2
        
        recon = np.einsum('rij,rab,rcd->ijabcd', U_r, V_r, W_r)
        # i, j from U
        # a, b from V (a=j2, b=k)
        # c, d from W (c=k2, d=i2)
        
        return recon.flatten() - T_flat

    def compute_error(U, V, W):
        """Compute squared error norm."""
        res = compute_residual(U, V, W)
        return np.dot(res, res)

    def get_schoolbook_solution():
        """Return the standard schoolbook decomposition (Rank 84)."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * m, rank)) # Wait, V is (m*p, rank)
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

        return U, V, W

    def try_local_search(R, max_steps=2000, restarts=10):
        """
        Try to find a rank R solution using local search on discrete values.
        """
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for restart in range(restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with random sparse matrices
            U = np.zeros((size_U, R))
            V = np.zeros((size_V, R))
            W = np.zeros((size_W, R))
            
            # Randomly set some entries to non-zero values
            for _ in range(size_U * R // 4):
                i, j = np.random.randint(0, size_U), np.random.randint(0, R)
                U[i, j] = random.choice(values)
            for _ in range(size_V * R // 4):
                i, j = np.random.randint(0, size_V), np.random.randint(0, R)
                V[i, j] = random.choice(values)
            for _ in range(size_W * R // 4):
                i, j = np.random.randint(0, size_W), np.random.randint(0, R)
                W[i, j] = random.choice(values)
                
            current_error = compute_error(U, V, W)
            
            if current_error < 1e-9:
                return U, V, W
            
            for step in range(max_steps):
                if time.time() > deadline:
                    return None
                    
                # Pick a random matrix and entry
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    mat = U
                    rows, cols = size_U, R
                elif mat_choice == 1:
                    mat = V
                    rows, cols = size_V, R
                else:
                    mat = W
                    rows, cols = size_W, R
                    
                r_idx = np.random.randint(0, rows)
                c_idx = np.random.randint(0, cols)
                
                old_val = mat[r_idx, c_idx]
                
                # Generate a new value from neighbors
                # Neighbors are other discrete values
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                    
                mat[r_idx, c_idx] = new_val
                
                new_error = compute_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                    if current_error < 1e-9:
                        return U, V, W
                else:
                    # Revert
                    mat[r_idx, c_idx] = old_val

        return None

    # Search for the minimum rank
    best_solution = None
    
    # Lower bound estimation
    # Known lower bound for 3x4x7 is likely around 55-60.
    # We scan from a conservative lower bound up to 84.
    # To save time, we might start from 55.
    # If we find a solution at R, we return it immediately as it's minimal in this scan.
    
    # However, local search is stochastic. It might miss a solution at R=55 but find one at R=56.
    # To be safe, we should verify if a lower rank is possible?
    # No, the score is the rank. Lower is better.
    # If we find a valid solution at R, it's a valid score.
    # We want the *minimum* rank.
    # So we must scan from low to high.
    
    # Let's start from 55.
    for R in range(55, 85):
        if time.time() > deadline:
            break
            
        result = try_local_search(R, max_steps=1000, restarts=5)
        
        if result is not None:
            U, V, W = result
            # Double check validity
            if compute_error(U, V, W) < 1e-9:
                best_solution = (U, V, W)
                break

    if best_solution is None:
        U, V, W = get_schoolbook_solution()
    else:
        U, V, W = best_solution

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
