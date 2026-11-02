import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 2, 2, 6
    
    # --- Rank 21 Solution (Strassen) ---
    rank_21 = 21
    U21 = np.zeros((n * m, rank_21))
    V21 = np.zeros((m * p, rank_21))
    W21 = np.zeros((p * n, rank_21))
    
    alpha = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    beta = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    gamma = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]

    for b in range(3):
        for s in range(7):
            r = b * 7 + s
            for (i, j), val in alpha[s].items():
                U21[i * m + j, r] = val
            for (j, k_local), val in beta[s].items():
                k_global = b * 2 + k_local
                V21[j * p + k_global, r] = val
            for (k_local, i), val in gamma[s].items():
                k_global = b * 2 + k_local
                W21[k_global * n + i, r] = val

    def check_decomposition(U, V, W):
        R = U.shape[1]
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i, j, :] * V_mat[j, k, :] * W_mat[k, i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_find_lower_rank(R, time_limit):
        # Strategy: Randomly initialize U, V with small integers.
        # Solve for W using least squares.
        # Check if W is close to half-integers.
        # If yes, round and verify.
        
        # We have 24 equations.
        # Variables: W (12 * R).
        # Equations: For each (i, j, k), sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
        # Let's build the linear system A W_flat = b.
        # b is a vector of 24 ones.
        # A is 24 x (12*R).
        
        # Indices:
        # Row index in A: idx = i * m * p + j * p + k ? No, just a counter.
        # Col index in A: (k, i, r) -> (k * n + i) * R + r.
        
        def build_A(U, V):
            A = np.zeros((n * m * p, p * n * R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i * m * p + j * p + k
                        for r in range(R):
                            col = (k * n + i) * R + r
                            A[row, col] = U[i * m + j, r] * V[j * p + k, r]
            return A

        b = np.ones(n * m * p)
        
        best_error = float('inf')
        best_W = None
        best_U = None
        best_V = None
        
        num_restarts = 500 # Adjust based on time
        
        for restart in range(num_restarts):
            if time.time() > time_limit:
                break
            
            # Initialize U, V with small integers
            # Use a set of values like {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            
            # Ensure some non-zero structure?
            # Random might be too sparse.
            
            A = build_A(U, V)
            
            # Solve A W_flat = b
            # Use least squares
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(p * n, R)
                
                # Check distance to nearest half-integer
                # Round to nearest 0.5
                W_rounded = np.round(W * 2) / 2
                
                # Calculate error of rounded W in the original equation
                # Error = || A @ W_rounded_flat - b ||
                W_rounded_flat = W_rounded.flatten()
                error = np.linalg.norm(A @ W_rounded_flat - b)
                
                if error < 1e-9:
                    # Check if it's exactly valid
                    if check_decomposition(U, V, W_rounded):
                        return U, V, W_rounded
                elif error < best_error:
                    best_error = error
                    best_W = W_rounded
                    best_U = U.copy()
                    best_V = V.copy()
                    
            except Exception:
                continue
        
        # If we didn't find an exact solution, we might try to polish the best one?
        # For now, return None if no exact solution found.
        return None

    # Try to find rank 19, 18, 20
    for R in [19, 18, 20]:
        if time.time() > deadline - 5:
            break
        res = try_find_lower_rank(R, deadline - 5)
        if res is not None:
            U, V, W = res
            if check_decomposition(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }

    # Fallback to Rank 21
    return {
        "U": U21.tolist(),
        "V": V21.tolist(),
        "W": W21.tolist()
    }
