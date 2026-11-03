import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_target = 23
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Initialize with schoolbook (rank 27)
    # But we want rank 23.
    # Let's try to optimize a rank-23 decomposition.
    
    # We will use a simple gradient descent or similar.
    # Since we need exact integers/half-integers, we will search in a discretized space or round at the end.
    
    # Strategy:
    # 1. Create U, V, W with rank 23, initialized with random small values or from schoolbook (trimmed).
    # 2. Minimize error.
    # 3. Round to half-integers.
    # 4. Verify.
    
    best_score = n*m*p # 27
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback schoolbook
    U_school = np.zeros((n*m, n*m*p), dtype=int)
    V_school = np.zeros((m*p, n*m*p), dtype=int)
    W_school = np.zeros((p*n, n*m*p), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try to find rank 23
    # We will use a random restart local search
    # Objective: minimize sum of squared differences
    
    def error(U, V, W):
        # Compute tensor from U, V, W
        # T_calc[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
        # Using einsum
        T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
        # Reshape to match T
        # U index: i*m+j -> i, j
        # V index: j*p+k -> j, k
        # W index: k*n+i -> k, i
        # T_calc shape is (n, m, p)? No.
        # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
        # np.einsum('ir,jr,kr->ijk', U, V, W) gives shape (n*m, m*p, p*n).
        diff = T_calc - T
        return np.sum(diff**2)
        
    # We need to optimize U, V, W.
    # This is a non-convex problem.
    # We will try a few random initializations.
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Initialize
        # Start with schoolbook and randomly zero out some columns?
        # Or random small integers.
        # Let's try random integers in [-2, 2]
        U = np.random.randint(-2, 3, (n*m, rank_target))
        V = np.random.randint(-2, 3, (m*p, rank_target))
        W = np.random.randint(-2, 3, (p*n, rank_target))
        
        # Simple local search: flip a random entry in one of the matrices
        current_err = error(U, V, W)
        
        for step in range(1000):
            if time.time() > deadline:
                break
            if current_err < 1e-9:
                # Found a solution?
                # Check if it's exact
                U_int = U.copy()
                V_int = V.copy()
                W_int = W.copy()
                # Round to half-integers
                U_round = np.round(U_int * 2) / 2.0
                V_round = np.round(V_int * 2) / 2.0
                W_round = np.round(W_int * 2) / 2.0
                
                # Verify rounded
                T_round = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
                if np.allclose(T_round, T, atol=1e-9):
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                break
                
            # Pick random matrix and entry
            mat_idx = np.random.randint(0, 3)
            if mat_idx == 0:
                r, c = np.random.randint(0, n*m), np.random.randint(0, rank_target)
                old_val = U[r, c]
                U[r, c] += 1
                new_err = error(U, V, W)
                if new_err < current_err:
                    current_err = new_err
                else:
                    U[r, c] = old_val
            elif mat_idx == 1:
                r, c = np.random.randint(0, m*p), np.random.randint(0, rank_target)
                old_val = V[r, c]
                V[r, c] += 1
                new_err = error(U, V, W)
                if new_err < current_err:
                    current_err = new_err
                else:
                    V[r, c] = old_val
            else:
                r, c = np.random.randint(0, p*n), np.random.randint(0, rank_target)
                old_val = W[r, c]
                W[r, c] += 1
                new_err = error(U, V, W)
                if new_err < current_err:
                    current_err = new_err
                else:
                    W[r, c] = old_val
    
    # If no rank 23 found, try rank 26? (Known to be easier?)
    # Or just return schoolbook.
    # Let's try to find rank 26 quickly if 23 fails?
    # Actually, 27 is guaranteed.
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
