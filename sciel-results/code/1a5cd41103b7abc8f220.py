import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T
    # Indices: a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if consistent, else 0.
    
    # We will search for rank R
    # Start from a reasonable lower bound
    # Schoolbook is 36.
    # Let's try to find a lower rank.
    
    # Precompute indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    target_vals = np.ones(len(target_indices))
    
    # Function to check if U, V, W form a valid decomposition
    def is_valid(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check all target indices
        for idx, (a, b, c) in enumerate(target_indices):
            val = np.dot(U[a], V[b] * W[c])
            if not np.isclose(val, 1.0, atol=1e-9):
                return False
        # Check some random non-target indices to ensure they are 0?
        # The problem statement says "exactly decompose the matrix multiplication tensor".
        # The tensor has 0s elsewhere.
        # We should check a sample of non-target indices.
        # There are (n*m)*(m*p)*(p*n) - 36 non-target indices.
        # Total size: 6 * 18 * 12 = 1296.
        # Non-target: 1260.
        # We can check all of them if it's fast enough, or a random subset.
        # 1260 checks * R operations. If R=20, 25200 ops. Fast.
        
        # Create full tensor check
        # To be safe, let's check all.
        # But iterating 1296 times in Python is slow.
        # Vectorized check:
        # We need to compute the full tensor T_calc = sum_r U[:,r] V[:,r] W[:,r] (outer products)
        # This is large: 6x18x12.
        # We can do:
        # T_calc = np.zeros((n*m, m*p, p*n))
        # for r in range(R):
        #     T_calc += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        # Then check T_calc[target_indices] == 1 and T_calc[non_target] == 0.
        
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_calc += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        
        # Check target
        for a, b, c in target_indices:
            if not np.isclose(T_calc[a,b,c], 1.0, atol=1e-9):
                return False
        
        # Check non-target (sample)
        # Generate random indices
        # Actually, checking all is safer.
        # 1296 is small.
        # We can flatten and compare.
        # But we need to know which are target.
        # Let's just check if any non-target is non-zero.
        # Create a mask of target indices.
        mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for a, b, c in target_indices:
            mask[a,b,c] = True
            
        non_target_vals = T_calc[~mask]
        if np.any(np.abs(non_target_vals) > 1e-9):
            return False
            
        return True

    # Search
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Try ranks from low to high
    # Lower bound for 2x3x6?
    # 2x2x2 is 7.
    # 2x3x3 is likely 11.
    # 2x3x6 might be around 20.
    # Let's start from 15.
    
    best_result = None
    
    for R in range(15, 37):
        if time.time() > deadline:
            break
            
        # Initialize
        # Use random integers or half-integers
        # Start with schoolbook-like structure but with noise?
        # Or just random.
        
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Scale: entries should be small.
            # Random integers in {-1, 0, 1} or similar.
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            W = np.random.randint(-1, 2, (p*n, R))
            
            # ALS
            for step in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                # Variables: U (n*m, R)
                # Equations: for each (i,j,k), sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                # This is linear in U.
                # Let's form the matrix A_u such that A_u * U_flat = b
                # U_flat is length n*m*R
                # For each equation (i,j,k), the coefficients for U[i*m+j, r] are V[j*p+k, r] * W[k*n+i, r]
                # But U[i*m+j, r] only appears in equations where the first two indices match i,j.
                # So for a fixed (i,j), we have p equations (for k=0..p-1).
                # The variables involved are U[i*m+j, 0..R-1].
                # So we can solve a small system for each (i,j).
                
                for i in range(n):
                    for j in range(m):
                        row_idx = i * m + j
                        # Equations for k=0..p-1
                        # LHS: sum_r U[row_idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Matrix M (p x R)
                        M = np.zeros((p, R))
                        for k in range(p):
                            for r in range(R):
                                M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Solve M @ U[row_idx] = ones(p)
                        # Use least squares
                        U[row_idx], _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        
                # Solve for V
                # For fixed (j,k), variables V[j*p+k, 0..R-1]
                # Equations for i=0..n-1
                for j in range(m):
                    for k in range(p):
                        row_idx = j * p + k
                        M = np.zeros((n, R))
                        for i in range(n):
                            for r in range(R):
                                M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        V[row_idx], _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        
                # Solve for W
                # For fixed (k,i), variables W[k*n+i, 0..R-1]
                # Equations for j=0..m-1
                for k in range(p):
                    for i in range(n):
                        row_idx = k * n + i
                        M = np.zeros((m, R))
                        for j in range(m):
                            for r in range(R):
                                M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        W[row_idx], _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        
                # Round to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Check validity
                if is_valid(U, V, W):
                    # Convert to list of lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # If not found, return schoolbook
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
