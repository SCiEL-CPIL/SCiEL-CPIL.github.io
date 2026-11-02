import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_solution = None
    best_rank = n * m * p
    
    # Schoolbook fallback
    def get_schoolbook():
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
        return {"U": U, "V": V, "W": W, "rank": rank}

    schoolbook = get_schoolbook()
    best_solution = schoolbook
    best_rank = schoolbook["rank"]

    # Try to find lower rank
    # Known minimal rank for 3x3x5 is 21
    for target_rank in range(21, best_rank + 1):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple restarts for each rank
        max_restarts = 5 if target_rank == 21 else 2
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            try:
                sol = _als_decompose(n, m, p, target_rank, deadline)
                if sol is not None:
                    # Verify
                    if _verify(sol, n, m, p):
                        best_solution = sol
                        best_rank = target_rank
                        found = True
                        break # Break restarts if found
            except Exception:
                pass

        if found:
            break # Break rank loop if found minimal or better

    # Convert to list of lists if needed, though numpy arrays might be accepted?
    # Prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # The seed returns lists. I will return lists to be safe.
    
    U = [[float(x) for x in row] for row in best_solution["U"]]
    V = [[float(x) for x in row] for row in best_solution["V"]]
    W = [[float(x) for x in row] for row in best_solution["W"]]
    
    return {"U": U, "V": V, "W": W}

def _als_decompose(n, m, p, rank, deadline):
    """
    Attempts to find a rank-`rank` decomposition using ALS.
    Returns dict with U, V, W as numpy arrays if successful (converged), else None.
    """
    np.random.seed(42 + rank) # Deterministic for reproducibility in this run? 
    # Actually, random restarts should use different seeds.
    
    # Precompute index mappings
    # Consistent tuples (i, j, k)
    # I = i*m + j
    # J = j*p + k
    # K = k*n + i
    
    # For ALS, we need to solve linear systems.
    # We will iterate.
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Normalize to prevent divergence
    # U, V, W can be scaled.
    
    for it in range(100):
        if time.time() > deadline:
            return None
            
        # Update W
        # For each K (0 to p*n - 1)
        # i = K % n
        # k = K // n
        # j varies 0 to m-1
        # I = i*m + j
        # J = j*p + k
        # Eq: sum_r W[K, r] * U[I, r] * V[J, r] = 1
        
        for K in range(p * n):
            i = K % n
            k = K // n
            # Build matrix A_K (m x rank)
            A_K = np.zeros((m, rank))
            for j in range(m):
                I = i * m + j
                J = j * p + k
                A_K[j, :] = U[I, :] * V[J, :]
            # Solve A_K @ w_K = 1
            # w_K is W[K, :]
            try:
                w_K, _, _, _ = np.linalg.lstsq(A_K, np.ones(m), rcond=None)
                W[K, :] = w_K
            except:
                pass
                
        # Update U
        # For each I (0 to n*m - 1)
        # i = I // m
        # j = I % m
        # k varies 0 to p-1
        # J = j*p + k
        # K = k*n + i
        # Eq: sum_r U[I, r] * V[J, r] * W[K, r] = 1
        
        for I in range(n * m):
            i = I // m
            j = I % m
            A_I = np.zeros((p, rank))
            for k in range(p):
                J = j * p + k
                K = k * n + i
                A_I[k, :] = V[J, :] * W[K, :]
            try:
                u_I, _, _, _ = np.linalg.lstsq(A_I, np.ones(p), rcond=None)
                U[I, :] = u_I
            except:
                pass
                
        # Update V
        # For each J (0 to m*p - 1)
        # j = J // p
        # k = J % p
        # i varies 0 to n-1
        # I = i*m + j
        # K = k*n + i
        # Eq: sum_r V[J, r] * U[I, r] * W[K, r] = 1
        
        for J in range(m * p):
            j = J // p
            k = J % p
            A_J = np.zeros((n, rank))
            for i in range(n):
                I = i * m + j
                K = k * n + i
                A_J[i, :] = U[I, :] * W[K, :]
            try:
                v_J, _, _, _ = np.linalg.lstsq(A_J, np.ones(n), rcond=None)
                V[J, :] = v_J
            except:
                pass
        
        # Check convergence / exactness
        # Snap to half-integers and verify
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if _verify_numpy(U_snap, V_snap, W_snap, n, m, p):
            return {"U": U_snap, "V": V_snap, "W": W_snap}
            
    return None

def _verify_numpy(U, V, W, n, m, p):
    """
    Verifies if the numpy arrays provide an exact decomposition.
    """
    # Check consistent indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                val = np.dot(U[I, :], V[J, :]) # Wait, this is not right.
                # The term is sum_r U[I,r] V[J,r] W[K,r]
                # This is not a simple dot product of U and V.
                # It's a trilinear form.
                # We can compute it as:
                # tmp = U[I,:] * V[J,:]
                # val = np.dot(tmp, W[K,:])
                
                val = np.dot(U[I, :] * V[J, :], W[K, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
                    
    # Check inconsistent indices (sample a few or all?)
    # For safety, check all.
    # Total 2025.
    for I in range(n * m):
        i = I // m
        j = I % m
        for J in range(m * p):
            j2 = J // p
            k = J % p
            if j != j2:
                continue # Inconsistent j
            for K in range(p * n):
                i2 = K % n
                k2 = K // n
                if i != i2 or k != k2:
                    # Inconsistent
                    val = np.dot(U[I, :] * V[J, :], W[K, :])
                    if not np.isclose(val, 0.0, atol=1e-9):
                        return False
                        
    return True

def _verify(sol, n, m, p):
    """
    Verifies the solution dict.
    """
    U = np.array(sol["U"])
    V = np.array(sol["V"])
    W = np.array(sol["W"])
    return _verify_numpy(U, V, W, n, m, p)
