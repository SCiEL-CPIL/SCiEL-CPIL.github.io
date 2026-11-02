import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor.
    
    For matrix C = A * B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i (with matching indices)
    Otherwise T[a,b,c] = 0.
    
    A rank-R decomposition satisfies: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c.
    """
    R = U.shape[1]
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(R):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check a sample of zero entries
    # A zero entry occurs when indices don't align: e.g., different j or i or k
    import random
    rng = random.Random(42)
    checked = 0
    max_checks = 2000
    while checked < max_checks:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, m - 1)
        k = rng.randint(0, p - 1)
        a = i * m + j
        
        # Pick a "wrong" b index
        j2 = rng.randint(0, m - 1)
        k2 = rng.randint(0, p - 1)
        b = j2 * p + k2
        
        # Pick a "wrong" c index  
        i2 = rng.randint(0, n - 1)
        k3 = rng.randint(0, p - 1)
        c = k3 * n + i2
        
        # Skip if this is actually a non-zero entry
        if j2 == j and i2 == i and k3 == k:
            continue
            
        s = 0.0
        for r in range(R):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s) > tol:
            return False
        checked += 1
    
    return True

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _try_rank_reduction(n, m, p, target_rank, deadline, seed=0):
    """Attempt to find a rank-target_rank decomposition using numerical optimization.
    
    Strategy: Start from a perturbed schoolbook solution and try to zero out
    columns while maintaining the tensor decomposition constraints.
    """
    rng = np.random.RandomState(seed)
    R = n * m * p
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    # Generate random initial U, V, W with small values
    # We'll use a different approach: try to find linear dependencies
    
    # For rank reduction, we need to find a decomposition where some combinations
    # cancel out. Let's try a direct approach:
    
    # The key equation is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # and for all "invalid" (a,b,c), the sum = 0
    
    # This is a system of bilinear equations. Let's try a heuristic:
    # 1. Start with random U, V of size (a_dim x target_rank) and (b_dim x target_rank)
    # 2. Solve for W using the constraints
    # 3. Check if the solution is consistent
    
    # Actually, this is very hard. Let's try a simpler approach:
    # Use the fact that for 3x3*3x6, we can split into two 3x3*3x3 multiplications.
    # If we have a rank-7 decomposition for 2x2*2x2 (Strassen), we can try to
    # extend it.
    
    # For now, let's try a random search with local optimization.
    
    # Initialize with random matrices
    U = rng.randn(a_dim, target_rank) * 0.1
    V = rng.randn(b_dim, target_rank) * 0.1
    W = rng.randn(c_dim, target_rank) * 0.1
    
    # Define the loss function
    def loss(U, V, W):
        total = 0.0
        # Non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    total += (s - 1.0) ** 2
        # Sample zero entries
        for _ in range(500):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, m - 1)
            k = rng.randint(0, p - 1)
            a = i * m + j
            j2 = rng.randint(0, m - 1)
            k2 = rng.randint(0, p - 1)
            b = j2 * p + k2
            i2 = rng.randint(0, n - 1)
            k3 = rng.randint(0, p - 1)
            c = k3 * n + i2
            if j2 == j and i2 == i and k3 == k:
                continue
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            total += s ** 2
        return total
    
    # Try gradient descent
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(200):
        if time.time() > deadline:
            break
        
        # Compute gradients (approximate)
        eps = 1e-7
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # This is too slow for full gradient computation. Let's use a simpler approach.
        # Just do random perturbations and accept if loss decreases.
        
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Perturb one matrix
            mat_choice = rng.randint(0, 3)
            delta = rng.randn(1, 1) * lr
            
            if mat_choice == 0:
                row = rng.randint(0, a_dim)
                col = rng.randint(0, target_rank)
                old = U[row, col]
                U[row, col] += delta[0, 0]
                new_loss = loss(U, V, W)
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    U[row, col] = old
                    
            elif mat_choice == 1:
                row = rng.randint(0, b_dim)
                col = rng.randint(0, target_rank)
                old = V[row, col]
                V[row, col] += delta[0, 0]
                new_loss = loss(U, V, W)
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    V[row, col] = old
                    
            else:
                row = rng.randint(0, c_dim)
                col = rng.randint(0, target_rank)
                old = W[row, col]
                W[row, col] += delta[0, 0]
                new_loss = loss(U, V, W)
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    W[row, col] = old
        
        U, V, W = best_U, best_V, best_W
    
    # Check if we found a valid solution
    if best_loss < 1e-6:
        if _verify(best_U, best_V, best_W, n, m, p, tol=1e-6):
            return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: return best found before wall-clock limit
    deadline = time.time() + 120 - 10  # 110s safety margin
    
    # Schoolbook baseline: rank 54
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    
    # Try to find a lower-rank decomposition
    # For 3x3*3x6, the theoretical minimum is unknown but likely < 54
    # Let's try ranks from 53 down to some lower bound
    
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try a few candidate ranks
    for target_rank in [53, 52, 50, 48, 45, 40]:
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_reduction(n, m, p, target_rank, deadline, seed=target_rank)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify(U_try, V_try, W_try, n, m, p, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Return the best result found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
