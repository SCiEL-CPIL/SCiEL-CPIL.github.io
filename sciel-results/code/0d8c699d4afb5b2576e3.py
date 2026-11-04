"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6, tol=1e-9):
    """Check if given factor matrices produce correct matmul tensor (vectorized)."""
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    
    R = U_np.shape[1]
    
    # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # T = einsum('ar,br,cr->abc', U_np, V_np, W_np)
    T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    # Check non-zero entries: T[i*m+j, j*p+k, k*n+i] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                    return False
    
    # Check zero entries: T[a,b,c] == 0 when indices don't align
    # a=i*m+j, b=j'*p+k, c=k'*n+i'
    # Non-zero only if j==j', k==k', i==i'
    for a in range(n*m):
        i_a = a // m
        j_a = a % m
        for b in range(m*p):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(p*n):
                k_c = c // n
                i_c = c % n
                if k_b != k_c or i_a != i_c:
                    if abs(T[a, b, c]) > tol:
                        return False
    
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _try_rank_optimization(R, n, m, p, time_budget=10):
    """Try to find a rank-R decomposition using structured random restarts."""
    import random
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    # Precompute target and zero-entry index lists
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    zero_indices = []
    for a in range(n*m):
        i_a = a // m
        j_a = a % m
        for b in range(m*p):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(p*n):
                k_c = c // n
                i_c = c % n
                if k_b != k_c or i_a != i_c:
                    zero_indices.append((a, b, c))
    
    U_size = n * m * R
    V_size = m * p * R
    W_size = p * n * R
    total_vars = U_size + V_size + W_size
    
    def objective(x):
        U_flat = x[:U_size]
        V_flat = x[U_size:U_size+V_size]
        W_flat = x[U_size+V_size:]
        
        U_np = U_flat.reshape(n*m, R)
        V_np = V_flat.reshape(m*p, R)
        W_np = W_flat.reshape(p*n, R)
        
        # Vectorized error computation
        # Non-zero entries
        err = 0.0
        for (a, b, c) in target_indices:
            val = np.dot(U_np[a], V_np[b] * W_np[c])
            err += (val - 1.0) ** 2
        
        # Zero entries
        for (a, b, c) in zero_indices:
            val = np.dot(U_np[a], V_np[b] * W_np[c])
            err += val ** 2
        
        return err
    
    def _polish(x0, maxiter=500):
        """High-precision polish pass."""
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-14})
            return result
        except Exception:
            return None
    
    def _extract_and_verify(x):
        U_flat = x[:U_size]
        V_flat = x[U_size:U_size+V_size]
        W_flat = x[U_size+V_size:]
        
        U_np = U_flat.reshape(n*m, R)
        V_np = V_flat.reshape(m*p, R)
        W_np = W_flat.reshape(p*n, R)
        
        U_list = U_np.tolist()
        V_list = V_np.tolist()
        W_list = W_np.tolist()
        
        if _verify_rank(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
        return None
    
    random.seed(42)
    np.random.seed(42)
    
    # Multiple initialization strategies
    strategies = []
    
    # Strategy 1: Small random
    strategies.append(lambda: np.random.randn(total_vars) * 0.1)
    
    # Strategy 2: Sparse random (many zeros)
    def sparse_init():
        x = np.zeros(total_vars)
        mask = np.random.rand(total_vars) < 0.3
        x[mask] = np.random.randn(mask.sum()) * 0.5
        return x
    strategies.append(sparse_init)
    
    # Strategy 3: Based on schoolbook structure with perturbations
    def schoolbook_perturb():
        x = np.zeros(total_vars)
        # Place 1s at schoolbook positions, then perturb
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < R:
                        x[i*m*j + col] = 1.0  # placeholder, will reshape
                    col += 1
        # Actually let's do this properly
        U_flat = np.zeros(U_size)
        V_flat = np.zeros(V_size)
        W_flat = np.zeros(W_size)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < R:
                        U_flat[(i*m+j)*R + col] = 1.0
                        V_flat[(j*p+k)*R + col] = 1.0
                        W_flat[(k*n+i)*R + col] = 1.0
                    col += 1
        # Add small perturbations to break symmetry
        U_flat += np.random.randn(U_size) * 0.05
        V_flat += np.random.randn(V_size) * 0.05
        W_flat += np.random.randn(W_size) * 0.05
        return np.concatenate([U_flat, V_flat, W_flat])
    strategies.append(schoolbook_perturb)
    
    # Strategy 4: Block-structured initialization
    def block_init():
        U_flat = np.zeros(U_size)
        V_flat = np.zeros(V_size)
        W_flat = np.zeros(W_size)
        # Assign each rank slice to cover a subset of (i,j,k) tuples
        slices_per_rank = max(1, (n*m*p) // R)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = idx % R
                    U_flat[(i*m+j)*R + r] = 1.0
                    V_flat[(j*p+k)*R + r] = 1.0
                    W_flat[(k*n+i)*R + r] = 1.0
                    idx += 1
        # Perturb
        U_flat += np.random.randn(U_size) * 0.1
        V_flat += np.random.randn(V_size) * 0.1
        W_flat += np.random.randn(W_size) * 0.1
        return np.concatenate([U_flat, V_flat, W_flat])
    strategies.append(block_init)
    
    max_restarts = 40
    restart = 0
    best_x = None
    best_err = float('inf')
    
    while restart < max_restarts and time.time() - start_time < time_budget:
        strategy = strategies[restart % len(strategies)]
        x0 = strategy()
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 80, 'ftol': 1e-10})
            
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x.copy()
            
            if result.fun < 1e-6:
                # Try polish
                polished = _polish(result.x, maxiter=200)
                if polished is not None:
                    if polished.fun < best_err:
                        best_err = polished.fun
                        best_x = polished.x.copy()
                    if polished.fun < 1e-10:
                        sol = _extract_and_verify(polished.x)
                        if sol is not None:
                            return sol
            
            if result.fun < 1e-10:
                sol = _extract_and_verify(result.x)
                if sol is not None:
                    return sol
                    
        except Exception:
            pass
        
        restart += 1
    
    # Final attempt with best found
    if best_x is not None and best_err < 1e-4:
        polished = _polish(best_x, maxiter=500)
        if polished is not None:
            sol = _extract_and_verify(polished.x)
            if sol is not None:
                return sol
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    best = _schoolbook_rank()
    best_rank = n * m * p  # 60
    
    # Verify schoolbook
    if not _verify_rank(best["U"], best["V"], best["W"]):
        return best
    
    # Try to find a lower rank solution
    # Focus on ranks just below schoolbook first, then wider search
    # Allocate time budget: more time for promising ranks
    rank_order = list(range(best_rank - 1, max(best_rank - 10, 0), -1))
    
    for R in rank_order:
        if time.time() > deadline - 25:  # Leave 25s buffer
            break
        
        # Allocate time based on rank proximity to schoolbook
        time_budget = min(20, max(5, (deadline - time.time()) - 25))
        
        result = _try_rank_optimization(R, n, m, p, time_budget=time_budget)
        if result is not None:
            best = result
            best_rank = R
            break
    
    # If we haven't found improvement, try a few more ranks with less time
    if best_rank == n * m * p:
        for R in range(best_rank - 10, max(best_rank - 15, 0), -1):
            if time.time() > deadline - 25:
                break
            result = _try_rank_optimization(R, n, m, p, time_budget=5)
            if result is not None:
                best = result
                best_rank = R
                break
    
    return best

# EVOLVE-BLOCK-END
