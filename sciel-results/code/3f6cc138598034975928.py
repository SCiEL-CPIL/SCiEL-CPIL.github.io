import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    # Target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Build the set of target indices (a, b, c) where T[a,b,c] = 1
    target_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.add((a, b, c))

    # Strategy: Search for sparse decompositions using F2 support patterns
    # For each rank R, try to find a support pattern (which entries of U, V, W are nonzero)
    # such that the F2 tensor contraction matches T mod 2, then solve for rational coefficients.

    def try_solve_with_support(U_sup, V_sup, W_sup, R):
        """
        Given boolean support patterns U_sup (n*m x R), V_sup (m*p x R), W_sup (p*n x R),
        solve for rational coefficients such that the tensor decomposition is exact.
        Returns (U, V, W) as Fraction matrices or None if no solution.
        """
        # Collect all variable positions
        vars_U = []
        vars_V = []
        vars_W = []
        var_map = {}  # (matrix, row, col) -> var index
        var_idx = 0

        for r in range(R):
            for a in range(n * m):
                if U_sup[a, r]:
                    var_map[('U', a, r)] = var_idx
                    vars_U.append((a, r))
                    var_idx += 1
            for b in range(m * p):
                if V_sup[b, r]:
                    var_map[('V', b, r)] = var_idx
                    vars_V.append((b, r))
                    var_idx += 1
            for c in range(p * n):
                if W_sup[c, r]:
                    var_map[('W', c, r)] = var_idx
                    vars_W.append((c, r))
                    var_idx += 1

        num_vars = var_idx
        if num_vars == 0:
            return None

        # Build linear system: for each (a, b, c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # This is a system of equations where each equation involves products of variables.
        # Since we have a fixed support, each (a,b,c) equation becomes:
        # sum over r where U[a,r], V[b,r], W[c,r] are all supported: U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # This is still nonlinear! We need a different approach.
        
        # Alternative: Fix values of some variables to 1 and solve for others.
        # Or use the fact that for a valid decomposition, we can set one factor's entries to 1
        # and solve for the others.
        
        # Better approach: Use the structure. For each rank component r, the contribution is
        # u_r * v_r * w_r where u_r is a vector in R^(n*m), v_r in R^(m*p), w_r in R^(p*n).
        # The sum over r must equal T.
        
        # For sparse patterns, we can try: set all W entries to 1 where supported,
        # then for each (a,b), we need sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all c.
        # This is still coupled.
        
        # Simpler: Try a greedy assignment. For each target (a,b,c), assign it to a rank r
        # and set U[a,r]=1, V[b,r]=1, W[c,r]=1. This gives the trivial decomposition.
        # To get lower rank, we need multiple targets to share rank components.
        
        # Let's try: for each rank r, pick a set of (a,b,c) triples that share common a, b, or c.
        # Actually, the key insight: if we set U[a,r]=1, V[b,r]=1, W[c,r]=1, then
        # component r contributes 1 to T[a,b,c]. For multiple triples to share component r,
        # we need the product to be 1 for each.
        
        # A better strategy: Use the F2 CSP to find a valid assignment, then lift.
        # In F2, 1*1*1 = 1, so the support pattern just needs each target (a,b,c) to have
        # an odd number of rank components r where U[a,r]=V[b,r]=W[c,r]=1.
        
        # For the rational lift: if we set all supported entries to 1, then in Q the product
        # is 1, and the sum gives the count mod... no, in Q it gives the actual count.
        # So we need the count to be exactly 1 for targets and 0 for non-targets.
        
        # This means: find a 0-1 support pattern such that for each (a,b,c):
        # sum_r U[a,r]*V[b,r]*W[c,r] = 1 if (a,b,c) in target, 0 otherwise.
        # With 0-1 entries, this is exactly the F2 condition PLUS the condition that
        # the count is exactly 0 or 1 (not 2, 3, etc.).
        
        # So we need a "perfect" matching-like structure.
        
        # Let's try a different approach: random 0-1 supports and check if they work over Q.
        return None

    # Alternative approach: Directly search for low-rank decompositions using the structure
    # of matrix multiplication.
    
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) / something... 
    # Actually, for 3x3x3, the minimum rank is 23 (Strassen-like).
    # For 3x3x7, let's try ranks from a reasonable lower bound upward.
    
    # The trivial rank is 63. Let's try to find something better.
    
    # Use random search with the structure of matrix multiplication.
    # Initialize U, V, W with random sparse patterns and try to solve.
    
    def random_sparse_support(num_rows, R, density):
        """Generate a random boolean support matrix."""
        support = np.zeros((num_rows, R), dtype=bool)
        for r in range(R):
            # Each column has a certain number of nonzeros
            num_nonzeros = max(1, int(num_rows * density))
            rows = np.random.choice(num_rows, size=num_nonzeros, replace=False)
            support[rows, r] = True
        return support

    # Try different ranks
    # Lower bound: at least max(n*m, m*p, p*n) / min(n, m, p) roughly
    # For 3x3x7: n*m=9, m*p=21, p*n=21
    # A reasonable lower bound might be around 30-40
    
    min_rank_to_try = 25
    max_rank_to_try = 62
    
    # Try ranks from low to high, but prioritize lower ranks
    for R in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
            
        # Try a few random support patterns
        num_trials = 50
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            np.random.seed(R * 1000 + trial)
            
            # Generate random supports with varying densities
            density_U = np.random.uniform(0.1, 0.4)
            density_V = np.random.uniform(0.1, 0.4)
            density_W = np.random.uniform(0.1, 0.4)
            
            U_sup = random_sparse_support(n * m, R, density_U)
            V_sup = random_sparse_support(m * p, R, density_V)
            W_sup = random_sparse_support(p * n, R, density_W)
            
            # Check if this support pattern works over Q with all 1s
            U = U_sup.astype(float)
            V = V_sup.astype(float)
            W = W_sup.astype(float)
            
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Check if T_comp matches T exactly (or very closely)
            if np.allclose(T_comp, T, atol=1e-10):
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_U is not None:
            break

    # If no sparse 0-1 solution found, try a numerical optimization for a few ranks
    if best_U is None:
        # Use scipy minimize for a few promising ranks
        from scipy.optimize import minimize
        
        for R in [40, 35, 30]:
            if time.time() > deadline:
                break
                
            for restart in range(3):
                if time.time() > deadline:
                    break
                    
                np.random.seed(10000 + R * 100 + restart)
                U0 = np.random.randn(n*m, R) * 0.1
                V0 = np.random.randn(m*p, R) * 0.1
                W0 = np.random.randn(p*n, R) * 0.1
                
                x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
                
                def objective(params):
                    size_U = n * m * R
                    size_V = m * p * R
                    U = params[0:size_U].reshape(n*m, R)
                    V = params[size_U:size_U+size_V].reshape(m*p, R)
                    W = params[size_U+size_V:].reshape(p*n, R)
                    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                    return np.sum((T - T_comp) ** 2)
                
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
                
                if res.fun < 1e-10:
                    size_U = n * m * R
                    size_V = m * p * R
                    U = res.x[0:size_U].reshape(n*m, R)
                    V = res.x[size_U:size_U+size_V].reshape(m*p, R)
                    W = res.x[size_U+size_V:].reshape(p*n, R)
                    
                    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.allclose(T_comp, T, atol=1e-8):
                        best_rank = R
                        best_U = U
                        best_V = V
                        best_W = W
                        break
            
            if best_U is not None:
                break

    if best_U is not None:
        # Convert to lists
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to trivial decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
        return {"U": U, "V": V, "W": W}
