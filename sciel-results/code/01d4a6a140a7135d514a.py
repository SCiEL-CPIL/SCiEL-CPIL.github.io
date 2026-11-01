import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = delta (i.e., T[i,j,k] = 1 if this is the (i,j,k) entry)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _solve_rank(rank, n, m, p, deadline):
    """
    Attempt to find a rank-rank decomposition by solving a linear system.
    We fix U and V to simple integer/half-integer values (e.g., 0, 0.5, 1) 
    and solve for W, or vice versa. 
    Actually, a better approach for small ranks is to assume a specific sparsity 
    pattern or use a greedy algorithm to zero out elements.
    
    For this specific small case, we can try to construct the decomposition 
    by assuming the first few terms are "diagonal" (identity-like) and 
    solving for the rest.
    """
    # Strategy:
    # 1. Initialize U, V with random/simple patterns.
    # 2. Solve for W given U, V.
    # 3. Check if W is "simple" (integer/half-integer).
    # 4. If not, perturb U, V.
    
    # However, a more robust method for small ranks is to use the fact that 
    # the tensor is small. We can iterate over possible values for U and V 
    # from a small set {-1, -0.5, 0, 0.5, 1} and solve for W.
    
    # Let's try a smarter initialization: 
    # The trivial decomposition has rank 16. 
    # We know rank is likely around 10-14 for 2x2*2x4.
    
    # Let's try to fix U and V to be "mostly identity" plus some extra terms.
    
    # For a more systematic approach, let's use the linear system:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # This is non-linear. But if we fix U and V, it's linear in W.
    # Let's try random U, V from {-1, 0, 1} and solve for W.
    
    np.random.seed(42)
    
    # Number of equations: n*m*p = 16
    # Number of unknowns in W: p*n*rank = 4*2*rank = 8*rank
    
    # If rank is large enough, the system is underdetermined.
    # We want a sparse solution.
    
    # Let's try to find a solution where U and V are "nice".
    # Actually, let's just try to solve for W given random U, V.
    
    for attempt in range(100):
        if time.time() > deadline:
            return None
        
        # Generate random U, V with entries in {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(n*m, rank))
        V = np.random.choice([-1, 0, 1], size=(m*p, rank))
        
        # Build the linear system for W
        # For each (i, j, k), we have:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Let's flatten W into a vector of size (p*n*rank)
        # Let W_flat[r * (p*n) + (k*n+i)] = W[k*n+i, r]
        
        num_eqs = n * m * p
        num_unknowns = p * n * rank
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        # Coefficient of W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
                        coeff = U[i*m+j, r] * V[j*p+k, r]
                        if coeff != 0:
                            # Index in W_flat: r * (p*n) + (k*n + i)
                            w_idx = r * (p*n) + (k*n + i)
                            A[eq_idx, w_idx] = coeff
                    eq_idx += 1
        
        # Solve A * W_flat = b
        # Use least squares
        try:
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
            
        # Check if the solution is "simple" (close to integer or half-integer)
        # And check if it satisfies the equation
        if not np.allclose(A @ W_flat, b, atol=1e-6):
            continue
            
        # Reshape W_flat back to W
        W = W_flat.reshape(rank, p*n).T # (p*n, rank)
        
        # Check if W entries are close to half-integers
        W_rounded = np.round(W * 2) / 2
        if np.allclose(W, W_rounded, atol=1e-6):
            # Verify the full decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_rounded.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin for 120s limit
    
    # Start with the trivial solution
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
    
    best_rank = rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank
    # We know rank must be at least max(n*m, m*p, p*n) = 8
    # And likely higher due to structure.
    # Let's try ranks from 9 to 15.
    for try_rank in range(9, best_rank):
        if time.time() > deadline:
            break
        
        result = _solve_rank(try_rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = try_rank
                best_result = {"U": U, "V": V, "W": W}
                break  # Found lower rank, stop
    
    return best_result
