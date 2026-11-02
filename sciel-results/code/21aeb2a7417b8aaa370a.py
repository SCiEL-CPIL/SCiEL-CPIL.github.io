"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _try_decomposition(rank, seed=0, max_iters=5000, tol=1e-9):
    """
    Try to find a rank-`rank` decomposition for 2x4 * 4x4 matrix multiplication.
    Uses random initialization and gradient-free coordinate descent / local search
    to minimize the residual tensor norm.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # Shapes
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (8, rank)
    
    # Initialize U, V, W randomly
    # We can use a heuristic: start with schoolbook and perturb, or pure random.
    # Let's try a mix: start with a structured random init.
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Normalize columns to avoid scale drift
    for c in range(rank):
        norm_u = np.linalg.norm(U[:, c])
        norm_v = np.linalg.norm(V[:, c])
        norm_w = np.linalg.norm(W[:, c])
        if norm_u > 1e-12 and norm_v > 1e-12 and norm_w > 1e-12:
            # Scale so that product of norms is 1
            scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
            U[:, c] /= scale
            V[:, c] /= scale
            W[:, c] /= scale
            # Re-normalize to unit norm for stability? No, let's just keep scale 1.
            # Actually, for tensor decomposition, we want U[i,j,c] V[j,k,c] W[k,l,c] = sum_c ...
            # The scale is arbitrary per rank.
            # Let's just normalize each column to unit norm to prevent overflow/underflow.
            U[:, c] /= np.linalg.norm(U[:, c])
            V[:, c] /= np.linalg.norm(V[:, c])
            W[:, c] /= np.linalg.norm(W[:, c])

    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # We want to minimize || T - sum_c U_c \otimes V_c \otimes W_c ||^2
    # T[i, j, k] = delta_{i,k} (since C = AB, C[i,k] = sum_j A[i,j]B[j,k])
    # So T[i,j,k] = 1 if i==k, else 0.
    
    # Precompute T as a 3D array
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            if i == k:
                for j in range(m):
                    T[i, j, k] = 1.0
    
    # Residual calculation
    def calc_residual(U, V, W):
        # Res = T - sum_c U[:,c] \otimes V[:,c] \otimes W[:,c]
        # Using einsum for speed
        # U: (8, r), V: (16, r), W: (8, r)
        # Sum over c of U[i,c] V[j,c] W[k,c]
        # Note: V index is j*p+k? No, V is indexed by (j,k) flattened.
        # U index is i*m+j? No, U is indexed by (i,j) flattened?
        # Let's check the seed code:
        # U[i * m + j][column] = 1
        # V[j * p + k][column] = 1
        # W[k * n + i][column] = 1
        # So U row index = i*m + j.
        # V row index = j*p + k.
        # W row index = k*n + i.
        
        # We need to reshape U, V, W to 3D to match T[i,j,k].
        # U: (n*m, r) -> (n, m, r)
        # V: (m*p, r) -> (m, p, r)
        # W: (p*n, r) -> (p, n, r) -> transpose to (n, p, r)? 
        # Wait, W[k*n+i] means row index is k*n + i. 
        # So W is (p, n) flattened. We need W[k, i].
        # So W_3d[k, i, c] = W[k*n + i, c].
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Let's use einsum: 'ijc,jkc,kic->ijk'
        # But W is indexed by k, i. So W_3d is (p, n, r).
        # We need W[k, i, c].
        # So: np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Check dimensions:
        # U_3d: (n, m, r) -> i, j, c
        # V_3d: (m, p, r) -> j, k, c
        # W_3d: (p, n, r) -> k, i, c
        # Result: (n, m, p) -> i, j, k
        
        prod = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        res = T - prod
        return np.linalg.norm(res)
    
    # Optimization loop
    # Simple local search: perturb one matrix, one column, one element.
    # Or use a simple gradient approximation.
    # Given the 150s limit, we need to be careful.
    # Let's try a few random restarts with simple coordinate descent.
    
    for it in range(max_iters):
        if time.time() > 140: break
        
        # Randomly pick a matrix, column, and row to perturb
        mat_idx = rng.randint(3)
        col = rng.randint(rank)
        
        if mat_idx == 0:
            mat = U
            rows = n * m
        elif mat_idx == 1:
            mat = V
            rows = m * p
        else:
            mat = W
            rows = p * n
            
        row = rng.randint(rows)
        
        # Calculate current residual
        err = calc_residual(U, V, W)
        
        # Try small perturbations
        # We can use a simple line search or just random steps
        # Let's try a few random steps
        for _ in range(5):
            step_size = 0.1 * (0.5 ** rng.randint(0, 3))
            direction = rng.choice([-1, 1])
            old_val = mat[row, col]
            mat[row, col] = old_val + direction * step_size
            
            new_err = calc_residual(U, V, W)
            if new_err < err:
                err = new_err
                # Keep the change
            else:
                # Revert
                mat[row, col] = old_val
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_err < tol:
                break
                
    return best_U, best_V, best_W, best_err

def construct():
    # Schoolbook decomposition is rank 32.
    # We want to find a lower rank.
    # Known bounds for 2x4x4:
    # Schoolbook: 32
    # Strassen-like?
    # Let's try ranks 27, 28, 29, 30, 31.
    # 27 is the Strassen rank for 2x2x2 scaled up?
    # 2x2x2 rank is 7.
    # 2x4x4 can be seen as 2x2 blocks of 2x2?
    # A is 2x4, B is 4x4.
    # Split A into two 2x2 blocks: A1, A2.
    # Split B into four 2x2 blocks: B11, B12, B21, B22.
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    # Each 2x2x2 multiplication has rank 7.
    # Total rank 4 * 7 = 28?
    # Or can we share?
    # If we use 7 ranks for each of the 4 products, that's 28.
    # Can we do better?
    # Let's try 27 first. If it fails, 28.
    
    for rank in [27, 28, 29, 30, 31]:
        # Try a few seeds
        for seed in range(10):
            U, V, W, err = _try_decomposition(rank, seed=seed, max_iters=1000)
            if err < 1e-8:
                # Success!
                # Convert to lists of ints/floats
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        # If we found a good but not perfect solution, maybe we can polish it?
        # For now, just move to next rank if not found.
        
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
