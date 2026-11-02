"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    def _verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            U_r = U_arr[:, r].reshape(n, m)
            V_r = V_arr[:, r].reshape(m, p)
            W_r = W_arr[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
        
        target = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(T, target, atol=1e-9)

    def _compute_tensor(U, V, W):
        """Compute the tensor from factor matrices."""
        rank = U.shape[1]
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
        return T

    def _schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
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

    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p

    def _als_decomposition(rank, max_iters=500, seed=42):
        """Alternating Least Squares for tensor decomposition.
        
        For each rank-r term: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        When fixing V and W, the problem for U is linear:
        For each (i,j), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        
        This is a linear system in U[i*m+j, :] for each (i,j).
        """
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), solve linear system
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            for i in range(n):
                for j in range(m):
                    # Build A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    # Solve A @ U[i*m+j, :] = b
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve linear system
            # sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    # Build A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve linear system
            # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
            for k in range(p):
                for i in range(n):
                    # Build A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            T = _compute_tensor(U, V, W)
            error = np.max(np.abs(T - 1.0))
            if error < 1e-10:
                if _verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Final check
        T = _compute_tensor(U, V, W)
        if np.max(np.abs(T - 1.0)) < 1e-9:
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    def _structured_search(rank):
        """Try structured decompositions with specific sparsity patterns."""
        if time.time() > deadline:
            return None
        
        # Try a block structure: partition indices into groups
        # For 4x5x5, try various structured patterns
        
        # Pattern 1: Use Kronecker-like structure
        # Split i into groups, j into groups, k into groups
        # T[i,j,k] = sum over blocks of products
        
        # Try random structured patterns
        for trial in range(50):
            if time.time() > deadline:
                return None
            
            rng = np.random.RandomState(42 + trial)
            
            # Initialize with sparse structure
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Assign each (i,j) pair to a subset of rank dimensions
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    # Choose a few random rank indices
                    num_active = max(1, rank // 10)
                    active = rng.choice(rank, size=min(num_active, rank), replace=False)
                    U[idx, active] = rng.randn(len(active)) * 0.5
            
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    num_active = max(1, rank // 10)
                    active = rng.choice(rank, size=min(num_active, rank), replace=False)
                    V[idx, active] = rng.randn(len(active)) * 0.5
            
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    num_active = max(1, rank // 10)
                    active = rng.choice(rank, size=min(num_active, rank), replace=False)
                    W[idx, active] = rng.randn(len(active)) * 0.5
            
            # Run ALS from this initialization
            result = _als_decomposition(rank, max_iters=200, seed=42 + trial)
            if result:
                return result
        
        return None

    # Try ranks from schoolbook-1 downward
    for rank in range(n * m * p - 1, max(n * m * p - 30, 1), -1):
        if time.time() > deadline:
            break
        
        # Try direct ALS first
        result = _als_decomposition(rank, max_iters=300, seed=42)
        if result:
            best = result
            best_rank = rank
            break
        
        # Try structured search
        if time.time() < deadline - 5:
            result = _structured_search(rank)
            if result:
                best = result
                best_rank = rank
                break

    return best
# EVOLVE-BLOCK-END
