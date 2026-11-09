"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 72
DEADLINE = 130.0  # safety margin before 150s kill


def verify_fast(U, V, W, rank):
    """Faster verification using numpy."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=1e-9)


def schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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


def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 72)
    2. Try to exploit structure by decomposing into blocks
    3. Use known low-rank decompositions for smaller tensors
    4. Random search with structured initializations
    
    The key insight is that for rectangular matrices, we can sometimes
    achieve better than n*m*p by sharing computations across rows/columns.
    """
    start_time = time.time()
    
    # Start with schoolbook as guaranteed valid solution
    best = schoolbook()
    best_rank = N * M * P
    
    # Strategy 1: Try block decomposition
    # Decompose 3x4x6 into blocks and use known results
    # For 2x2x3, optimal rank is 7 (not 6)
    # For 2x2x2, optimal rank is 7 (Strassen)
    
    # Try to construct a better decomposition using structure
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # One approach: use the fact that we can factor through intermediate dimensions
    # If we can find matrices A, B, C such that the contraction works...
    
    # Let's try a systematic search for lower rank
    # Start with ranks that might be achievable
    target_ranks = [70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50]
    
    for target_rank in target_ranks:
        if time.time() > DEADLINE - 30:
            break
        
        # Try random search with this rank
        result = _try_random_decomposition(target_rank, time_budget=15)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    # If we didn't find anything better, try structured approaches
    if best_rank == N * M * P:
        # Try to use the structure of the problem
        # For 3x4x6, we can think of it as:
        # C[3x6] = A[3x4] * B[4x6]
        # Each C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Try to find a decomposition that shares some terms
        # This is equivalent to finding a low-rank factorization of the tensor
        
        # Use gradient-based optimization to find a valid decomposition
        result = _try_gradient_optimization(60, time_budget=30)
        if result is not None:
            best = result
            best_rank = 60
    
    # Final verification
    if not verify_fast(best["U"], best["V"], best["W"], len(best["U"][0])):
        best = schoolbook()
    
    return best


def _try_random_decomposition(rank, time_budget):
    """Try random search for a valid decomposition of given rank."""
    n, m, p = N, M, P
    start = time.time()
    rng = np.random.RandomState(42)
    
    # Try different initialization strategies
    for strategy in range(5):
        if time.time() - start > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        if strategy == 0:
            # Pure random
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
        elif strategy == 1:
            # Sparse random
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for _ in range(rank * 2):
                U[np.random.randint(n * m), np.random.randint(rank)] = rng.choice([-1, 1])
                V[np.random.randint(m * p), np.random.randint(rank)] = rng.choice([-1, 1])
                W[np.random.randint(p * n), np.random.randint(rank)] = rng.choice([-1, 1])
        elif strategy == 2:
            # Block structure
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            # Assign blocks of rows to different columns
            for i in range(n):
                for j in range(m):
                    cols = np.random.choice(rank, size=min(3, rank), replace=False)
                    for c in cols:
                        U[i * m + j, c] = rng.choice([-1, 1])
                    V[j * p + np.random.randint(p), c] = rng.choice([-1, 1])
                    W[np.random.randint(p) * n + i, c] = rng.choice([-1, 1])
        elif strategy == 3:
            # Try to use Kronecker product structure
            # 3x4x6 = (1x2x3) ⊗ (3x2x2) approximately
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            # Initialize with structured patterns
            for c in range(rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            # Add random perturbations
            U += rng.randn(n * m, rank) * 0.1
            V += rng.randn(m * p, rank) * 0.1
            W += rng.randn(p * n, rank) * 0.1
        else:
            # Integer entries from {-1, 0, 1, 2}
            U = rng.randint(-1, 3, size=(n * m, rank)).astype(float)
            V = rng.randint(-1, 3, size=(m * p, rank)).astype(float)
            W = rng.randint(-1, 3, size=(p * n, rank)).astype(float)
        
        if verify_fast(U, V, W, rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def _try_gradient_optimization(rank, time_budget):
    """Try gradient-based optimization to find a valid decomposition."""
    n, m, p = N, M, P
    start = time.time()
    rng = np.random.RandomState(123)
    
    # Initialize with schoolbook-like structure
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Set up initial values
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = (i * m * p + j * p + k) % rank
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    
    # Add small random perturbations
    U += rng.randn(n * m, rank) * 0.01
    V += rng.randn(m * p, rank) * 0.01
    W += rng.randn(p * n, rank) * 0.01
    
    # Simple iterative refinement
    for iteration in range(100):
        if time.time() - start > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        # Check if we're close to a solution
        if verify_fast(U, V, W, rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Compute error
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        error = np.sum((T - 1.0) ** 2)
        
        if error < 1e-6:
            # Try to polish
            for _ in range(10):
                if verify_fast(U, V, W, rank):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                U += rng.randn(n * m, rank) * 0.001
                V += rng.randn(m * p, rank) * 0.001
                W += rng.randn(p * n, rank) * 0.001
        
        # Simple gradient step (approximate)
        # For each (i,j,k), we want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Gradient with respect to U[i*m+j,c] is V[j*p+k,c]*W[k*n+i,c]
        # But this is complex, so use a simpler approach: perturb and accept if better
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_error = error
        
        for _ in range(5):
            U_trial = U + rng.randn(n * m, rank) * 0.01
            V_trial = V + rng.randn(m * p, rank) * 0.01
            W_trial = W + rng.randn(p * n, rank) * 0.01
            
            Ur_t = U_trial.reshape(n, m, rank)
            Vr_t = V_trial.reshape(m, p, rank)
            Wr_t = W_trial.reshape(p, n, rank)
            T_t = np.einsum('ijc,jkc,kic->ijk', Ur_t, Vr_t, Wr_t)
            error_t = np.sum((T_t - 1.0) ** 2)
            
            if error_t < best_error:
                best_U, best_V, best_W = U_trial, V_trial, W_trial
                best_error = error_t
        
        U, V, W = best_U, best_V, best_W
    
    return None


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Valid: {verify_fast(result['U'], result['V'], result['W'], len(result['U'][0]))}")
