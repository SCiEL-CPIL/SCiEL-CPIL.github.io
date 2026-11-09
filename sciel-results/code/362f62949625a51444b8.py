"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m + j, j*p + k, k*n + i] = 1 for all i, j, k, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, T):
    """Verify that U, V, W exactly decompose T."""
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(recon, T, atol=1e-10)

def _check_half_integer(U, V, W):
    """Check if entries are close to integers or half-integers."""
    max_dev_int = max(np.max(np.abs(U - np.round(U))),
                     np.max(np.abs(V - np.round(V))),
                     np.max(np.abs(W - np.round(W))))
    max_dev_half = max(np.max(np.abs(2*U - np.round(2*U))),
                      np.max(np.abs(2*V - np.round(2*V))),
                      np.max(np.abs(2*W - np.round(2*W))))
    return max_dev_int, max_dev_half

def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 48
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # Schoolbook decomposition (rank = n*m*p = 48)
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Precompute tensor slices for efficient ALS updates
    M_i_list = []
    for i in range(n):
        M_i = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                M_i[j, k] = T[i * m + j, j * p + k, k * n + i]
        M_i_list.append(M_i)
    
    M_j_list = []
    for j in range(m):
        M_j = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                M_j[i, k] = T[i * m + j, j * p + k, k * n + i]
        M_j_list.append(M_j)
    
    M_w_list = {}
    for k in range(p):
        for i in range(n):
            M_w = np.zeros(m)
            for j in range(m):
                M_w[j] = T[i * m + j, j * p + k, k * n + i]
            M_w_list[(k, i)] = M_w
    
    # Try ranks from a reasonable lower bound up to schoolbook-1
    start_rank = 30
    for rank in range(start_rank, rank_schoolbook):
        if time.time() > deadline:
            break
        
        # Use ALS (Alternating Least Squares) to try to find exact decomposition
        best_solution = None
        n_restarts = 15
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with different scales
            seed = 42 + restart * 1000 + rank * 7
            np.random.seed(seed)
            
            # Try different initialization strategies
            if restart % 5 == 0:
                # Small random
                U_curr = np.random.randn(n * m, rank) * 0.5
                V_curr = np.random.randn(m * p, rank) * 0.5
                W_curr = np.random.randn(p * n, rank) * 0.5
            elif restart % 5 == 1:
                # Sparse random with some structure
                U_curr = np.zeros((n * m, rank))
                V_curr = np.zeros((m * p, rank))
                W_curr = np.zeros((p * n, rank))
                for c in range(rank):
                    for _ in range(3):
                        idx = np.random.randint(0, n * m)
                        U_curr[idx, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    for _ in range(3):
                        idx = np.random.randint(0, m * p)
                        V_curr[idx, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    for _ in range(3):
                        idx = np.random.randint(0, p * n)
                        W_curr[idx, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            elif restart % 5 == 2:
                # Integer-like initialization
                U_curr = np.random.randint(-2, 3, (n * m, rank)) * 0.5
                V_curr = np.random.randint(-2, 3, (m * p, rank)) * 0.5
                W_curr = np.random.randint(-2, 3, (p * n, rank)) * 0.5
            elif restart % 5 == 3:
                # Mixed scale initialization
                U_curr = np.random.randn(n * m, rank) * np.random.uniform(0.3, 1.0)
                V_curr = np.random.randn(m * p, rank) * np.random.uniform(0.3, 1.0)
                W_curr = np.random.randn(p * n, rank) * np.random.uniform(0.3, 1.0)
            else:
                # Structured initialization based on embedding idea
                # Try to initialize with patterns that respect the rectangular structure
                U_curr = np.zeros((n * m, rank))
                V_curr = np.zeros((m * p, rank))
                W_curr = np.zeros((p * n, rank))
                
                # First n*m*p//3 columns: basic schoolbook-like
                for col in range(min(rank, n*m*p//3)):
                    i = col % n
                    j = (col // n) % m
                    k = (col // (n*m)) % p
                    U_curr[i*m + j, col] = 1
                    V_curr[j*p + k, col] = 1
                    W_curr[k*n + i, col] = 1
                
                # Remaining columns: random perturbations
                for col in range(n*m*p//3, rank):
                    for _ in range(2):
                        idx = np.random.randint(0, n * m)
                        U_curr[idx, col] = np.random.choice([-1, 1]) * np.random.uniform(0.3, 1.0)
                    for _ in range(2):
                        idx = np.random.randint(0, m * p)
                        V_curr[idx, col] = np.random.choice([-1, 1]) * np.random.uniform(0.3, 1.0)
                    for _ in range(2):
                        idx = np.random.randint(0, p * n)
                        W_curr[idx, col] = np.random.choice([-1, 1]) * np.random.uniform(0.3, 1.0)
            
            # ALS iterations
            n_iterations = 80
            prev_residual = np.inf
            for iteration in range(n_iterations):
                if time.time() > deadline:
                    break
                
                # Update U: For each i
                for i in range(n):
                    D_i = np.zeros((m * p, rank))
                    for j in range(m):
                        for k in range(p):
                            D_i[j * p + k, :] = V_curr[j * p + k, :] * W_curr[k * n + i, :]
                    
                    try:
                        U_curr[i * m:(i+1) * m, :] = np.linalg.lstsq(D_i.T, M_i_list[i].flatten(), rcond=None)[0].reshape(m, rank)
                    except:
                        pass
                
                # Update V: For each j
                for j in range(m):
                    D_j = np.zeros((n * p, rank))
                    for i in range(n):
                        for k in range(p):
                            D_j[i * p + k, :] = U_curr[i * m + j, :] * W_curr[k * n + i, :]
                    
                    try:
                        V_curr[j * p:(j+1) * p, :] = np.linalg.lstsq(D_j.T, M_j_list[j].flatten(), rcond=None)[0].reshape(p, rank)
                    except:
                        pass
                
                # Update W: For each (k,i) pair
                for k in range(p):
                    for i in range(n):
                        D_w = np.zeros((m, rank))
                        for j in range(m):
                            D_w[j, :] = U_curr[i * m + j, :] * V_curr[j * p + k, :]
                        
                        try:
                            W_curr[k * n + i, :] = np.linalg.lstsq(D_w.T, M_w_list[(k, i)], rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if iteration % 5 == 0:
                    recon = np.einsum('ic,jc,kc->ijk', U_curr, V_curr, W_curr)
                    residual = np.max(np.abs(recon - T))
                    if residual < 1e-10:
                        break
                    if residual > prev_residual * 1.01 and iteration > 20:
                        # Diverging, break early
                        break
                    prev_residual = residual
            
            # Check if solution is exact
            reconstructed = np.einsum('ic,jc,kc->ijk', U_curr, V_curr, W_curr)
            if np.allclose(reconstructed, T, atol=1e-6):
                # Check if entries are close to integers or half-integers
                max_dev_int, max_dev_half = _check_half_integer(U_curr, V_curr, W_curr)
                
                if max_dev_int < 1e-4:
                    U_int = np.round(U_curr).astype(int)
                    V_int = np.round(V_curr).astype(int)
                    W_int = np.round(W_curr).astype(int)
                    
                    recon_int = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
                    if np.array_equal(recon_int, T):
                        best_solution = (U_int, V_int, W_int)
                        break
                elif max_dev_half < 1e-4:
                    U_half = np.round(2 * U_curr) / 2
                    V_half = np.round(2 * V_curr) / 2
                    W_half = np.round(2 * W_curr) / 2
                    
                    recon_half = np.einsum('ic,jc,kc->ijk', U_half, V_half, W_half)
                    if np.allclose(recon_half, T, atol=1e-10):
                        U_list = U_half.tolist()
                        V_list = V_half.tolist()
                        W_list = W_half.tolist()
                        best_solution = (U_list, V_list, W_list)
                        break
        
        if best_solution is not None:
            U_best, V_best, W_best = best_solution
            best_rank = rank
            if isinstance(U_best, np.ndarray):
                best_U = U_best.tolist()
                best_V = V_best.tolist()
                best_W = W_best.tolist()
            else:
                best_U = U_best
                best_V = V_best
                best_W = W_best
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
