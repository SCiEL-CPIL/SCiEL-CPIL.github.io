"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Build the tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # Shape: (n, m, p)
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Start with schoolbook rank
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    def verify(U, V, W):
        # Check if the decomposition is exact
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - T[i,j,k]) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return U, V, W

    # Try to find a lower rank decomposition using optimization
    # We'll use a random restart strategy with gradient descent or similar
    # But first, let's see if we can find a known lower bound or structure
    
    # For 3x5x6, the schoolbook rank is 90.
    # Can we do better?
    # Let's try a simple local search: randomly initialize U, V, W with a lower rank
    # and try to fit T.
    
    def random_init(rank):
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        return U, V, W

    def compute_tensor(U, V, W):
        n, m, p = DIMENSIONS
        T_comp = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_comp[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        return T_comp

    def error(U, V, W):
        T_comp = compute_tensor(U, V, W)
        return np.linalg.norm(T_comp - T, 'fro')

    # Try a few ranks below 90
    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]
    
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe
    
    # Try rank reduction
    for rank in range(best_rank - 1, 50, -1):
        if time.time() > deadline:
            break
        # Try a few random inits
        for attempt in range(10):
            if time.time() > deadline:
                break
            U, V, W = random_init(rank)
            # Simple gradient descent or alternating least squares
            # Let's use a simple alternating optimization
            for step in range(50):
                if time.time() > deadline:
                    break
                # Update U
                for i in range(n):
                    for j in range(m):
                        # T[i,j,:] = sum_k W[k*n+i, :] * V[j*p+k, :] * U[i*m+j, :]
                        # This is tricky because of the 3-way product.
                        # Let's use a simpler heuristic: randomly perturb and accept if error decreases
                        pass
                
                # Actually, let's just use a simple random search with local perturbations
                # Or better yet, use scipy.optimize if available, but we need to be careful with time.
                # Let's stick to a simpler approach: random initialization and check if it's close.
                # If not, we'll just return the schoolbook decomposition.
                
                # For now, let's just try random initialization and see if we get lucky.
                # This is unlikely to work for low ranks, but it's a start.
                pass
                
        # If we can't find a better rank, keep the schoolbook one.
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
