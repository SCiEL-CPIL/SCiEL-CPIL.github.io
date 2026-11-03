"""Matrix multiplication 3,3,5 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if i,j,k correspond to a valid multiplication term, else 0
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # where we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Check: for each (i,k), the sum over j should give the correct product
    # Actually, let's verify by checking that the decomposition reproduces the identity
    # tensor for matrix multiplication
    
    # The tensor T[i,j,k] for matrix multiplication 3x3 * 3x5 is 1 for all
    # i in range(n), j in range(m), k in range(p)
    # Wait, that's not right. Let me reconsider.
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid indices, meaning every term contributes
    # with coefficient 1.
    
    # So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # But that's only true if the decomposition is of the form where each term
    # corresponds to one (i,j,k) triple. In general, the tensor entries are all 1.
    
    # Let's verify by reconstructing the tensor
    T_reconstructed = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
    
    # The target tensor has all entries = 1
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_reconstructed, T_target, atol=tol)

def _try_rank(rank, n, m, p, max_iter=500, seed=None):
    """Try to find a rank-r decomposition using random initialization and gradient-free search."""
    if seed is not None:
        random.seed(seed)
    
    # Initialize with random values in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
    U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, rank))
    V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, rank))
    W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, rank))
    
    # Use a simple local search: randomly perturb entries and accept if error decreases
    def compute_error():
        T_reconstructed = np.zeros((n, m, p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_reconstructed[i, j, k] += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
        return np.sum((T_reconstructed - 1.0) ** 2)
    
    current_error = compute_error()
    if current_error < 1e-9:
        return U.tolist(), V.tolist(), W.tolist()
    
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Randomly perturb one entry
        mat = random.choice([0, 1, 2])
        if mat == 0:
            idx = (random.randint(0, n * m - 1), random.randint(0, rank - 1))
            old_val = U[idx]
            U[idx] = random.choice([-1, -0.5, 0, 0.5, 1])
        elif mat == 1:
            idx = (random.randint(0, m * p - 1), random.randint(0, rank - 1))
            old_val = V[idx]
            V[idx] = random.choice([-1, -0.5, 0, 0.5, 1])
        else:
            idx = (random.randint(0, p * n - 1), random.randint(0, rank - 1))
            old_val = W[idx]
            W[idx] = random.choice([-1, -0.5, 0, 0.5, 1])
        
        new_error = compute_error()
        if new_error < best_error:
            best_error = new_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_error < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()
        else:
            # Revert
            if mat == 0:
                U[idx] = old_val
            elif mat == 1:
                V[idx] = old_val
            else:
                W[idx] = old_val
    
    if best_error < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound for matrix multiplication rank
    lower_bound = max(n * m, m * p, n * p)
    
    # Try ranks from lower_bound to best_rank - 1
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            result = _try_rank(rank, n, m, p, max_iter=200, seed=rank * 1000 + attempt)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify_decomposition(U, V, W, n, m, p, rank):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break  # Found valid decomposition for this rank, try lower
    
    return best
# EVOLVE-BLOCK-END
