"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using structured search
    # Strategy: use sparse random initialization with local repair
    
    for target_rank in range(95, 50, -1):
        if time.time() > deadline:
            break
        
        result = _search_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
    
    # Convert to list format with integer rounding
    def to_list(mat):
        out = []
        for row in mat:
            r = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    r.append(int(round(x)))
                else:
                    r.append(float(x))
            out.append(r)
        return out
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}


def _search_decomposition(n, m, p, target_rank, deadline):
    """Search for a rank-target_rank decomposition."""
    # Multiple restarts with different strategies
    for restart in range(50):
        if time.time() > deadline:
            return None
        
        U, V, W = _initialize(n, m, p, target_rank, restart)
        U, V, W = _local_search(U, V, W, n, m, p, deadline, max_iters=2000)
        
        if _verify(U, V, W, n, m, p):
            return (U, V, W)
    
    return None


def _initialize(n, m, p, r, seed):
    """Create initial sparse decomposition."""
    np.random.seed(seed)
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        U[i*m+j, c] = np.random.choice([-1, 1])
        V[j*p+k, c] = np.random.choice([-1, 1])
        W[k*n+i, c] = np.random.choice([-1, 1])
        
        # Add some extra non-zeros for flexibility
        if np.random.random() < 0.5:
            i2 = np.random.randint(n)
            U[i2*m+j, c] = np.random.choice([-1, 1])
        if np.random.random() < 0.5:
            k2 = np.random.randint(p)
            W[k2*n+i, c] = np.random.choice([-1, 1])
        if np.random.random() < 0.5:
            j2 = np.random.randint(m)
            V[j2*p+k, c] = np.random.choice([-1, 1])
    
    return U, V, W


def _local_search(U, V, W, n, m, p, deadline, max_iters=2000):
    """Local search to fix violations."""
    r = U.shape[1]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find a random violation
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        
        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
        if abs(val - 1.0) < 1e-9:
            continue
        
        # Try to fix by modifying one entry
        c = np.random.randint(r)
        
        # Try U
        for new_val in [-1, 0, 1]:
            if U[i*m+j, c] == new_val:
                continue
            old = U[i*m+j, c]
            U[i*m+j, c] = new_val
            new_val_total = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
            if abs(new_val_total - 1.0) < abs(val - 1.0) or abs(new_val_total - 1.0) < 1e-9:
                break
            U[i*m+j, c] = old
        
        # Try V
        for new_val in [-1, 0, 1]:
            if V[j*p+k, c] == new_val:
                continue
            old = V[j*p+k, c]
            V[j*p+k, c] = new_val
            new_val_total = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
            if abs(new_val_total - 1.0) < abs(val - 1.0) or abs(new_val_total - 1.0) < 1e-9:
                break
            V[j*p+k, c] = old
        
        # Try W
        for new_val in [-1, 0, 1]:
            if W[k*n+i, c] == new_val:
                continue
            old = W[k*n+i, c]
            W[k*n+i, c] = new_val
            new_val_total = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
            if abs(new_val_total - 1.0) < abs(val - 1.0) or abs(new_val_total - 1.0) < 1e-9:
                break
            W[k*n+i, c] = old
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
