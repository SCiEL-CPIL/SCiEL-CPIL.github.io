import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 7]

def _build_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def _support_indices(n, m, p):
    """Return list of (idx_u, idx_v, idx_w) for all support entries."""
    sup = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                sup.append((i * m + j, j * p + k, k * n + i))
    return sup

def _check_exact(U, V, W, n, m, p, support):
    """
    Check that the decomposition is exact using Fraction.
    U, V, W are lists of lists of floats (representing half-integers).
    """
    rank = len(U[0]) if U else 0
    U_frac = [[Fraction(U[i][j]).limit_denominator(100) for j in range(rank)] for i in range(n*m)]
    V_frac = [[Fraction(V[i][j]).limit_denominator(100) for j in range(rank)] for i in range(m*p)]
    W_frac = [[Fraction(W[i][j]).limit_denominator(100) for j in range(rank)] for i in range(p*n)]
    
    support_set = set(support)
    
    # Check support entries sum to 1
    for (a, b, c) in support:
        s = sum(U_frac[a][r] * V_frac[b][r] * W_frac[c][r] for r in range(rank))
        if s != 1:
            return False
            
    # Check non-support entries sum to 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) in support_set:
                    continue
                s = sum(U_frac[a][r] * V_frac[b][r] * W_frac[c][r] for r in range(rank))
                if s != 0:
                    return False
    return True

def _build_structured_init(n, m, p, rank, seed_val):
    """
    Build a structured initialization using block-circulant embedding.
    Pad to a common block size and use Strassen-like identities.
    """
    np.random.seed(seed_val)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    val_set = np.array([-1, -0.5, 0, 0.5, 1], dtype=float)
    
    # Try to create structure by grouping columns into blocks
    # For each block, assign values that create algebraic relationships
    
    # Strategy: Use the fact that for matrix multiplication, we can use
    # identities like (a+b)(c+d) = ac + ad + bc + bd to reduce multiplications
    
    # For 2x3 * 3x7, we can think of it as:
    # Row i of result = sum_j A[i,j] * (column j of B)
    
    # Let's try a block structure where we group the inner dimension
    # m=3, so we can try to use 2 blocks of sizes 2 and 1
    
    # For each rank position, assign values based on a pattern
    for r in range(rank):
        # Assign random but structured values
        block_size = max(1, rank // 3)
        
        # U values
        for idx in range(n * m):
            i = idx // m
            j = idx % m
            # Create correlation between i, j, and r
            pattern = (i * 2 + j + r) % 5
            if np.random.rand() < 0.4:
                U[idx, r] = val_set[pattern]
        
        # V values
        for idx in range(m * p):
            j = idx // p
            k = idx % p
            pattern = (j * 2 + k + r) % 5
            if np.random.rand() < 0.4:
                V[idx, r] = val_set[pattern]
        
        # W values
        for idx in range(p * n):
            k = idx // n
            i = idx % n
            pattern = (k * 2 + i + r) % 5
            if np.random.rand() < 0.4:
                W[idx, r] = val_set[pattern]
    
    return U, V, W

def _calculate_error_fast(U, V, W, n, m, p, target_tensor):
    """
    Calculate the squared error between the reconstructed tensor and the target.
    Uses einsum for efficiency.
    """
    rank = U.shape[1]
    UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
    R = np.einsum('abr,cr->abc', UV, W)
    diff = R - target_tensor
    return np.sum(diff**2)

def _local_search(U, V, W, n, m, p, target_tensor, support, deadline, max_steps=3000):
    """
    Perform local search with simulated annealing.
    Returns (U, V, W, error, found_exact)
    """
    val_set = np.array([-1, -0.5, 0, 0.5, 1], dtype=float)
    current_error = _calculate_error_fast(U, V, W, n, m, p, target_tensor)
    
    T_temp = 0.5
    cooling = 0.995
    min_temp = 0.01
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Propose a move: change one entry in one matrix
        mat_choice = np.random.randint(0, 3)
        if mat_choice == 0:
            mat = U
            rows = n * m
        elif mat_choice == 1:
            mat = V
            rows = m * p
        else:
            mat = W
            rows = p * n
        
        idx = np.random.randint(0, rows)
        col = np.random.randint(0, mat.shape[1])
        old_val = mat[idx, col]
        new_val = val_set[np.random.randint(0, len(val_set))]
        
        if old_val == new_val:
            continue
        
        mat[idx, col] = new_val
        new_error = _calculate_error_fast(U, V, W, n, m, p, target_tensor)
        
        delta = new_error - current_error
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T_temp, 1e-10)):
            current_error = new_error
            if current_error < 1e-9:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _check_exact(U_list, V_list, W_list, n, m, p, support):
                    return U, V, W, current_error, True
        else:
            mat[idx, col] = old_val
        
        T_temp *= cooling
        if T_temp < min_temp:
            T_temp = min_temp
    
    return U, V, W, current_error, False

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    support = _support_indices(n, m, p)
    support_set = set(support)
    
    # Build target tensor
    target_tensor = np.zeros((n*m, m*p, p*n))
    for (a, b, c) in support:
        target_tensor[a, b, c] = 1.0

    # Schoolbook fallback
    best_result = _build_schoolbook(n, m, p)
    best_rank = n * m * p

    np.random.seed(42)
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But we know from theory that for matrix multiplication, the rank is at least
    # the maximum of the three dimensions' products
    
    # Try a range of ranks, focusing on promising ones
    ranks_to_try = list(range(25, 42))
    
    # Add some higher ranks for safety
    ranks_to_try = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Number of attempts per rank - more for lower ranks (harder)
        max_attempts = 8 if rank <= 30 else 5
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Build structured initialization
            U, V, W = _build_structured_init(n, m, p, rank, seed_val=42 + attempt * 7 + rank)
            
            # Check if already a solution (unlikely but possible)
            current_error = _calculate_error_fast(U, V, W, n, m, p, target_tensor)
            if current_error < 1e-9:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _check_exact(U_list, V_list, W_list, n, m, p, support):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
            
            # Local search
            U, V, W, final_error, found = _local_search(
                U, V, W, n, m, p, target_tensor, support,
                deadline=deadline - 3,
                max_steps=2000
            )
            
            if found:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                return best_result
    
    # If we haven't found a better solution, try a few more with random init
    for rank in [30, 32, 34, 36, 38]:
        if time.time() > deadline - 5:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 3:
                break
            
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            val_set = np.array([-1, -0.5, 0, 0.5, 1], dtype=float)
            density = 0.3
            
            mask_u = np.random.rand(n*m, rank) < density
            mask_v = np.random.rand(m*p, rank) < density
            mask_w = np.random.rand(p*n, rank) < density
            
            U[mask_u] = val_set[np.random.randint(0, len(val_set), size=np.sum(mask_u))]
            V[mask_v] = val_set[np.random.randint(0, len(val_set), size=np.sum(mask_v))]
            W[mask_w] = val_set[np.random.randint(0, len(val_set), size=np.sum(mask_w))]
            
            U, V, W, final_error, found = _local_search(
                U, V, W, n, m, p, target_tensor, support,
                deadline=deadline - 2,
                max_steps=1500
            )
            
            if found:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                return best_result
    
    return best_result
