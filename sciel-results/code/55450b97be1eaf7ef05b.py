"""Program for matrix multiplication 2x3x7 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7):
    """Verify the decomposition. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"""
    rank = U.shape[1]
    # Pre-compute indices for speed
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            for k in range(p):
                idx_v = j * p + k
                idx_w = k * n + i
                s = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_schoolbook():
    """Return the schoolbook rank-42 decomposition."""
    n, m, p = DIMENSIONS
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

def _to_list(mat):
    """Convert numpy array to list of lists with half-integer precision."""
    return [[float(round(x * 2)) / 2.0 for x in row] for row in mat]

def _random_search(rank, n, m, p, deadline, seed=42):
    """Try to find a rank-rank decomposition using random search + local refinement."""
    np.random.seed(seed)
    random.seed(seed)
    
    val_set = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Try multiple random initializations
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random sparse values
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # Fill some entries randomly
        density = 0.5
        for _ in range(int(rank * density)):
            r = np.random.randint(0, rank)
            u = np.random.randint(0, n * m)
            v = np.random.randint(0, m * p)
            w = np.random.randint(0, p * n)
            val = val_set[np.random.randint(0, len(val_set))]
            if val != 0:
                U[u, r] = val
                V[v, r] = val
                W[w, r] = val
        
        # Check initial
        if _verify(U, V, W, n, m, p):
            return _to_list(U), _to_list(V), _to_list(W), rank
        
        # Local search: coordinate descent
        improved = True
        steps = 0
        while improved and steps < 200 and time.time() < deadline - 3:
            improved = False
            steps += 1
            
            # Pick a random matrix and entry to modify
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                idx = np.random.randint(0, n * m)
                col = np.random.randint(0, rank)
                old_val = U[idx, col]
                new_val = val_set[np.random.randint(0, len(val_set))]
                if new_val == old_val:
                    continue
                U[idx, col] = new_val
                if _verify(U, V, W, n, m, p):
                    return _to_list(U), _to_list(V), _to_list(W), rank
                U[idx, col] = old_val
            elif mat_choice == 1:
                idx = np.random.randint(0, m * p)
                col = np.random.randint(0, rank)
                old_val = V[idx, col]
                new_val = val_set[np.random.randint(0, len(val_set))]
                if new_val == old_val:
                    continue
                V[idx, col] = new_val
                if _verify(U, V, W, n, m, p):
                    return _to_list(U), _to_list(V), _to_list(W), rank
                V[idx, col] = old_val
            else:
                idx = np.random.randint(0, p * n)
                col = np.random.randint(0, rank)
                old_val = W[idx, col]
                new_val = val_set[np.random.randint(0, len(val_set))]
                if new_val == old_val:
                    continue
                W[idx, col] = new_val
                if _verify(U, V, W, n, m, p):
                    return _to_list(U), _to_list(V), _to_list(W), rank
                W[idx, col] = old_val
    
    return None

def construct():
    """Construct the tensor decomposition for 2x3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Fallback: schoolbook
    schoolbook = _build_schoolbook()
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Try ranks from 21 up to 41
    best_result = schoolbook
    best_rank = schoolbook_rank
    
    # Try several ranks with random search
    ranks_to_try = [21, 22, 23, 24, 25, 28, 30, 32, 35, 38, 40, 41]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        if rank >= best_rank:
            continue
        
        # Try random search for this rank
        result = _random_search(rank, n, m, p, deadline, seed=42 + rank)
        if result is not None:
            U, V, W, r = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break  # Found a solution, stop
    
    return best_result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
