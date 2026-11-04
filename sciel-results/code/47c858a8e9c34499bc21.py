"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NM = N * M
MP = M * P
PN = P * N
TENSOR_SIZE = N * M * P  # 105

def _verify(U, V, W, rank, n, m, p):
    """Check if the decomposition is valid."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_decomposition_from_patterns(patterns, n, m, p):
    """
    Given a list of patterns (each pattern is a dict mapping (i,j,k) to a value),
    construct the U, V, W matrices.
    Each pattern corresponds to one rank-1 term.
    """
    rank = len(patterns)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, pattern in enumerate(patterns):
        for (i, j, k), val in pattern.items():
            U[i*m+j, r] += val
            V[j*p+k, r] += val
            W[k*n+i, r] += val
    
    return U, V, W

def _random_sparse_search(n, m, p, max_rank, deadline, seed=42):
    """
    Random search for low-rank decomposition.
    We try to find patterns where each rank-1 term covers multiple (i,j,k) positions.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    best_rank = max_rank + 1
    best_result = None
    
    # We'll try different ranks and see if we can find a valid decomposition
    # For each rank, we'll try to construct a sparse solution
    
    for target_rank in range(40, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random attempts for this rank
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Generate random sparse patterns
            patterns = []
            for r in range(target_rank):
                pattern = {}
                # Each pattern covers a small number of positions
                num_positions = random.randint(1, 5)
                positions = random.sample(range(n*m*p), num_positions)
                for pos in positions:
                    i = pos // (m*p)
                    rem = pos % (m*p)
                    j = rem // p
                    k = rem % p
                    # Assign a random value
                    val = random.choice([1, -1, 0.5, -0.5, 2, -2])
                    pattern[(i, j, k)] = val
                patterns.append(pattern)
            
            U, V, W = _construct_decomposition_from_patterns(patterns, n, m, p)
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_result is not None:
            break
    
    return best_result, best_rank

def _iterative_hard_thresholding_search(n, m, p, deadline, seed=42):
    """
    Use iterative hard-thresholding to find sparse solutions.
    Start with a dense random initialization and iteratively threshold small values.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    best_rank = n * m * p + 1
    best_result = None
    
    # Start with a moderate rank
    for initial_rank in [60, 50, 45, 40, 35, 30]:
        if time.time() > deadline:
            break
        
        U = np.random.randn(n*m, initial_rank) * 0.1
        V = np.random.randn(m*p, initial_rank) * 0.1
        W = np.random.randn(p*n, initial_rank) * 0.1
        
        # Iteratively threshold and verify
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Compute the current tensor
            tensor = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(initial_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        tensor[i, j, k] = s
            
            # Check how close we are to the identity tensor
            error = np.abs(tensor - 1.0).max()
            
            # Threshold small values
            threshold = 0.01 * (1 + iteration * 0.01)
            U[np.abs(U) < threshold] = 0
            V[np.abs(V) < threshold] = 0
            W[np.abs(W) < threshold] = 0
            
            # Count non-zero columns
            U_nonzero = np.any(U != 0, axis=0)
            V_nonzero = np.any(V != 0, axis=0)
            W_nonzero = np.any(W != 0, axis=0)
            
            # Keep only columns that are non-zero in all three
            keep_mask = U_nonzero & V_nonzero & W_nonzero
            if np.sum(keep_mask) < initial_rank:
                U = U[:, keep_mask]
                V = V[:, keep_mask]
                W = W[:, keep_mask]
                initial_rank = np.sum(keep_mask)
            
            # Check if valid
            if error < 1e-9:
                best_rank = initial_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_result is not None:
            break
    
    return best_result, best_rank

def _structured_decomposition(n, m, p):
    """
    Try to construct a structured decomposition based on known results.
    For 3x5x7, we can try to use a combination of block decompositions.
    """
    # This is a placeholder for a more sophisticated approach
    # For now, return None
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook as fallback
    schoolbook_result = _schoolbook(n, m, p)
    best_result = schoolbook_result
    best_rank = n * m * p  # 105
    
    # Try structured decomposition first
    structured = _structured_decomposition(n, m, p)
    if structured is not None:
        rank = len(structured["U"][0])
        if rank < best_rank:
            best_result = structured
            best_rank = rank
    
    # Try iterative hard-thresholding search
    if time.time() < deadline - 60:
        result, rank = _iterative_hard_thresholding_search(n, m, p, deadline, seed=12345)
        if result is not None and rank < best_rank:
            best_result = result
            best_rank = rank
    
    # Try random sparse search
    if time.time() < deadline - 30:
        result, rank = _random_sparse_search(n, m, p, best_rank, deadline, seed=67890)
        if result is not None and rank < best_rank:
            best_result = result
            best_rank = rank
    
    # Final verification
    if time.time() < deadline - 10:
        U = np.array(best_result["U"])
        V = np.array(best_result["V"])
        W = np.array(best_result["W"])
        rank = U.shape[1]
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fall back to schoolbook
            best_result = _schoolbook(n, m, p)
    
    return best_result

# EVOLVE-BLOCK-END
