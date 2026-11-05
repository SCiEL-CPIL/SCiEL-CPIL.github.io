"""Rewritten program: search for low-rank exact tensor decomposition of 3x4 * 4x5 multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

def _construct_schoolbook():
    """Return the schoolbook rank-60 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _validate(U, V, W, n=3, m=4, p=5, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all (i,j,k) and 0 otherwise (but all (i,j,k) are valid inputs)
    # Actually the multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
    # Wait: the matrix multiplication tensor T satisfies (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = delta_{i,j,k} where the tensor is trilinear form: T(A,B) = A*B
    # The tensor decomposition requires: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # And T[i,j,k] should be 1 for all i,j,k (since every entry of A*B is a sum of products)
    # Actually no: T[i,j,k] is the coefficient such that (A*B)[i,k] = sum_j A[i,j]*B[j,k]*T[i,j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all i,j,k
    # Using einsum
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # We need T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _try_random_search(n=3, m=4, p=5, max_rank=56, time_limit=120, seed=42):
    """Try to find a low-rank decomposition via random initialization + local search."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    deadline = time.time() + time_limit - 5  # 5s safety margin
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, try the known rank-56 structure if we can construct it
    # For 3x4x5, the Strassen-like or other known decompositions
    # Let's try random rank r from low to high
    
    for target_rank in range(40, min(max_rank + 1, n*m*p + 1)):
        if time.time() > deadline:
            break
        
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random initialization with small integers/half-integers
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with random values from {-1, 0, 1, 0.5, -0.5}
            for c in range(target_rank):
                # Each column has a sparse pattern
                for idx in range(n*m):
                    if rng.random() < 0.3:
                        U[idx, c] = rng.choice([-1, 1, 0.5, -0.5])
                for idx in range(m*p):
                    if rng.random() < 0.3:
                        V[idx, c] = rng.choice([-1, 1, 0.5, -0.5])
                for idx in range(p*n):
                    if rng.random() < 0.3:
                        W[idx, c] = rng.choice([-1, 1, 0.5, -0.5])
            
            # Check if this random assignment works
            if _validate(U.tolist(), V.tolist(), W.tolist()):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
            
            # Try local perturbation: flip individual entries
            for _ in range(200):
                if time.time() > deadline:
                    break
                # Pick a random matrix and entry to flip
                mat_idx = rng.choice([0, 1, 2])
                if mat_idx == 0:
                    idx = rng.randint(0, n*m-1)
                    col = rng.randint(0, target_rank-1)
                    old_val = U[idx, col]
                    U[idx, col] = rng.choice([-1, 1, 0.5, -0.5, 0])
                    if _validate(U.tolist(), V.tolist(), W.tolist()):
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break
                    else:
                        U[idx, col] = old_val
                elif mat_idx == 1:
                    idx = rng.randint(0, m*p-1)
                    col = rng.randint(0, target_rank-1)
                    old_val = V[idx, col]
                    V[idx, col] = rng.choice([-1, 1, 0.5, -0.5, 0])
                    if _validate(U.tolist(), V.tolist(), W.tolist()):
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break
                    else:
                        V[idx, col] = old_val
                else:
                    idx = rng.randint(0, p*n-1)
                    col = rng.randint(0, target_rank-1)
                    old_val = W[idx, col]
                    W[idx, col] = rng.choice([-1, 1, 0.5, -0.5, 0])
                    if _validate(U.tolist(), V.tolist(), W.tolist()):
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break
                    else:
                        W[idx, col] = old_val
            
            if best_result is not None and best_rank <= target_rank:
                break
    
    if best_result is not None:
        return best_result
    return None

def _try_structured_search(n=3, m=4, p=5, time_limit=120):
    """Try structured patterns based on known low-rank decompositions."""
    deadline = time.time() + time_limit - 5
    
    # Try a pattern inspired by Strassen's algorithm generalized
    # For 3x4x5, we can try rank 48, 49, 50, ...
    # Use a systematic approach: try all sparse patterns with limited support
    
    # Actually, let's try a more principled approach:
    # The multiplication tensor for n x m by m x p has rank at most n*m*p (schoolbook)
    # Lower bounds exist. For 3x4x5, let's just try to find any improvement over 60.
    
    # Try rank 55-60 with structured random search
    best_result = None
    best_rank = n * m * p
    
    for target_rank in range(55, 61):
        if time.time() > deadline:
            break
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Structured initialization: each column corresponds to a subset
            for c in range(target_rank):
                # Random subset of indices for each matrix
                u_idx = set()
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.4:
                            u_idx.add(i*m+j)
                v_idx = set()
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.4:
                            v_idx.add(j*p+k)
                w_idx = set()
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.4:
                            w_idx.add(k*n+i)
                
                for idx in u_idx:
                    U[idx, c] = random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, c] = random.choice([-1, 1])
                for idx in w_idx:
                    W[idx, c] = random.choice([-1, 1])
            
            if _validate(U.tolist(), V.tolist(), W.tolist()):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_result is not None:
            break
    
    return best_result

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s, leaving 5s margin
    
    # Try structured search first
    result = _try_structured_search(time_limit=deadline - time.time())
    if result is not None:
        return result
    
    # Try random search
    remaining = deadline - time.time()
    if remaining > 10:
        result = _try_random_search(time_limit=remaining)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _construct_schoolbook()
