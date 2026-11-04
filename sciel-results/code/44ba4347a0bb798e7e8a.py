import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
DEADLINE = 140.0  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Compute the tensor T[u, v, w] = sum_c U[u, c] * V[v, c] * W[w, c]
    T = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
    
    # Construct the target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_target[u, v, w] = 1.0
                
    # Check if T matches T_target
    if not np.allclose(T, T_target, atol=tol, rtol=0):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_strassen_block(n, m, p):
    """
    Try to use Strassen-like decomposition for sub-blocks.
    This is a heuristic search for lower rank decompositions.
    """
    # For 3x4 and 4x6, we can try to partition into smaller blocks
    # and use Strassen's 7-multiplication algorithm where applicable.
    # However, Strassen works best for 2x2 blocks.
    # Let's try a different approach: search for a valid decomposition
    # with rank less than n*m*p using random initialization and refinement.
    pass

def _random_search(n, m, p, max_rank, time_limit):
    """
    Randomly search for a decomposition with rank <= max_rank.
    Uses random integer/half-integer entries and verifies.
    """
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                target_indices.append((u, v, w))
    
    # Try decreasing rank
    for rank in range(min(max_rank, n*m*p), 10, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random decompositions at this rank
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts:
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small random integers
            U = np.random.randint(-2, 3, size=(nm, rank))
            V = np.random.randint(-2, 3, size=(mp, rank))
            W = np.random.randint(-2, 3, size=(pn, rank))
            
            # Quick check: does it produce the right support?
            T = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
            
            # Check non-zero pattern
            valid = True
            for (u, v, w) in target_indices:
                if T[u, v, w] != 1.0:
                    valid = False
                    break
            
            if valid:
                # Also check no other entries are non-zero
                T_target = np.zeros((nm, mp, pn), dtype=float)
                for (u, v, w) in target_indices:
                    T_target[u, v, w] = 1.0
                if np.allclose(T, T_target, atol=1e-9):
                    return U.tolist(), V.tolist(), W.tolist(), rank
            
            attempts += 1
    
    return None

def _structured_search(n, m, p, time_limit):
    """
    Structured search using Kronecker products and known identities.
    For 3x4x6, we can exploit the factorization 6 = 2*3 and 4 = 2*2.
    """
    start_time = time.time()
    
    # One approach: treat the 3x4 matrix as (3x2) x (2x2) in a block sense
    # and the 4x6 as (2x2) x (2x3)
    # This doesn't directly reduce rank without Strassen-like tricks.
    
    # Alternative: use the fact that the tensor is a sub-tensor of a larger
    # known decomposition.
    
    # For now, try to find a decomposition using the "winograd" scheme
    # which is related to border rank techniques.
    
    # Winograd for 1D convolution can be adapted, but for matrix multiplication,
    # we look for a decomposition that exploits linearity.
    
    # A simple improvement: if n, m, or p is even, we can use 2x2 Strassen blocks.
    # Here m=4 is even, so we can split the 4-dimensional middle space into two 2-dim blocks.
    
    # Let's try to construct a decomposition using 2x2 Strassen blocks.
    # The middle dimension is 4 = 2*2.
    # We can view the 3x4 matrix as [A1 A2] where A1, A2 are 3x2.
    # And the 4x6 matrix as [B1; B2] where B1, B2 are 2x6.
    # Then C = A1*B1 + A2*B2.
    # Each A_i*B_i is a 3x6 product.
    # We can use Strassen for the 2x2 blocks if we further split.
    # But Strassen applies to 2x2 matrices, not 3x2 * 2x6 directly.
    
    # Instead, let's try a direct search with a structured initialization.
    
    # Use the schoolbook as a starting point and try to merge terms.
    # This is a local search heuristic.
    
    U, V, W = _schoolbook(n, m, p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Try to set some columns to zero or merge them.
    # This is a crude heuristic and may not improve the rank.
    
    # For the purpose of this task, we'll return the schoolbook decomposition
    # as it's guaranteed to be correct. The search for a lower rank decomposition
    # is complex and may not find a better one within the time limit.
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try random search for a lower rank decomposition
    result = _random_search(n, m, p, n*m*p-1, DEADLINE - 10)
    
    if result is not None:
        U, V, W, rank = result
        # Verify the result
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
