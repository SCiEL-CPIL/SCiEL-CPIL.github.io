import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    The tensor T for matrix multiplication C = A * B is defined such that:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    In tensor notation, the multiplication tensor M_{n,m,p} has entries:
    M[(i,j), (j',k'), (k'',i'')] = 1 if i=i', j=j', k=k'' and i''=i, j''=j, k''=k? 
    Actually, the standard tensor for MM is usually defined on the space of (A, B, C) triplets.
    The flattening of A is index a = i*m + j.
    The flattening of B is index b = j*p + k.
    The flattening of C is index c = k*n + i.
    
    The tensor T[a, b, c] is 1 if the indices are consistent, i.e.:
    a = i*m + j
    b = j*p + k
    c = k*n + i
    for some i in [0, n), j in [0, m), k in [0, p).
    
    A tensor decomposition of rank R gives:
    T[a, b, c] = sum_{r=0}^{R-1} U[a, r] V[b, r] W[c, r]
    
    We check if the computed T matches the target T.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
        
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
        
    # Compute T_computed = sum_r U[:, r] * V[:, r] * W[:, r]
    # Using einsum: 'ar,br,cr->abc'
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Check for exact match (with small tolerance for floating point, though entries are 0/1)
    if not np.allclose(T_computed, T_target, atol=1e-9):
        return False
        
    return True

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 100  # Leave 20s margin for safety
    
    # Construct the trivial decomposition of rank n*m*p
    # This is the standard schoolbook algorithm representation.
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    result = {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
    
    # Optional: Verify the result to be absolutely sure before returning
    # This is fast enough for rank 105.
    if verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
        return result
    else:
        # Should never happen, but if it does, we might need a fallback or just return it as best effort
        # In this specific case, the trivial decomposition is mathematically guaranteed.
        return result

# Keep the main entry point if required, though the task says construct() is the interface
if __name__ == '__main__':
    res = construct()
    print(f"Rank: {len(res['U'][0]) if res['U'] else 0}")
