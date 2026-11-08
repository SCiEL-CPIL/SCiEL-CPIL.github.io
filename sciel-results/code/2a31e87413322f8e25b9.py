"""Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 130  # Safety margin before 150s kill

def _build_schoolbook():
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
    return U, V, W

def _verify(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                # Wait, that's not right. The tensor is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T_{i,j,k} = 1 for all i,j,k
    # Use einsum for speed
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a 3-index tensor. Let's compute it.
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is a contraction over r with indices i,j,k
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)

def _try_reduce_rank(U, V, W, max_iter=5000):
    """Try to reduce rank by finding linear dependencies."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    start = time.time()
    
    for _ in range(max_iter):
        if time.time() - start > DEADLINE - 20:
            break
        
        # Try to find two columns that can be merged
        # If U[:,a] = c*U[:,b] and V[:,a] = c*V[:,b] and W[:,a] = c*W[:,b], we can merge
        # More generally, look for linear dependencies
        
        # Simple approach: try removing one column and checking if the remaining
        # can still span the tensor (this is expensive)
        
        # Better: use the fact that if the rank-1 tensors are linearly dependent,
        # we can reduce. Check if any column is a linear combination of others.
        
        # Try random pairs to see if one is redundant
        a = np.random.randint(rank)
        b = np.random.randint(rank - 1)
        if b >= a:
            b += 1
        
        # Check if removing column a is still valid
        # This is expensive, so do it sparsely
        if np.random.random() < 0.1:  # Only 10% of the time do full check
            U_test = np.delete(U, a, axis=1)
            V_test = np.delete(V, a, axis=1)
            W_test = np.delete(W, a, axis=1)
            if _verify_fast(U_test, V_test, W_test, tol=1e-8):
                U, V, W = U_test, V_test, W_test
                rank -= 1
    
    return U, V, W

def construct():
    """Construct the matrix multiplication decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Start with schoolbook
    U, V, W = _build_schoolbook()
    rank = U.shape[1]
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(U, V, W)
    rank = U.shape[1]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Valid: {_verify_fast(U, V, W)}")
