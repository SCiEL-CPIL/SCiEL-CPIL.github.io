"""Matrix multiplication 4,4,5 rank-minimal decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]
START_TIME = time.time()
TIME_LIMIT = 135.0  # safety margin before 150s kill

def time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def verify_decomposition(U, V, W, n=4, m=4, p=5):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix mult, T[i,j,k] should be 1 if (i,j) is in A, (j,k) in B, else 0
    # Actually the tensor is: T[i,j,k] = 1 for all valid i,j,k
    # Let's verify: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = A[i,j] * B[j,k] * C[k,i]... no
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (since we multiply all entries)
    # Wait, let me re-read the task.
    
    # The matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 for all i,j,k
    # A decomposition of rank r means: T[i,j,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Check a sample of entries for correctness
    # For efficiency, check all entries if rank is small enough
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T[i,j,k] should equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Vectorized check
    # Create index arrays
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                t_val = np.dot(np.dot(u_val, v_val), w_val)
                if abs(t_val - 1.0) > 1e-9:
                    return False
    return True

def verify_decomposition_fast(U, V, W, n=4, m=4, p=5):
    """Faster verification using numpy broadcasting."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    
    # Build the tensor via einsum
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Check if T is all ones
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def schoolbook_decomposition(n=4, m=4, p=5):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def random_decomposition(rank, n=4, m=4, p=5):
    """Generate a random decomposition of given rank."""
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    return U, V, W

def try_rank(rank, n=4, m=4, p=5, max_iters=2000, seed=None):
    """Try to find a decomposition of given rank using iterative refinement."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_iters):
        if time_remaining() < 15:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to avoid scaling issues
        for c in range(rank):
            norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
            if norm > 1e-10:
                U[:, c] /= norm
                V[:, c] /= norm
                W[:, c] /= norm
        
        # Iterative refinement (alternating least squares style)
        for inner in range(50):
            if time_remaining() < 10:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :] given V and W
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # For fixed (i,j), we have p equations in rank unknowns
            # Let A[i,j] be a p x rank matrix where A[i,j][k,c] = V[j*p+k,c] * W[k*n+i,c]
            # Then A[i,j] @ U[i*m+j, :] = ones(p)
            
            # Similarly for V and W
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build A: p x rank
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A @ u = ones(p) in least squares sense
                    # If rank > p, use pseudo-inverse
                    try:
                        u_new = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        continue
                    U[i * m + j, :] = u_new
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # For fixed (j,k): sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v_new = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        continue
                    V[j * p + k, :] = v_new
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # For fixed (k,i): sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w_new = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        continue
                    W[k * n + i, :] = w_new
            
            # Check error
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            err = np.max(np.abs(T - 1.0))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-9:
                # Found exact solution
                return U.tolist(), V.tolist(), W.tolist()
            
            # Early stop if error is not decreasing
            if err > 0.5 and inner > 10:
                break
    
    # Return best found
    if best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    schoolbook = schoolbook_decomposition(n, m, p)
    
    # Try ranks from low to high
    # Lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # For 4x4x5, schoolbook is 80.
    # Known lower bounds: rank >= (n+m+p-2)*something... 
    # Let's try a range of ranks
    
    # Theoretical lower bound for matmul tensor: at least max(n*m, m*p, p*n) / min(n,m,p) * something
    # Simple lower bound: rank >= max(n, m, p) * max(n, m, p) / min(n, m, p) ... not quite
    # For n=m=p, rank >= n^2. For 4,4,5: at least 16? 
    # Let's try ranks from 16 upward, but be strategic
    
    # Try a set of candidate ranks
    candidate_ranks = []
    # Start from reasonable lower bound
    for r in range(16, 81):
        candidate_ranks.append(r)
    
    # Sort by likelihood (try middle ranges first? or low to high?)
    # Let's try low to high, but skip some to save time
    # Actually, let's try a curated set
    
    best_result = schoolbook
    best_rank = n * m * p  # 80
    
    # Try each rank
    for rank in candidate_ranks:
        if time_remaining() < 20:
            break
        if rank >= best_rank:
            break
        
        # Try multiple random seeds for this rank
        found = False
        for attempt in range(3):
            if time_remaining() < 15:
                break
            
            U, V, W = try_rank(rank, n, m, p, max_iters=100, seed=rank * 1000 + attempt)
            
            if U is not None:
                # Verify
                if verify_decomposition_fast(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
                else:
                    # Check how close we got
                    U_arr = np.array(U)
                    V_arr = np.array(V)
                    W_arr = np.array(W)
                    U3 = U_arr.reshape(n, m, rank)
                    V3 = V_arr.reshape(m, p, rank)
                    W3 = W_arr.reshape(p, n, rank)
                    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                    err = np.max(np.abs(T - 1.0))
                    # If very close, try to polish
                    if err < 1e-6:
                        # Try to fix with one more ALS iteration
                        for _ in range(20):
                            for i in range(n):
                                for j in range(m):
                                    A = np.zeros((p, rank))
                                    for k in range(p):
                                        A[k, :] = V_arr[j * p + k, :] * W_arr[k * n + i, :]
                                    u_new = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                                    U_arr[i * m + j, :] = u_new
                            for j in range(m):
                                for k in range(p):
                                    A = np.zeros((n, rank))
                                    for i in range(n):
                                        A[i, :] = U_arr[i * m + j, :] * W_arr[k * n + i, :]
                                    v_new = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                                    V_arr[j * p + k, :] = v_new
                            for k in range(p):
                                for i in range(n):
                                    A = np.zeros((m, rank))
                                    for j in range(m):
                                        A[j, :] = U_arr[i * m + j, :] * V_arr[j * p + k, :]
                                    w_new = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                                    W_arr[k * n + i, :] = w_new
                            
                            U3 = U_arr.reshape(n, m, rank)
                            V3 = V_arr.reshape(m, p, rank)
                            W3 = W_arr.reshape(p, n, rank)
                            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                            err = np.max(np.abs(T - 1.0))
                            if err < 1e-9:
                                break
                        
                        if err < 1e-9:
                            best_result = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
                            best_rank = rank
                            found = True
                            break
        
        if found:
            # Found a valid decomposition, try lower ranks
            continue
    
    return best_result

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Rank: {rank}")
