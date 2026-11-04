"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _als_refine(U, V, W, n, m, p, max_iters=100, tol=1e-10):
    """Vectorized Alternating Least Squares refinement for tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Precompute index arrays for vectorized operations
    I = np.array([i for i in range(n) for j in range(m)])
    J = np.array([j for i in range(n) for j in range(m)])
    K_idx = np.array([k for j in range(m) for k in range(p)])
    J_idx = np.array([j for j in range(m) for k in range(p)])
    
    for it in range(max_iters):
        # Compute error
        errors = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    errors[idx] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
                    idx += 1
        old_err = np.sum(errors**2)
        
        # Update U: for each (i,j)
        for i in range(n):
            for j in range(m):
                A = np.zeros((rank, rank))
                b = np.zeros(rank)
                for k in range(p):
                    A += np.outer(V[j*p+k], W[k*n+i])
                    b += V[j*p+k] * W[k*n+i]
                try:
                    U[i*m+j] = np.linalg.solve(A + 1e-12*np.eye(rank), b)
                except:
                    U[i*m+j] = np.linalg.lstsq(A + 1e-12*np.eye(rank), b, rcond=None)[0]
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                A = np.zeros((rank, rank))
                b = np.zeros(rank)
                for i in range(n):
                    A += np.outer(U[i*m+j], W[k*n+i])
                    b += U[i*m+j] * W[k*n+i]
                try:
                    V[j*p+k] = np.linalg.solve(A + 1e-12*np.eye(rank), b)
                except:
                    V[j*p+k] = np.linalg.lstsq(A + 1e-12*np.eye(rank), b, rcond=None)[0]
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                A = np.zeros((rank, rank))
                b = np.zeros(rank)
                for j in range(m):
                    A += np.outer(U[i*m+j], V[j*p+k])
                    b += U[i*m+j] * V[j*p+k]
                try:
                    W[k*n+i] = np.linalg.solve(A + 1e-12*np.eye(rank), b)
                except:
                    W[k*n+i] = np.linalg.lstsq(A + 1e-12*np.eye(rank), b, rcond=None)[0]
        
        errors = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    errors[idx] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
                    idx += 1
        new_err = np.sum(errors**2)
        
        if abs(old_err - new_err) < tol:
            break
    
    return U, V, W, np.sqrt(new_err / (n*m*p))

def _svd_reduce(U, V, W, target_rank, n, m, p):
    """Try to reduce rank via SVD on combined factors."""
    rank = U.shape[1]
    if rank <= target_rank:
        return U, V, W, rank
    
    # Try to find a low-rank approximation by projecting
    # Use the fact that if we have a rank-r decomposition, we can try to
    # find a lower rank one by optimizing over a lower-dimensional subspace
    
    # Approach: parameterize U = U0 @ P where P is rank x target_rank
    # and optimize P, Q, R such that U=U0@P, V=V0@Q, W=W0@R
    # with P, Q, R being orthogonal or having some structure
    
    # Simpler: just try random projections and ALS
    best_U, best_V, best_W, best_err = None, None, None, float('inf')
    
    for attempt in range(10):
        np.random.seed(42 + attempt)
        P = np.random.randn(rank, target_rank)
        Q = np.random.randn(rank, target_rank) 
        R = np.random.randn(rank, target_rank)
        
        U_new = U @ P
        V_new = V @ Q
        W_new = W @ R
        
        U_ref, V_ref, W_ref, err = _als_refine(U_new, V_new, W_new, n, m, p, max_iters=50)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_ref, V_ref, W_ref
    
    if best_err < 1e-8:
        return best_U, best_V, best_W, target_rank
    return None, None, None, float('inf')

def _structured_init(n, m, p, rank):
    """Create a structured initial guess based on matrix multiplication structure."""
    np.random.seed(123)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Distribute rank across (i,j,k) triples with some structure
    # Try to use the fact that T[i,j,k] = 1 for all i,j,k
    
    # Initialize with random but structured values
    for c in range(rank):
        # Each rank component contributes to a subset of (i,j,k)
        # Use a hash-like function to assign
        i0 = c % n
        j0 = (c // n) % m
        k0 = (c // (n*m)) % p
        
        U[i0 * m + j0, c] = 1.0
        V[j0 * p + k0, c] = 1.0
        W[k0 * n + i0, c] = 1.0
        
        # Add some cross terms
        if c < rank // 2:
            i1 = (i0 + 1) % n
            j1 = (j0 + 1) % m
            k1 = (k0 + 1) % p
            U[i1 * m + j1, c] += 0.5
            V[j1 * p + k1, c] += 0.5
            W[k1 * n + i1, c] += 0.5
    
    # Add random noise to break symmetry
    U += 0.1 * np.random.randn(n * m, rank)
    V += 0.1 * np.random.randn(m * p, rank)
    W += 0.1 * np.random.randn(p * n, rank)
    
    return U, V, W

def _try_rank_als(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition using ALS with structured init."""
    if time.time() > deadline:
        return None
    
    # Try multiple structured inits
    for init_type in range(5):
        if time.time() > deadline:
            break
        
        if init_type == 0:
            U, V, W = _structured_init(n, m, p, rank)
        elif init_type == 1:
            np.random.seed(100 + init_type)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            # Add schoolbook-like structure
            for c in range(min(rank, n*m*p)):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] += 1.0
                V[j*p+k, c] += 1.0
                W[k*n+i, c] += 1.0
        elif init_type == 2:
            # Block structure
            U, V, W = np.zeros((n*m, rank)), np.zeros((m*p, rank)), np.zeros((p*n, rank))
            block_size = max(1, rank // (n*m*p // 10))
            for c in range(rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                # Add neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            ii = (i + di) % n
                            jj = (j + dj) % m
                            kk = (k + dk) % p
                            if (ii, jj, kk) != (i, j, k):
                                U[ii*m+jj, c] += 0.3
                                V[jj*p+kk, c] += 0.3
                                W[kk*n+ii, c] += 0.3
        elif init_type == 3:
            # Fourier-like structure
            U, V, W = np.zeros((n*m, rank)), np.zeros((m*p, rank)), np.zeros((p*n, rank))
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, c] = np.cos(2*np.pi*c*i/(n*rank)) * np.sin(2*np.pi*c*j/(m*rank))
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, c] = np.cos(2*np.pi*c*j/(m*rank)) * np.sin(2*np.pi*c*k/(p*rank))
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, c] = np.cos(2*np.pi*c*k/(p*rank)) * np.sin(2*np.pi*c*i/(n*rank))
        else:
            # Pure random with scaling
            np.random.seed(200 + init_type)
            scale = 1.0 / (rank ** 0.5)
            U = np.random.randn(n * m, rank) * scale
            V = np.random.randn(m * p, rank) * scale
            W = np.random.randn(p * n, rank) * scale
        
        U_ref, V_ref, W_ref, err = _als_refine(U, V, W, n, m, p, max_iters=200)
        
        if err < 1e-8:
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U_ref]
            V_list = [[float(x) for x in row] for row in V_ref]
            W_list = [[float(x) for x in row] for row in W_ref]
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Known lower bounds for matrix multiplication:
    # For 3x5x6, the rank is at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But actually the tensor rank is at least the max of the support sizes
    # Let's try ranks from 45 down to 30
    
    for rank in range(45, 29, -1):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank_als(n, m, p, rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best = {"U": U, "V": V, "W": W}
                # Try to go lower
                continue
            else:
                # Verification failed, skip
                continue
    
    # If we found something, also try SVD reduction from schoolbook
    if best_rank == n * m * p and time.time() < deadline - 10:
        U_arr = np.array([[float(x) for x in row] for row in best["U"]])
        V_arr = np.array([[float(x) for x in row] for row in best["V"]])
        W_arr = np.array([[float(x) for x in row] for row in best["W"]])
        
        for target_rank in [60, 50, 45, 40]:
            if time.time() > deadline - 5:
                break
            U_r, V_r, W_r, actual_rank = _svd_reduce(U_arr, V_arr, W_arr, target_rank, n, m, p)
            if U_r is not None and actual_rank < best_rank:
                U_list = [[float(x) for x in row] for row in U_r]
                V_list = [[float(x) for x in row] for row in V_r]
                W_list = [[float(x) for x in row] for row in W_r]
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_rank = actual_rank
                    best = {"U": U_list, "V": V_list, "W": W_list}
    
    return best
# EVOLVE-BLOCK-END
