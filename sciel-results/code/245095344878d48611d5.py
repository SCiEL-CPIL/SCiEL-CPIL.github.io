"""Matrix multiplication 3x4x5 - rank-minimal tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS
N = n * m  # 12
M = m * p  # 20
P = p * n  # 15

# Build the matrix multiplication tensor T[i,j,k] = 1 if indices align
# U shape: (N, rank), V shape: (M, rank), W shape: (P, rank)
# Convention: W[k*n+i, r]
# The tensor to decompose: T_{(i,j),(j,k),(k,i)} = 1
# We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid (i,j,k)
# and = 0 for all invalid (i,j,k)

# Let's define the full tensor of size N*M*P
# Index (a, b, c) where a = i*m+j (0..N-1), b = j'*p+k (0..M-1), c = k'*n+i' (0..P-1)
# The tensor T[a,b,c] = 1 if the indices are consistent (j==j' and k==k' and i==i')
# i.e., a = i*m+j, b = j*p+k, c = k*n+i

def build_tensor():
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def verify_decomposition(U, V, W, T=None):
    """Verify that U, V, W exactly decompose the tensor."""
    if T is None:
        T = build_tensor()
    rank = U.shape[1]
    # Compute sum_r U[:, :, r] * V[:, :, r] * W[:, :, r]
    # Using einsum
    result = np.einsum('ar,br,cr->abc', U, V, W)
    # Check if it matches T
    if not np.allclose(result, T, atol=1e-10):
        return False
    # Also check that entries are integer or half-integer
    for mat in [U, V, W]:
        for val in mat:
            if not (abs(val * 2 - round(val * 2)) < 1e-10):
                return False
    return True

def schoolbook_decomposition():
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((N, rank), dtype=np.float64)
    V = np.zeros((M, rank), dtype=np.float64)
    W = np.zeros((P, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def try_rank(r, T, max_iters=200, timeout=140.0):
    """
    Try to find a rank-r decomposition using iterative refinement.
    Uses a simple multiplicative update approach.
    """
    start_time = time.time()
    
    # Initialize randomly
    U = np.random.randn(N, r)
    V = np.random.randn(M, r)
    W = np.random.randn(P, r)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > timeout - 5:
            break
        
        # Update U: fix V, W, solve for U
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each a: T[a,:,:] = U[a,:] @ diag(V[b,r]*W[c,r]) ... 
        # Actually: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
        # Let's reshape: for fixed a, T[a,:,:] is a (M,P) matrix
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = (V * U[a,:])_b * W[c,:] ... no
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Let X = V * diag(U[a,:]) ... no
        # T[a] (M x P) = sum_r U[a,r] * (V[:,r] * W[:,r].T)  -- outer product V[:,r] @ W[:,r].T
        # So T[a] = V @ diag(U[a,:]) @ W.T  ... let me verify
        # (V @ diag(U[a,:]) @ W.T)[b,c] = sum_r V[b,r] * U[a,r] * W[c,r]  YES!
        
        # So for each a: T[a,:,:] = V @ diag(U[a,:]) @ W.T
        # We want to find U[a,:] that minimizes ||T[a,:,:] - V @ diag(U[a,:]) @ W.T||
        # This is a linear system in U[a,:]
        # Let's vectorize: vec(T[a]) = (W ⊗ V) @ vec(U[a,:])  ... 
        # Actually: (V @ diag(U[a,:]) @ W.T).vec = (W ⊗ V) @ U[a,:]
        # Where W ⊗ V is the Kronecker product, shape (M*P, r)
        
        WV = np.kron(W, V)  # (M*P, r)
        
        # For each a, solve: WV @ U[a,:] = T[a,:,:].flatten()
        # Use least squares
        for a in range(N):
            target = T[a, :, :].flatten()
            # Solve WV @ x = target
            try:
                U[a, :] = np.linalg.lstsq(WV, target, rcond=None)[0]
            except:
                pass
        
        # Update V
        VW = np.kron(W, U)  # (N*P, r)
        for b in range(M):
            # T[:,b,:] = U @ diag(V[b,:]) @ W.T
            # vec = (W ⊗ U) @ V[b,:]
            target = T[:, b, :].flatten()
            try:
                V[b, :] = np.linalg.lstsq(VW, target, rcond=None)[0]
            except:
                pass
        
        # Update W
        WU = np.kron(U, V)  # (N*M, r)
        for c in range(P):
            # T[:,:,c] = U @ diag(V[:,r]) @ W[c,r] ... 
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed c: T[:,:,c] (N x M) = sum_r W[c,r] * (U[:,r] @ V[:,r].T)
            # vec(T[:,:,c]) = (V ⊗ U) @ W[c,:]
            target = T[:, :, c].flatten()
            try:
                W[c, :] = np.linalg.lstsq(WU, target, rcond=None)[0]
            except:
                pass
        
        # Compute error
        result = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(result - T)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-8:
                # Snap to nearest half-integers
                best_U = np.round(best_U * 2) / 2
                best_V = np.round(best_V * 2) / 2
                best_W = np.round(best_W * 2) / 2
                if verify_decomposition(best_U, best_V, best_W, T):
                    return best_U, best_V, best_W
    
    if best_error < 1e-8:
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        if verify_decomposition(best_U, best_V, best_W, T):
            return best_U, best_V, best_W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    T = build_tensor()
    
    # First try the schoolbook (always works, rank 60)
    U, V, W = schoolbook_decomposition()
    best_rank = n * m * p
    best_result = (U, V, W)
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(N, M, P) = max(12, 20, 15) = 20
    # Known bounds for 3x4x5 matrix multiplication rank:
    # The rank is at least max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Let's try ranks from 20 upward
    
    random.seed(42)
    np.random.seed(42)
    
    for r in range(20, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        found = None
        max_trials = 10 if r <= 25 else 5
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            if trial == 0:
                # Random normal
                U = np.random.randn(N, r)
                V = np.random.randn(M, r)
                W = np.random.randn(P, r)
            elif trial == 1:
                # Sparse random
                U = np.zeros((N, r))
                V = np.zeros((M, r))
                W = np.zeros((P, r))
                for col in range(r):
                    # Pick random non-zero pattern
                    nnz_u = max(1, N // 3)
                    nnz_v = max(1, M // 3)
                    nnz_w = max(1, P // 3)
                    rows_u = np.random.choice(N, nnz_u, replace=False)
                    rows_v = np.random.choice(M, nnz_v, replace=False)
                    rows_w = np.random.choice(P, nnz_w, replace=False)
                    U[rows_u, col] = np.random.choice([-1, 1], nnz_u)
                    V[rows_v, col] = np.random.choice([-1, 1], nnz_v)
                    W[rows_w, col] = np.random.choice([-1, 1], nnz_w)
            else:
                U = np.random.randn(N, r) * 0.5
                V = np.random.randn(M, r) * 0.5
                W = np.random.randn(P, r) * 0.5
            
            # Iterative refinement
            iters = 100
            best_err = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                WV = np.kron(W, V)
                for a in range(N):
                    target = T[a, :, :].flatten()
                    try:
                        U[a, :] = np.linalg.lstsq(WV, target, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                VW = np.kron(W, U)
                for b in range(M):
                    target = T[:, b, :].flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(VW, target, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                WU = np.kron(U, V)
                for c in range(P):
                    target = T[:, :, c].flatten()
                    try:
                        W[c, :] = np.linalg.lstsq(WU, target, rcond=None)[0]
                    except:
                        pass
                
                result = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(result - T)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < 1e-8:
                        break
            
            if best_err < 1e-8:
                # Snap to half-integers
                best_U = np.round(best_U * 2) / 2
                best_V = np.round(best_V * 2) / 2
                best_W = np.round(best_W * 2) / 2
                if verify_decomposition(best_U, best_V, best_W, T):
                    found = (best_U, best_V, best_W)
                    break
        
        if found is not None:
            best_rank = r
            best_result = found
            break  # Found minimal rank for this r, try lower... but we're going up
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
