"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p  # 112
    
    # Build schoolbook factor matrices as numpy arrays
    U = np.zeros((n * m, rank_sb), dtype=np.float64)
    V = np.zeros((m * p, rank_sb), dtype=np.float64)
    W = np.zeros((p * n, rank_sb), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Build the matrix multiplication tensor T_{ijk} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # Actually, the tensor for matrix mult: T[i,j,k] where result C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T_{i j k} = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The tensor is: T[i, j, k] = 1 (it's the standard matmul tensor)
    # In index notation: C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T has indices (i, j, k) with T[i,j,k] = 1 for all valid i,j,k
    
    # Actually the matmul tensor is: T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # Wait no. The tensor is defined by: sum over j of A[i,j] * B[j,k] = C[i,k]
    # The tensor T_{ijk} = 1 means: the (i,j,k) entry contributes to C[i,k]
    # So T[i,j,k] = 1 for all i,j,k (it's just the identity tensor in the sense that
    # every triple (i,j,k) where i indexes A's row, j indexes A's col/B's row, k indexes B's col)
    
    # Let's think more carefully. The matmul tensor M has:
    # M[i, j, k] = 1  (for the standard matmul, the tensor is all-ones in the sense
    # that the (i,j,k) component picks out A[i,j]*B[j,k])
    
    # Actually, the standard way: the matmul tensor T satisfies
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # where T[i,j,k] = 1 for all i,j,k.
    
    # So the tensor is just an n x m x p array of ones.
    T = np.ones((n, m, p), dtype=np.float64)
    
    # The tensor decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # And the mapping is: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Verify schoolbook:
    # For each (i,j,k): U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # Reconstructed: sum_col U[i*m+j,col]*V[j*p+k,col]*W[k*n+i,col]
    # For the (i,j,k) triple: only col corresponding to (i,j,k) has all three = 1
    # So reconstructed[i,j,k] = 1. Correct!
    
    # Now try to find a lower rank decomposition.
    # For 4x4 * 4x7, the known bounds:
    # Lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # Upper bound (schoolbook): 112
    # Known results: For 4x4*4x4, rank is 49 (Strassen-like)
    # For 4x4*4x7, we need to search.
    
    # Strategy: Try to find a decomposition of rank r < 112
    # using numerical tensor decomposition (Tucker/CP) and then round.
    
    # Reshape tensor to matrix form for SVD-based approaches
    # T as (n*m) x (p*n) with middle index m... 
    
    # Actually, let's use the approach: 
    # T[i,j,k] where we want sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = T[i,j,k]
    
    # This is a 3-way tensor CP decomposition with specific index mappings.
    
    # Let's try a different approach: use the fact that for small matrices,
    # we can try random initializations and do alternating least squares or 
    # gradient descent to find a low-rank decomposition.
    
    # First, let's try a simpler approach: exploit structure.
    # For 4x4*4x7, we can try rank 28 (theoretical lower bound) up to 112.
    
    # Let's try to use a known technique: for matmul, there are known
    # fast multiplication algorithms. For 4x4, Strassen gives rank 49.
    # For 4x4*4x7, we can potentially combine.
    
    # Actually, let's just try to numerically find a low-rank decomposition.
    # We'll use the following approach:
    # 1. Start with schoolbook (rank 112)
    # 2. Try to reduce rank by finding linear dependencies
    
    # A better approach: use the "tensor rank" computation via
    # alternating minimization with a target rank.
    
    # For 4x4*4x7, let's try ranks from 28 upward.
    
    # Actually, a practical approach: use the known fact that 
    # R(4,4,4) = 49 (Strassen)
    # R(4,4,7) might be achievable with some combination.
    
    # Let me try a numerical approach: CP decomposition via ALS.
    
    def try_rank(r, max_iters=200, seed=42):
        """Try to find a rank-r decomposition using ALS."""
        if r < 28:  # below theoretical lower bound
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U_r = rng.randn(n * m, r)
        V_r = rng.randn(m * p, r)
        W_r = rng.randn(p * n, r)
        
        # Normalize to prevent blowup
        norms_U = np.linalg.norm(U_r, axis=0)
        norms_V = np.linalg.norm(V_r, axis=0)
        norms_W = np.linalg.norm(W_r, axis=0)
        for c in range(r):
            if norms_U[c] > 0 and norms_V[c] > 0 and norms_W[c] > 0:
                scale = (norms_U[c] * norms_V[c] * norms_W[c]) ** (1.0/3.0)
                U_r[:, c] /= scale
                V_r[:, c] /= scale
                W_r[:, c] /= scale
        
        best_loss = float('inf')
        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve least squares
            # T[i,j,:] = sum_r U[i*m+j, r] * (V[:,r] * W[:,r] reshaped)
            # Actually: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            # For fixed (i,j): T[i,j,:] = sum_r U[i*m+j,r] * (V[j*p+0:r] * W[0::n+i])
            # Hmm, this is getting complex. Let me think differently.
            
            # The tensor T[i,j,k] = 1 for all i,j,k.
            # We want: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # For ALS update of U:
            # For each (i,j), we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is: U[i*m+j,:] @ (V[j*p:,:]*W[0::n+i, :])^T... no
            # V[j*p+k, r] * W[k*n+i, r] for k=0..p-1
            # So for fixed (i,j): the k-th component of (V[j*p+0:j*p+p, :] * W[i:i+p*n:n, :])
            # Wait, W[k*n+i, r] for k=0..p-1 gives rows i, i+n, i+2n, ... of W
            
            # Let me define: for fixed (i,j), define M_{k,r} = V[j*p+k, r] * W[k*n+i, r]
            # Then we need: sum_r U[i*m+j, r] * M_{k,r} = 1 for all k
            # So: U[i*m+j, :] @ M.T = ones(p)
            # i.e., U[i*m+j, :] = ones(p) @ M^{-1} (if invertible) or least squares
            
            # Similarly for V and W.
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    M = V[j*p:(j+1)*p, :] * W[i::n, :].T  # shape (p, r)
                    # Solve: U[i*m+j, :] @ M.T = ones(p)
                    # i.e., M @ U[i*m+j, :]^T = ones(p)
                    # U[i*m+j, :] = argmin ||M @ x - 1||^2
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_r[i*m+j, :] = x
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k): sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
                    # N[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    N = U[j::m, :] * W[k*n::n, :].T  # shape (n, r)
                    # V[j*p+k, :] @ N.T = ones(n)
                    # N @ V[j*p+k, :]^T = ones(n)
                    try:
                        x, _, _, _ = np.linalg.lstsq(N, np.ones(n), rcond=None)
                        V_r[j*p+k, :] = x
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i): sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
                    # P[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    P = U[i*m::m, :] * V[k::p, :].T  # shape (m, r)
                    # W[k*n+i, :] @ P.T = ones(m)
                    # P @ W[k*n+i, :]^T = ones(m)
                    try:
                        x, _, _, _ = np.linalg.lstsq(P, np.ones(m), rcond=None)
                        W_r[k*n+i, :] = x
                    except:
                        pass
            
            # Compute loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        loss += (val - 1.0) ** 2
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            if best_loss < 1e-10:
                break
        
        if best_loss < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from lower bound up
    # Known: R(4,4,4) = 49, so R(4,4,7) >= 28
    # Let's try a range of ranks
    
    best_result = None
    best_rank = rank_sb
    
    # Try some candidate ranks
    candidate_ranks = sorted(set([28, 32, 35, 36, 40, 42, 44, 45, 48, 49, 50, 52, 54, 56, 60, 64, 70, 72, 76, 80, 84, 90, 96, 100, 104, 108, 112]))
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        # Try multiple seeds
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            result = try_rank(r, max_iters=100, seed=seed)
            if result is not None:
                best_result = result
                best_rank = r
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    # If no better found, return schoolbook
    if best_result is None:
        U_out, V_out, W_out = U, V, W
    else:
        U_out, V_out, W_out = best_result
    
    # Convert to lists of lists
    U_list = [[float(U_out[i, j]) for j in range(U_out.shape[1])] for i in range(U_out.shape[0])]
    V_list = [[float(V_out[i, j]) for j in range(V_out.shape[1])] for i in range(V_out.shape[0])]
    W_list = [[float(W_out[i, j]) for j in range(W_out.shape[1])] for i in range(W_out.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
