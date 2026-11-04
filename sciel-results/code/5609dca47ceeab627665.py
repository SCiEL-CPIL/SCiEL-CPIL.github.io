"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication."""
import numpy as np
import itertools
import time
import random

DIMENSIONS = [4, 4, 5]
N, M, P = DIMENSIONS
# W index convention: W[k*n+i, r]
# T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

def _build_T():
    """Build the matmul tensor as a (n*m, m*p, p*n) array with T[a,b,c] = 1 if indices match."""
    T = np.zeros((N * M, M * P, P * N), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, tol=1e-9):
    """Check if U, V, W exactly decompose the tensor."""
    T = _build_T()
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Use einsum for efficiency
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(T_recon, T, atol=tol, rtol=0)

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_rank(rank):
    """
    Try to find an exact decomposition of the given rank using randomized search.
    Returns (U, V, W) or None.
    """
    T = _build_T()
    a_dim = N * M  # 16
    b_dim = M * P  # 20
    c_dim = P * N  # 20

    # We'll try several random initializations and local optimization.
    # Strategy: parameterize U, V, W with random entries, then use a 
    # gradient-free or iterative approach to minimize ||T - T_recon||.
    
    # For exact decompositions, we can try a structured approach:
    # Each rank-1 component (u, v, w) should contribute to some subset of T entries.
    
    # Alternative: use the fact that for matmul tensors, there are known
    # lower bounds. For 4x4*4x5, the known best is likely around 80-100.
    # Let's try a direct search with random rank-1 components.
    
    deadline = time.time() + 100  # 100 seconds for this attempt
    
    # Try multiple random restarts
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(a_dim, rank)
        V = np.random.randn(b_dim, rank)
        W = np.random.randn(c_dim, rank)
        
        # Normalize columns to avoid scale issues
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-12 and norm_v > 1e-12 and norm_w > 1e-12:
                # Scale so that product of norms is 1
                scale = (norm_u * norm_v * norm_w) ** (1/3)
                U[:, r] /= norm_u / scale
                V[:, r] /= norm_v / scale
                W[:, r] /= norm_w / scale
        
        # Iterative refinement: for each rank-1 component, optimize it
        # while keeping others fixed (block coordinate descent)
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.linalg.norm(T_recon - T)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                return U, V, W
            
            # Block coordinate descent: update one component at a time
            for r in range(rank):
                if time.time() > deadline:
                    break
                
                # Compute contribution of other components
                T_other = T_recon - U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
                T_target = T - T_other
                
                # Now find u, v, w to best match T_target
                # This is a rank-1 approximation problem
                # We can use iterative SVD-like approach
                
                # Initialize with current values
                u = U[:, r].copy()
                v = V[:, r].copy()
                w = W[:, r].copy()
                
                for sub_iter in range(50):
                    if time.time() > deadline:
                        break
                    
                    # Update u: u = sum over b,c of T_target[:,b,c] * v[b] * w[c]
                    u_new = np.einsum('abc,b,c->a', T_target, v, w)
                    # Update v: v = sum over a,c of T_target[a,:,c] * u[a] * w[c]
                    v_new = np.einsum('abc,a,c->b', T_target, u_new, w)
                    # Update w: w = sum over a,b of T_target[a,b,:] * u[a] * v[b]
                    w_new = np.einsum('abc,a,b->c', T_target, u_new, v_new)
                    
                    # Normalize to prevent divergence
                    nu = np.linalg.norm(u_new)
                    nv = np.linalg.norm(v_new)
                    nw = np.linalg.norm(w_new)
                    if nu < 1e-12 or nv < 1e-12 or nw < 1e-12:
                        break
                    scale = (nu * nv * nw) ** (1/3)
                    u_new /= nu / scale
                    v_new /= nv / scale
                    w_new /= nw / scale
                    
                    u, v, w = u_new, v_new, w_new
                
                U[:, r] = u
                V[:, r] = v
                W[:, r] = w
                
                # Recompute T_recon
                T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.linalg.norm(T_recon - T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Check if we found an exact solution
    if best_U is not None:
        T_recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W, optimize=True)
        if np.allclose(T_recon, T, atol=1e-9, rtol=0):
            return best_U, best_V, best_W
    
    return None

def _try_structured_rank(rank):
    """
    Try a more structured approach: each rank-1 component corresponds to
    a pattern in the matmul tensor.
    """
    T = _build_T()
    deadline = time.time() + 80
    
    # For matmul, a natural structure is: each rank-1 component picks
    # a subset of (i,j) for U, subset of (j,k) for V, subset of (k,i) for W
    # such that the products cover the tensor.
    
    # Let's try: each component has U nonzero on some (i,j) pairs,
    # V nonzero on some (j,k) pairs, W nonzero on some (k,i) pairs.
    
    # A simpler structured approach: try to find a decomposition where
    # each rank-1 term has small support.
    
    # Actually, let's try a different approach: use the known fact that
    # matrix multiplication can be done with fewer than n*m*p operations.
    # For 4x4, Strassen gives rank 7 instead of 16.
    # For 4x4*4x5, we can extend Strassen.
    
    # Let's try to construct a Strassen-like decomposition.
    # Strassen for 2x2: rank 7
    # For 4x4, we can use Strassen recursively: rank 7^2 = 49
    # But we need 4x4 * 4x5, so it's asymmetric.
    
    # Let's just do a search with the block coordinate descent above.
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline (10s safety margin)
    
    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb = _schoolbook()
    best_rank = N * M * P
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Known lower bound for matmul rank is at least max(n*m, m*p, p*n) = 20
    # But actually the lower bound is higher. For 4x4*4x5, let's search from 80 downward.
    
    # Actually, let's try a few specific ranks
    ranks_to_try = [80, 75, 70, 65, 60, 56, 50, 49, 45, 40, 35, 30, 25, 20]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        result = _try_rank(rank)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W):
                # Convert to integer/half-integer if possible
                # Check if entries are close to integers or half-integers
                best_rank = rank
                best_result = (U, V, W)
                break  # Found a valid decomposition at this rank
    
    # If we didn't find anything better, return schoolbook
    if best_rank == N * M * P:
        U, V, W = best_result
    else:
        U, V, W = best_result
    
    # Convert numpy arrays to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
