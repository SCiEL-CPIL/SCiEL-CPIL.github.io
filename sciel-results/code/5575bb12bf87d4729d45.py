"""
Rewritten program for matrix multiplication tensor decomposition.
Task: M_{2,3,10}
Dimensions: n=2, m=3, p=10
Standard rank: 60
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dense 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _normalize(U, V, W):
    """Normalize the factor matrices to prevent numerical drift."""
    max_u = np.max(np.abs(U))
    max_v = np.max(np.abs(V))
    max_w = np.max(np.abs(W))
    
    if max_u > 1e-12:
        U = U / max_u
    if max_v > 1e-12:
        V = V / max_v
    if max_w > 1e-12:
        W = W / max_w
    return U, V, W

def _try_rank(n, m, p, rank, max_iters=1000, seed=42, T=None):
    """Try to find a rank-r decomposition using alternating least squares with normalization."""
    if T is None:
        T = _build_tensor(n, m, p)
        
    rng = np.random.RandomState(seed)
    
    # Initialize
    # Scale initialization to be roughly consistent with the tensor magnitude
    # The tensor entries are 0 or 1.
    # If we pick random normal, the products will be small.
    # Let's initialize with small random values.
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Pre-compute flat tensors for speed
    T_flat_u = T.reshape(n * m, -1)
    T_flat_v = T.reshape(m * p, -1)
    T_flat_w = T.reshape(p * n, -1)
    
    # Dimensions for reshaping
    dim_vw = (m * p, p * n)
    dim_uw = (n * m, p * n)
    dim_uv = (n * m, m * p)
    
    for it in range(max_iters):
        # Update U
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Fix V, W.
        # M[b, c, r] = V[b, r] W[c, r]
        # T_flat_u[a, idx(b,c)] = sum_r U[a, r] M[idx(b,c), r]
        # Let M_flat[r, idx(b,c)] = M[b, c, r]
        # T_flat_u = U @ M_flat.T
        # U = T_flat_u @ M_flat.T^+
        
        M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, rank)
        M_flat = M.transpose(2, 0, 1).reshape(rank, -1)  # (rank, m*p*p*n)
        
        try:
            # Solve U @ M_flat.T = T_flat_u
            # U = T_flat_u @ pinv(M_flat.T)
            # Using lstsq is safer for rectangular
            U = np.linalg.lstsq(M_flat.T, T_flat_u.T, rcond=None)[0].T
        except np.linalg.LinAlgError:
            pass
            
        # Update V
        N = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, rank)
        N_flat = N.transpose(2, 0, 1).reshape(rank, -1)  # (rank, n*m*p*n)
        try:
            V = np.linalg.lstsq(N_flat.T, T_flat_v.T, rcond=None)[0].T
        except np.linalg.LinAlgError:
            pass
            
        # Update W
        P = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
        P_flat = P.transpose(2, 0, 1).reshape(rank, -1)  # (rank, n*m*m*p)
        try:
            W = np.linalg.lstsq(P_flat.T, T_flat_w.T, rcond=None)[0].T
        except np.linalg.LinAlgError:
            pass
            
        # Normalize to keep values in check
        U, V, W = _normalize(U, V, W)
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < 1e-10:
            break
            
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with trivial decomposition
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # Pre-build tensor
    T = _build_tensor(n, m, p)
    
    # Try to find lower-rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = 30
    # We try from 59 down to 30
    # Given time constraints, we prioritize ranks close to 60 first, as they are more likely to be reachable if the rank is just slightly below 60.
    # If the rank is much lower, ALS might struggle, but we'll try.
    
    # We allocate more time/seeds to higher ranks (closer to 60) initially
    # If we find a solution, we record it and continue to see if we can go lower.
    
    # Strategy:
    # 1. Try rank 59 with many seeds.
    # 2. If found, try 58, etc.
    # 3. If not found, move to next rank.
    
    # To maximize chance, we iterate ranks in descending order.
    # We use a dynamic number of seeds based on remaining time.
    
    for r in range(rank_trivial - 1, 29, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Determine number of seeds to try for this rank
        # Use more seeds for ranks closer to the trivial rank, as the "gap" is smaller
        # and local minima might be easier to escape or the solution space is denser.
        # Actually, if the rank is 59, it's very close to 60.
        # Let's try 20 seeds for 59, 10 for 58, etc.
        seeds_to_try = 20 if r > 55 else 10
        
        found = False
        for seed in range(seeds_to_try):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            U_arr, V_arr, W_arr, success = _try_rank(n, m, p, r, max_iters=500, seed=seed, T=T)
            if success:
                # Snap to half-integers
                U_list = [[int(round(val * 2)) / 2 for val in row] for row in U_arr]
                V_list = [[int(round(val * 2)) / 2 for val in row] for row in V_arr]
                W_list = [[int(round(val * 2)) / 2 for val in row] for row in W_arr]
                
                # Verify
                U_chk = np.array(U_list, dtype=np.float64)
                V_chk = np.array(V_list, dtype=np.float64)
                W_chk = np.array(W_list, dtype=np.float64)
                T_recon = np.einsum('ar,br,cr->abc', U_chk, V_chk, W_chk)
                if np.allclose(T_recon, T, atol=1e-10):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    found = True
                    break
        
        # If we found a solution, we continue the loop to try for even lower rank.
        # If we didn't find one, we also continue, as lower ranks might be easier (unlikely) 
        # or we just run out of time.
        
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
