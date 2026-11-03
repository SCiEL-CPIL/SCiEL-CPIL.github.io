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
DEADLINE = 140  # seconds, safety margin before 150s kill

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

def _check_exact(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    # Compute the reconstructed tensor
    # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] ... wait, that's not right
    # The decomposition is: T[a, b, c] = sum_r U[a, r] * V[b, r] * ... no
    # Actually for a rank-r decomposition of a 3-tensor:
    # T[a, b, c] = sum_{r=0}^{rank-1} U[a, r] * V[b, r] * W[c, r]
    # But wait, the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # So T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    rank = U.shape[1]
    # Use einsum for efficiency
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _try_rank(n, m, p, rank, max_iters=500, seed=42):
    """Try to find a rank-r decomposition using alternating least squares."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    # Initialize with larger variance to escape local minima
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Pre-compute flat tensors for speed
    T_flat_u = T.reshape(n * m, -1)
    T_flat_v = T.reshape(m * p, -1)
    T_flat_w = T.reshape(p * n, -1)
    
    for it in range(max_iters):
        # Update U
        M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, rank)
        M_flat = M.transpose(2, 0, 1).reshape(rank, -1)  # (rank, m*p*p*n)
        try:
            U = np.linalg.lstsq(M_flat.T, T_flat_u.T, rcond=None)[0].T
        except:
            pass
        
        # Update V
        N = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, rank)
        N_flat = N.transpose(2, 0, 1).reshape(rank, -1)  # (rank, n*m*p*n)
        try:
            V = np.linalg.lstsq(N_flat.T, T_flat_v.T, rcond=None)[0].T
        except:
            pass
        
        # Update W
        P = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
        P_flat = P.transpose(2, 0, 1).reshape(rank, -1)  # (rank, n*m*m*p)
        try:
            W = np.linalg.lstsq(P_flat.T, T_flat_w.T, rcond=None)[0].T
        except:
            pass
        
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
    
    # Try to find lower-rank decompositions
    T = _build_tensor(n, m, p)
    
    # Try ranks from best_rank - 1 down to the lower bound.
    # It is easier to find a solution for rank 59 than 30 if one exists.
    # If we find a solution, we stop.
    lower_bound = max(n * m, m * p, p * n)
    
    for r in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try multiple seeds for this rank
        found = False
        # Use more seeds for higher ranks (closer to trivial) as they might be more robust?
        # Or just standard.
        for seed in range(10):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            U_arr, V_arr, W_arr, success = _try_rank(n, m, p, r, max_iters=300, seed=seed)
            if success:
                # Convert to list of lists with integer/half-integer check
                # Snap to half-integers
                U_list = [[int(round(val * 2)) / 2 for val in row] for row in U_arr]
                V_list = [[int(round(val * 2)) / 2 for val in row] for row in V_arr]
                W_list = [[int(round(val * 2)) / 2 for val in row] for row in W_arr]
                
                # Verify the rounded solution is still exact
                U_chk = np.array(U_list, dtype=np.float64)
                V_chk = np.array(V_list, dtype=np.float64)
                W_chk = np.array(W_list, dtype=np.float64)
                T_recon = np.einsum('ar,br,cr->abc', U_chk, V_chk, W_chk)
                if np.allclose(T_recon, T, atol=1e-10):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    found = True
                    break
        
        if found:
            # We found a lower rank. We could continue to see if we can go lower,
            # but given the time budget and difficulty, stopping at the first found
            # significant improvement is a safe strategy to ensure we return a valid
            # improved result rather than running out of time.
            # However, the goal is "rank-minimal".
            # If we have time, we should try to go lower.
            # But if we just found 59, trying 58 is a new search.
            # Let's continue the loop to try lower ranks if time permits.
            continue
        else:
            # If we couldn't find rank r, it's possible lower ranks are also hard.
            # But we must check.
            pass
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
