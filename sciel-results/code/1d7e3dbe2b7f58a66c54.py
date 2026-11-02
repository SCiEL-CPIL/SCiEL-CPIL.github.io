"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Fast verification using NumPy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum: T[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need to be careful with indices
    # Let's compute it as: for each c, add U_r[:,:,c] * V_r[:,:,c] * W_r[:,:,c] but with proper index mapping
    # W_r is indexed by [k, i, c], so W_r[k,i,c]
    # We need: sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    
    # Create the full tensor
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U_r[:, :, c]  # (n, m)
        vc = V_r[:, :, c]  # (m, p)
        wc = W_r[:, :, c]  # (p, n)
        # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        # This is: for each i,j,k: uc[i,j] * vc[j,k] * wc[k,i]
        # We can compute this as an outer product chain
        # tmp[j,k] = vc[j,k] * wc[k,i] for each i... but wc depends on i
        # Let's do it with broadcasting
        # tmp[i,j,k] = uc[i,j] * vc[j,k] * wc[k,i]
        # = uc[:,j,None] * vc[None,:,k] * wc[k,None,i]... this gets complex
        # Simpler: loop over i
        for i in range(n):
            # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            # For fixed i: T[i,:,:] += uc[i,:] * (vc * wc[:,i])
            # vc[j,k] * wc[k,i] is element-wise product of vc (m,p) and wc[:,i] (p,) broadcast to (m,p)
            wc_i = wc[:, i]  # (p,)
            # vc[j,k] * wc_i[k] -> (m,p)
            prod = vc * wc_i[None, :]  # (m, p)
            T[i, :, :] += uc[i, :] * prod  # (m,) * (m,p) broadcast -> (m,p)
    
    return np.allclose(T, 1.0, atol=1e-6)

def _verify_slow(U, V, W, n, m, p):
    """Slow but reliable verification."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > 1e-6:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _als_decomposition(n, m, p, rank, iters=30, seed=0):
    """Attempt to find a decomposition of given rank using ALS."""
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for it in range(iters):
        # Update U: for each (i,j), solve U[i,j,:] @ M[i,j] = ones(p)
        # where M[i,j][c,k] = V_r[j,k,c] * W_r[k,i,c]
        for i in range(n):
            for j in range(m):
                # M[c,k] = V_r[j,k,c] * W_r[k,i,c]
                # V_r[j] is (p, rank), W_r[:,i] is (p, rank)
                M = (V_r[j, :, :] * W_r[:, i, :]).T  # (rank, p)
                try:
                    U_r[i, j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve V[j,k,:] @ B[j,k] = ones(n)
        # where B[j,k][c,i] = U_r[i,j,c] * W_r[k,i,c]
        for j in range(m):
            for k in range(p):
                # B[c,i] = U_r[i,j,c] * W_r[k,i,c]
                # U_r[:,j] is (n, rank), W_r[k] is (n, rank)
                B = (U_r[:, j, :] * W_r[k, :, :]).T  # (rank, n)
                try:
                    V_r[j, k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve W[k,i,:] @ C[k,i] = ones(m)
        # where C[k,i][c,j] = U_r[i,j,c] * V_r[j,k,c]
        for k in range(p):
            for i in range(n):
                # C[c,j] = U_r[i,j,c] * V_r[j,k,c]
                # U_r[i] is (m, rank), V_r[:,k] is (m, rank)
                C = (U_r[i, :, :] * V_r[:, k, :]).T  # (rank, m)
                try:
                    W_r[k, i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Resync
        U = U_r.reshape(n * m, rank)
        V = V_r.reshape(m * p, rank)
        W = W_r.reshape(p * n, rank)
    
    # Verify
    if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _structured_init(n, m, p, rank, pattern="block"):
    """Create a structured initialization."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    if pattern == "block":
        # Try to assign blocks
        # For simplicity, use a semi-random structured init
        rng = np.random.RandomState(42)
        for c in range(rank):
            # Each column c corresponds to a "term"
            # Assign random but structured indices
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            if i < n and j < m and k < p:
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = max(20, 25, 20) = 25
    # But for exact integer decompositions, likely higher
    
    # First, try a few structured approaches
    # Then use ALS with various ranks and seeds
    
    ranks_to_try = list(range(30, 60)) + list(range(60, 80, 2)) + list(range(80, 100, 5))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple seeds
        max_seeds = 5 if target_rank < 50 else 3
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            result = _als_decomposition(n, m, p, target_rank, iters=20, seed=seed)
            if result is not None:
                # Attempt to round to integer/half-integer
                U_rounded = [[round(2 * x) / 2 for x in row] for row in result['U']]
                V_rounded = [[round(2 * x) / 2 for x in row] for row in result['V']]
                W_rounded = [[round(2 * x) / 2 for x in row] for row in result['W']]
                
                # Verify the rounded version
                if _verify_slow(U_rounded, V_rounded, W_rounded, n, m, p):
                    if target_rank < best_rank:
                        best = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                        best_rank = target_rank
                        if best_rank <= 30:
                            return best
                else:
                    # Try the unrounded version if it's close
                    if _verify_slow(result['U'], result['V'], result['W'], n, m, p):
                        # Check if values are close to half-integers
                        all_close = True
                        for mat in [result['U'], result['V'], result['W']]:
                            for row in mat:
                                for x in row:
                                    if abs(x * 2 - round(x * 2)) > 1e-4:
                                        all_close = False
                                        break
                            if not all_close:
                                break
                        if all_close:
                            if target_rank < best_rank:
                                best = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                                best_rank = target_rank
    
    # If we found something better, use it
    if best_rank < n * m * p:
        return best
    
    # Otherwise, return schoolbook
    return best
# EVOLVE-BLOCK-END
