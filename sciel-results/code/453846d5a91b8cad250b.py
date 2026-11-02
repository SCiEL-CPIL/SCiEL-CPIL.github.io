import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s budget with margin

    # Build the target tensor
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    # We'll try to find a rank-r decomposition
    # Using the Strassen-like block construction as initial guess
    
    # Standard Rank 39 Construction
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    r = 0

    def add_strassen_block(k0, k1):
        nonlocal r
        v_e, v_f = k0, k1
        v_g, v_h = 11 + k0, 11 + k1
        w_c00, w_c10 = 2 * k0, 2 * k0 + 1
        w_c01, w_c11 = 2 * k1, 2 * k1 + 1

        U[0, r] = 1; U[3, r] = 1
        V[v_e, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1; W[w_c11, r] = 1
        r += 1

        U[2, r] = 1; U[3, r] = 1
        V[v_e, r] = 1
        W[w_c10, r] = 1; W[w_c11, r] = -1
        r += 1

        U[0, r] = 1
        V[v_f, r] = 1; V[v_h, r] = -1
        W[w_c01, r] = 1; W[w_c11, r] = 1
        r += 1

        U[3, r] = 1
        V[v_g, r] = 1; V[v_e, r] = -1
        W[w_c00, r] = 1; W[w_c10, r] = 1
        r += 1

        U[0, r] = 1; U[1, r] = 1
        V[v_h, r] = 1
        W[w_c00, r] = -1; W[w_c01, r] = 1
        r += 1

        U[2, r] = 1; U[0, r] = -1
        V[v_e, r] = 1; V[v_f, r] = 1
        W[w_c11, r] = 1
        r += 1

        U[1, r] = 1; U[3, r] = -1
        V[v_g, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1
        r += 1

    for block in range(5):
        if time.time() > deadline - 30:
            break
        add_strassen_block(block * 2, block * 2 + 1)

    # Last k=10 (odd) handled with 4 additional terms
    k2 = 10
    U[0, r] = 1; V[k2, r] = 1; W[2*k2, r] = 1; r += 1
    U[1, r] = 1; V[11+k2, r] = 1; W[2*k2, r] = 1; r += 1
    U[2, r] = 1; V[k2, r] = 1; W[2*k2+1, r] = 1; r += 1
    U[3, r] = 1; V[11+k2, r] = 1; W[2*k2+1, r] = 1; r += 1

    assert r == rank, f"Expected rank {rank}, got {r}"
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    # Verify the base construction
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    base_err = np.linalg.norm(T_recon.flatten() - T_flat)
    
    # Try to reduce rank using SVD-based linear algebra
    # Strategy: try removing terms and see if the remaining can still span the tensor
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # First, check if any single term can be removed
    for idx in range(rank):
        if time.time() > deadline - 20:
            break
        
        # Remove term idx
        mask = np.ones(rank, dtype=bool)
        mask[idx] = False
        U_red = U[:, mask]
        V_red = V[:, mask]
        W_red = W[:, mask]
        
        T_red = np.einsum('ar,br,cr->abc', U_red, V_red, W_red)
        err = np.linalg.norm(T_red.flatten() - T_flat)
        
        if err < 1e-10:
            best_U, best_V, best_W = U_red.copy(), V_red.copy(), W_red.copy()
            best_rank = rank - 1
            U, V, W = U_red, V_red, W_red
            rank = rank - 1
    
    # Try removing pairs
    for i in range(rank):
        if time.time() > deadline - 15:
            break
        for j in range(i+1, rank):
            mask = np.ones(rank, dtype=bool)
            mask[[i, j]] = False
            U_red = U[:, mask]
            V_red = V[:, mask]
            W_red = W[:, mask]
            
            T_red = np.einsum('ar,br,cr->abc', U_red, V_red, W_red)
            err = np.linalg.norm(T_red.flatten() - T_flat)
            
            if err < 1e-10:
                best_U, best_V, best_W = U_red.copy(), V_red.copy(), W_red.copy()
                best_rank = rank - 2
    
    # Use SVD to check for more subtle dependencies
    # Try to find a minimal set by examining the tensor's structure
    if time.time() < deadline - 25 and best_rank == 39:
        # Build the 39 rank-1 tensors and check their span
        # If the tensor has intrinsic rank < 39, we can find a smaller basis
        
        # Try rank 38 with random initialization + optimization
        # This is a search for a different decomposition
        
        # For each subset of size 38, check if T is in the span
        # This is expensive, so we use a smarter approach:
        
        # The matrix multiplication tensor for 2x2 * 2x11 has a specific structure.
        # Let's try to exploit the fact that p=11 is odd.
        
        # Alternative: try a different block structure
        # Use 4 blocks of size 2 + 1 term for k=10, but try a different decomposition
        
        # Actually, let's try to see if rank 38 is achievable by solving a least squares problem
        # Fix 38 terms and see if we can find U,V,W that work
        
        # This is underdetermined in general, so let's try a different approach:
        # Use the fact that the tensor has 8*22*22 = 3872 entries, but only 44 are nonzero
        # Wait, let me recount: n*m*p = 2*2*11 = 44 nonzero entries
        
        # The tensor is sparse. Let's try to find a rank decomposition using iterative methods.
        
        pass  # Keep the simple removal strategy for now

    # Final verification
    T_final = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    final_err = np.linalg.norm(T_final.flatten() - T_flat)
    
    if final_err > 1e-8:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook))
        V_sb = np.zeros((m*p, rank_schoolbook))
        W_sb = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
