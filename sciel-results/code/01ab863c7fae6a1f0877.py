import numpy as np
import time
from scipy.linalg import solve

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-9):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1  # A[0,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 0, col] = 1  # C[0,6]
            col += 1
            U[1 * m + j, col] = 1  # A[1,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 1, col] = 1  # C[1,6]
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    def get_structured_rank48():
        """
        Exploit n=2 structure to get rank 48.
        
        Use 5 Strassen blocks (35 rank) for columns 0-5, then handle
        columns 5-6 with a custom 2x2 x 2x2 decomposition that shares
        with the previous block, saving 2 ranks.
        
        Actually, let's use a different approach: use 6 Strassen blocks
        but modify the last one to share with column 6.
        
        For now, return the same rank-50 decomposition but with a flag
        that we'll try to reduce via ALS.
        """
        return get_rank50()

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()
    
    # Verify rank 50
    if not verify_decomp(U50, V50, W50):
        # Fallback: schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()

    def try_reduce_rank(U, V, W, target_rank, rng, max_attempts=5):
        """
        Try to reduce rank from current to target_rank by finding a valid
        lower-rank decomposition. Uses structured perturbation + ALS polish.
        """
        current_rank = U.shape[1]
        if target_rank >= current_rank:
            return None
        
        # Precompute target tensor slices for efficient ALS
        T_U = T.reshape(n*m, m*p*p*n)  # shape (n*m, m*p*p*n)
        T_V = T.reshape(m*p, p*n*n*m)  # shape (m*p, p*n*n*m)
        T_W = T.reshape(p*n, n*m*m*p)  # shape (p*n, n*m*m*p)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
                
            # Strategy: keep first target_rank columns, zero out the rest,
            # then use ALS to find a valid decomposition
            U_init = U[:, :target_rank].copy()
            V_init = V[:, :target_rank].copy()
            W_init = W[:, :target_rank].copy()
            
            # Add small structured perturbation
            noise_scale = 0.05 / (attempt + 1)
            U_init += rng.normal(0, noise_scale, U_init.shape)
            V_init += rng.normal(0, noise_scale, V_init.shape)
            W_init += rng.normal(0, noise_scale, W_init.shape)
            
            # ALS with multiple iterations
            R = target_rank
            U_cur, V_cur, W_cur = U_init, V_init, W_init
            
            for it in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Build Khatri-Rao product of V and W (column-wise)
                # KR_VW has shape (m*p*p*n, R) where each column r is kron(V[:,r], W[:,r])
                KR_VW = np.zeros((m*p*p*n, R))
                for r in range(R):
                    KR_VW[:, r] = np.kron(V_cur[:, r], W_cur[:, r])
                try:
                    U_new = T_U @ np.linalg.pinv(KR_VW)
                except:
                    break
                
                # KR_WU has shape (p*n*n*m, R)
                KR_WU = np.zeros((p*n*n*m, R))
                for r in range(R):
                    KR_WU[:, r] = np.kron(W_cur[:, r], U_new[:, r])
                try:
                    V_new = T_V @ np.linalg.pinv(KR_WU)
                except:
                    break
                
                # KR_UV has shape (n*m*m*p, R)
                KR_UV = np.zeros((n*m*m*p, R))
                for r in range(R):
                    KR_UV[:, r] = np.kron(U_new[:, r], V_new[:, r])
                try:
                    W_new = T_W @ np.linalg.pinv(KR_UV)
                except:
                    break
                
                change = (np.linalg.norm(U_new - U_cur) + 
                         np.linalg.norm(V_new - V_cur) + 
                         np.linalg.norm(W_new - W_cur))
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                if change < 1e-14:
                    break
            
            # Verify with tight tolerance
            if verify_decomp(U_cur, V_cur, W_cur, atol=1e-7):
                # Polish: try to round to half-integers
                U_pol = np.round(U_cur * 2) / 2
                V_pol = np.round(V_cur * 2) / 2
                W_pol = np.round(W_cur * 2) / 2
                if verify_decomp(U_pol, V_pol, W_pol, atol=1e-7):
                    return U_pol, V_pol, W_pol
                return U_cur, V_cur, W_cur
        
        return None

    rng = np.random.default_rng(42)
    
    # First, try a direct structural approach for rank 49:
    # The rank-50 decomposition uses 5 Strassen blocks (35 ranks) + 8 rank-1 terms (8 ranks) + 7 more = 50
    # Actually it's 6 Strassen blocks (42) + 8 rank-1 = 50
    # Try to see if we can save 1 rank by modifying the last Strassen block to absorb one rank-1 term
    
    def try_structural_rank49():
        """Try to construct a rank-49 decomposition directly."""
        if time.time() > deadline - 10:
            return None
        
        rank = 49
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            if col >= rank:
                return False
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            
            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1
            
            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            
            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1
            
            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1
            
            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            
            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1
            
            return col < rank
        
        # Use 5 Strassen blocks (35 ranks) for j0 in [0,2] and k0 in [0,2,4]
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                if not add_strassen(0, j0, k0):
                    return None
        
        # Now we have 35 ranks used. We need 14 more for the remaining work.
        # The original used 8 rank-1 terms for column 6 + 7 more = 15, total 50.
        # Let's see what's left. The Strassen blocks cover:
        # j0=0: columns 0,1 of B (and C)
        # j0=2: columns 2,3 of B (and C)  
        # k0=0: rows 0,1 of B
        # k0=2: rows 2,3 of B
        # k0=4: rows 4,5 of B
        # So Strassen covers B[0:4, 0:6] and C[0:6, 0:2]
        # Remaining: B[0:4, 6] (column 6) and C[6, 0:2]
        
        # For the remaining part, we need to handle:
        # C[0,6] = sum_j A[0,j]*B[j,6]
        # C[1,6] = sum_j A[1,j]*B[j,6]
        
        # That's 8 rank-1 terms (4 for row 0, 4 for row 1)
        # But we only have 14 ranks left. Let's try a different tiling.
        
        # Actually, let me reconsider. The issue is that with n=2, we can potentially
        # do better than Strassen for the last column.
        
        # For now, just add the 8 rank-1 terms and see if we're at rank 43+8=51... no
        # 35 + 8 = 43, but we need rank 49. So we have 6 extra ranks.
        # This means the original rank-50 had 6 Strassen blocks (42) + 8 rank-1 = 50.
        # Let me recount the original.
        
        # Original: j0 in [0,2], k0 in [0,2,4] = 2*3 = 6 Strassen blocks = 42 ranks
        # Plus 8 rank-1 terms = 50 total.
        
        # So for rank 49, I need to save 1 rank. Let me try using only 5 Strassen blocks
        # and handle the rest more efficiently.
        
        # This is getting complex. Let me just try the ALS approach with the fixed code.
        return None
    
    # Try structural approach first
    struct_result = try_structural_rank49()
    if struct_result is not None:
        best_rank = 49
        best_U, best_V, best_W = struct_result
    
    # Try to reduce rank from 49 down using fixed ALS
    for target_rank in range(49, 35, -1):
        if time.time() > deadline - 8:
            break
        
        result = try_reduce_rank(best_U, best_V, best_W, target_rank, rng, max_attempts=3)
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_rank = target_rank
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            # Continue trying lower ranks
    
    # Final polish: ensure exactness
    if not verify_decomp(best_U, best_V, best_W, atol=1e-9):
        # Try one more time with the rank-50
        if verify_decomp(U50, V50, W50, atol=1e-9):
            best_U, best_V, best_W = U50, V50, W50
            best_rank = 50

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
