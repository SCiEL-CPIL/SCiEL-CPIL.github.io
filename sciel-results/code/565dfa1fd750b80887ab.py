import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Schoolbook decomposition (Rank 30) as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[d, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find lower rank decompositions
    # Conjectured rank for 2x3x5 is 18.
    # We search from 18 up to 29.
    
    def round_to_half(x):
        # Round to nearest integer or half-integer
        # x is a numpy array
        # 2*x should be close to an integer
        return np.round(2 * x) / 2.0

    def is_valid(U, V, W, T):
        # Check if sum_r U[a,r] V[b,r] W[d,r] == T[a,b,d]
        # Efficient check:
        # Residual = T - sum_r U[:,r] V[:,r] W[:,r]
        # We can compute this using einsum or loops.
        # Given small sizes, loops or einsum are fine.
        # Let's use einsum for speed if possible, but loops are safer for clarity.
        # Actually, we can compute the full tensor and compare.
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # return np.allclose(T_recon, T, atol=1e-9)
        
        # To be faster, we can check only non-zero entries of T, 
        # but we must ensure non-zero entries of T_recon are zero where T is zero.
        # Since T is sparse, we can check:
        # 1. T_recon[a,b,d] == T[a,b,d] for all (a,b,d) where T != 0
        # 2. T_recon[a,b,d] == 0 for all (a,b,d) where T == 0
        
        # Let's just compute the full tensor, it's 6*15*10 = 900 elements.
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)

    def try_rank(target_rank, trials=50):
        # Try to find a decomposition of target_rank
        # Using ALS with rounding
        
        for trial in range(trials):
            if time.time() > deadline:
                return None
            
            # Initialize
            # Random initialization can be tricky. 
            # Maybe start with schoolbook and zero out some columns?
            # Or random.
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize to avoid huge numbers
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(200): # Max iterations
                if time.time() > deadline:
                    return None
                
                # Update U
                # For each a, solve U[a, :]
                for a in range(n*m):
                    # T[a, :, :] is (m*p, p*n)
                    # We need M such that T[a, b, d] = sum_r U[a, r] M[b, d, r]
                    # M[b, d, r] = V[b, r] W[d, r]
                    # Flatten M to (m*p*p*n, target_rank)
                    M = np.einsum('br,dr->bdr', V, W) # shape (m*p, p*n, target_rank)
                    M_flat = M.reshape(-1, target_rank)
                    T_flat = T[a, :, :].reshape(-1)
                    
                    # Solve U[a, :] such that U[a, :] M_flat.T approx T_flat
                    # U_a = T_flat M_flat (M_flat.T M_flat)^-1 ?
                    # Least squares: M_flat.T U_a = T_flat
                    # U_a = (M_flat.T M_flat)^-1 M_flat.T T_flat
                    try:
                        # Use lstsq for stability
                        U_a, residuals, rank_m, s = np.linalg.lstsq(M_flat.T, T_flat, rcond=None)
                        U[a, :] = U_a
                    except:
                        pass
                
                # Round U
                U = round_to_half(U)
                
                # Update V
                for b in range(m*p):
                    M = np.einsum('ar,dr->adr', U, W) # shape (n*m, p*n, target_rank)
                    M_flat = M.reshape(-1, target_rank)
                    T_flat = T[:, b, :].reshape(-1)
                    try:
                        V_b, residuals, rank_m, s = np.linalg.lstsq(M_flat.T, T_flat, rcond=None)
                        V[b, :] = V_b
                    except:
                        pass
                V = round_to_half(V)
                
                # Update W
                for d in range(p*n):
                    M = np.einsum('ar,br->abr', U, V) # shape (n*m, m*p, target_rank)
                    M_flat = M.reshape(-1, target_rank)
                    T_flat = T[:, :, d].reshape(-1)
                    try:
                        W_d, residuals, rank_m, s = np.linalg.lstsq(M_flat.T, T_flat, rcond=None)
                        W[d, :] = W_d
                    except:
                        pass
                W = round_to_half(W)
                
                # Check convergence
                if is_valid(U, V, W, T):
                    return (U, V, W)
                    
                # Optional: check if residual is decreasing, but we rely on exact check.
                
        return None

    # Search for lower ranks
    for r in range(18, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Increase trials for smaller ranks as they are harder? Or keep constant?
        # 50 trials * 200 iters * 3 updates * (small matrices)
        # Should be fast enough.
        res = try_rank(r, trials=50)
        if res:
            U, V, W = res
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break # Found the lowest rank in this range (since we iterate up)
            # Note: We iterate 18, 19, ... so first found is minimal in this sequence.
            
    # Convert to lists
    def to_list(matrix):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(x*2 - round(x*2)) < 1e-9 else x) for x in row] for row in matrix]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
