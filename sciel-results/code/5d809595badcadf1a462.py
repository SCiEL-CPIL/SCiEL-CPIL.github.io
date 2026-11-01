"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor T[a, b, c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition (rank = n*m*p = 50)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Check if schoolbook is valid
    T_recon = np.einsum('ar,br,cr->abc', U_sb, V_sb, W_sb)
    if np.max(np.abs(T_recon - T)) < 1e-9:
        pass  # valid
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Strategy 1: Try to find lower rank by structured perturbation
    # The tensor has a special structure: T[i*m+j, j*p+k, k*n+i] = 1
    # This is a "matrix multiplication" tensor with specific indexing
    
    # Try rank from 49 down to 30 with structured search
    np.random.seed(42)
    
    for target_rank in range(49, 29, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple strategies per rank
        found = False
        
        # Strategy A: Start from schoolbook, zero out random columns, then fix
        for attempt in range(20):
            if time.time() > deadline - 10 or found:
                break
            
            # Start with schoolbook
            U_r = U_sb.copy()
            V_r = V_sb.copy()
            W_r = W_sb.copy()
            
            # Randomly set some entries to 0 or adjust
            # Try to merge some terms
            mask = np.random.random(rank_sb) < (target_rank / rank_sb)
            # Keep only ~target_rank columns initially
            keep_cols = np.where(mask)[0]
            if len(keep_cols) < target_rank:
                # Add random columns
                extra = np.random.choice(rank_sb, target_rank - len(keep_cols), replace=False)
                keep_cols = np.concatenate([keep_cols, extra])
            
            U_r = U_sb[:, keep_cols]
            V_r = V_sb[:, keep_cols]
            W_r = W_sb[:, keep_cols]
            
            # Now try to adjust entries to make it work
            # Use iterative refinement
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err = T_recon - T
                if np.max(np.abs(err)) < 1e-9:
                    found = True
                    break
                
                # Gradient-like update: adjust entries to reduce error
                # dT/dU[a,r] = sum_{b,c} V[b,r]*W[c,r] * err[a,b,c]
                # Simplified: just do random perturbations with annealing
                
                # Random perturbation with decreasing magnitude
                scale = 0.5 / (iteration + 1)
                pert_U = np.random.uniform(-scale, scale, U_r.shape)
                pert_V = np.random.uniform(-scale, scale, V_r.shape)
                pert_W = np.random.uniform(-scale, scale, W_r.shape)
                
                # Snap to half-integers
                U_new = np.round((U_r + pert_U) * 2) / 2
                V_new = np.round((V_r + pert_V) * 2) / 2
                W_new = np.round((W_r + pert_W) * 2) / 2
                
                T_recon_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                err_new = np.max(np.abs(T_recon_new - T))
                err_old = np.max(np.abs(err))
                
                if err_new <= err_old * 1.01:  # Accept if not much worse
                    U_r, V_r, W_r = U_new, V_new, W_new
            
            if found:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        # Strategy B: Pure random half-integer search
        if not found:
            for attempt in range(30):
                if time.time() > deadline - 10 or found:
                    break
                
                U_r = np.round(np.random.uniform(-1, 1, (nm, target_rank)) * 2) / 2
                V_r = np.round(np.random.uniform(-1, 1, (mp, target_rank)) * 2) / 2
                W_r = np.round(np.random.uniform(-1, 1, (pn, target_rank)) * 2) / 2
                
                T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.max(np.abs(T_recon - T)) < 1e-9:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
        
        if best_rank == target_rank:
            break
    
    # Final validation
    T_recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    if np.max(np.abs(T_recon - T)) < 1e-9:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Return schoolbook as fallback
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
