import time
import numpy as np
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Precompute valid indices
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.append((i*m+j, j*p+k, k*n+i))
                
    # Total number of entries in the tensor
    total_entries = (n*m) * (m*p) * (p*n)
    
    # Build the target tensor T as a flat array? No, too big (2304 is small, actually).
    # T[alpha, beta, gamma] = 1 if (alpha, beta, gamma) in valid_indices else 0
    
    def verify(U, V, W, rank):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        Check sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c]
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Efficient check:
        # For each valid (a,b,c), check sum is 1
        # For each invalid (a,b,c), check sum is 0
        
        # We can compute the full tensor product sum for all a,b,c?
        # T_reconstructed[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a 3-way tensor of size 6x12x8 = 576? No, 6*12*8 = 576.
        # Wait, nm=6, mp=12, pn=8. 6*12*8 = 576.
        # 576 is very small. We can compute the full tensor.
        
        # Reshape U, V, W to (dim, rank)
        # Compute T_rec[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        
        # Use einsum
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check valid entries
        for a, b, c in valid_indices:
            if abs(T_rec[a, b, c] - 1.0) > 1e-9:
                return False
        
        # Check invalid entries
        # Create a mask of valid entries
        mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for a, b, c in valid_indices:
            mask[a, b, c] = True
            
        invalid_mask = ~mask
        invalid_vals = T_rec[invalid_mask]
        
        if np.any(np.abs(invalid_vals) > 1e-9):
            return False
            
        return True

    def to_half_integers(U, V, W, tol=1e-6):
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if np.allclose(U, U2, atol=tol) and np.allclose(V, V2, atol=tol) and np.allclose(W, W2, atol=tol):
            return U2, V2, W2
        return None

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
                
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb
    
    deadline = time.time() + 110
    
    # Search for lower ranks
    for rank in range(12, best_rank):
        if time.time() > deadline:
            break
            
        # Try random initialization and local search?
        # Or just random search with more trials?
        # Given the small size, we can try random integer/half-integer search with more iterations.
        
        val_set = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        n_trials = 2000
        for trial in range(n_trials):
            if time.time() > deadline:
                break
                
            U_try = np.random.choice(val_set, size=(n*m, rank))
            V_try = np.random.choice(val_set, size=(m*p, rank))
            W_try = np.random.choice(val_set, size=(p*n, rank))
            
            # Quick check on valid entries first
            T_rec = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
            valid_ok = True
            for a, b, c in valid_indices:
                if abs(T_rec[a, b, c] - 1.0) > 1e-9:
                    valid_ok = False
                    break
            
            if not valid_ok:
                continue
                
            # Full check
            if verify(U_try, V_try, W_try, rank):
                snapped = to_half_integers(U_try, V_try, W_try)
                if snapped is not None:
                    U_s, V_s, W_s = snapped
                    if verify(U_s, V_s, W_s, rank):
                        best_result = (U_s, V_s, W_s)
                        best_rank = rank
                        break
        
        if best_rank == rank:
            break
            
    U_out, V_out, W_out = best_result
    # Convert to lists for JSON compatibility if needed, but numpy arrays are usually fine?
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # Returning numpy arrays should be fine.
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
