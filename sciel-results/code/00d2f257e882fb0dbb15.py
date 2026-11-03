"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition"""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 130  # seconds, with safety margin

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a flat array."""
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    return tensor

def _check_decomposition(U, V, W, tensor, tol=1e-9):
    """Check if U, V, W exactly decompose the tensor."""
    r = U.shape[1]
    # Efficient check: tensor[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We can compute this as: for each r, outer(U[:,r], V[:,r]) * W[c,r] summed over r
    # Or more efficiently: reshape and use einsum
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is equivalent to: for each a,b, T[a,b,:] = (U[a,:] * V[b,:]) @ W.T
    # Or: T = sum_r np.einsum('r,br,cr->abc', ...) -- no, let's just do it directly
    
    # Efficient method: 
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # = sum_r (U[a,:] * V[b,:])_r * W[c,r]
    # For fixed a,b: T[a,b,:] = np.dot(U[a,:] * V[b,:], W.T)
    # So we can compute all T[a,b,:] for all a,b and compare
    
    # Even better: use the fact that we only need to check where tensor != 0
    # The tensor has exactly n*m*p non-zero entries, all 1.0
    # For entries where tensor=0, we need sum_r U[a,r]*V[b,r]*W[c,r] = 0
    # For entries where tensor=1, we need sum_r U[a,r]*V[b,r]*W[c,r] = 1
    
    # Let's compute the full tensor and compare
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    if not np.allclose(T_computed, tensor, atol=tol, rtol=0):
        return False
    return True

def _strassen_2x2():
    """Return Strassen decomposition for 2x2x2. Rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A00=0, A01=1, A10=2, A11=3
    # B00=0, B01=1, B10=2, B11=3
    # C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W

def _support_pruning_search(n, m, p, initial_rank=40, max_time=100, seed_base=42):
    """
    Support-pruning heuristic:
    1. Start with a random rank-40 ensemble
    2. Iteratively remove the term with smallest contribution to residual
    3. Try to find exact low-rank solution
    """
    start_time = time.time()
    tensor = _build_tensor(n, m, p)
    
    # Flatten tensor for efficient operations
    tensor_flat = tensor.flatten()
    tensor_indices = np.argwhere(tensor != 0)
    
    # Try multiple initializations
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        rng = np.random.default_rng(seed_base + attempt * 1000)
        
        # Initialize with random sparse factors
        r = initial_rank
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        for c in range(r):
            # Sparse random initialization with ±1 values
            nz_U = rng.choice(n*m, size=min(4, n*m), replace=False)
            nz_V = rng.choice(m*p, size=min(4, m*p), replace=False)
            nz_W = rng.choice(p*n, size=min(4, p*n), replace=False)
            
            for idx in nz_U:
                U[idx, c] = rng.choice([-1, 1])
            for idx in nz_V:
                V[idx, c] = rng.choice([-1, 1])
            for idx in nz_W:
                W[idx, c] = rng.choice([-1, 1])
        
        # Iterative pruning
        active = list(range(r))
        
        for prune_iter in range(30):
            if time.time() - start_time > max_time:
                break
            
            if len(active) <= 28:  # Lower bound is 28
                break
            
            # Compute current tensor approximation
            U_active = U[:, active]
            V_active = V[:, active]
            W_active = W[:, active]
            
            T_current = np.einsum('ar,br,cr->abc', U_active, V_active, W_active, optimize=True)
            residual = T_current - tensor
            
            # For each active term, compute its contribution to the residual norm
            # Term r contributes: U[:,r] * V[:,r] * W[:,r] (outer product)
            # We want to remove the term whose removal least increases ||residual||
            # Actually, we want to remove the term with smallest "importance"
            
            # Compute contribution of each term to the tensor
            # term_r[a,b,c] = U[a,r] * V[b,r] * W[c,r]
            # ||term_r||_F^2 = sum_{a,b,c} (U[a,r]*V[b,r]*W[c,r])^2
            #                 = (sum_a U[a,r]^2) * (sum_b V[b,r]^2) * (sum_c W[c,r]^2)
            
            # But more importantly, we want to know how much each term contributes
            # to matching the target tensor. Let's use the dot product with the target.
            
            # Actually, a better metric: for each term, compute how well it matches
            # the residual pattern. Terms that are "wasted" (contribute to zero entries
            # or don't help match ones) should be pruned.
            
            # Simpler approach: compute the squared norm of each term's contribution
            # to the residual, and remove the smallest.
            
            min_contrib = float('inf')
            min_idx = -1
            
            for pos, r_idx in enumerate(active):
                # Contribution of this term to the current approximation
                term = np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                
                # How much does removing this term change the residual?
                # New residual = residual + term
                # ||residual + term||^2 = ||residual||^2 + 2*<residual, term> + ||term||^2
                # We want to minimize ||residual + term||^2 after removal
                # So we want to maximize <residual, term> + ||term||^2/2
                # Or equivalently, remove the term that minimizes the increase in residual norm.
                
                # Actually, if we remove term r, new_residual = residual + term_r
                # We want to minimize ||new_residual||^2
                # = ||residual||^2 + 2*<residual, term_r> + ||term_r||^2
                # Since ||residual||^2 is constant, minimize 2*<residual, term_r> + ||term_r||^2
                
                dot_residual_term = np.sum(residual * term)
                norm_term_sq = np.sum(term * term)
                
                # If removing this term decreases the residual norm, it's a good candidate
                change = 2 * dot_residual_term + norm_term_sq
                
                if change < min_contrib:
                    min_contrib = change
                    min_idx = pos
            
            if min_idx < 0:
                break
            
            # Check if removing this term would make things worse
            if min_contrib > 0 and prune_iter > 5:
                # Only prune if it doesn't significantly increase error
                current_res_norm_sq = np.sum(residual * residual)
                if min_contrib > current_res_norm_sq * 0.1:
                    break
            
            # Remove the term
            active.pop(min_idx)
        
        # Final check with pruned factors
        if len(active) < r:
            U_final = U[:, active]
            V_final = V[:, active]
            W_final = W[:, active]
            
            if _check_decomposition(U_final, V_final, W_final, tensor):
                return U_final, V_final, W_final, len(active)
    
    return None

def _try_strassen_block_decomposition(n, m, p):
    """
    Try to decompose n*m*p multiplication using block structure.
    For 3x4*4x7: split the 7 columns into 3+4.
    """
    start_time = time.time()
    tensor = _build_tensor(n, m, p)
    
    # Try ranks from a lower bound upward
    for rank in range(35, 85):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        max_attempts = 20 if rank < 50 else 8
        
        for attempt in range(max_attempts):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            rng = np.random.default_rng(rank * 10000 + attempt * 7 + 42)
            
            # Sparse random initialization
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for c in range(rank):
                nz_U = rng.choice(n * m, size=min(3, n * m), replace=False)
                nz_V = rng.choice(m * p, size=min(3, m * p), replace=False)
                nz_W = rng.choice(p * n, size=min(3, p * n), replace=False)
                
                for idx in nz_U:
                    U[idx, c] = rng.choice([-1, 1])
                for idx in nz_V:
                    V[idx, c] = rng.choice([-1, 1])
                for idx in nz_W:
                    W[idx, c] = rng.choice([-1, 1])
            
            if _check_decomposition(U, V, W, tensor):
                return U, V, W, rank
    
    return None

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x7."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try support-pruning heuristic first (new strategy)
    result = _support_pruning_search(n, m, p, initial_rank=40, max_time=TIME_LIMIT - 30)
    if result is not None:
        U_found, V_found, W_found, rank_found = result
        if rank_found < best_rank:
            best_rank = rank_found
            best_U, best_V, best_W = U_found, V_found, W_found
    
    # Try block decomposition approach
    if time.time() - start_time < TIME_LIMIT - 10:
        result = _try_strassen_block_decomposition(n, m, p)
        if result is not None:
            U_found, V_found, W_found, rank_found = result
            if rank_found < best_rank:
                best_rank = rank_found
                best_U, best_V, best_W = U_found, V_found, W_found
    
    # Convert to lists for return
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_U]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_V]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
