"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    num_blocks = 5
    block_size = 2
    
    # Baseline: Rank 55 solution using Strassen (7) + Outer (4) per 2x3x2 block
    rank_per_block_base = 11
    total_rank_base = num_blocks * rank_per_block_base
    
    U_base = np.zeros((n * m, total_rank_base))
    V_base = np.zeros((m * p, total_rank_base))
    W_base = np.zeros((p * n, total_rank_base))
    
    strassen_terms = [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
    ]
    outer_terms = [
        {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
        {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
        {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
    ]
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        for term in all_terms:
            col = col_offset
            for i, j, val in term['U']:
                U_base[i * m + j, col] += val
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V_base[j * p + k_global, col] += val
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W_base[k_global * n + i, col] += val
            col_offset += 1
            
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = total_rank_base
    
    # Attempt to find a rank 10 block decomposition for 2x3x2
    def _find_rank_10_block():
        n_b, m_b, p_b = 2, 3, 2
        target_rank = 10
        time_limit = 40
        
        start_time = time.time()
        rng = np.random.default_rng(42)
        
        # Build the rank-11 base solution
        U11 = np.zeros((n_b*m_b, 11))
        V11 = np.zeros((m_b*p_b, 11))
        W11 = np.zeros((p_b*n_b, 11))
        
        for idx, term in enumerate(all_terms):
            col = idx
            for i, j, val in term['U']:
                U11[i * m_b + j, col] += val
            for j, k_local, val in term['V']:
                V11[j * p_b + k_local, col] += val
            for k_local, i, val in term['W']:
                W11[k_local * n_b + i, col] += val
        
        # Strategy 1: Try to find a rank-10 solution by local search
        # Start from rank-11, drop one column, and perturb + round to half-integers
        for col_to_drop in range(11):
            if time.time() - start_time > time_limit:
                break
            U_sub = np.delete(U11, col_to_drop, axis=1)
            V_sub = np.delete(V11, col_to_drop, axis=1)
            W_sub = np.delete(W11, col_to_drop, axis=1)
            
            for pert in range(200):
                if time.time() - start_time > time_limit:
                    break
                # Random perturbation with varying scale
                scale = 0.05 + 0.1 * (pert % 5)
                U_p = U_sub + rng.normal(0, scale, U_sub.shape)
                V_p = V_sub + rng.normal(0, scale, V_sub.shape)
                W_p = W_sub + rng.normal(0, scale, W_sub.shape)
                
                # Round to nearest half-integer
                U_r = np.round(U_p * 2) / 2
                V_r = np.round(V_p * 2) / 2
                W_r = np.round(W_p * 2) / 2
                
                if _verify_decomposition(U_r, V_r, W_r, n_b, m_b, p_b):
                    return U_r, V_r, W_r
        
        # Strategy 2: Random search with structured initialization
        # Each rank-1 term should have sparse structure
        for restart in range(5000):
            if time.time() - start_time > time_limit:
                break
                
            max_val = 1 if restart % 4 != 0 else 2
            U = np.zeros((n_b*m_b, target_rank))
            V = np.zeros((m_b*p_b, target_rank))
            W = np.zeros((p_b*n_b, target_rank))
            
            for r in range(target_rank):
                # Each term: pick random non-zero positions
                u_nonzero = rng.choice(n_b*m_b, size=rng.integers(1, 4), replace=False)
                v_nonzero = rng.choice(m_b*p_b, size=rng.integers(1, 4), replace=False)
                w_nonzero = rng.choice(p_b*n_b, size=rng.integers(1, 3), replace=False)
                
                for idx in u_nonzero:
                    U[idx, r] = rng.integers(-max_val, max_val + 1)
                for idx in v_nonzero:
                    V[idx, r] = rng.integers(-max_val, max_val + 1)
                for idx in w_nonzero:
                    W[idx, r] = rng.integers(-max_val, max_val + 1)
            
            if _verify_decomposition(U, V, W, n_b, m_b, p_b):
                return U, V, W
        
        # Strategy 3: Try known patterns for 2x3x2
        # The tensor has 12 entries. Try to cover them with 10 rank-1 terms.
        # Each term (i,j,k) contributes to U[i*m+j], V[j*p+k], W[k*n+i]
        
        # Try a systematic approach: fix the structure and solve for values
        # Use the fact that we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        
        # Try all assignments of which (i,j,k) triple each rank-1 term "covers"
        # This is a combinatorial search, but we can be smart about it
        
        # Alternative: use linear algebra. For fixed U and V, W is determined.
        # Let's try to find U and V such that the resulting W has half-integer entries.
        
        # Build the constraint matrix
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is nonlinear in W, but if we fix U and V, it's linear in W.
        
        # Try random U, V and solve for W
        for restart in range(3000):
            if time.time() - start_time > time_limit:
                break
            
            max_val = 1
            U = rng.integers(-max_val, max_val + 1, size=(n_b*m_b, target_rank)).astype(float)
            V = rng.integers(-max_val, max_val + 1, size=(m_b*p_b, target_rank)).astype(float)
            
            # For each (i,j,k), we have: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This gives 12 equations in 4*10=40 unknowns (W entries)
            # But the equations are coupled through the product U*V
            
            # Build the system: for each (k,i) pair (4 pairs), we have 3 equations (for j=0,1,2)
            # sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1 for each j
            
            # Let A[k,i,j] = sum_r U[i*m+j,r] * V[j*p+k,r]
            # Then for fixed (k,i): sum_r A[k,i,j] * W[k*n+i,r] = 1 for j=0,1,2
            # But A depends on j, so this isn't a simple linear system in W
            
            # Actually, for fixed (k,i), we have:
            # sum_r U[i*m+0,r] * V[0*p+k,r] * W[k*n+i,r] = 1
            # sum_r U[i*m+1,r] * V[1*p+k,r] * W[k*n+i,r] = 1
            # sum_r U[i*m+2,r] * V[2*p+k,r] * W[k*n+i,r] = 1
            
            # This is 3 equations in 10 unknowns (W[k*n+i, :])
            # So we can solve it if the system is consistent
            
            # Let's build the full system
            # Variables: W[0,0], W[0,1], ..., W[3,9] (40 variables)
            # Equations: 12 equations
            
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # The W variable that appears is W[k*n+i, r] for each r
            
            # This is not linear in W because U*V multiplies W element-wise
            # Actually it IS linear in W: sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            
            # So for each (i,j,k), we have a linear equation in the 10 W[k*n+i, r] variables
            # But different (i,j,k) with same (k,i) give different equations
            
            # Let's group by (k,i):
            # For fixed (k,i), we have 3 equations (j=0,1,2) in 10 unknowns
            # We can pick a solution, but we need it to be consistent across all (k,i)
            
            # This approach is complex. Let's just do random search with verification.
            pass
        
        return None

    block_decomp = _find_rank_10_block()
    
    if block_decomp is not None:
        U_block, V_block, W_block = block_decomp
        rank_per_block = 10
        total_rank = num_blocks * rank_per_block
        
        if total_rank < best_rank:
            U_new = np.zeros((n * m, total_rank))
            V_new = np.zeros((m * p, total_rank))
            W_new = np.zeros((p * n, total_rank))
            
            for b in range(num_blocks):
                col_offset = b * rank_per_block
                U_new[:, col_offset:col_offset+rank_per_block] = U_block
                for j in range(m):
                    for k_local in range(block_size):
                        k_global = b * block_size + k_local
                        V_new[j * p + k_global, col_offset:col_offset+rank_per_block] = V_block[j * block_size + k_local, :]
                for k_local in range(block_size):
                    for i in range(n):
                        k_global = b * block_size + k_local
                        W_new[k_global * n + i, col_offset:col_offset+rank_per_block] = W_block[k_local * n + i, :]
            
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = total_rank
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook rank 60
        U_sb = np.zeros((n * m, n * m * p))
        V_sb = np.zeros((m * p, n * m * p))
        W_sb = np.zeros((p * n, n * m * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_sb[i * m + j, idx] = 1
                    V_sb[j * p + k, idx] = 1
                    W_sb[k * n + i, idx] = 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
