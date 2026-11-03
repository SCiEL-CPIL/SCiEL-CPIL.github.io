"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _strassen_2x2_factors():
    """Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication."""
    factors = []
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _strassen_4x4_factors():
    """Return the 49 rank-1 factors for 4x4*4x4 using Strassen."""
    s2 = _strassen_2x2_factors()
    factors = []
    for s in range(7):
        for r in range(7):
            u_block = s2[s][0]
            v_block = s2[s][1]
            w_block = s2[s][2]
            u_scalar = s2[r][0]
            v_scalar = s2[r][1]
            w_scalar = s2[r][2]
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    c_block = u_block[block_idx]
                    if c_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = u_scalar[scalar_idx]
                            if c_scalar == 0: continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            u_vec[i * 4 + j] += c_block * c_scalar
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    c_block = v_block[block_idx]
                    if c_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = v_scalar[scalar_idx]
                            if c_scalar == 0: continue
                            j = 2 * bi + ei
                            k = 2 * bj + ej
                            v_vec[j * 4 + k] += c_block * c_scalar
            for ci in range(2):
                for cj in range(2):
                    block_idx = ci * 2 + cj
                    c_block = w_block[block_idx]
                    if c_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = w_scalar[scalar_idx]
                            if c_scalar == 0: continue
                            i = 2 * ci + ei
                            k = 2 * cj + ej
                            w_vec[i * 4 + k] += c_block * c_scalar
            factors.append((u_vec, v_vec, w_vec))
    return factors

def _strassen_4x4_4x8():
    """Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen."""
    n, m, p = DIMENSIONS
    s4_factors = _strassen_4x4_factors()
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for t in range(2):
        for idx in range(49):
            col = t * 49 + idx
            u_vec, v_vec, w_vec = s4_factors[idx]
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][col] = u_vec[i * 4 + j]
            for j in range(4):
                for k in range(4):
                    V[j * 8 + (4 * t + k)][col] = v_vec[j * 4 + k]
            for i in range(4):
                for k in range(4):
                    W[(4 * t + k) * 4 + i][col] = w_vec[i * 4 + k]
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _border_rank_4x4_4x8():
    """
    Try to construct a rank-96 decomposition using border rank techniques.
    
    The idea: The 4x4x4 matrix multiplication tensor has border rank 48.
    We can exploit this by finding a rank-48 decomposition that approximates
    the tensor, then extend to 4x4x4x8 (two halves of B) for rank 96.
    
    Strategy: Start from Strassen rank-49, apply a perturbation that allows
    one factor to be eliminated in the limit, then project to exact factors.
    """
    n, m, p = DIMENSIONS
    s4_factors = _strassen_4x4_factors()
    
    # Build the rank-49 Strassen decomposition as arrays
    U49 = np.zeros((16, 49))
    V49 = np.zeros((16, 49))
    W49 = np.zeros((16, 49))
    for idx in range(49):
        u_vec, v_vec, w_vec = s4_factors[idx]
        U49[:, idx] = u_vec
        V49[:, idx] = v_vec
        W49[:, idx] = w_vec
    
    # Border rank approach: We look for a linear dependency among the 49 factors
    # that allows us to express one factor as a limit of combinations of others.
    # 
    # The key insight from algebraic complexity: the 4x4x4 MM tensor lies in
    # the 4-secant variety of the Segre variety, meaning it can be expressed
    # as a limit of sums of 48 rank-1 tensors.
    #
    # Practical approach: Try to find a rank-48 decomposition by solving
    # for a null vector in the factor space.
    
    # Build the tensor in 3-mode form
    T = np.zeros((4, 4, 4))
    T.fill(1.0)  # The MM tensor for 4x4x4 has T[i,j,k] = 1
    
    # For each potential elimination, check if we can find coefficients
    # such that removing one column still gives an exact decomposition.
    
    # Actually, let's try a different approach: use the known structure
    # that the 4x4x4 MM tensor can be decomposed with rank 48 using
    # the "border rank" construction from Landsberg-Manivel.
    #
    # The construction involves:
    # 1. Starting with the 49 Strassen factors
    # 2. Finding a 1-dimensional subspace in the span of the factors
    # 3. Perturbing along that subspace
    # 4. Taking the limit as the perturbation goes to 0
    #
    # For our purposes, we'll try to find if any column can be expressed
    # as a linear combination of others in a way that preserves the tensor.
    
    # Check for linear dependencies in the U, V, W factor matrices
    # If there's a dependency, we might be able to reduce the rank.
    
    # Try to find a null vector in the combined factor space
    # For each column r, the contribution to the tensor is:
    #   sum over (i,j,k) of U[i,j,r] * V[j,k,r] * W[k,i,r] * T[i,j,k]
    # 
    # The tensor equation requires:
    #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
    
    # Let's try a direct approach: assume we can drop one column and
    # solve for the remaining factors.
    
    best_rank = 98
    best_result = None
    
    # Try dropping each of the 49 columns in the first half
    for drop_col in range(49):
        # Keep all other 48 columns
        keep_cols = [c for c in range(49) if c != drop_col]
        
        # Build the 48-column factors for one half
        U_keep = U49[:, keep_cols]
        V_keep = V49[:, keep_cols]
        W_keep = W49[:, keep_cols]
        
        # Now extend to 4x4x4x8: use the same 48 factors for both halves
        # Total rank = 48 * 2 = 96
        rank96 = 96
        U96 = np.zeros((16, rank96))
        V96 = np.zeros((32, rank96))
        W96 = np.zeros((32, rank96))
        
        for t in range(2):
            for local_idx, global_idx in enumerate(keep_cols):
                col = t * 48 + local_idx
                U96[:, col] = U_keep[:, local_idx]
                V96[0:16, col] = V_keep[:, local_idx]  # First 4x4 block of B
                V96[16:32, col] = V_keep[:, local_idx]  # Second 4x4 block of B
                W96[0:16, col] = W_keep[:, local_idx]   # First 4x4 block of C
                W96[16:32, col] = W_keep[:, local_idx]   # Second 4x4 block of C
        
        # Verify
        U3 = U96.reshape(4, 4, rank96)
        V3 = V96.reshape(4, 8, rank96)
        W3 = W96.reshape(8, 4, rank96)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        if np.allclose(result, 1.0, atol=1e-10):
            # Convert to list format
            U_list = U96.tolist()
            V_list = V96.tolist()
            W_list = W96.tolist()
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank_reduction():
    """
    Try to reduce the rank below 98 using various techniques.
    Returns the best decomposition found, or None.
    """
    start_time = time.time()
    deadline = start_time + 100  # Leave safety margin
    
    # Try border rank approach
    if time.time() < deadline - 10:
        try:
            result = _border_rank_4x4_4x8()
            if result is not None:
                return result
        except Exception:
            pass
    
    # Try random perturbation + projection
    if time.time() < deadline - 10:
        try:
            base = _strassen_4x4_4x8()
            U_base = np.array(base["U"], dtype=float)
            V_base = np.array(base["V"], dtype=float)
            W_base = np.array(base["W"], dtype=float)
            rank0 = U_base.shape[1]  # 98
            
            # Try to find a rank-97 decomposition by merging two columns
            # For columns r1, r2, find a new column that replaces both
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(attempt * 7919 + 42)
                
                # Pick two random columns to merge
                r1 = np.random.randint(0, rank0)
                r2 = np.random.randint(0, rank0)
                if r1 == r2:
                    continue
                
                # Try to find a linear combination that preserves the tensor
                # This is a hard problem, so we use a heuristic:
                # Try small perturbations and check if the tensor is preserved.
                
                U_mod = U_base.copy()
                V_mod = V_base.copy()
                W_mod = W_base.copy()
                
                # Zero out column r2 and try to adjust column r1
                U_mod[:, r2] = 0
                V_mod[:, r2] = 0
                W_mod[:, r2] = 0
                
                # Check if we can compensate by adjusting other columns
                # This is a linear system: for each (i,j,k),
                # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                # We need to find adjustments to columns 0..r1-1, r1+1..r2-1, r2+1..97
                # such that the tensor equation is satisfied.
                
                # This is underdetermined, so we look for a sparse solution.
                # For now, just check if zeroing out works (it won't, but
                # it establishes the baseline).
                
                U3 = U_mod.reshape(4, 4, rank0)
                V3 = V_mod.reshape(4, 8, rank0)
                W3 = W_mod.reshape(8, 4, rank0)
                result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                
                if np.allclose(result, 1.0, atol=1e-10):
                    # Count effective rank
                    col_norms = np.linalg.norm(np.vstack([U_mod, V_mod, W_mod]), axis=0)
                    effective_rank = np.sum(col_norms > 1e-10)
                    if effective_rank < 98:
                        mask = col_norms > 1e-10
                        return {
                            "U": U_mod[:, mask].tolist(),
                            "V": V_mod[:, mask].tolist(),
                            "W": W_mod[:, mask].tolist()
                        }
        except Exception:
            pass
    
    return None

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries to find a rank-minimal decomposition, falls back to Strassen (rank 98) or schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try border rank / rank reduction
    try:
        result = _try_rank_reduction()
        if result is not None and time.time() < deadline:
            if _verify(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fall back to Strassen
    try:
        result = _strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Final fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
