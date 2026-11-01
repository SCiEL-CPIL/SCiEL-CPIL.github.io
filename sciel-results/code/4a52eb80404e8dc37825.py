"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    # Build a set of valid (a,b,c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True

def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
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
    return U, V, W

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    """
    n, m, p = N, M, P
    # Strassen terms for M_{2,2,2}
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    num_blocks = p // 2  # 4 blocks
    
    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U: index is i*m+j, but since we're doing 2x2 blocks of A,
            # u is indexed by (i, j) where i,j in {0,1}, flattened as i*2+j
            for i in range(2):
                for j in range(2):
                    a_idx = i * M + j
                    U[a_idx][col] = u[i * 2 + j]
                    
            # Fill V: index is j*p+k_global
            # v is indexed by (j, k_local) where j in {0,1}, k_local in {0,1}
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    V[b_idx][col] = v[j * 2 + k_local]
                    
            # Fill W: index is k_global*n+i
            # w is indexed by (i, k_local) where i in {0,1}, k_local in {0,1}
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    W[c_idx][col] = w[i * 2 + k_local]
                    
    return U, V, W

def _try_rank_27_search(deadline):
    """
    Attempt to find a rank-27 decomposition via systematic search.
    Start from the rank-28 Strassen solution and try to eliminate one term
    by adjusting the remaining 27 terms.
    Returns (U, V, W) if found, None otherwise.
    """
    U28, V28, W28 = strassen_decomposition_228()
    if not verify(U28, V28, W28):
        return None
    
    rank28 = 28
    
    # For each term to eliminate, try to find corrections to the other 27 terms
    for elim_term in range(rank28):
        if time.time() > deadline:
            return None
        
        # Remove the eliminated term
        U27 = [U28[r][:elim_term] + U28[r][elim_term+1:] for r in range(NM)]
        V27 = [V28[r][:elim_term] + V28[r][elim_term+1:] for r in range(MP)]
        W27 = [W28[r][:elim_term] + W28[r][elim_term+1:] for r in range(PN)]
        
        # The tensor contribution of the eliminated term is:
        # T_elim[a,b,c] = U28[a][elim_term] * V28[b][elim_term] * W28[c][elim_term]
        # We need to add corrections delta_U, delta_V, delta_W to the 27 terms
        # such that sum_{r in 27} (U27[a][r]+dU[a][r])*(V27[b][r]+dV[b][r])*(W27[c][r]+dW[c][r])
        #   = T_elim[a,b,c] for all (a,b,c)
        #
        # This is nonlinear. Instead, try a linearized approach:
        # Assume small corrections and linearize around the 27-term solution.
        # The residual we need to cancel is T_elim.
        # 
        # Alternative: try specific structured modifications.
        # Since the 28 terms come in 4 identical blocks, try to merge two blocks.
        
        # Try: can we express the eliminated term as a linear combination
        # of the other 27 terms? This requires the tensor to be linearly dependent.
        # Check if the 28 rank-1 tensors are linearly dependent.
        
        # Build the tensor as a 4x16x16 array and check linear dependence
        # of the 28 rank-1 tensors U_r ⊗ V_r ⊗ W_r
        tensors = []
        for r in range(rank28):
            t = np.zeros((NM, MP, PN))
            for a in range(NM):
                for b in range(MP):
                    for c in range(PN):
                        t[a,b,c] = U28[a][r] * V28[b][r] * W28[c][r]
            tensors.append(t.flatten())
        
        tensors_arr = np.array(tensors)  # shape (28, 1024)
        
        # Check if removing term elim_term makes the remaining 27 span the eliminated one
        remaining = [t for idx, t in enumerate(tensors) if idx != elim_term]
        remaining_arr = np.array(remaining)  # shape (27, 1024)
        elim_tensor = tensors[elim_term]
        
        # Solve: find coefficients x such that remaining_arr^T @ x ≈ elim_tensor
        # This is an overdetermined system (27 unknowns, 1024 equations)
        try:
            # Use least squares
            coeffs, residuals, rank_r, sv = np.linalg.lstsq(remaining_arr.T, elim_tensor, rcond=None)
            # Check if the fit is exact
            recon = remaining_arr.T @ coeffs
            if np.allclose(recon, elim_tensor, atol=1e-10):
                # The eliminated term is a linear combination of the others!
                # We can absorb it. But this gives a rank-27 decomposition
                # where the 27th term is modified.
                # Actually, if T_elim = sum x_r * T_r, then we can just
                # remove T_elim and the tensor is still represented by the other 27.
                # Wait, that's not right. The tensor is sum of all 28.
                # If T_elim = sum x_r T_r, then total = sum_{r!=elim} T_r + sum x_r T_r
                # = sum_{r!=elim} (1+x_r) T_r
                # So we can modify the coefficients.
                # But in our factorization, each term has fixed U,V,W vectors.
                # We'd need to scale them, which changes the factorization.
                # 
                # Actually, if T_elim = sum x_r T_r exactly, then:
                # Total tensor = sum_{all r} T_r = sum_{r!=elim} T_r + sum_{r!=elim} x_r T_r
                # = sum_{r!=elim} (1 + x_r) T_r
                # So we can create a new rank-27 factorization where term r has
                # U'_r = (1+x_r) U_r, V'_r = V_r, W'_r = W_r (or distribute the scale).
                
                U_new = [U28[r][:elim_term] + U28[r][elim_term+1:] for r in range(NM)]
                V_new = [V28[r][:elim_term] + V28[r][elim_term+1:] for r in range(MP)]
                W_new = [W28[r][:elim_term] + W28[r][elim_term+1:] for r in range(PN)]
                
                # Apply scaling: for the i-th remaining term (i from 0 to 26),
                # the original index is i if i < elim_term, else i+1
                for i in range(27):
                    orig_idx = i if i < elim_term else i + 1
                    scale = 1 + coeffs[i]
                    for a in range(NM):
                        U_new[a][i] = U_new[a][i] * scale
                    # Keep V and W unchanged (scale absorbed into U)
                
                if verify(U_new, V_new, W_new):
                    return U_new, V_new, W_new
        except Exception:
            pass
        
        # Also try: modify one of the 27 terms to absorb the eliminated term
        # Try adding the eliminated term's contribution to each of the 27 terms
        for target in range(27):
            if time.time() > deadline:
                return None
            orig_target = target if target < elim_term else target + 1
            
            U_try = [row[:] for row in U27]
            V_try = [row[:] for row in V27]
            W_try = [row[:] for row in W27]
            
            # Add eliminated term's vectors to the target term
            for a in range(NM):
                U_try[a][target] += U28[a][elim_term]
            for b in range(MP):
                V_try[b][target] += V28[b][elim_term]
            for c in range(PN):
                W_try[c][target] += W28[c][elim_term]
            
            # This won't work directly because (u+v)(v2+w2)(w2+z2) != uvw + v2w2z2
            # Skip this approach as it's mathematically incorrect for rank-1 tensors
            
    return None

def _try_rank_27_structural(deadline):
    """
    Try to find a rank-27 decomposition by exploiting structure.
    The 2x8 matrix multiplication can be seen as 4 independent 2x2 multiplications
    sharing the same A matrix. This gives rank 4*7=28 with Strassen.
    
    Can we do better by sharing more structure? The key observation is that
    all 4 blocks use the same A, so the U factors are identical across blocks.
    This might allow for a more compact representation.
    """
    if time.time() > deadline:
        return None
    
    # The rank-28 solution has U factors that repeat every 7 columns.
    # Specifically, U[:, 0:7] == U[:, 7:14] == U[:, 14:21] == U[:, 21:28]
    # This means we have redundancy in the U factors.
    # 
    # A rank-27 solution would need to break this symmetry somehow.
    # One approach: try to find a single term that can be shared differently.
    # 
    # For now, try a more aggressive local search with structured perturbations.
    U28, V28, W28 = strassen_decomposition_228()
    if not verify(U28, V28, W28):
        return None
    
    # Try merging two Strassen terms across different blocks
    # Term t in block b1 and term t' in block b2
    # If we can find a single rank-1 tensor that approximates their sum...
    # This is complex. Instead, try random local search with better heuristics.
    
    random.seed(42)
    best_result = None
    
    for elim_term in range(28):
        if time.time() > deadline:
            break
        
        U27 = [U28[r][:elim_term] + U28[r][elim_term+1:] for r in range(NM)]
        V27 = [V28[r][:elim_term] + V28[r][elim_term+1:] for r in range(MP)]
        W27 = [W28[r][:elim_term] + W28[r][elim_term+1:] for r in range(PN)]
        
        # Try perturbing multiple entries with small integer/half-integer values
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            U_try = [row[:] for row in U27]
            V_try = [row[:] for row in V27]
            W_try = [row[:] for row in W27]
            
            # Perturb 1-3 entries
            num_pert = random.randint(1, 3)
            for _ in range(num_pert):
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    arr = U_try
                elif mat == 'V':
                    arr = V_try
                else:
                    arr = W_try
                
                row = random.randrange(len(arr))
                col = random.randrange(27)
                # Use small perturbations
                pert = random.choice([-1, -0.5, 0.5, 1])
                arr[row][col] += pert
            
            if verify(U_try, V_try, W_try):
                return U_try, V_try, W_try
    
    return None

def construct():
    """Entry point: return the best decomposition found."""
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # First try the known rank-28 Strassen decomposition
    best_result = None
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try to find rank-27 via linear dependence check
    try:
        r27 = _try_rank_27_search(deadline)
        if r27 is not None:
            U27, V27, W27 = r27
            if verify(U27, V27, W27):
                return {"U": U27, "V": V27, "W": W27}
    except Exception:
        pass
    
    # Try structural rank-27 search
    try:
        r27 = _try_rank_27_structural(deadline)
        if r27 is not None:
            U27, V27, W27 = r27
            if verify(U27, V27, W27):
                return {"U": U27, "V": V27, "W": W27}
    except Exception:
        pass
    
    # Return best found
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
