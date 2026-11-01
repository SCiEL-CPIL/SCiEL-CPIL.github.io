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
    
    # Build tensor vectors for all 28 rank-1 components (done once)
    tensor_vecs = []
    for r in range(rank28):
        t = np.zeros((NM, MP, PN))
        for a in range(NM):
            for b in range(MP):
                for c in range(PN):
                    t[a,b,c] = U28[a][r] * V28[b][r] * W28[c][r]
        tensor_vecs.append(t.flatten())
    
    # Check linear dependence of the 28 rank-1 tensors
    tensors_arr = np.array(tensor_vecs)  # shape (28, 1024)
    
    for elim_term in range(rank28):
        if time.time() > deadline:
            return None
        
        # Check if the eliminated tensor is in the span of the other 27
        remaining_indices = [i for i in range(rank28) if i != elim_term]
        remaining_arr = tensors_arr[remaining_indices]  # shape (27, 1024)
        elim_tensor = tensors_arr[elim_term]
        
        try:
            coeffs, residuals, rank_r, sv = np.linalg.lstsq(remaining_arr.T, elim_tensor, rcond=None)
            recon = remaining_arr.T @ coeffs
            if np.allclose(recon, elim_tensor, atol=1e-10):
                # Exact linear dependence found!
                U_new = [U28[r][:elim_term] + U28[r][elim_term+1:] for r in range(NM)]
                V_new = [V28[r][:elim_term] + V28[r][elim_term+1:] for r in range(MP)]
                W_new = [W28[r][:elim_term] + W28[r][elim_term+1:] for r in range(PN)]
                
                for i in range(27):
                    scale = 1 + coeffs[i]
                    for a in range(NM):
                        U_new[a][i] = U_new[a][i] * scale
                
                if verify(U_new, V_new, W_new):
                    return U_new, V_new, W_new
        except Exception:
            pass
    
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
    
    U28, V28, W28 = strassen_decomposition_228()
    if not verify(U28, V28, W28):
        return None
    
    # Approach: Try to find a rank-27 solution by systematically modifying
    # the Strassen decomposition. Since all 4 blocks share the same U factors,
    # we can try to find a single modification that absorbs one term.
    #
    # Key insight: The tensor T = sum_{b=0}^{3} sum_{t=0}^{6} U_t ⊗ V_t^{(b)} ⊗ W_t^{(b)}
    # where V_t^{(b)} and W_t^{(b)} are the b-th block versions.
    # 
    # Try: modify V and W in one block to absorb a term from another block.
    
    random.seed(42)
    
    # Try all pairs of (elim_term, target_term) where we try to absorb
    # elim_term into target_term by modifying U, V, W entries
    for elim_term in range(28):
        if time.time() > deadline:
            break
        
        for target_term in range(28):
            if target_term == elim_term or time.time() > deadline:
                continue
            
            # Start with the 28-term solution, modify target_term to absorb elim_term
            U_try = [row[:] for row in U28]
            V_try = [row[:] for row in V28]
            W_try = [row[:] for row in W28]
            
            # We need: sum_{r=0}^{27} U'[a][r]*V'[b][r]*W'[c][r] = T[a,b,c]
            # where U', V', W' differ from U28, V28, W28 only in columns target_term and elim_term
            # and we set column elim_term to zero.
            # 
            # This is: original_sum - U28[a][elim]*V28[b][elim]*W28[c][elim]
            #   + delta = T[a,b,c]
            # So delta = U28[a][elim]*V28[b][elim]*W28[c][elim]
            # 
            # delta must be expressible as a modification of column target_term:
            # (U28[a][target]+dU[a])*(V28[b][target]+dV[b])*(W28[c][target]+dW[c])
            #   - U28[a][target]*V28[b][target]*W28[c][target]
            # = U28[a][elim]*V28[b][elim]*W28[c][elim]
            #
            # This is highly nonlinear. Instead, try: set U'[a][target] = U28[a][target] + U28[a][elim]
            # and similarly for V, W. But this gives (u1+u2)(v1+v2)(w1+w2) which has
            # extra cross terms.
            #
            # Alternative: try specific small modifications.
            
            # Try: add elim's U contribution to target's U
            for a in range(NM):
                U_try[a][target_term] = U28[a][target_term] + U28[a][elim_term]
            # Zero out elim column
            for a in range(NM):
                U_try[a][elim_term] = 0
            for b in range(MP):
                V_try[b][elim_term] = 0
            for c in range(PN):
                W_try[c][elim_term] = 0
            
            # This won't work in general, but check anyway
            if verify(U_try, V_try, W_try):
                # Remove the zero column
                U_res = [U_try[r][:target_term] + U_try[r][target_term+1:elim_term] + U_try[r][elim_term+1:] for r in range(NM)]
                V_res = [V_try[r][:target_term] + V_try[r][target_term+1:elim_term] + V_try[r][elim_term+1:] for r in range(MP)]
                W_res = [W_try[r][:target_term] + W_try[r][target_term+1:elim_term] + W_try[r][elim_term+1:] for r in range(PN)]
                return U_res, V_res, W_res
    
    # Try: modify multiple entries in a structured way
    # Focus on the fact that U is identical across blocks
    # Try to find a "correction" that works across all blocks
    
    for elim_term in range(28):
        if time.time() > deadline:
            break
        
        # Remove the eliminated term
        U27 = [U28[r][:elim_term] + U28[r][elim_term+1:] for r in range(NM)]
        V27 = [V28[r][:elim_term] + V28[r][elim_term+1:] for r in range(MP)]
        W27 = [W28[r][:elim_term] + W28[r][elim_term+1:] for r in range(PN)]
        
        # Try structured perturbations: modify entries in the same column
        # across all rows of one factor
        for col in range(27):
            if time.time() > deadline:
                break
            
            # Try scaling one factor's column
            for scale in [-1, -0.5, 0.5, 1, 1.5, 2]:
                if time.time() > deadline:
                    break
                
                U_try = [row[:] for row in U27]
                V_try = [row[:] for row in V27]
                W_try = [row[:] for row in W27]
                
                for a in range(NM):
                    U_try[a][col] = U_try[a][col] + scale * U28[a][elim_term]
                
                if verify(U_try, V_try, W_try):
                    return U_try, V_try, W_try
                
                # Try scaling V
                U_try = [row[:] for row in U27]
                V_try = [row[:] for row in V27]
                W_try = [row[:] for row in W27]
                
                for b in range(MP):
                    V_try[b][col] = V_try[b][col] + scale * V28[b][elim_term]
                
                if verify(U_try, V_try, W_try):
                    return U_try, V_try, W_try
                
                # Try scaling W
                U_try = [row[:] for row in U27]
                V_try = [row[:] for row in V27]
                W_try = [row[:] for row in W27]
                
                for c in range(PN):
                    W_try[c][col] = W_try[c][col] + scale * W28[c][elim_term]
                
                if verify(U_try, V_try, W_try):
                    return U_try, V_try, W_try
    
    # Try random search with more attempts and better perturbation patterns
    for elim_term in range(28):
        if time.time() > deadline:
            break
        
        U27 = [U28[r][:elim_term] + U28[r][elim_term+1:] for r in range(NM)]
        V27 = [V28[r][:elim_term] + V28[r][elim_term+1:] for r in range(MP)]
        W27 = [W28[r][:elim_term] + W28[r][elim_term+1:] for r in range(PN)]
        
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            U_try = [row[:] for row in U27]
            V_try = [row[:] for row in V27]
            W_try = [row[:] for row in W27]
            
            # Perturb 1-4 entries with half-integer values
            num_pert = random.randint(1, 4)
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
                pert = random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                arr[row][col] += pert
            
            if verify(U_try, V_try, W_try):
                return U_try, V_try, W_try
    
    return None

def construct():
    """Entry point: return the best decomposition found."""
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # First try the known rank-28 Strassen decomposition
    best_result = None
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try to find rank-27 via linear dependence check (fast)
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
