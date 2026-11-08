"""Tensor decomposition for 4x4 * 4x6 matrix multiplication using randomized rank-reduction search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _build_schoolbook(n, m, p):
    """Build the trivial schoolbook decomposition with rank n*m*p."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check exact decomposition: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _snap_half(x):
    """Snap values to nearest half-integer for cleaner output."""
    return np.round(x * 2.0) / 2.0


def _try_random_rank(n, m, p, rank, rng, max_iters=50):
    """
    Try to find an exact rank-`rank` decomposition by:
    1. Generate random U, V with integer/half-integer entries
    2. Solve for W via least squares
    3. Check if reconstruction is exact
    """
    for _ in range(max_iters):
        # Random U and V with entries from {-1, -0.5, 0, 0.5, 1}
        U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, rank))
        V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, rank))

        # Build the tensor T as a matrix: T[i*m+j, j*p+k] doesn't quite work.
        # The tensor is T[i,j,k] = delta(i*m+j, j*p+k, k*n+i) = 1 if consistent.
        # Actually T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor).
        # We need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k.
        
        # This is a 3-linear constraint. For fixed U,V, each (i,j,k) gives:
        # sum_t (U[i*m+j,t] * V[j*p+k,t]) * W[k*n+i,t] = 1
        # This is linear in W! For each (i,j,k), it's one equation in the rank unknowns W[*,t].
        # But W has p*n*rank unknowns and n*m*p equations.
        
        # Let's set up the linear system for W.
        # Unknowns: W[k*n+i, t] for k in range(p), i in range(n), t in range(rank)
        # Total unknowns: p * n * rank
        # Equations: n * m * p (one for each i,j,k)
        
        n_unknowns = p * n * rank
        n_equations = n * m * p
        
        # Build A matrix and b vector
        # Equation (i,j,k): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
        # W index: k*n+i, t -> flat index (k*n+i)*rank + t
        
        if n_equations >= n_unknowns:
            # Overdetermined or square - use least squares
            A = np.zeros((n_equations, n_unknowns))
            b = np.ones(n_equations)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(rank):
                            coeff = U[i * m + j, t] * V[j * p + k, t]
                            if coeff != 0:
                                w_idx = (k * n + i) * rank + t
                                A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            if len(residuals) == 0:
                # Check residual manually
                resid = np.linalg.norm(A @ W_flat - b)
                if resid > 1e-8:
                    continue
            
            W = W_flat.reshape(p * n, rank)
            
            # Snap to half-integers
            U_s = _snap_half(U)
            V_s = _snap_half(V)
            W_s = _snap_half(W)
            
            # Verify with snapped values
            if _verify(U_s, V_s, W_s, n, m, p, tol=1e-6):
                return U_s, V_s, W_s
            else:
                # Try without snapping
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    return U, V, W
        else:
            # Underdetermined - still try least squares (will have free variables)
            A = np.zeros((n_equations, n_unknowns))
            b = np.ones(n_equations)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(rank):
                            coeff = U[i * m + j, t] * V[j * p + k, t]
                            if coeff != 0:
                                w_idx = (k * n + i) * rank + t
                                A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            if len(residuals) == 0:
                resid = np.linalg.norm(A @ W_flat - b)
                if resid > 1e-8:
                    continue
            
            W = W_flat.reshape(p * n, rank)
            U_s = _snap_half(U)
            V_s = _snap_half(V)
            W_s = _snap_half(W)
            
            if _verify(U_s, V_s, W_s, n, m, p, tol=1e-6):
                return U_s, V_s, W_s
            else:
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    return U, V, W
    
    return None


def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2*2x2."""
    terms = []
    terms.append({'U': {(0,0):1,(1,1):1}, 'V': {(0,0):1,(1,1):1}, 'W': {(0,0):1,(1,1):1}})
    terms.append({'U': {(1,0):1,(1,1):1}, 'V': {(0,0):1}, 'W': {(1,0):1}})
    terms.append({'U': {(0,0):1}, 'V': {(0,1):1,(1,1):-1}, 'W': {(0,1):1,(1,1):1}})
    terms.append({'U': {(1,1):1}, 'V': {(1,0):1,(0,0):-1}, 'W': {(0,0):1,(1,0):1}})
    terms.append({'U': {(0,0):1,(0,1):1}, 'V': {(1,1):1}, 'W': {(0,0):-1,(0,1):1}})
    terms.append({'U': {(1,0):1,(0,0):-1}, 'V': {(0,0):1,(0,1):1}, 'W': {(1,1):1}})
    terms.append({'U': {(0,1):1,(1,1):-1}, 'V': {(1,0):1,(1,1):1}, 'W': {(0,0):1}})
    return terms


def _block_strassen(n, m, p):
    """
    Build block Strassen decomposition for 4x4 * 4x6.
    Split into 2x2 blocks. A: 2x2 blocks, B: 2x2 blocks (3 cols), C: 2x2 blocks (3 cols).
    Each C block = sum of two 2x2*2x2 products, each done with Strassen (7 terms).
    Total: 2*3 * 2 * 7 = 84 terms.
    """
    strassen_terms = _strassen_2x2x2()
    full_U = []
    full_V = []
    full_W = []
    
    for ci in range(2):
        for ck in range(3):
            for aj in range(2):
                for st in strassen_terms:
                    U_row = np.zeros(n * m)
                    V_row = np.zeros(m * p)
                    W_row = np.zeros(p * n)
                    
                    for (si, sj), coeff in st['U'].items():
                        gi = 2 * ci + si
                        gj = 2 * aj + sj
                        U_row[gi * m + gj] = coeff
                    
                    for (sj, sk), coeff in st['V'].items():
                        gj = 2 * aj + sj
                        gk = 2 * ck + sk
                        V_row[gj * p + gk] = coeff
                    
                    for (si, sk), coeff in st['W'].items():
                        gi = 2 * ci + si
                        gk = 2 * ck + sk
                        W_row[gk * n + gi] = coeff
                    
                    full_U.append(U_row)
                    full_V.append(V_row)
                    full_W.append(W_row)
    
    U = np.array(full_U)
    V = np.array(full_V)
    W = np.array(full_W)
    return U, V, W


def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p  # 96
    
    # Try block Strassen (rank 84)
    U_bs, V_bs, W_bs = _block_strassen(n, m, p)
    if _verify(U_bs, V_bs, W_bs, n, m, p):
        best_U, best_V, best_W = U_bs, V_bs, W_bs
        best_rank = U_bs.shape[1]  # 84
    
    # Try randomized search for lower ranks
    rng = np.random.default_rng(42)
    
    # Try ranks from best_rank-1 down, with time budget
    search_ranks = list(range(min(best_rank - 1, 80), 40, -1))
    
    for rank in search_ranks:
        if time.time() > deadline - 20:
            break
        
        # Limit iterations per rank based on remaining time
        remaining = deadline - time.time() - 15
        if remaining < 5:
            break
        
        # Estimate time per iteration: building A matrix is O(n*m*p*rank)
        # Be conservative: try a few iterations
        max_iters = max(5, int(remaining / 2.0))
        
        result = _try_random_rank(n, m, p, rank, rng, max_iters=max_iters)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank
            break  # Found a valid decomposition at this rank
    
    # Also try a few specific structured approaches
    if time.time() < deadline - 30 and best_rank > 50:
        # Try: use Strassen on the 4x4 part and schoolbook on the extra 2 columns
        # A(4x4) * B(4x6) = A(4x4) * [B1(4x4) | B2(4x2)]
        # = [A*B1 (4x4) | A*B2 (4x2)]
        # A*B1: use 2x2 block Strassen -> rank 4*4/4 * 7 = 28? No.
        # Actually for 4x4*4x4: 2x2 blocks, 4 block products, each Strassen 7 -> 28
        # For 4x4*4x2: 2x2 blocks for A, 2x1 blocks for B... tricky.
        # Let's just try random search at a few more ranks
        for rank in [best_rank - 2, best_rank - 5, best_rank - 10]:
            if rank < 40 or time.time() > deadline - 20:
                break
            result = _try_random_rank(n, m, p, rank, rng, max_iters=20)
            if result is not None:
                U_r, V_r, W_r = result
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
