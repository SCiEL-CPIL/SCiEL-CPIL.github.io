import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin

def _build_structured_rank77():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 x 2x2
    # A blocks: 0=00, 1=01, 2=10, 3=11
    # B blocks: 0=00, 1=01, 2=10, 3=11
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        # Inner Strassen for 2x2 x 2x2 (cols 0,1)
        for si in range(7):
            # U part
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Schoolbook for 2x2 x 2x1 (col 2)
        for u in range(2):
            for v in range(2):
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    return U, V, W

def _build_expected_tensor():
    """Build the expected matrix multiplication tensor T[a,b,c] for 4x4 * 4x6."""
    n, m, p = 4, 4, 6
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

_EXPECTED_T = None

def _get_expected_tensor():
    global _EXPECTED_T
    if _EXPECTED_T is None:
        _EXPECTED_T = _build_expected_tensor()
    return _EXPECTED_T

def _verify_decomposition(U, V, W, tol=1e-6):
    """Check if U,V,W exactly decompose the 4x4*4x6 matrix multiplication tensor (vectorized)."""
    R = U.shape[1]
    if R == 0:
        return False
    # Compute M[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    M = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    T = _get_expected_tensor()
    if np.max(np.abs(M - T)) > tol:
        return False
    return True

def _rank_reduce(U, V, W, max_time=130.0):
    """Try to reduce rank by identifying and removing redundant terms (vectorized)."""
    R = U.shape[1]
    start = time.time()
    T = _get_expected_tensor()
    
    if not _verify_decomposition(U, V, W):
        return U, V, W
    
    # Precompute full tensor
    M_full = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    # Check for all-zero columns
    zero_cols = [r for r in range(R) 
                 if np.allclose(U[:, r], 0, atol=1e-9) or 
                     np.allclose(V[:, r], 0, atol=1e-9) or 
                     np.allclose(W[:, r], 0, atol=1e-9)]
    
    if zero_cols:
        keep = [x for x in range(R) if x not in zero_cols]
        U_r, V_r, W_r = U[:, keep], V[:, keep], W[:, keep]
        M_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
        if np.max(np.abs(M_r - T)) < 1e-6:
            return U_r, V_r, W_r
    
    # Try single removals using residual check
    # M_removed = M_full - M_r, so M_r = T iff M_full - M_r = 0 iff M_removed = M_full - T
    # Actually: M_r = M_full - sum_{r in removed} term_r. We need M_r = T, so sum_{removed} term_r = M_full - T = 0
    # So we need the sum of removed terms to be zero.
    
    # Precompute each term's contribution
    # term_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
    # For single removal: need U[:,r]*V[:,r]*W[:,r] = 0 (outer product zero)
    # For pair removal: need term_r + term_s = 0
    
    # Single removals: check if term is zero
    for r in range(R):
        if time.time() - start > max_time - 20:
            break
        # Check if U[:,r], V[:,r], W[:,r] outer product is zero
        # This is zero iff any of the three vectors is zero
        if np.allclose(U[:, r], 0, atol=1e-9) or np.allclose(V[:, r], 0, atol=1e-9) or np.allclose(W[:, r], 0, atol=1e-9):
            keep = [x for x in range(R) if x != r]
            U_r, V_r, W_r = U[:, keep], V[:, keep], W[:, keep]
            M_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
            if np.max(np.abs(M_r - T)) < 1e-6:
                return U_r, V_r, W_r
    
    # Try pair removals: need term_r + term_s = 0
    # term_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
    # So we need U[a,r]*V[b,r]*W[c,r] + U[a,s]*V[b,s]*W[c,s] = 0 for all a,b,c
    # This means U[:,r]*V[:,r]*W[:,r] = -U[:,s]*V[:,s]*W[:,s]
    
    # Compute all term tensors
    terms = np.zeros((R, 16, 24, 24))
    for r in range(R):
        terms[r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r], optimize=True)
    
    # For pairs, check if terms[r] + terms[s] == 0
    if time.time() - start < max_time - 25:
        pairs_tried = 0
        for r in range(R):
            if time.time() - start > max_time - 20:
                break
            for s in range(r + 1, R):
                if time.time() - start > max_time - 20 or pairs_tried >= 2000:
                    break
                # Check if terms[r] + terms[s] is zero
                if np.max(np.abs(terms[r] + terms[s])) < 1e-6:
                    keep = [x for x in range(R) if x != r and x != s]
                    U_r, V_r, W_r = U[:, keep], V[:, keep], W[:, keep]
                    M_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                    if np.max(np.abs(M_r - T)) < 1e-6:
                        return U_r, V_r, W_r
                pairs_tried += 1
    
    return U, V, W

def to_list(mat):
    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

def construct():
    # Build the structured rank-77 decomposition
    U, V, W = _build_structured_rank77()
    
    # Verify
    if not _verify_decomposition(U, V, W):
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Try to reduce rank with time guard
    remaining = TIME_LIMIT - (time.time() - START_TIME) - 5.0
    if remaining > 10:
        U, V, W = _rank_reduce(U, V, W, max_time=max(10, remaining))
    
    # Final verification
    if not _verify_decomposition(U, V, W):
        # Fallback to original
        U, V, W = _build_structured_rank77()
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
