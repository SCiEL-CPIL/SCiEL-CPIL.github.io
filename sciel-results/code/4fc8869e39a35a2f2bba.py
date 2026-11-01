"""Tensor decomposition for 2x2 x 2x9 matrix multiplication."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 130  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _build_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _validate(U, V, W, n, m, p, atol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T, T_recon, atol=atol)


def _strassen_2x2x2():
    """
    Return Strassen (rank 7) decomposition for 2x2x2 matrix multiplication.
    
    The tensor for 2x2x2 has entries T[i*2+j, j*2+k, k*2+i] = 1.
    U indices: 0=A00, 1=A01, 2=A10, 3=A11
    V indices: 0=B00, 1=B01, 2=B10, 3=B11
    W indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    Strassen terms:
    P0=(A00+A11)(B00+B11) -> C00+=P0, C11+=P0
    P1=(A10+A11)B00       -> C10+=P1, C11-=P1
    P2=A00(B01-B11)       -> C01+=P2, C11-=P2
    P3=A11(B10-B00)       -> C10+=P3, C11+=P3
    P4=(A00+A01)B11       -> C01+=P4, C11+=P4
    P5=(A10-A00)(B00+B01) -> C00-=P5, C10+=P5
    P6=(A01-A11)(B10+B11) -> C00+=P6, C01+=P6
    """
    terms = [
        # P0
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # P1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # P2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, -1)]),
        # P3
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (3, 1)]),
        # P4
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, -1), (2, 1)]),
        # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)]),
    ]
    out = []
    for uc, vc, wc in terms:
        u = np.zeros(4)
        for idx, c in uc:
            u[idx] += c
        v = np.zeros(4)
        for idx, c in vc:
            v[idx] += c
        w = np.zeros(4)
        for idx, c in wc:
            w[idx] += c
        out.append((u, v, w))
    return out


def _kron_decomposition(n, m, p):
    """
    Build decomposition using block approach:
    Split k into 4 blocks of 2 (using Strassen rank 7 each) + 1 block of 1 (rank 4).
    Total rank: 4*7 + 4 = 32
    """
    U_terms = []
    V_terms = []
    W_terms = []
    
    strassen = _strassen_2x2x2()
    
    for block in range(4):
        k_start = block * 2
        for u, v, w in strassen:
            U_terms.append(u.copy())
            v_full = np.zeros(m * p)
            v_full[0 * p + k_start] = v[0]       # B0k
            v_full[0 * p + k_start + 1] = v[1]   # B0(k+1)
            v_full[1 * p + k_start] = v[2]       # B1k
            v_full[1 * p + k_start + 1] = v[3]   # B1(k+1)
            V_terms.append(v_full)
            w_full = np.zeros(p * n)
            w_full[k_start * n + 0] = w[0]       # Ck0
            w_full[k_start * n + 1] = w[1]       # Ck1
            w_full[(k_start + 1) * n + 0] = w[2] # C(k+1)0
            w_full[(k_start + 1) * n + 1] = w[3] # C(k+1)1
            W_terms.append(w_full)
    
    # Block 5: k=8 (single column)
    # T[i*2+j, j*9+8, 8*2+i] = 1 for i,j in {0,1}
    # Entries: (0,8,16), (1,8,17), (2,17,16), (3,17,17)
    for i in range(2):
        for j in range(2):
            u = np.zeros(4)
            u[i * 2 + j] = 1
            v = np.zeros(m * p)
            v[j * p + 8] = 1
            w = np.zeros(p * n)
            w[8 * n + i] = 1
            U_terms.append(u)
            V_terms.append(v)
            W_terms.append(w)
    
    if not U_terms:
        return None
    
    U = np.array(U_terms).T  # (4, rank)
    V = np.array(V_terms).T  # (18, rank)
    W = np.array(W_terms).T  # (18, rank)
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _try_improve_with_greedy(U, V, W, n, m, p, max_iter=50, time_limit=60):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    If we can express some terms as linear combinations of others, we can remove them.
    
    This is a structural search: we look for subsets of terms that are linearly dependent
    in the tensor space.
    """
    rank = U.shape[1]
    if rank <= 32:
        return U, V, W
    
    # Build the full tensor
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Build matrix where each column is a rank-1 term flattened
    # Term r: u_r ⊗ v_r ⊗ w_r flattened
    n_u, n_v, n_w = U.shape[0], V.shape[0], W.shape[0]
    terms_flat = np.zeros((n_u * n_v * n_w, rank))
    for r in range(rank):
        term = np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).flatten()
        terms_flat[:, r] = term
    
    # Check for linear dependencies using SVD
    # If some columns are linearly dependent, we can potentially reduce rank
    # But this is a necessary condition, not sufficient for valid decomposition
    
    # For now, just return the input if no improvement found
    return U, V, W


def construct():
    n, m, p = DIMENSIONS

    best = None
    best_rank = None

    # Try Kronecker decomposition (rank 32)
    try:
        res = _kron_decomposition(n, m, p)
        if res is not None:
            U, V, W = res
            if _validate(U, V, W, n, m, p):
                rank = U.shape[1]
                if best_rank is None or rank < best_rank:
                    best = (U, V, W)
                    best_rank = rank
    except Exception:
        pass

    # Try to improve
    if best is not None and _elapsed() < _TIMEOUT - 30:
        try:
            U, V, W = best
            U_improved, V_improved, W_improved = _try_improve_with_greedy(
                U, V, W, n, m, p, max_iter=50, time_limit=60
            )
            if U_improved.shape[1] < U.shape[1]:
                if _validate(U_improved, V_improved, W_improved, n, m, p):
                    best = (U_improved, V_improved, W_improved)
                    best_rank = U_improved.shape[1]
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook (rank 36)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
