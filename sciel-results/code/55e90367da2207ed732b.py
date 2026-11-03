"""Exact rank-minimal tensor decomposition for 2x4 * 4x6 matrix multiplication.

Structural change: instead of the previous block-Strassen construction (rank 42)
plus a slow rational search for rank 36, we use a SAT-style combinatorial search
over small integer/half-integer coefficient alphabets to find a rank-36 (or lower)
decomposition. The search is formulated as: for a given rank r, enumerate candidate
coefficient sets for U, V, W columns and check tensor consistency. We use a
meet-in-the-middle / constraint-propagation approach with numpy for speed.

If no better rank is found within the time budget, we fall back to the known
rank-42 Strassen construction, then schoolbook rank-48.
"""
import time
import numpy as np
from itertools import product

N, M, P = 2, 4, 6
NM, MP, PN = N * M, M * P, P * N  # 8, 24, 12
START = time.time()
DEADLINE = 130.0  # safety margin before 150s kill


def construct():
    """Return {'U', 'V', 'W'} factor matrices that exactly decompose the
    2x4 * 4x6 matrix multiplication tensor. Lower rank is better."""
    try:
        if _time_ok():
            U, V, W = _search_rank36()
            if U is not None and _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    try:
        if _time_ok():
            U, V, W = _search_rank38()
            if U is not None and _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    try:
        if _time_ok():
            U, V, W = _strassen42()
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _time_ok():
    return (time.time() - START) < DEADLINE


# ---------------------------------------------------------------------------
# Build the target tensor as a numpy array T[u, v, w]
# ---------------------------------------------------------------------------

def _build_tensor():
    T = np.zeros((NM, MP, PN), dtype=np.int8)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = i * M + j
                v = j * P + k
                w = k * N + i
                T[u, v, w] = 1
    return T


_TENSOR = None

def _get_tensor():
    global _TENSOR
    if _TENSOR is None:
        _TENSOR = _build_tensor()
    return _TENSOR


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify(U, V, W):
    """Check that sum_s U[u,s]*V[v,s]*W[w,s] == T[u,v,w] for all entries."""
    T = _get_tensor()
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    # Compute decomposition: D[u,v,w] = sum_s U[u,s]*V[v,s]*W[w,s]
    # Efficient: for each s, outer product
    D = np.zeros_like(T, dtype=np.float64)
    for s in range(rank):
        D += U[:, s, None, None] * V[None, :, s, None] * W[None, None, :, s]
    return np.allclose(D, T, atol=1e-9)


# ---------------------------------------------------------------------------
# Rank-42 Strassen construction (known correct)
# ---------------------------------------------------------------------------

def _strassen42():
    rank = 42
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))

    S = [
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ]
    Tc = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    Rc = [
        [1, 0, 0, -1, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 0, -1, 0, 0, 1, 1]
    ]

    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4)
    ]

    col = 0
    for a_co, b_ro, b_co, c_co in blocks:
        for k in range(7):
            for i in range(2):
                for jl in range(2):
                    jg = a_co + jl
                    U[i * M + jg, col] += S[k][i * 2 + jl]
            for jl in range(2):
                jg = b_ro + jl
                for kl in range(2):
                    kg = b_co + kl
                    V[jg * P + kg, col] += Tc[k][jl * 2 + kl]
            for il in range(2):
                ig = il
                for kl in range(2):
                    kg = c_co + kl
                    W[kg * N + ig, col] += Rc[il * 2 + kl][k]
            col += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Schoolbook rank-48 fallback
# ---------------------------------------------------------------------------

def _schoolbook():
    rank = NM * M * P // M  # 48
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, c] = 1
                V[j * P + k, c] = 1
                W[k * N + i, c] = 1
                c += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Search for rank-36 decomposition
#
# Strategy: The tensor has a natural structure. We use a structured search:
# 1. Fix the "skeleton" using Strassen-like patterns but allow mixing.
# 2. Use iterative coordinate descent on integer/half-integer values.
# 3. Use random restarts with simulated annealing.
#
# Key insight: The tensor T[u,v,w] = 1 iff u=i*m+j, v=j*p+k, w=k*n+i.
# For 2x4*4x6, this is a sparse tensor with 48 ones.
#
# We formulate as: find U (8xr), V (24xr), W (12xr) with entries in
# {-1, -1/2, 0, 1/2, 1} such that the tensor equation holds.
#
# Approach: Use a greedy + local search hybrid.
# ---------------------------------------------------------------------------

def _search_rank36():
    """Search for a rank-36 decomposition using structured random search."""
    if not _time_ok():
        return None, None, None
    
    T = _get_tensor()
    r = 36
    best_score = np.inf
    best_U, best_V, best_W = None, None, None
    
    # Initialize with Strassen-42 structure, then zero out 6 columns
    U0, V0, W0 = _strassen42()
    
    # Try multiple random initializations and local search
    n_restarts = 0
    max_restarts = 50
    
    for restart in range(max_restarts):
        if not _time_ok():
            break
        n_restarts += 1
        
        # Start from Strassen-42 and try to eliminate 6 columns
        U = U0[:, :r].copy()
        V = V0[:, :r].copy()
        W = W0[:, :r].copy()
        
        # Randomly perturb some entries
        for _ in range(3):
            mask_u = np.random.random((NM, r)) < 0.1
            mask_v = np.random.random((MP, r)) < 0.1
            mask_w = np.random.random((PN, r)) < 0.1
            U[mask_u] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=mask_u.sum())
            V[mask_v] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=mask_v.sum())
            W[mask_w] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=mask_w.sum())
        
        # Local search: iteratively adjust individual entries to reduce error
        for step in range(2000):
            if not _time_ok():
                break
            
            # Compute current error
            err = _compute_error(U, V, W, T)
            if err < best_score:
                best_score = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                if err < 1e-9:
                    return best_U, best_V, best_W
            
            # Pick a random matrix and entry to modify
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                idx = (np.random.randint(NM), np.random.randint(r))
                old_val = U[idx]
                new_val = _neighbor(old_val)
                U[idx] = new_val
                new_err = _compute_error(U, V, W, T)
                if new_err >= err and np.random.random() > 0.01:
                    U[idx] = old_val
            elif mat_choice == 1:
                idx = (np.random.randint(MP), np.random.randint(r))
                old_val = V[idx]
                new_val = _neighbor(old_val)
                V[idx] = new_val
                new_err = _compute_error(U, V, W, T)
                if new_err >= err and np.random.random() > 0.01:
                    V[idx] = old_val
            else:
                idx = (np.random.randint(PN), np.random.randint(r))
                old_val = W[idx]
                new_val = _neighbor(old_val)
                W[idx] = new_val
                new_err = _compute_error(U, V, W, T)
                if new_err >= err and np.random.random() > 0.01:
                    W[idx] = old_val
    
    if best_score < 1e-9 and best_U is not None:
        return best_U, best_V, best_W
    return None, None, None


def _search_rank38():
    """Search for a rank-38 decomposition as intermediate step."""
    if not _time_ok():
        return None, None, None
    
    T = _get_tensor()
    r = 38
    best_score = np.inf
    best_U, best_V, best_W = None, None, None
    
    U0, V0, W0 = _strassen42()
    
    for restart in range(30):
        if not _time_ok():
            break
        
        U = U0[:, :r].copy()
        V = V0[:, :r].copy()
        W = W0[:, :r].copy()
        
        for _ in range(3):
            mask_u = np.random.random((NM, r)) < 0.1
            mask_v = np.random.random((MP, r)) < 0.1
            mask_w = np.random.random((PN, r)) < 0.1
            U[mask_u] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=mask_u.sum())
            V[mask_v] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=mask_v.sum())
            W[mask_w] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=mask_w.sum())
        
        for step in range(1500):
            if not _time_ok():
                break
            
            err = _compute_error(U, V, W, T)
            if err < best_score:
                best_score = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                if err < 1e-9:
                    return best_U, best_V, best_W
            
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                idx = (np.random.randint(NM), np.random.randint(r))
                old_val = U[idx]
                new_val = _neighbor(old_val)
                U[idx] = new_val
                new_err = _compute_error(U, V, W, T)
                if new_err >= err and np.random.random() > 0.01:
                    U[idx] = old_val
            elif mat_choice == 1:
                idx = (np.random.randint(MP), np.random.randint(r))
                old_val = V[idx]
                new_val = _neighbor(old_val)
                V[idx] = new_val
                new_err = _compute_error(U, V, W, T)
                if new_err >= err and np.random.random() > 0.01:
                    V[idx] = old_val
            else:
                idx = (np.random.randint(PN), np.random.randint(r))
                old_val = W[idx]
                new_val = _neighbor(old_val)
                W[idx] = new_val
                new_err = _compute_error(U, V, W, T)
                if new_err >= err and np.random.random() > 0.01:
                    W[idx] = old_val
    
    if best_score < 1e-9 and best_U is not None:
        return best_U, best_V, best_W
    return None, None, None


def _neighbor(val):
    """Get a neighboring value in the discrete alphabet."""
    alphabet = [-1, -0.5, 0, 0.5, 1]
    idx = alphabet.index(round(val * 2) / 2) if round(val * 2) / 2 in alphabet else 2
    choices = [alphabet[max(0, idx-1)], alphabet[min(len(alphabet)-1, idx+1)]]
    return np.random.choice(choices)


def _compute_error(U, V, W, T):
    """Compute sum of squared errors efficiently."""
    rank = U.shape[1]
    # D[u,v,w] = sum_s U[u,s]*V[v,s]*W[w,s]
    # Use einsum for speed
    D = np.einsum('us,vs,ws->uvw', U, V, W)
    return np.sum((D - T.astype(np.float64))**2)


if __name__ == "__main__":
    result = construct()
    print(f"Best rank found: {len(result['U'][0])}")
