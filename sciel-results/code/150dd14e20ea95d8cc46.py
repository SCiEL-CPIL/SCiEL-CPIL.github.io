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

# Build the multiplication tensor once (as a flat vector of length NM*MP*PN = 1024)
# T[a,b,c] = 1 if (a,b,c) in support, else 0
_SUPPORT = []
for _i in range(N):
    for _j in range(M):
        for _k in range(P):
            _a = _i * M + _j
            _b = _j * P + _k
            _c = _k * N + _i
            _SUPPORT.append((_a, _b, _c))
_SUPPORT = list(set(_SUPPORT))
_T_FLAT = np.zeros(NM * MP * PN, dtype=float)
for (_a, _b, _c) in _SUPPORT:
    _T_FLAT[_a * MP * PN + _b * PN + _c] = 1.0

# Strassen M_{2,2,2} terms: u,v,w are length-4 vectors (flattened 2x2)
_STRASSEN = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
    ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
]


def _strassen_block_vectors(block_idx):
    """Return (U_block, V_block, W_block) for one 2x2x2 Strassen block."""
    U = np.zeros((NM, 7))
    V = np.zeros((MP, 7))
    W = np.zeros((PN, 7))
    for t in range(7):
        u, v, w = _STRASSEN[t]
        for i in range(2):
            for j in range(2):
                U[i * M + j, t] = u[i * 2 + j]
        for j in range(2):
            for kl in range(2):
                kg = 2 * block_idx + kl
                V[j * P + kg, t] = v[j * 2 + kl]
        for i in range(2):
            for kl in range(2):
                kg = 2 * block_idx + kl
                W[kg * N + i, t] = w[i * 2 + kl]
    return U, V, W


def strassen_decomposition_228():
    """Rank-28 decomposition: 4 Strassen blocks."""
    U = np.zeros((NM, 28))
    V = np.zeros((MP, 28))
    W = np.zeros((PN, 28))
    for b in range(4):
        Ub, Vb, Wb = _strassen_block_vectors(b)
        U[:, b*7:(b+1)*7] = Ub
        V[:, b*7:(b+1)*7] = Vb
        W[:, b*7:(b+1)*7] = Wb
    return U, V, W


def verify(U, V, W, n=N, m=M, p=P):
    """Verify exact decomposition. Accepts lists or numpy arrays."""
    if U is None or V is None or W is None:
        return False
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    if rank == 0:
        return False
    # Compute T = sum_r U[:,r] V[:,r] W[:,r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    for (a, b, c) in _SUPPORT:
        if abs(T[a, b, c] - 1.0) > 1e-9:
            return False
    # All other entries must be 0
    mask = np.ones((n*m, m*p, p*n), dtype=bool)
    for (a, b, c) in _SUPPORT:
        mask[a, b, c] = False
    if np.any(np.abs(T[mask]) > 1e-9):
        return False
    return True


def schoolbook():
    """Rank-32 schoolbook decomposition."""
    rank = N * M * P
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W


def _to_list(arr):
    """Convert numpy array to list of lists (for output)."""
    return [[float(x) for x in row] for row in arr]


def _random_half_integer(max_abs=3):
    """Random integer or half-integer in [-max_abs, max_abs]."""
    r = random.uniform(-max_abs, max_abs)
    return round(2 * r) / 2.0


def _random_rank1_vectors():
    """Generate random rank-1 vectors (u, v, w) with integer/half-integer entries."""
    # Use a mix of sparse and dense vectors
    style = random.random()
    if style < 0.3:
        # Very sparse: 1-2 nonzeros
        u = np.zeros(NM)
        v = np.zeros(MP)
        w = np.zeros(PN)
        for _ in range(random.randint(1, 2)):
            u[random.randrange(NM)] = random.choice([-1, -0.5, 0.5, 1])
        for _ in range(random.randint(1, 2)):
            v[random.randrange(MP)] = random.choice([-1, -0.5, 0.5, 1])
        for _ in range(random.randint(1, 2)):
            w[random.randrange(PN)] = random.choice([-1, -0.5, 0.5, 1])
    elif style < 0.6:
        # Sparse: 3-5 nonzeros
        u = np.zeros(NM)
        v = np.zeros(MP)
        w = np.zeros(PN)
        for _ in range(random.randint(3, 5)):
            u[random.randrange(NM)] = random.choice([-1, -0.5, 0.5, 1])
        for _ in range(random.randint(3, 5)):
            v[random.randrange(MP)] = random.choice([-1, -0.5, 0.5, 1])
        for _ in range(random.randint(3, 5)):
            w[random.randrange(PN)] = random.choice([-1, -0.5, 0.5, 1])
    else:
        # Dense but small
        u = np.array([_random_half_integer(2) for _ in range(NM)])
        v = np.array([_random_half_integer(2) for _ in range(MP)])
        w = np.array([_random_half_integer(2) for _ in range(PN)])
    return u, v, w


def _try_random_rank_k_search(deadline, target_rank=26, max_restarts=5000):
    """
    Search for a rank-k decomposition by randomly sampling k rank-1 tensors
    and solving the linear system for coefficients.
    """
    T = _T_FLAT  # shape (1024,)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Generate k random rank-1 tensors
        U = np.zeros((NM, target_rank))
        V = np.zeros((MP, target_rank))
        W = np.zeros((PN, target_rank))
        
        for r in range(target_rank):
            u, v, w = _random_rank1_vectors()
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
        
        # Build the linear system: A @ c = T
        # A[:, r] = (u_r ⊗ v_r ⊗ w_r).flatten()
        A = np.zeros((NM * MP * PN, target_rank))
        for r in range(target_rank):
            A[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
        
        # Solve A @ c = T
        try:
            c, residuals, rank_A, sv = np.linalg.lstsq(A, T, rcond=1e-12)
        except Exception:
            continue
        
        # Check if solution is exact
        recon = A @ c
        if not np.allclose(recon, T, atol=1e-9):
            continue
        
        # Check coefficients are "nice" (integer or half-integer)
        # If coefficients are nice, we can absorb them into U
        nice = True
        for ci in c:
            if abs(ci - round(2 * ci) / 2.0) > 1e-6:
                nice = False
                break
        
        if not nice:
            # Try to round coefficients and re-verify
            c_rounded = np.round(2 * c) / 2.0
            recon_r = A @ c_rounded
            if np.allclose(recon_r, T, atol=1e-9):
                c = c_rounded
            else:
                continue
        
        # Absorb coefficients into U
        U_final = U * c  # scale each column by its coefficient
        # Remove zero columns
        nonzero = np.where(np.any(U_final != 0, axis=0))[0]
        if len(nonzero) < target_rank:
            # Fewer terms than target — even better!
            U_final = U_final[:, nonzero]
            V_final = V[:, nonzero]
            W_final = W[:, nonzero]
        else:
            V_final = V
            W_final = W
        
        # Final verification
        if verify(U_final, V_final, W_final):
            return U_final, V_final, W_final
    
    return None


def _try_strassen_mutation_search(deadline, target_rank=26, max_restarts=3000):
    """
    Start from Strassen rank-28 and try to reduce rank by:
    1. Removing 2 terms and adding corrections to remaining terms.
    2. Replacing 2 terms with 1 new term.
    """
    U28, V28, W28 = strassen_decomposition_228()
    T = _T_FLAT
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Strategy: remove 2 terms, keep 26, solve for corrections
        # Actually, let's try: keep 26 of the 28 Strassen terms, 
        # and solve for a rank-26 system where 2 terms are "merged"
        
        # Simpler: take 26 random terms from the 28 Strassen terms
        # plus 0 new terms, and see if they span T
        # (This won't work since 28 are independent)
        
        # Better: take all 28 Strassen terms, remove 2, add 0 new
        # The remaining 26 won't span T (since all 28 are needed)
        
        # Alternative: perturb 2 of the 28 terms to create dependency
        # This is complex. Instead, try: take 26 random terms from a 
        # larger pool that includes Strassen terms and random terms.
        
        pool_size = 28 + random.randint(0, 4)
        # Build pool: 28 Strassen terms + random extras
        U_pool = np.zeros((NM, pool_size))
        V_pool = np.zeros((MP, pool_size))
        W_pool = np.zeros((PN, pool_size))
        
        # Copy Strassen terms
        for r in range(28):
            U_pool[:, r] = U28[:, r]
            V_pool[:, r] = V28[:, r]
            W_pool[:, r] = W28[:, r]
        
        # Add random extras
        for r in range(28, pool_size):
            u, v, w = _random_rank1_vectors()
            U_pool[:, r] = u
            V_pool[:, r] = v
            W_pool[:, r] = w
        
        # Select target_rank terms from pool
        selected = random.sample(range(pool_size), target_rank)
        
        U_sel = U_pool[:, selected]
        V_sel = V_pool[:, selected]
        W_sel = W_pool[:, selected]
        
        A = np.zeros((NM * MP * PN, target_rank))
        for r in range(target_rank):
            A[:, r] = np.einsum('a,b,c->abc', U_sel[:, r], V_sel[:, r], W_sel[:, r]).flatten()
        
        try:
            c, residuals, rank_A, sv = np.linalg.lstsq(A, T, rcond=1e-12)
        except Exception:
            continue
        
        recon = A @ c
        if not np.allclose(recon, T, atol=1e-9):
            continue
        
        # Check nice coefficients
        nice = True
        for ci in c:
            if abs(ci - round(2 * ci) / 2.0) > 1e-6:
                nice = False
                break
        
        if not nice:
            c_rounded = np.round(2 * c) / 2.0
            recon_r = A @ c_rounded
            if np.allclose(recon_r, T, atol=1e-9):
                c = c_rounded
            else:
                continue
        
        U_final = U_sel * c
        V_final = V_sel
        W_final = W_sel
        
        nonzero = np.where(np.any(U_final != 0, axis=0))[0]
        if len(nonzero) < target_rank:
            U_final = U_final[:, nonzero]
            V_final = V_final[:, nonzero]
            W_final = W_final[:, nonzero]
        
        if verify(U_final, V_final, W_final):
            return U_final, V_final, W_final
    
    return None


def _try_rank_25_search(deadline, target_rank=25, max_restarts=2000):
    """Try to find a rank-25 decomposition (even better!)."""
    return _try_random_rank_k_search(deadline, target_rank=target_rank, max_restarts=max_restarts)


def construct():
    """Entry point: return the best decomposition found."""
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Start with known rank-28 solution
    best_result = None
    best_rank = 999
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            best_result = {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
            best_rank = 28
    except Exception:
        pass
    
    # Try rank-26 via random search
    if time.time() < deadline:
        try:
            r26 = _try_random_rank_k_search(deadline, target_rank=26, max_restarts=3000)
            if r26 is not None:
                U26, V26, W26 = r26
                if verify(U26, V26, W26):
                    return {"U": _to_list(U26), "V": _to_list(V26), "W": _to_list(W26)}
        except Exception:
            pass
    
    # Try rank-26 via Strassen mutation
    if time.time() < deadline:
        try:
            r26b = _try_strassen_mutation_search(deadline, target_rank=26, max_restarts=2000)
            if r26b is not None:
                U26, V26, W26 = r26b
                if verify(U26, V26, W26):
                    return {"U": _to_list(U26), "V": _to_list(V26), "W": _to_list(W26)}
        except Exception:
            pass
    
    # Try rank-25 (ambitious)
    if time.time() < deadline:
        try:
            r25 = _try_rank_25_search(deadline, target_rank=25, max_restarts=1000)
            if r25 is not None:
                U25, V25, W25 = r25
                if verify(U25, V25, W25):
                    return {"U": _to_list(U25), "V": _to_list(V25), "W": _to_list(W25)}
        except Exception:
            pass
    
    # Return best found
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
