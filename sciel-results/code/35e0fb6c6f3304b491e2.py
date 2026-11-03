import numpy as np
import time
import random
from itertools import combinations

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS
NM = n * m  # 9
MP = m * p  # 12
PN = p * n  # 12
TOTAL = n * m * p  # 36

def _verify_fast(U, V, W):
    """Fast verification: check tensor reconstruction only (no half-integer check)."""
    r = U.shape[1]
    if r == 0:
        return False
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
    return np.allclose(T, 1.0, atol=1e-10, rtol=1e-10)

def _verify_full(U, V, W):
    """Full verification including half-integer constraint."""
    if not _verify_fast(U, V, W):
        return False
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=1e-9): return False
    if not np.allclose(V2, np.round(V2), atol=1e-9): return False
    if not np.allclose(W2, np.round(W2), atol=1e-9): return False
    return True

def _rank1_term(i, j, k):
    """Return (u, v, w) vectors for a single rank-1 term at (i,j,k)."""
    u = np.zeros(NM)
    v = np.zeros(MP)
    w = np.zeros(PN)
    u[i * m + j] = 1.0
    v[j * p + k] = 1.0
    w[k * n + i] = 1.0
    return u, v, w

def _strassen_2x2(i0, j0, k0):
    """
    7-term Strassen decomposition for the 2x2x2 block at (i0,j0,k0).
    Returns list of 7 (u, v, w) vectors.
    """
    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i
    
    terms = []
    # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11, -c10, -c01
    u = np.zeros(NM); u[ui(i0,j0)] = 1; u[ui(i0+1,j0+1)] = 1
    v = np.zeros(MP); v[vi(j0,k0)] = 1; v[vi(j0+1,k0+1)] = 1
    w = np.zeros(PN); w[wi(k0,i0)] = 1; w[wi(k0+1,i0+1)] = 1; w[wi(k0,i0+1)] = -1; w[wi(k0+1,i0)] = -1
    terms.append((u, v, w))
    
    # M2 = (a10+a11)b00 -> c10
    u = np.zeros(NM); u[ui(i0+1,j0)] = 1; u[ui(i0+1,j0+1)] = 1
    v = np.zeros(MP); v[vi(j0,k0)] = 1
    w = np.zeros(PN); w[wi(k0,i0+1)] = 1
    terms.append((u, v, w))
    
    # M3 = a00(b10-b00) -> c00
    u = np.zeros(NM); u[ui(i0,j0)] = 1
    v = np.zeros(MP); v[vi(j0+1,k0)] = 1; v[vi(j0,k0)] = -1
    w = np.zeros(PN); w[wi(k0,i0)] = 1
    terms.append((u, v, w))
    
    # M4 = a11(b01-b11) -> c11
    u = np.zeros(NM); u[ui(i0+1,j0+1)] = 1
    v = np.zeros(MP); v[vi(j0,k0+1)] = 1; v[vi(j0+1,k0+1)] = -1
    w = np.zeros(PN); w[wi(k0+1,i0+1)] = 1
    terms.append((u, v, w))
    
    # M5 = (a00+a01)b11 -> c01
    u = np.zeros(NM); u[ui(i0,j0)] = 1; u[ui(i0,j0+1)] = 1
    v = np.zeros(MP); v[vi(j0+1,k0+1)] = 1
    w = np.zeros(PN); w[wi(k0+1,i0)] = 1
    terms.append((u, v, w))
    
    # M6 = (a10-a00)(b00+b01) -> c10, c11
    u = np.zeros(NM); u[ui(i0+1,j0)] = 1; u[ui(i0,j0)] = -1
    v = np.zeros(MP); v[vi(j0,k0)] = 1; v[vi(j0,k0+1)] = 1
    w = np.zeros(PN); w[wi(k0,i0+1)] = 1; w[wi(k0+1,i0+1)] = 1
    terms.append((u, v, w))
    
    # M7 = (a01-a11)(b10+b11) -> c00, c01
    u = np.zeros(NM); u[ui(i0,j0+1)] = 1; u[ui(i0+1,j0+1)] = -1
    v = np.zeros(MP); v[vi(j0+1,k0)] = 1; v[vi(j0+1,k0+1)] = 1
    w = np.zeros(PN); w[wi(k0,i0)] = 1; w[wi(k0+1,i0)] = 1
    terms.append((u, v, w))
    
    return terms

def _build_from_terms(term_list):
    """Build U, V, W matrices from a list of (u, v, w) tuples."""
    if not term_list:
        return None
    U = np.column_stack([t[0] for t in term_list])
    V = np.column_stack([t[1] for t in term_list])
    W = np.column_stack([t[2] for t in term_list])
    return U, V, W

def _local_search(U, V, W, deadline, max_rounds=50):
    """
    Local search: iteratively try to remove columns or merge pairs.
    Returns (U, V, W) with reduced rank if possible.
    """
    def _time_exceeded():
        return time.time() > deadline
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    r = current_U.shape[1]
    
    for round_num in range(max_rounds):
        if _time_exceeded() or r <= 1:
            break
        improved = False
        
        # Try removing each column
        for col in range(r):
            if _time_exceeded():
                break
            mask = [i for i in range(r) if i != col]
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            if _verify_fast(U_test, V_test, W_test):
                current_U, current_V, current_W = U_test, V_test, W_test
                r -= 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try merging pairs: replace two columns with one if possible
        # This is harder; skip for now and just do removal
        
        if not improved:
            break
    
    return current_U, current_V, current_W

def _try_strassen_plus_direct(deadline):
    """Try all combinations of Strassen blocks + direct terms for uncovered cells."""
    def _time_exceeded():
        return time.time() > deadline
    
    best_rank = float('inf')
    best_result = None
    
    # All possible 2x2x2 block positions
    block_positions = [(ib, jb, kb) for ib in range(n-1) for jb in range(m-1) for kb in range(p-1)]
    
    # Try single blocks
    for ib, jb, kb in block_positions:
        if _time_exceeded():
            break
        terms = _strassen_2x2(ib, jb, kb)
        covered = set()
        for i in range(ib, ib+2):
            for j in range(jb, jb+2):
                for k in range(kb, kb+2):
                    covered.add((i, j, k))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u, v, w = _rank1_term(i, j, k)
                        terms.append((u, v, w))
        
        result = _build_from_terms(terms)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W) and U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_result = (U, V, W)
    
    # Try pairs of non-overlapping blocks
    for idx1 in range(len(block_positions)):
        if _time_exceeded():
            break
        for idx2 in range(idx1+1, len(block_positions)):
            ib1, jb1, kb1 = block_positions[idx1]
            ib2, jb2, kb2 = block_positions[idx2]
            
            # Check if blocks overlap
            overlap = False
            for i in range(max(ib1,ib2), min(ib1+2,ib2+2)):
                for j in range(max(jb1,jb2), min(jb1+2,jb2+2)):
                    for k in range(max(kb1,kb2), min(kb1+2,kb2+2)):
                        if i < n and j < m and k < p:
                            overlap = True
            if overlap:
                continue
            
            terms = _strassen_2x2(ib1, jb1, kb1) + _strassen_2x2(ib2, jb2, kb2)
            covered = set()
            for ib, jb, kb in [(ib1,jb1,kb1), (ib2,jb2,kb2)]:
                for i in range(ib, ib+2):
                    for j in range(jb, jb+2):
                        for k in range(kb, kb+2):
                            covered.add((i, j, k))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) not in covered:
                            u, v, w = _rank1_term(i, j, k)
                            terms.append((u, v, w))
            
            result = _build_from_terms(terms)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W) and U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_result = (U, V, W)
    
    return best_result

def _random_search(deadline, seed=0):
    """Randomized local search from random starting points."""
    def _time_exceeded():
        return time.time() > deadline
    
    rng = random.Random(seed)
    best_rank = float('inf')
    best_result = None
    
    # Strategy: start with schoolbook (36 terms), then try to reduce
    # by replacing groups of rank-1 terms with structured terms
    
    for attempt in range(50):
        if _time_exceeded():
            break
        
        # Build a random starting decomposition
        # Mix of Strassen blocks and direct terms
        terms = []
        covered = set()
        
        # Randomly place some Strassen blocks
        possible_blocks = [(ib, jb, kb) for ib in range(n-1) for jb in range(m-1) for kb in range(p-1)]
        rng.shuffle(possible_blocks)
        
        for ib, jb, kb in possible_blocks:
            if _time_exceeded():
                break
            # Check if this block would overlap with already covered
            block_cells = set()
            for i in range(ib, ib+2):
                for j in range(jb, jb+2):
                    for k in range(kb, kb+2):
                        block_cells.add((i, j, k))
            
            # Only add if it covers at least one new cell
            new_cells = block_cells - covered
            if len(new_cells) < 4:  # Need at least 4 new cells to be worth 7 terms
                continue
            
            terms.extend(_strassen_2x2(ib, jb, kb))
            covered.update(block_cells)
        
        # Add direct terms for uncovered cells
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u, v, w = _rank1_term(i, j, k)
                        terms.append((u, v, w))
        
        result = _build_from_terms(terms)
        if result is None:
            continue
        
        U, V, W = result
        if not _verify_fast(U, V, W):
            continue
        
        # Local search to reduce rank
        U, V, W = _local_search(U, V, W, deadline, max_rounds=20)
        
        if _verify_fast(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    
    return best_result

def _try_hankel_structure(deadline):
    """
    Try Hankel-like structures that are known to give good matmul decompositions.
    For 3x3*3x4, we can try to exploit the rectangular structure.
    """
    def _time_exceeded():
        return time.time() > deadline
    
    best_rank = float('inf')
    best_result = None
    
    # Try: use 3 Strassen blocks (one for each "layer" in k)
    # Block at (0,0,0): covers i=0,1; j=0,1; k=0,1
    # Block at (1,0,1): covers i=1,2; j=0,1; k=1,2  
    # Block at (0,1,2): covers i=0,1; j=1,2; k=2,3
    # This might not work due to overlaps, but let's try various combos
    
    combos = [
        # Single blocks
        [(0,0,0)],
        [(0,0,1)],
        [(0,0,2)],
        [(1,0,0)],
        [(1,0,1)],
        [(1,0,2)],
        [(0,1,0)],
        [(0,1,1)],
        [(0,1,2)],
        [(1,1,0)],
        [(1,1,1)],
        [(1,1,2)],
    ]
    
    for combo in combos:
        if _time_exceeded():
            break
        terms = []
        covered = set()
        for ib, jb, kb in combo:
            terms.extend(_strassen_2x2(ib, jb, kb))
            for i in range(ib, ib+2):
                for j in range(jb, jb+2):
                    for k in range(kb, kb+2):
                        covered.add((i, j, k))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u, v, w = _rank1_term(i, j, k)
                        terms.append((u, v, w))
        
        result = _build_from_terms(terms)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W) and U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_result = (U, V, W)
    
    return best_result

def _polish(U, V, W):
    """
    Final polish: ensure exactness by rounding to nearest half-integers
    and verifying. If rounding breaks things, try small adjustments.
    """
    # Round to nearest 0.5
    U_rounded = np.round(2 * U) / 2
    V_rounded = np.round(2 * V) / 2
    W_rounded = np.round(2 * W) / 2
    
    if _verify_fast(U_rounded, V_rounded, W_rounded):
        return U_rounded, V_rounded, W_rounded
    
    # Try original
    if _verify_fast(U, V, W):
        # Check if already half-integer
        if (np.allclose(2*U, np.round(2*U)) and 
            np.allclose(2*V, np.round(2*V)) and 
            np.allclose(2*W, np.round(2*W))):
            return U, V, W
        return U_rounded, V_rounded, W_rounded
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def _time_exceeded():
        return time.time() > deadline
    
    best_rank = float('inf')
    best_result = None
    
    # 1. Try structured Strassen + direct decompositions
    result = _try_strassen_plus_direct(deadline)
    if result is not None:
        U, V, W = result
        if _verify_fast(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    
    if _time_exceeded():
        if best_result is not None:
            U, V, W = best_result
            U, V, W = _polish(U, V, W)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return _schoolbook_fallback()
    
    # 2. Try random searches with different seeds
    for seed in range(10):
        if _time_exceeded():
            break
        result = _random_search(deadline, seed=seed)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W) and U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_result = (U, V, W)
    
    if _time_exceeded():
        if best_result is not None:
            U, V, W = best_result
            U, V, W = _polish(U, V, W)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return _schoolbook_fallback()
    
    # 3. Aggressive local search on best result
    if best_result is not None:
        U, V, W = best_result
        U, V, W = _local_search(U, V, W, deadline, max_rounds=100)
        if _verify_fast(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    
    if _time_exceeded():
        if best_result is not None:
            U, V, W = best_result
            U, V, W = _polish(U, V, W)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return _schoolbook_fallback()
    
    # 4. More random searches if time permits
    for seed in range(10, 30):
        if _time_exceeded():
            break
        result = _random_search(deadline, seed=seed)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W) and U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_result = (U, V, W)
    
    if best_result is not None:
        U, V, W = best_result
        U, V, W = _polish(U, V, W)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return _schoolbook_fallback()

def _schoolbook_fallback():
    U_sb = np.zeros((NM, TOTAL))
    V_sb = np.zeros((MP, TOTAL))
    W_sb = np.zeros((PN, TOTAL))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print("Rank:", U.shape[1])
    print("Fast valid:", _verify_fast(U, V, W))
    print("Full valid:", _verify_full(U, V, W))
