"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
import random
from fractions import Fraction
from itertools import product

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
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
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
    # Correct Strassen terms for M_{2,2,2}
    # U, V, W are vectors of length 4 (indices 00, 01, 10, 11)
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
    
    num_blocks = p // 2 # 4 blocks
    
    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U: indices are i*m + j. Since m=2, indices are 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
            for i in range(4):
                U[i][col] = u[i]
                
            # Fill V: indices are j*p + k.
            # Block block_idx uses k = 2*block_idx and 2*block_idx+1.
            # Local v indices 0,1,2,3 correspond to:
            # 0: j=0, k_local=0 -> j=0, k_global=2*block_idx
            # 1: j=0, k_local=1 -> j=0, k_global=2*block_idx+1
            # 2: j=1, k_local=0 -> j=1, k_global=2*block_idx
            # 3: j=1, k_local=1 -> j=1, k_global=2*block_idx+1
            for l in range(4):
                if l == 0:
                    b_idx = 0 * p + (2 * block_idx)
                elif l == 1:
                    b_idx = 0 * p + (2 * block_idx + 1)
                elif l == 2:
                    b_idx = 1 * p + (2 * block_idx)
                elif l == 3:
                    b_idx = 1 * p + (2 * block_idx + 1)
                V[b_idx][col] = v[l]
                
            # Fill W: indices are k*n + i.
            # Block block_idx uses k = 2*block_idx and 2*block_idx+1.
            # Local w indices 0,1,2,3 correspond to:
            # 0: i=0, k_local=0 -> i=0, k_global=2*block_idx
            # 1: i=0, k_local=1 -> i=0, k_global=2*block_idx+1
            # 2: i=1, k_local=0 -> i=1, k_global=2*block_idx
            # 3: i=1, k_local=1 -> i=1, k_global=2*block_idx+1
            for l in range(4):
                if l == 0:
                    c_idx = (2 * block_idx) * n + 0
                elif l == 1:
                    c_idx = (2 * block_idx + 1) * n + 0
                elif l == 2:
                    c_idx = (2 * block_idx) * n + 1
                elif l == 3:
                    c_idx = (2 * block_idx + 1) * n + 1
                W[c_idx][col] = w[l]
                    
    return U, V, W

def _try_rank_27_search(deadline):
    """
    Attempt to find a rank-27 decomposition.
    Start from the rank-28 Strassen solution and try to eliminate one term.
    Returns (U, V, W) if found, None otherwise.
    """
    U28, V28, W28 = strassen_decomposition_228()
    if not verify(U28, V28, W28):
        return None
    
    # Try each term elimination with structured absorption
    for elim_term in range(28):
        if time.time() > deadline:
            return None
        
        # Compute residual after removing elim_term
        residual = {}
        valid = set()
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    valid.add((i*M+j, j*P+k, k*N+i))
        
        for a in range(NM):
            for b in range(MP):
                for c in range(PN):
                    s = 0
                    for r in range(28):
                        if r == elim_term:
                            continue
                        s += U28[a][r] * V28[b][r] * W28[c][r]
                    expected = 1 if (a, b, c) in valid else 0
                    res = s - expected
                    if res != 0:
                        residual[(a, b, c)] = res
        
        if not residual:
            # Already zero residual - just remove the term
            U27 = [U28[r][:elim_term] + U28[r][elim_term+1:] for r in range(NM)]
            V27 = [V28[r][:elim_term] + V28[r][elim_term+1:] for r in range(MP)]
            W27 = [W28[r][:elim_term] + W28[r][elim_term+1:] for r in range(PN)]
            return U27, V27, W27
        
        # Try adjusting entries in U, V, W to absorb the residual
        U_try = [row[:] for row in U28]
        V_try = [row[:] for row in V28]
        W_try = [row[:] for row in W28]
        
        # Try small integer/half-integer perturbations
        perturb_values = [1, -1, 0.5, -0.5]
        
        # Try adjusting entries in U
        for a in range(NM):
            for r in range(28):
                if r == elim_term:
                    continue
                for pert in perturb_values:
                    if time.time() > deadline:
                        return None
                    U_try[a][r] = U28[a][r] + pert
                    if verify(U_try, V_try, W_try):
                        U27 = [U_try[r2][:elim_term] + U_try[r2][elim_term+1:] for r2 in range(NM)]
                        V27 = [V_try[r2][:elim_term] + V_try[r2][elim_term+1:] for r2 in range(MP)]
                        W27 = [W_try[r2][:elim_term] + W_try[r2][elim_term+1:] for r2 in range(PN)]
                        return U27, V27, W27
                    U_try[a][r] = U28[a][r]
        
        # Try adjusting entries in V
        for b in range(MP):
            for r in range(28):
                if r == elim_term:
                    continue
                for pert in perturb_values:
                    if time.time() > deadline:
                        return None
                    V_try[b][r] = V28[b][r] + pert
                    if verify(U_try, V_try, W_try):
                        U27 = [U_try[r2][:elim_term] + U_try[r2][elim_term+1:] for r2 in range(NM)]
                        V27 = [V_try[r2][:elim_term] + V_try[r2][elim_term+1:] for r2 in range(MP)]
                        W27 = [W_try[r2][:elim_term] + W_try[r2][elim_term+1:] for r2 in range(PN)]
                        return U27, V27, W27
                    V_try[b][r] = V28[b][r]
        
        # Try adjusting entries in W
        for c in range(PN):
            for r in range(28):
                if r == elim_term:
                    continue
                for pert in perturb_values:
                    if time.time() > deadline:
                        return None
                    W_try[c][r] = W28[c][r] + pert
                    if verify(U_try, V_try, W_try):
                        U27 = [U_try[r2][:elim_term] + U_try[r2][elim_term+1:] for r2 in range(NM)]
                        V27 = [V_try[r2][:elim_term] + V_try[r2][elim_term+1:] for r2 in range(MP)]
                        W27 = [W_try[r2][:elim_term] + W_try[r2][elim_term+1:] for r2 in range(PN)]
                        return U27, V27, W27
                    W_try[c][r] = W28[c][r]
    
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
    
    # Try to find rank-27
    try:
        r27 = _try_rank_27_search(deadline)
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
