"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

N, M, P = 2, 2, 8
NM, MP, PN = N*M, M*P, P*N  # 4, 16, 16

def verify(U, V, W, n=N, m=M, p=P):
    """Verify exact decomposition. U: (nm x rank), V: (mp x rank), W: (pn x rank)."""
    nm, mp, pn = n*m, m*p, p*n
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
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
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

def strassen_228():
    """Rank-28 Strassen block decomposition for M_{2,2,8}."""
    rank = 28
    U = [[0]*rank for _ in range(NM)]
    V = [[0]*rank for _ in range(MP)]
    W = [[0]*rank for _ in range(PN)]
    
    # Strassen terms for 2x2x2
    # A flat: [A00, A01, A10, A11] indices 0,1,2,3
    # For each block t (0..3), B columns are 2t, 2t+1
    # B flat index: j*8 + k
    # C flat index: k*2 + i
    
    terms = [
        # (u, v_pattern, w_pattern)
        # u: 4-vector for A
        # v: 4-vector for local B [B00, B01, B10, B11]
        # w: 4-vector for local C [C00, C01, C10, C11] where Cij = C[i, 2t+j]
        ([1,0,0,1], [1,0,0,0], [1,0,0,0]),  # M1 = (A00+A11)*B00 -> but wait
    ]
    
    # Let me redo this carefully.
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M5 + M3
    # C10 = M2 + M4
    # C11 = M1 - M2 - M6 + M3
    #
    # Wait, let me use the standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M5 + M3
    # C10 = M2 + M4
    # C11 = M1 - M2 - M6 + M3
    
    # u vectors (A-side), v vectors (B-side local), w vectors (C-side local)
    # Local B: [B00, B01, B10, B11]
    # Local C: [C00, C01, C10, C11]
    
    terms = [
        # (u, v, w)
        ([1,0,0,1], [1,0,0,0], [1,0,0,0]),  # M1: (A00+A11)*B00? No...
    ]
    
    # I'm getting confused. Let me just use the standard formulation.
    # Each term r contributes u_r[i] * v_r[j] * w_r[k] to C[i,k]
    # where i is row of C (0,1), j is row of B (0,1), k is col of B/C (0,1 local)
    
    # Term 1: M1 = (A00+A11)*(B00+B11)
    # u = [1,0,0,1] (A00+A11)
    # v = [1,0,1,0] (B00+B11)
    # w: this contributes to C00 and C11
    # C00 gets +M1, C11 gets +M1
    # So w = [1, 0, 0, 1] (C00 and C11)
    
    # Term 2: M2 = (A10+A11)*B00
    # u = [0,0,1,1]
    # v = [1,0,0,0]
    # C10 gets +M2
    # w = [0,0,1,0]
    
    # Term 3: M3 = A00*(B10-B11)
    # u = [1,0,0,0]
    # v = [0,0,1,-1]
    # C01 gets +M3, C11 gets +M3
    # w = [0,1,0,1]
    
    # Term 4: M4 = A11*(B01-B00)
    # u = [0,0,0,1]
    # v = [-1,1,0,0]
    # C00 gets -M4, C10 gets +M4
    # w = [-1,0,1,0]
    
    # Term 5: M5 = (A00+A01)*B11
    # u = [1,1,0,0]
    # v = [0,0,0,1]
    # C00 gets -M5, C01 gets +M5
    # w = [-1,1,0,0]
    
    # Term 6: M6 = (A10-A00)*(B00+B01)
    # u = [-1,0,1,0]
    # v = [1,1,0,0]
    # C11 gets -M6
    # w = [0,0,0,-1]
    
    # Term 7: M7 = (A01-A11)*(B10+B11)
    # u = [0,1,0,-1]
    # v = [0,0,1,1]
    # C00 gets +M7
    # w = [1,0,0,0]
    
    terms = [
        ([1,0,0,1], [1,0,1,0], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,1,0,0], [-1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,-1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    for t in range(4):
        for s in range(7):
            col = t*7 + s
            u, v, w = terms[s]
            # Fill U: u[i] for i in 0..3
            for i in range(4):
                U[i][col] = u[i]
            # Fill V: v[j*2+k] for j in 0..1, k in 0..1
            # Global B index: j*8 + (2t+k)
            for j in range(2):
                for k in range(2):
                    b_idx = j*8 + (2*t + k)
                    v_idx = j*2 + k
                    V[b_idx][col] = v[v_idx]
            # Fill W: w[i*2+k] for i in 0..1, k in 0..1
            # Global C index: (2t+k)*2 + i
            for i in range(2):
                for k in range(2):
                    c_idx = (2*t + k)*2 + i
                    w_idx = i*2 + k
                    W[c_idx][col] = w[w_idx]
    
    return U, V, W

def schoolbook():
    """Rank-32 schoolbook decomposition."""
    rank = 32
    U = [[0]*rank for _ in range(NM)]
    V = [[0]*rank for _ in range(MP)]
    W = [[0]*rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j][col] = 1
                V[j*P+k][col] = 1
                W[k*N+i][col] = 1
                col += 1
    return U, V, W

def _build_strassen_228():
    """Build and verify a rank-28 Strassen decomposition for 2x2x8.
    
    Uses the block structure: 4 blocks of 2x2x2, each with 7 Strassen terms.
    Returns (U, V, W) if valid, else (None, None, None).
    """
    # Find a valid rank-7 decomposition of the 2x2x2 tensor by search.
    # The 2x2x2 tensor T[a,b,c] = 1 iff (a,b,c) = (i*2+j, j*2+k, k*2+i)
    valid_222 = set()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                valid_222.add((i*2+j, j*2+k, k*2+i))
    
    # Try known Strassen formulations
    candidate_sets = [
        # Set 1: Standard Strassen
        [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,1,0,0], [0,0,1,0]),
            ([1,0,-1,0], [0,0,1,1], [0,1,0,0]),
            ([0,0,0,1], [1,0,0,-1], [0,0,0,1]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,1,1,0], [1,0,1,0], [0,0,1,0]),
            ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),
        ],
        # Set 2: Alternative
        [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,1,0,0], [0,0,1,0]),
            ([1,0,-1,0], [0,0,1,1], [0,1,0,0]),
            ([0,0,0,1], [-1,0,0,1], [0,0,0,1]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,1,1,0], [1,0,1,0], [0,0,1,0]),
            ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),
        ],
        # Set 3: Another variant
        [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,1,0,0], [0,0,1,0]),
            ([1,0,-1,0], [0,0,1,1], [0,1,0,0]),
            ([0,0,0,1], [1,0,0,-1], [0,0,1,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,1,1,0], [1,0,1,0], [0,0,1,0]),
            ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),
        ],
    ]
    
    terms = None
    for cand in candidate_sets:
        ok = True
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    s = 0
                    for u, v, w in cand:
                        s += u[a] * v[b] * w[c]
                    expected = 1 if (a, b, c) in valid_222 else 0
                    if s != expected:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            terms = cand
            break
    
    if terms is None:
        # Fallback: use a verified rank-8 decomposition (one per nonzero entry)
        # This gives rank 32 total, same as schoolbook, but we return None
        # to signal failure and let construct() use schoolbook.
        return None, None, None
    
    # Build the full 2x2x8 decomposition
    rank = 28
    U = [[0]*rank for _ in range(NM)]
    V = [[0]*rank for _ in range(MP)]
    W = [[0]*rank for _ in range(PN)]
    
    for t in range(4):
        for s in range(7):
            col = t*7 + s
            u, v, w = terms[s]
            for i in range(2):
                for j in range(2):
                    U[i*2+j][col] = u[i*2+j]
            for j in range(2):
                for k in range(2):
                    V[j*8 + (2*t+k)][col] = v[j*2+k]
            for i in range(2):
                for k in range(2):
                    W[(2*t+k)*2 + i][col] = w[i*2+k]
    
    if verify(U, V, W):
        return U, V, W
    return None, None, None

def random_search_rank(rank, time_limit=10.0):
    """Random search for rank-`rank` decomposition with half-integer entries."""
    start = time.time()
    nm, mp, pn = NM, MP, PN
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i*M+j, j*P+k, k*N+i] = 1.0
    
    best_err = float('inf')
    best_sol = None
    
    while time.time() - start < time_limit:
        # Random half-integer initialization
        U = np.random.randint(-4, 5, size=(nm, rank)) / 2.0
        V = np.random.randint(-4, 5, size=(mp, rank)) / 2.0
        W = np.random.randint(-4, 5, size=(pn, rank)) / 2.0
        
        # Compute error
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((T_calc - T)**2)
        
        if err < best_err:
            best_err = err
            best_sol = (U.copy(), V.copy(), W.copy())
            if err < 1e-9:
                # Verify with exact arithmetic
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if verify(U_list, V_list, W_list):
                    return U_list, V_list, W_list
    
    return None

def local_search(U, V, W, time_limit=10.0):
    """Local search: perturb individual entries and accept if error decreases."""
    start = time.time()
    nm, mp, pn = NM, MP, PN
    rank = len(U[0])
    
    T = np.zeros((nm, mp, pn))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i*M+j, j*P+k, k*N+i] = 1.0
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    def compute_error(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_calc - T)**2)
    
    current_err = compute_error(U, V, W)
    
    while time.time() - start < time_limit:
        improved = False
        # Try perturbing random entries
        for _ in range(100):
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                a, r = random.randint(0, nm-1), random.randint(0, rank-1)
                old = U[a, r]
                delta = random.choice([-0.5, -0.25, 0.25, 0.5])
                U[a, r] = old + delta
            elif mat == 'V':
                b, r = random.randint(0, mp-1), random.randint(0, rank-1)
                old = V[b, r]
                delta = random.choice([-0.5, -0.25, 0.25, 0.5])
                V[b, r] = old + delta
            else:
                c, r = random.randint(0, pn-1), random.randint(0, rank-1)
                old = W[c, r]
                delta = random.choice([-0.5, -0.25, 0.25, 0.5])
                W[c, r] = old + delta
            
            new_err = compute_error(U, V, W)
            if new_err < current_err:
                current_err = new_err
                improved = True
            else:
                if mat == 'U':
                    U[a, r] = old
                elif mat == 'V':
                    V[b, r] = old
                else:
                    W[c, r] = old
        
        if not improved:
            break
        
        if current_err < 1e-9:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            if verify(U_list, V_list, W_list):
                return U_list, V_list, W_list
    
    return None

def construct():
    """Entry point: return the best decomposition found."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Start with schoolbook rank-32 (always valid)
    best_U, best_V, best_W = schoolbook()
    best_rank = 32
    
    # Try to find a valid rank-28 Strassen
    try:
        U, V, W = strassen_228()
        if U is not None and verify(U, V, W):
            best_U, best_V, best_W = U, V, W
            best_rank = 28
    except Exception:
        pass
    
    # Try to find lower rank using systematic search
    for target_rank in [24, 25, 26, 27]:
        if time.time() > deadline - 5:
            break
        if target_rank >= best_rank:
            continue
        
        time_left = deadline - time.time() - 5
        if time_left < 3:
            break
        
        # Random search
        t_limit = min(8.0, time_left * 0.5)
        try:
            result = random_search_rank(target_rank, t_limit)
            if result:
                U_new, V_new, W_new = result
                if verify(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = target_rank
                    if target_rank <= 24:
                        break
        except Exception:
            pass
        
        # Local search from random init
        if time.time() > deadline - 5:
            break
        t_limit = min(8.0, deadline - time.time() - 5)
        if t_limit < 3:
            break
        try:
            U_init = np.random.randint(-2, 3, size=(NM, target_rank)) / 2.0
            V_init = np.random.randint(-2, 3, size=(MP, target_rank)) / 2.0
            W_init = np.random.randint(-2, 3, size=(PN, target_rank)) / 2.0
            
            U_list = [[float(x) for x in row] for row in U_init]
            V_list = [[float(x) for x in row] for row in V_init]
            W_list = [[float(x) for x in row] for row in W_init]
            result = local_search(U_list, V_list, W_list, t_limit)
            if result:
                U_new, V_new, W_new = result
                if verify(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = target_rank
                    if target_rank <= 24:
                        break
        except Exception:
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
