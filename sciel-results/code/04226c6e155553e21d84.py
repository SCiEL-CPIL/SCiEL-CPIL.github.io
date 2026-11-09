"""Optimized 3x3x3 matrix multiplication tensor decomposition.

Combines the current program's structured search with an SAT-inspired
Boolean-satisfaction approach over GF(2) and small finite fields to find
rank-minimal exact decompositions. The key insight: if we can find a
decomposition over GF(2) with small integer entries, it also works over
the reals (since the tensor is all-ones, which is 1 in any field).

We search for rank r such that there exist U, V, W with entries in {0,1}
(or {-1,0,1}) satisfying:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  (mod 2 or over Z)

For the all-ones 3x3x3 tensor, this is a combinatorial problem.
"""
import time
import itertools
import random
from collections import defaultdict

DIMENSIONS = [3, 3, 3]

_DEADLINE = 150.0
_STARTED = time.time()


def _verify(U, V, W, n=3, m=3, p=3):
    """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _make_schoolbook(n=3, m=3, p=3):
    """Return the rank-27 schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _try_gf2_search(r, n=3, m=3, p=3, max_trials=5000, seed=42):
    """
    SAT-inspired search over GF(2) for binary entries {0,1}.
    We need: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] ≡ 1 (mod 2).
    
    This is equivalent to: the XOR of the products equals 1 for each (i,j,k).
    """
    rng = random.Random(seed)
    nm = n * m  # 9
    mp = m * p  # 9
    pn = p * n  # 9
    
    # Precompute index mappings
    U_idx = [(i * m + j) for i in range(n) for j in range(m)]
    V_idx = [(j * p + k) for j in range(m) for k in range(p)]
    W_idx = [(k * n + i) for k in range(p) for i in range(n)]
    
    for trial in range(max_trials):
        # Random binary matrices
        U = [[rng.randint(0, 1) for _ in range(r)] for _ in range(nm)]
        V = [[rng.randint(0, 1) for _ in range(r)] for _ in range(mp)]
        W = [[rng.randint(0, 1) for _ in range(r)] for _ in range(pn)]
        
        # Check all 27 constraints mod 2
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for rr in range(r):
                        s = (s + U[ui][rr] * V[vj][rr] * W[wk][rr]) % 2
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            # Convert to float for the interface
            U_f = [[float(x) for x in row] for row in U]
            V_f = [[float(x) for x in row] for row in V]
            W_f = [[float(x) for x in row] for row in W]
            # Verify over reals
            if _verify(U_f, V_f, W_f, n, m, p):
                return {"U": U_f, "V": V_f, "W": W_f}
    
    return None


def _try_ternary_search(r, n=3, m=3, p=3, max_trials=3000, seed=42):
    """
    Search with entries in {-1, 0, 1} over the integers.
    Need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 exactly.
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    for trial in range(max_trials):
        U = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
        
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for rr in range(r):
                        s += U[ui][rr] * V[vj][rr] * W[wk][rr]
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            U_f = [[float(x) for x in row] for row in U]
            V_f = [[float(x) for x in row] for row in V]
            W_f = [[float(x) for x in row] for row in W]
            if _verify(U_f, V_f, W_f, n, m, p):
                return {"U": U_f, "V": V_f, "W": W_f}
    
    return None


def _try_structured_strassen_boundary(r, n=3, m=3, p=3, max_trials=2000, seed=42):
    """
    Structured search: fix the 2x2x2 Strassen core (rank 7) and search
    for the boundary terms (involving index 2) with the remaining rank.
    
    For 3x3x3, the boundary consists of entries where at least one of i,j,k is 2.
    There are 27 - 8 = 19 such entries.
    
    We use rank 7 for the core and r-7 for the boundary.
    """
    if r < 7:
        return None
    
    rng = random.Random(seed)
    nm = n * m  # 9
    mp = m * p  # 9
    pn = p * n  # 9
    
    # Strassen 2x2x2 core: entries (i,j,k) with i,j,k in {0,1}
    # 7 terms
    # M0 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M1 = (A10+A11)B00       -> contributes to C00, C11
    # M2 = A00(B01-B11)       -> contributes to C00, C11
    # M3 = A11(B10-B00)       -> contributes to C00, C11
    # M4 = (A00+A01)B11       -> contributes to C01, C11
    # M5 = (A10-A00)(B00+B01) -> contributes to C01, C11
    # M6 = (A01-A11)(B10+B11) -> contributes to C00, C11
    
    # For the tensor T[i,j,k]=1, we need the bilinear form to give 1 for each (i,j,k).
    # The Strassen algorithm computes C[i,k] = sum_j A[i,j]B[j,k] using 7 multiplications.
    # In the tensor decomposition T[i,j,k] = sum_r U[i,j,r]V[j,k,r]W[k,i,r],
    # with W=1 for the core, we need sum_r U[i,j,r]V[j,k,r] = 1 for (i,j,k) in {0,1}^3.
    
    # Let's set up the Strassen core with W=1 for the 8 core entries.
    # Actually, let's just do a full search but bias the core entries.
    
    # For each trial, initialize with Strassen-like structure for core, random for boundary
    for trial in range(max_trials):
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        W = [[1] * r for _ in range(pn)]  # Start with W=1
        
        # Core: i,j,k in {0,1}
        # We'll use 7 columns for the core
        # Term 0: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1
        U[0*3+0][0] = 1
        U[1*3+1][0] = 1
        V[0*3+0][0] = 1
        V[1*3+1][0] = 1
        W[0*3+0][0] = 1
        W[1*3+1][0] = 1
        
        # Term 1: U[1,0]=1, U[1,1]=1; V[0,0]=1
        U[1*3+0][1] = 1
        U[1*3+1][1] = 1
        V[0*3+0][1] = 1
        W[0*3+0][1] = -1
        W[1*3+1][1] = 1
        
        # Term 2: U[0,0]=1; V[0,1]=1, V[1,1]=-1
        U[0*3+0][2] = 1
        V[0*3+1][2] = 1
        V[1*3+1][2] = -1
        W[0*3+0][2] = 1
        W[1*3+1][2] = -1
        
        # Term 3: U[1,1]=1; V[1,0]=1, V[0,0]=-1
        U[1*3+1][3] = 1
        V[1*3+0][3] = 1
        V[0*3+0][3] = -1
        W[0*3+0][3] = 1
        W[1*3+1][3] = -1
        
        # Term 4: U[0,0]=1, U[0,1]=1; V[1,1]=1
        U[0*3+0][4] = 1
        U[0*3+1][4] = 1
        V[1*3+1][4] = 1
        W[0*3+1][4] = 1
        W[1*3+1][4] = 1
        
        # Term 5: U[1,0]=1, U[0,0]=-1; V[0,0]=1, V[0,1]=1
        U[1*3+0][5] = 1
        U[0*3+0][5] = -1
        V[0*3+0][5] = 1
        V[0*3+1][5] = 1
        W[0*3+1][5] = -1
        W[1*3+1][5] = 1
        
        # Term 6: U[0,1]=1, U[1,1]=-1; V[1,0]=1, V[1,1]=1
        U[0*3+1][6] = 1
        U[1*3+1][6] = -1
        V[1*3+0][6] = 1
        V[1*3+1][6] = 1
        W[0*3+0][6] = -1
        W[1*3+1][6] = 1
        
        # Now fill in boundary terms (columns 7 to r-1) randomly
        # Boundary entries: those with i=2 or j=2 or k=2
        # For each such (i,j,k), we need the sum over all r to equal 1.
        # The core terms already contribute to some of these if they involve index 2,
        # but they don't (core is only {0,1}). So boundary entries get 0 from core.
        # We need sum_{r=7}^{r-1} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for boundary (i,j,k).
        
        # Randomly assign boundary entries
        for rr in range(7, r):
            for ui in range(nm):
                U[ui][rr] = rng.choice([-1, 0, 1])
            for vj in range(mp):
                V[vj][rr] = rng.choice([-1, 0, 1])
            for wk in range(pn):
                W[wk][rr] = rng.choice([-1, 0, 1])
        
        # Verify all 27 constraints
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for rr in range(r):
                        s += U[ui][rr] * V[vj][rr] * W[wk][rr]
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            U_f = [[float(x) for x in row] for row in U]
            V_f = [[float(x) for x in row] for row in V]
            W_f = [[float(x) for x in row] for row in W]
            if _verify(U_f, V_f, W_f, n, m, p):
                return {"U": U_f, "V": V_f, "W": W_f}
    
    return None


def _try_w_ones_search(r, n=3, m=3, p=3, max_trials=10000, seed=42):
    """
    Search with W = all ones, so we need:
    sum_r U[i,j,r] * V[j,k,r] = 1 for all i,j,k.
    This is the standard bilinear MM rank problem.
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    for trial in range(max_trials):
        U = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
        W = [[1] * r for _ in range(pn)]
        
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    for rr in range(r):
                        s += U[ui][rr] * V[vj][rr]
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            U_f = [[float(x) for x in row] for row in U]
            V_f = [[float(x) for x in row] for row in V]
            W_f = [[float(x) for x in row] for row in W]
            if _verify(U_f, V_f, W_f, n, m, p):
                return {"U": U_f, "V": V_f, "W": W_f}
    
    return None


def _try_half_integer_search(r, n=3, m=3, p=3, max_trials=5000, seed=42):
    """
    Search with half-integer entries (multiples of 0.5).
    Scale by 2 to work with integers.
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    for trial in range(max_trials):
        U2 = [[rng.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(nm)]
        V2 = [[rng.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(mp)]
        W2 = [[rng.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(pn)]
        
        # Check: sum_r (U2/2)*(V2/2)*(W2/2) = 1 => sum_r U2*V2*W2 = 8
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for rr in range(r):
                        s += U2[ui][rr] * V2[vj][rr] * W2[wk][rr]
                    if s != 8:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            U_f = [[x / 2.0 for x in row] for row in U2]
            V_f = [[x / 2.0 for x in row] for row in V2]
            W_f = [[x / 2.0 for x in row] for row in W2]
            if _verify(U_f, V_f, W_f, n, m, p):
                return {"U": U_f, "V": V_f, "W": W_f}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = _STARTED + _DEADLINE - 5.0  # 5s safety margin
    
    # Try increasing ranks from a lower bound
    # Theoretical lower bound for 3x3x3 MM is 23 (Lader), but for 3-factor it may differ.
    # Let's try from rank 10 upwards, but focus on 20-26 where solutions likely exist.
    
    best_result = None
    
    for r in range(10, 28):
        if time.time() > deadline:
            break
        
        # Try W=ones bilinear search (most likely to succeed)
        result = _try_w_ones_search(r, n, m, p, max_trials=2000, seed=r * 137)
        if result is not None:
            return result
        
        # Try full 3-factor ternary search
        result = _try_ternary_search(r, n, m, p, max_trials=1000, seed=r * 251)
        if result is not None:
            return result
        
        # Try GF(2) search
        result = _try_gf2_search(r, n, m, p, max_trials=1000, seed=r * 389)
        if result is not None:
            return result
        
        # Try structured Strassen + boundary
        result = _try_structured_strassen_boundary(r, n, m, p, max_trials=500, seed=r * 503)
        if result is not None:
            return result
        
        # Try half-integer search
        result = _try_half_integer_search(r, n, m, p, max_trials=500, seed=r * 617)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _make_schoolbook(n, m, p)


# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
