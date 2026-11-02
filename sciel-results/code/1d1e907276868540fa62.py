import time

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 120.0
_START = time.time()

def _time_left():
    return _DEADLINE_S - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify_exact(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    Ud = [[int(round(2 * x)) for x in row] for row in U]
    Vd = [[int(round(2 * x)) for x in row] for row in V]
    Wd = [[int(round(2 * x)) for x in row] for row in W]
    for M in (Ud, Vd, Wd):
        for row in M:
            for x in row:
                if x not in (-2, -1, 0, 1, 2):
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(r):
                    s += Ud[ui][c] * Vd[vj][c] * Wd[wk][c]
                if s != 8:
                    return False
    return True

def _rank24_construct():
    """Rank-24 exact 3x3x3 matmul decomposition.

    Structure:
      - Top-left 2x2 Strassen core (7 terms) -> 2x2 contributions to C[0,0],C[0,1],C[1,0],C[1,1]
      - Bottom-right 1x1 core (1 term)       -> C[2,2]
      - 16 cross terms (schoolbook products) -> the A[i,2]B[2,k] part of the 2x2 block
                                                + the 12 third-row/col entries
    Total rank = 7 + 1 + 16 = 24.

    Index conventions (n=m=p=3):
      U row = i*3+j  (A[i,j])
      V row = j*3+k  (B[j,k])
      W row = k*3+i  (C[i,k] output coefficient)
    """
    n = m = p = 3
    rank = 24
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]

    def add_term(u, v, w):
        nonlocal cur_c
        for a, val in u.items():
            U[a][cur_c] += val
        for b, val in v.items():
            V[b][cur_c] += val
        for d, val in w.items():
            W[d][cur_c] += val
        cur_c += 1

    cur_c = 0

    # ---- Top-left 2x2 Strassen core (7 terms) ----
    # A sub-block rows/cols 0,1 : A00=0,A01=1,A10=3,A11=4
    # B sub-block               : B00=0,B01=1,B10=3,B11=4
    # C sub-block (W idx)       : C00=0,C01=3,C10=1,C11=4
    # M1 = (A00+A11)(B00+B11)  -> C00,C01,C10,C11
    add_term({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 3: 1, 1: 1, 4: 1})
    # M2 = (A10+A11)B00        -> C10,C11
    add_term({3: 1, 4: 1}, {0: 1}, {1: 1, 4: 1})
    # M3 = A00(B10-B11)        -> C01, -C11
    add_term({0: 1}, {3: 1, 4: -1}, {3: 1, 4: -1})
    # M4 = A11(B01-B00)        -> C00, C10
    add_term({4: 1}, {1: 1, 0: -1}, {0: 1, 1: 1})
    # M5 = (A00+A01)B11        -> -C00, C01
    add_term({0: 1, 1: 1}, {4: 1}, {0: -1, 3: 1})
    # M6 = (A10-A00)(B00+B01)  -> C11
    add_term({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1})
    # M7 = (A01-A11)(B10+B11)  -> C00
    add_term({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1})

    # ---- Bottom-right 1x1 core (1 term) ----
    # A22=8, B22=8 -> C22 (W idx 8)
    add_term({8: 1}, {8: 1}, {8: 1})

    # ---- 16 cross terms (schoolbook) ----
    # The 2x2 block entries each miss the A[i,2]B[2,k] product:
    #   C00 += A02(5)*B20(6)
    #   C01 += A02(5)*B21(7)
    #   C10 += A12(8)*B20(6)
    #   C11 += A12(8)*B21(7)
    cross = [
        # (A index, B index, C W index)
        (5, 6, 0),  # C00
        (5, 7, 3),  # C01
        (8, 6, 1),  # C10
        (8, 7, 4),  # C11
        # C02 = A00B02 + A01B12 + A02B22
        (0, 2, 6),
        (1, 5, 6),
        (5, 8, 6),
        # C12 = A10B02 + A11B12 + A12B22
        (3, 2, 7),
        (4, 5, 7),
        (8, 8, 7),
        # C20 = A20B00 + A21B10 + A22B20
        (6, 0, 2),
        (7, 3, 2),
        (8, 6, 2),
        # C21 = A20B01 + A21B11 + A22B21
        (6, 1, 5),
        (7, 4, 5),
        (8, 7, 5),
    ]
    for (a, b, widx) in cross:
        add_term({a: 1}, {b: 1}, {widx: 1})

    assert cur_c == rank, "expected rank 24, got %d" % cur_c
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
import numpy as np

def _rank23_search():
    """Attempt to find a rank-23 decomposition via local search from rank-24."""
    n = m = p = 3
    rank = 23
    
    # Start from rank-24 and try to merge two terms
    base = _rank24_construct()
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    # Try to find pairs of terms that can be merged
    # A merge is valid if for all (i,j,k): sum of products works out
    # We'll try all pairs and check if we can eliminate one term
    
    best_rank = 24
    best_cand = base
    
    # Try merging pairs of terms
    for c1 in range(rank + 1):
        for c2 in range(c1 + 1, rank + 1):
            # Check if term c2 is "redundant" given term c1
            # i.e., can we adjust other terms to absorb c2's contribution?
            # Simple check: if the two terms have similar structure
            if np.allclose(U0[:, c1], U0[:, c2]) and \
               np.allclose(V0[:, c1], V0[:, c2]) and \
               np.allclose(W0[:, c1], W0[:, c2]):
                # Identical terms - can merge
                continue
            
            # Try a simple merge: see if we can express c2 as a linear combo
            # This is complex; instead try random perturbations
            pass
    
    # Try random search: start from rank-24, randomly perturb and see if rank-23 works
    np.random.seed(42)
    for trial in range(200):
        if _time_left() < 10.0:
            break
        
        # Start with rank-23 random initialization
        U = np.random.randn(9, rank) * 0.1
        V = np.random.randn(9, rank) * 0.1
        W = np.random.randn(9, rank) * 0.1
        
        # Optimize to match the tensor
        # The tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # We want T[i,j,k] = 1 for all (i,j,k)
        
        # Use simple gradient-free approach: try to match each entry
        # This is a hard problem; skip if too slow
        break
    
    return best_cand, best_rank

def _try_rank23_direct():
    """Try to construct a rank-23 decomposition directly."""
    n = m = p = 3
    rank = 23
    
    # Use the known structure: 2x2 Strassen (7) + 1x1 (1) + cross terms
    # Standard cross terms are 16, but maybe we can do 15?
    # The 16 cross terms are:
    # C00: A02*B20
    # C01: A02*B21
    # C10: A12*B20
    # C11: A12*B21
    # C02: A00*B02, A01*B12, A02*B22
    # C12: A10*B02, A11*B12, A12*B22
    # C20: A20*B00, A21*B10, A22*B20
    # C21: A20*B01, A21*B11, A22*B21
    
    # Can we combine any of these? For example, A02*B20 and A02*B21 share A02.
    # But they produce different outputs (C00 vs C01), so we can't simply merge.
    
    # What if we use a different 2x2 block decomposition?
    # Or what if we use a non-standard Strassen variant?
    
    # Let me try: what if we use 8 terms for the 2x2 part (standard) but save elsewhere?
    # No, Strassen is optimal for 2x2.
    
    # Alternative: use a different partitioning
    # Partition as 1x3 and 2x3 blocks?
    
    # Let me try a completely different approach:
    # Use the fact that 3x3 matmul might have rank 23
    # according to some literature
    
    # For now, return rank-24 as the best we can verify
    return _rank24_construct(), 24

def construct():
    best = _schoolbook()
    
    # Try rank-24 first (known to work)
    if _time_left() > 5.0:
        cand = _rank24_construct()
        if cand is not None and _verify_exact(cand["U"], cand["V"], cand["W"]):
            best = cand
    
    # Try to find rank-23
    if _time_left() > 15.0:
        cand23, r23 = _try_rank23_direct()
        if cand23 is not None and r23 < len(best["U"][0]):
            if _verify_exact(cand23["U"], cand23["V"], cand23["W"]):
                best = cand23
    
    return best
# EVOLVE-BLOCK-END
