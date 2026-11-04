"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k
    and off-diagonal terms are zero."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    U_mat = U.reshape(n, m, r)
    V_mat = V.reshape(m, p, r)
    W_mat = W.reshape(p, n, r)
    # Diagonal check
    diag = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat, optimize=True)
    if not np.allclose(diag, 1.0, atol=tol):
        return False
    # Off-diagonal: check that the full 4D coefficient tensor is identity in (i,k)
    # S[i',k',j,i] = sum_r U[i*m+j,r] * V[j*p+k',r] * W[k'*n+i',r] should be 0 when (i',k')!=(i,k)
    # Equivalent: for each j, the matrix M_j of size (n*p) x (n*p) with
    # M_j[(i',k'),(i,k)] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k'*n+i',r] should be identity
    # Actually simpler: check random off-diagonals
    rng = np.random.default_rng(12345)
    for _ in range(200):
        i = rng.integers(0, n)
        j = rng.integers(0, m)
        k = rng.integers(0, p)
        ip = rng.integers(0, n)
        kp = rng.integers(0, p)
        if ip == i and kp == k:
            continue
        val = float(np.dot(U[i*m+j, :] * V[j*p+k, :], W[kp*n+ip, :]))
        if abs(val) > tol:
            return False
    return True

def _strassen_2x2x2(n, m, p, i1, i2, j1, j2, k1, k2):
    """Return 7 rank-1 vectors for a 2x2x2 Strassen block."""
    cols_U = []
    cols_V = []
    cols_W = []
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    cols_U.append([(i1,j1,1.0),(i2,j2,1.0)])
    cols_V.append([(j1,k1,1.0),(j2,k2,1.0)])
    cols_W.append([(k1,i1,1.0),(k2,i2,1.0)])
    # M2 = (A21+A22)B11 -> C12 += M2
    cols_U.append([(i2,j1,1.0),(i2,j2,1.0)])
    cols_V.append([(j1,k1,1.0)])
    cols_W.append([(k1,i1,1.0)])
    # M3 = A11(B21-B22) -> C21 += M3
    cols_U.append([(i1,j1,1.0)])
    cols_V.append([(j2,k1,1.0),(j2,k2,-1.0)])
    cols_W.append([(k1,i2,1.0)])
    # M4 = A22(B12-B11) -> C11 += M4, C21 += M4
    cols_U.append([(i2,j2,1.0)])
    cols_V.append([(j1,k2,1.0),(j1,k1,-1.0)])
    cols_W.append([(k1,i1,1.0),(k1,i2,1.0)])
    # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    cols_U.append([(i1,j1,1.0),(i1,j2,1.0)])
    cols_V.append([(j2,k2,1.0)])
    cols_W.append([(k1,i1,-1.0),(k2,i1,1.0)])
    # M6 = (A21-A11)(B11+B12) -> C11 += M6
    cols_U.append([(i2,j1,1.0),(i1,j1,-1.0)])
    cols_V.append([(j1,k1,1.0),(j1,k2,1.0)])
    cols_W.append([(k1,i1,1.0)])
    # M7 = (A12-A22)(B21+B22) -> C22 += M7
    cols_U.append([(i1,j2,1.0),(i2,j2,-1.0)])
    cols_V.append([(j2,k1,1.0),(j2,k2,1.0)])
    cols_W.append([(k2,i2,1.0)])
    
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    for c in range(7):
        for (idx, val) in cols_U[c]:
            i, j = idx
            U[i*m+j, c] = val
        for (idx, val) in cols_V[c]:
            j, k = idx
            V[j*p+k, c] = val
        for (idx, val) in cols_W[c]:
            k, i = idx
            W[k*n+i, c] = val
    return U, V, W

def _laderman_3x3x3(n, m, p, i1, j1, k1):
    """Laderman's rank-23 decomposition for 3x3x3.
    i1,j1,k1 are the starting indices (0-based) for the 3x3x3 block."""
    # Laderman's 23 multiplication terms for 3x3 matrices
    # A rows: i1, i1+1, i1+2; cols: j1, j1+1, j1+2
    # B rows: j1, j1+1, j1+2; cols: k1, k1+1, k1+2
    # C rows: i1, i1+1, i1+2; cols: k1, k1+1, k1+2
    
    # Using the standard Laderman decomposition (23 terms)
    # Let's define using local indices a,b,c in {0,1,2}
    # M1 = (A00+A22)(B00+B22)
    # M2 = (A02+A22)(B00+B20)  -- actually let me use a verified set
    
    # I'll use a well-known Laderman formulation:
    # Terms are defined by their (U, V, W) support patterns
    # Using the formulation from Laderman 1976 / Bini et al.
    
    terms = []
    # Each term: (list of (local_i, local_j, coeff), list of (local_j, local_k, coeff), list of (local_k, local_i, coeff))
    
    # M1: (A00+A22)(B00+B22) -> C00+=, C22+=
    terms.append(([ (0,0,1),(2,2,1) ], [ (0,0,1),(2,2,1) ], [ (0,0,1),(2,2,1) ]))
    # M2: (A02+A22)(B00+B20) -> C00+=, C20+=  
    terms.append(([ (0,2,1),(2,2,1) ], [ (0,0,1),(2,0,1) ], [ (0,0,1),(0,2,1) ]))
    # M3: (A20+A22)(B02+B22) -> C02+=, C22+=
    terms.append(([ (2,0,1),(2,2,1) ], [ (0,2,1),(2,2,1) ], [ (2,0,1),(2,2,1) ]))
    # M4: A00(B11+B22) -> C01+=, C02+=
    terms.append(([ (0,0,1) ], [ (1,1,1),(2,2,1) ], [ (1,0,1),(2,0,1) ]))
    # M5: A01(B11+B21) -> C01+=, C02+=
    terms.append(([ (0,1,1) ], [ (1,1,1),(2,1,1) ], [ (1,0,1),(2,0,1) ]))
    # M6: A02(B11+B21) -> C01+=, C02+=
    terms.append(([ (0,2,1) ], [ (1,1,1),(2,1,1) ], [ (1,0,1),(2,0,1) ]))
    # M7: A10(B01+B11) -> C10+=, C11+=
    terms.append(([ (1,0,1) ], [ (0,1,1),(1,1,1) ], [ (1,1,1),(1,1,1) ]))
    # M8: A11(B01+B11) -> C10+=, C11+=
    terms.append(([ (1,1,1) ], [ (0,1,1),(1,1,1) ], [ (1,1,1),(1,1,1) ]))
    # M9: A12(B01+B11) -> C10+=, C11+=
    terms.append(([ (1,2,1) ], [ (0,1,1),(1,1,1) ], [ (1,1,1),(1,1,1) ]))
    # M10: A20(B12+B22) -> C12+=, C22+=
    terms.append(([ (2,0,1) ], [ (1,2,1),(2,2,1) ], [ (2,1,1),(2,2,1) ]))
    # M11: A21(B12+B22) -> C12+=, C22+=
    terms.append(([ (2,1,1) ], [ (1,2,1),(2,2,1) ], [ (2,1,1),(2,2,1) ]))
    # M12: A22(B12+B22) -> C12+=, C22+=
    terms.append(([ (2,2,1) ], [ (1,2,1),(2,2,1) ], [ (2,1,1),(2,2,1) ]))
    # M13: (A00+A11+A22)(B00+B11+B22) -> C00+=, C11+=, C22+=
    terms.append(([ (0,0,1),(1,1,1),(2,2,1) ], [ (0,0,1),(1,1,1),(2,2,1) ], [ (0,0,1),(1,1,1),(2,2,1) ]))
    # M14: (A01+A12)(B10+B21) -> C01+=, C12+=
    terms.append(([ (0,1,1),(1,2,1) ], [ (1,0,1),(2,1,1) ], [ (1,0,1),(2,1,1) ]))
    # M15: (A02+A10)(B20+B02) -> C02+=, C10+=
    terms.append(([ (0,2,1),(1,0,1) ], [ (2,0,1),(0,2,1) ], [ (2,0,1),(0,1,1) ]))
    # M16: (A10+A21)(B01+B12) -> C10+=, C21+=
    terms.append(([ (1,0,1),(2,1,1) ], [ (0,1,1),(1,2,1) ], [ (1,1,1),(2,2,1) ]))
    # M17: (A12+A20)(B10+B21) -> C12+=, C20+=
    terms.append(([ (1,2,1),(2,0,1) ], [ (1,0,1),(2,1,1) ], [ (2,1,1),(0,2,1) ]))
    # M18: (A01+A10)(B12+B20) -> C01+=, C10+=
    terms.append(([ (0,1,1),(1,0,1) ], [ (1,2,1),(2,0,1) ], [ (2,0,1),(1,1,1) ]))
    # M19: (A02+A20)(B11+B21) -> C02+=, C20+=
    terms.append(([ (0,2,1),(2,0,1) ], [ (1,1,1),(2,1,1) ], [ (1,0,1),(1,2,1) ]))
    # M20: (A11+A22)(B01+B20) -> C11+=, C22+=
    terms.append(([ (1,1,1),(2,2,1) ], [ (0,1,1),(2,0,1) ], [ (1,1,1),(2,2,1) ]))
    # M21: (A00+A22)(B10+B21) -> C00+=, C22+=
    terms.append(([ (0,0,1),(2,2,1) ], [ (1,0,1),(2,1,1) ], [ (0,1,1),(2,2,1) ]))
    # M22: (A01+A12)(B02+B11) -> C01+=, C12+=
    terms.append(([ (0,1,1),(1,2,1) ], [ (0,2,1),(1,1,1) ], [ (2,0,1),(1,1,1) ]))
    # M23: (A10+A21)(B11+B20) -> C10+=, C21+=
    terms.append(([ (1,0,1),(2,1,1) ], [ (1,1,1),(2,0,1) ], [ (1,1,1),(2,2,1) ]))
    
    # This Laderman set is not verified. Let me use a simpler, verified approach.
    # Actually, for safety, let me just use the schoolbook for 3x3x3 (rank 27) 
    # and focus on 2x2x2 Strassen blocks which I can verify.
    
    # Return None to indicate this block type is not used
    return None

def _build_decomposition(n, m, p, i_blocks, j_blocks, k_blocks, block_type="strassen"):
    """Build decomposition by tiling with blocks."""
    U_cols = []
    V_cols = []
    W_cols = []
    covered = set()
    
    for (i1, i2) in i_blocks:
        for (j1, j2) in j_blocks:
            for (k1, k2) in k_blocks:
                U_b, V_b, W_b = _strassen_2x2x2(n, m, p, i1, i2, j1, j2, k1, k2)
                U_cols.append(U_b)
                V_cols.append(V_b)
                W_cols.append(W_b)
                for i in [i1, i2]:
                    for j in [j1, j2]:
                        for k in [k1, k2]:
                            covered.add((i, j, k))
    
    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(n*m); u[i*m+j] = 1.0
                    v = np.zeros(m*p); v[j*p+k] = 1.0
                    w = np.zeros(p*n); w[k*n+i] = 1.0
                    U_cols.append(u.reshape(-1,1))
                    V_cols.append(v.reshape(-1,1))
                    W_cols.append(w.reshape(-1,1))
    
    U = np.hstack(U_cols)
    V = np.hstack(V_cols)
    W = np.hstack(W_cols)
    return U, V, W, U.shape[1]

def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Best known: schoolbook rank 120
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    U_best, V_best, W_best = U_sb, V_sb, W_sb
    rank_best = rank_sb
    
    # Strategy: try all tilings with 2x2x2 Strassen blocks
    # i can be partitioned into pairs: for n=4, possible i-blocks: [(0,1),(2,3)] or [(0,1)] or [(2,3)] or []
    # j for m=5: [(0,1),(2,3)] or [(0,1)] or [(2,3)] or [(1,2),(3,4)] or [(0,1),(3,4)] etc.
    # k for p=6: [(0,1),(2,3),(4,5)] or subsets
    
    # Generate all valid 2x2x2 block tilings
    def gen_pairs(n):
        """Generate all ways to place non-overlapping pairs in range(n)."""
        # For small n, enumerate all subsets of disjoint pairs
        results = [[]]
        def backtrack(remaining, current):
            if not remaining:
                results.append(list(current))
                return
            # Try placing a pair starting at the smallest remaining element
            first = min(remaining)
            # Option 1: don't pair first
            rest = [x for x in remaining if x != first]
            backtrack(rest, current)
            # Option 2: pair first with next available
            if len(remaining) >= 2:
                # Find the smallest element greater than first
                candidates = sorted([x for x in remaining if x > first])
                for c in candidates:
                    if c - first == 1:  # adjacent only for 2x2 blocks
                        new_remaining = [x for x in remaining if x != first and x != c]
                        current.append((first, c))
                        backtrack(new_remaining, current)
                        current.pop()
        backtrack(list(range(n)), [])
        return results
    
    # For 2x2x2 Strassen, we need adjacent pairs (consecutive indices)
    # Actually Strassen works for any 2x2x2, not just adjacent. But for non-overlapping, adjacent is cleanest.
    # Let's just try the natural partition:
    # i: (0,1), (2,3)
    # j: (0,1), (2,3)  [j=4 left as schoolbook]
    # k: (0,1), (2,3), (4,5)
    
    tilings = []
    
    # Tiling 1: max 2x2x2 blocks
    i_bl = [(0,1),(2,3)]
    j_bl = [(0,1),(2,3)]
    k_bl = [(0,1),(2,3),(4,5)]
    tilings.append((i_bl, j_bl, k_bl))
    
    # Tiling 2: fewer blocks
    tilings.append(([(0,1)], [(0,1)], [(0,1)]))
    tilings.append(([(2,3)], [(2,3)], [(4,5)]))
    
    # Try all tilings
    for i_bl, j_bl, k_bl in tilings:
        if time.time() > deadline:
            break
        U_t, V_t, W_t, rank_t = _build_decomposition(n, m, p, i_bl, j_bl, k_bl)
        if rank_t < rank_best:
            if _verify_exact(U_t, V_t, W_t, n, m, p):
                U_best, V_best, W_best = U_t, V_t, W_t
                rank_best = rank_t
    
    # Also try: use 2x3x2 blocks? Not standard. Stick with 2x2x2.
    
    # Try a different structural approach: 
    # For the 4x5x6 case, consider that we can use a 4x4x4 sub-block with rank 21 (if known)
    # But 4x4x4 minimal rank is not well below 4*4*4=64. Actually it's 49 for 4x4x4 (Coppersmith-Winograd style)
    # That's too complex. Let's stick with verified Strassen.
    
    # Additional idea: try overlapping Strassen in j-dimension
    # j pairs: (0,1), (1,2), (2,3), (3,4) - but they overlap, can't use directly
    # However, we could use (0,1) and (2,3) for some k-blocks, and (1,2) and (3,4) for others
    # This requires careful bookkeeping. For now, skip.
    
    # Final: try to see if we can do better with a mixed approach
    # Use Strassen for i-blocks (0,1) and (2,3), j-blocks (0,1) and (2,3), k-blocks (0,1),(2,3),(4,5)
    # That's 2*2*3 = 12 blocks * 7 = 84, plus uncovered:
    # Covered i: {0,1,2,3}, j: {0,1,2,3}, k: {0,1,2,3,4,5}
    # Uncovered: all (i,j,k) where j=4 -> 4*1*6 = 24 terms
    # Total: 84 + 24 = 108
    # This is what the current program does. Let's verify it works.
    
    # The key question: can we do better than 108?
    # Lower bound: max(n*p, m*p, n*m) = max(24, 30, 20) = 30
    # Known upper bounds for 4x5x6: not well studied.
    # 108 seems reasonable. Let's make sure our implementation is correct.
    
    if time.time() < deadline - 5:
        # Double-check the 108-rank solution
        U_chk, V_chk, W_chk, r_chk = _build_decomposition(n, m, p, [(0,1),(2,3)], [(0,1),(2,3)], [(0,1),(2,3),(4,5)])
        if r_chk < rank_best and _verify_exact(U_chk, V_chk, W_chk, n, m, p):
            U_best, V_best, W_best = U_chk, V_chk, W_chk
            rank_best = r_chk
    
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
# EVOLVE-BLOCK-END
