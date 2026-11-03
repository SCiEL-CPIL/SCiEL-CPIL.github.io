"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from collections import defaultdict

_start_time = time.time()
TIME_BUDGET = 120.0

def _time_left():
    return max(0.0, TIME_BUDGET - (time.time() - _start_time))

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _generate_candidates(n, m, p):
    """
    Generate a dense dictionary of candidate rank-1 terms.
    Each candidate is a tuple (U_vec, V_vec, W_vec) where each vec is a sparse dict.
    We use algebraic patterns: sums/differences of adjacent indices.
    """
    candidates = []
    
    # For each possible "pattern" of which indices are included with what sign
    # In a rank-1 term, U[i*m+j] can be 0, 1, or -1
    # Same for V[j*p+k] and W[k*n+i]
    
    # Strategy: generate terms where each factor is a sum of a small subset
    # of basis vectors with coefficients in {-1, 0, 1}
    
    # For efficiency, we'll generate terms by specifying for each factor
    # a list of (index, coeff) pairs
    
    # U factor: sum over some (i,j) pairs
    # V factor: sum over some (j,k) pairs  
    # W factor: sum over some (k,i) pairs
    
    # The product U*V*W contributes to tensor entry (i,j,k) as:
    # sum over all (i',j'),(j',k'),(k'',i'') in the three sums
    #   where j'=j, k''=i (for the contraction to be valid)
    #   coeff_U[i',j'] * coeff_V[j',k'] * coeff_W[k'',i'']
    #   if i'=i and k'=k
    
    # Actually, the tensor equation is:
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # So a single rank-1 term contributes to (i,j,k) the value:
    # U[i*m+j] * V[j*p+k] * W[k*n+i]
    
    # We need to find a set of such terms whose sum equals 1 at every (i,j,k).
    
    # Let's generate candidate terms where each factor has at most 2-3 nonzero entries
    # with values in {-1, 1}
    
    nU = n * m  # 9
    nV = m * p  # 24
    nW = p * n  # 24
    
    # Generate all subsets of size 1, 2, 3 for each factor
    # This could be large, so let's be selective
    
    # For U: subsets of {0,...,8} of size 1-3
    # For V: subsets of {0,...,23} of size 1-3
    # For W: subsets of {0,...,23} of size 1-3
    
    # But we need to be smart. Let's use the structure:
    # U[i*m+j] corresponds to row i, col j of A
    # V[j*p+k] corresponds to row j, col k of B
    # W[k*n+i] corresponds to row k, col i of C
    
    # A natural pattern: U picks a set of (i,j) pairs, V picks a set of (j,k) pairs,
    # W picks a set of (k,i) pairs. The product is nonzero at (i',j',k') iff
    # there exists j'' such that (i',j'') in U, (j'',k') in V, and (k',i') in W.
    
    # Let's generate candidates more systematically.
    
    # Pattern 1: Single terms (schoolbook)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {i*m+j: 1}
                v = {j*p+k: 1}
                w = {k*n+i: 1}
                candidates.append((u, v, w))
    
    # Pattern 2: Pairs in U (two (i,j) pairs), single in V and W
    # U has (i1,j1) and (i2,j2), V has (j,k), W has (k,i)
    # Contribution to (i',j',k'): U[i',j']*V[j',k']*W[k',i']
    # Nonzero if (i',j') is one of the two U entries, (j',k') is the V entry, (k',i') is the W entry
    
    for i1 in range(n):
        for j1 in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    if (i1,j1) == (i2,j2):
                        continue
                    for j in range(m):
                        for k in range(p):
                            for i in range(n):
                                u = {i1*m+j1: 1, i2*m+j2: 1}
                                v = {j*p+k: 1}
                                w = {k*n+i: 1}
                                candidates.append((u, v, w))
    
    # Pattern 3: Pairs in V
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    for k2 in range(p):
                        if (j,k) == (j2,k2):
                            continue
                        u = {i*m+j: 1}
                        v = {j*p+k: 1, j2*p+k2: 1}
                        w = {k*n+i: 1}
                        candidates.append((u, v, w))
    
    # Pattern 4: Pairs in W
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    for i2 in range(n):
                        if (k,i) == (k2,i2):
                            continue
                        u = {i*m+j: 1}
                        v = {j*p+k: 1}
                        w = {k*n+i: 1, k2*n+i2: 1}
                        candidates.append((u, v, w))
    
    # Pattern 5: Strassen-like with differences
    # U has (i1,j1) with +1 and (i2,j2) with -1
    # This creates cancellations
    
    for i1 in range(n):
        for j1 in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    if (i1,j1) == (i2,j2):
                        continue
                    for j in range(m):
                        for k in range(p):
                            for i in range(n):
                                u = {i1*m+j1: 1, i2*m+j2: -1}
                                v = {j*p+k: 1}
                                w = {k*n+i: 1}
                                candidates.append((u, v, w))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    for k2 in range(p):
                        if (j,k) == (j2,k2):
                            continue
                        u = {i*m+j: 1}
                        v = {j*p+k: 1, j2*p+k2: -1}
                        w = {k*n+i: 1}
                        candidates.append((u, v, w))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    for i2 in range(n):
                        if (k,i) == (k2,i2):
                            continue
                        u = {i*m+j: 1}
                        v = {j*p+k: 1}
                        w = {k*n+i: 1, k2*n+i2: -1}
                        candidates.append((u, v, w))
    
    # Pattern 6: Triple combinations (Strassen-like)
    # U has 3 terms, V has 2, W has 1
    # This is getting large, let's limit
    
    # For now, return what we have
    return candidates

def _greedy_set_cover(candidates, n, m, p, max_rank=None):
    """
    Greedy set cover: repeatedly pick the candidate that covers the most
    uncovered tensor entries (with correct sign).
    """
    if max_rank is None:
        max_rank = n * m * p
    
    # Initialize: we need each (i,j,k) to have sum = 1
    # Track current sum at each entry
    current = np.zeros((n, m, p))
    selected = []
    remaining = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Precompute contribution of each candidate
    # contrib[c][i,j,k] = value contributed by candidate c at entry (i,j,k)
    
    # For memory efficiency, compute on the fly or use sparse representation
    
    for iteration in range(max_rank):
        if not remaining:
            break
        
        if _time_left() < 2.0:
            break
        
        # For each candidate, compute how many remaining entries it helps
        best_idx = -1
        best_score = -1
        best_contrib = None
        
        for c_idx, (u, v, w) in enumerate(candidates):
            # Compute contribution at each remaining entry
            score = 0
            contrib = {}
            for (i, j, k) in remaining:
                val = 0.0
                # U[i*m+j] * V[j*p+k] * W[k*n+i]
                uv = u.get(i*m+j, 0)
                vv = v.get(j*p+k, 0)
                ww = w.get(k*n+i, 0)
                val = uv * vv * ww
                if val != 0:
                    # Check if this helps: we need current[i,j,k] + val to be closer to 1
                    # i.e., |current[i,j,k] + val - 1| < |current[i,j,k] - 1|
                    cur = current[i, j, k]
                    if abs(cur + val - 1) < abs(cur - 1) - 1e-10:
                        score += 1
                        contrib[(i,j,k)] = val
                    elif abs(cur + val - 1) == abs(cur - 1) - 1e-10 and val != 0:
                        # Neutral, but might help later
                        score += 0.1
                        contrib[(i,j,k)] = val
            
            if score > best_score:
                best_score = score
                best_idx = c_idx
                best_contrib = contrib
        
        if best_idx < 0 or best_score <= 0:
            break
        
        # Apply the best candidate
        u, v, w = candidates[best_idx]
        selected.append(best_idx)
        
        new_remaining = []
        for (i, j, k) in remaining:
            val = best_contrib.get((i,j,k), 0)
            current[i, j, k] += val
            if abs(current[i, j, k] - 1) > 1e-8:
                new_remaining.append((i, j, k))
            else:
                current[i, j, k] = 1.0
        
        remaining = new_remaining
    
    if remaining:
        return None  # Couldn't cover all entries
    
    # Build the result
    r = len(selected)
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c_idx, cand_idx in enumerate(selected):
        u, v, w = candidates[cand_idx]
        for idx, val in u.items():
            U[idx][c_idx] += val
        for idx, val in v.items():
            V[idx][c_idx] += val
        for idx, val in w.items():
            W[idx][c_idx] += val
    
    return {"U": U, "V": V, "W": W}

def _strassen_3x3x8():
    """
    Construct a Strassen-based decomposition for 3x3 * 3x8.
    
    Split A (3x3) into blocks:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    
    Split B (3x8) into blocks:
    B = [[B11, B12], [B21, B22]] where B11 is 2x4, B12 is 2x4, B21 is 1x4, B22 is 1x4
    
    Use Strassen's 7 products on the 2x2 sub-blocks, plus additional terms
    for the cross-block interactions.
    
    Actually, let's use a cleaner approach: pad to 4x4 and use Strassen recursively,
    or use a direct 3x3 Strassen variant.
    
    For 3x3, we can use the following 11-product scheme (or similar):
    But let's try a simpler approach: use 2x2 Strassen on the top-left 2x2,
    and handle the rest with schoolbook.
    """
    n, m, p = 3, 3, 8
    
    U = [[0]*0 for _ in range(n*m)]
    V = [[0]*0 for _ in range(m*p)]
    W = [[0]*0 for _ in range(p*n)]
    
    col = 0
    
    # Top-left 2x2 Strassen: handles C[0:2, :] = A[0:2,:] @ B[:, :]
    # But only for the 2x2 sub-block interaction
    # P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) ... etc.
    
    # Actually, for 3x3 * 3x8, let's use a different strategy:
    # Group the 8 columns into 4 pairs
    # For each pair, use a 2x2 Strassen-like approach on the 3x3 A
    
    # Let me try: for each pair of columns (k1, k2) in B,
    # compute A @ B[:, k1:k2] using a low-rank method
    
    # B[:, k1:k2] is a 3x2 matrix
    # A is 3x3
    # Result is 3x2
    
    # We can write this as:
    # C[:, k1:k2] = A @ B[:, k1:k2]
    # = A @ [[B[0,k1], B[0,k2]], [B[1,k1], B[1,k2]], [B[2,k1], B[2,k2]]]
    
    # Split A into rows: A = [A_row0; A_row1; A_row2] where each A_rowi is 1x3
    # Then C[i, k] = A_rowi @ B[:, k]
    
    # This is just schoolbook for each row.
    
    # Let's try a different approach: use the fact that we can share computations
    # across the 8 columns.
    
    # For now, return None and rely on the greedy approach
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try greedy set cover with generated candidates
    if _time_left() > 10.0:
        candidates = _generate_candidates(n, m, p)
        result = _greedy_set_cover(candidates, n, m, p, max_rank=best_rank - 1)
        if result is not None and _verify(result["U"], result["V"], result["W"], n, m, p):
            r = len(result["U"][0])
            if r < best_rank:
                best = result
                best_rank = r
    
    # Try Strassen-based
    if _time_left() > 5.0:
        cand = _strassen_3x3x8()
        if cand is not None and _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
