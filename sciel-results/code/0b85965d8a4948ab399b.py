"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify decomposition: for each rank term c, U_col @ V_col should equal W_col.T"""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for c in range(rank):
        u_col = U_arr[:, c].reshape(n, m)
        v_col = V_arr[:, c].reshape(m, p)
        w_col = W_arr[:, c].reshape(p, n)
        prod = u_col @ v_col
        if not np.allclose(prod, w_col.T, atol=1e-9):
            return False
    return True


def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _construct_strassen_block():
    """
    Block decomposition with Strassen for 2x2x2 blocks.
    Partition: i=[2,1], j=[2,2,1], k=[2,2,3]
    This gives a baseline better than pure schoolbook.
    """
    n, m, p = DIMENSIONS
    splits_i = [2, 1]
    splits_j = [2, 2, 1]
    splits_k = [2, 2, 3]
    
    i_offsets = [0, 2]
    j_offsets = [0, 2, 4]
    k_offsets = [0, 2, 4]
    
    # Count total rank
    total_rank = 0
    for ii in range(len(splits_i)):
        for jj in range(len(splits_j)):
            for kk in range(len(splits_k)):
                hi, hj, hk = splits_i[ii], splits_j[jj], splits_k[kk]
                if hi == 2 and hj == 2 and hk == 2:
                    total_rank += 7
                else:
                    total_rank += hi * hj * hk
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def strassen_2x2x2(i0, j0, k0, col):
        terms = [
            # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
            {'U': [(0,0,1),(1,1,1)], 'V': [(0,0,1),(1,1,1)], 'W': [(0,0,1),(1,1,1)]},
            # M2: (A10+A11)B00 -> C10+=1, C11-=1
            {'U': [(1,0,1),(1,1,1)], 'V': [(0,0,1)], 'W': [(1,0,1),(1,1,-1)]},
            # M3: A00(B10-B11) -> C01+=1, C11+=1
            {'U': [(0,0,1)], 'V': [(1,0,1),(1,1,-1)], 'W': [(0,1,1),(1,1,1)]},
            # M4: A11(B01-B00) -> C00+=1, C10+=1
            {'U': [(1,1,1)], 'V': [(0,1,1),(0,0,-1)], 'W': [(0,0,1),(1,0,1)]},
            # M5: (A00+A01)B11 -> C00-=1, C01+=1
            {'U': [(0,0,1),(0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1),(0,1,1)]},
            # M6: (A10-A00)(B00+B01) -> C11+=1
            {'U': [(1,0,1),(0,0,-1)], 'V': [(0,0,1),(0,1,1)], 'W': [(1,1,1)]},
            # M7: (A01-A11)(B10+B11) -> C00+=1
            {'U': [(0,1,1),(1,1,-1)], 'V': [(1,0,1),(1,1,1)], 'W': [(0,0,1)]},
        ]
        for t in terms:
            for (di, dj, sign) in t['U']:
                U[(i0+di)*m + (j0+dj)][col] += sign
            for (dj, dk, sign) in t['V']:
                V[(j0+dj)*p + (k0+dk)][col] += sign
            for (dk, di, sign) in t['W']:
                W[(k0+dk)*n + (i0+di)][col] += sign
            col += 1
        return col
    
    def schoolbook_block(i0, j0, k0, hi, hj, hk, col):
        for di in range(hi):
            for dj in range(hj):
                for dk in range(hk):
                    U[(i0+di)*m + (j0+dj)][col] = 1
                    V[(j0+dj)*p + (k0+dk)][col] = 1
                    W[(k0+dk)*n + (i0+di)][col] = 1
                    col += 1
        return col
    
    col = 0
    for ii in range(len(splits_i)):
        for jj in range(len(splits_j)):
            for kk in range(len(splits_k)):
                hi, hj, hk = splits_i[ii], splits_j[jj], splits_k[kk]
                i0, j0, k0 = i_offsets[ii], j_offsets[jj], k_offsets[kk]
                if hi == 2 and hj == 2 and hk == 2:
                    col = strassen_2x2x2(i0, j0, k0, col)
                else:
                    col = schoolbook_block(i0, j0, k0, hi, hj, hk, col)
    
    return {"U": U, "V": V, "W": W}


def _construct_direct_sparse(n, m, p, max_rank=105, time_limit=110.0):
    """
    Direct sparse linear system approach:
    The multiplication tensor has entries T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    A rank-r decomposition writes T = sum_c u_c * v_c * w_c where:
      u_c[i,j] * v_c[j,k] * w_c[k,i] contributes to T[i,j,k]
    
    For each rank term c, the contribution to entry (i,j,k) is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    We need: for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    This is a system of 105 trilinear equations. We use a greedy set-cover heuristic:
    Each potential rank-1 term is identified by its (i,j,k) support pattern.
    A term with U[i*m+j]=a, V[j*p+k]=b, W[k*n+i]=c contributes a*b*c to entry (i,j,k).
    
    Strategy: Try to find sparse rank-1 terms that cover multiple entries.
    A rank-1 term that has non-zero values at positions (i1,j1), (i2,j2), ... in U,
    (j1,k1), (j2,k2), ... in V, and (k1,i1), (k2,i2), ... in W
    will cover the intersection of these patterns.
    
    We use a randomized approach: generate candidate sparse rank-1 terms and greedily
    select those that cover the most uncovered entries.
    """
    start_time = time.time()
    
    # The target: all 105 entries must equal 1
    all_entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_entries = len(all_entries)  # 105
    
    # A rank-1 term is defined by three vectors:
    #   u in R^(n*m), v in R^(m*p), w in R^(p*n)
    # Its contribution to entry (i,j,k) is u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # For a term to contribute to entry (i,j,k), we need:
    #   u[i*m+j] != 0, v[j*p+k] != 0, w[k*n+i] != 0
    # The product must equal 1 (for a single-term covering that entry alone)
    # or combine with other terms to sum to 1.
    
    # Key insight: if we restrict to {0, 1, -1} entries, then a term covers entry (i,j,k)
    # iff all three factors are non-zero, and the product is +/-1.
    # We need the sum over all terms of these products to equal 1 for each entry.
    
    # Greedy set cover:
    # 1. Generate candidate terms (sparse patterns)
    # 2. For each candidate, compute which entries it covers and with what sign
    # 3. Greedily select terms to minimize total rank while satisfying all equations
    
    # Simplified approach: use the structure that each entry (i,j,k) needs exactly
    # one "unit" of contribution. A term that covers entry (i,j,k) with product +1
    # contributes 1, with product -1 contributes -1.
    
    # Let's think differently. The standard schoolbook uses 105 terms, each covering
    # exactly one entry. To reduce rank, we need terms that cover multiple entries.
    
    # A term covers entries (i,j,k) where u[i*m+j]*v[j*p+k]*w[k*n+i] != 0.
    # If u has support S_u (set of (i,j) pairs), v has support S_v (set of (j,k) pairs),
    # w has support S_w (set of (k,i) pairs), then the term covers entries (i,j,k)
    # where (i,j) in S_u, (j,k) in S_v, and (k,i) in S_w.
    
    # For the product to be +1 for all covered entries, we need consistent signs.
    
    # Let's try: terms where u, v, w are indicator vectors of "lines" or "planes".
    
    # A "line" in U: fix j, vary i -> covers entries (i,j,k) for all i, and k where
    # v[j,k] and w[k,i] are non-zero.
    
    # Actually, let's just do a direct greedy search with random sparse terms.
    
    best_rank = num_entries  # schoolbook
    best_solution = None
    
    # Precompute: for each entry (i,j,k), the three indices
    entry_indices = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entry_indices[(i,j,k)] = (i*m+j, j*p+k, k*n+i)
    
    def generate_random_term(rng):
        """Generate a random sparse rank-1 term with entries in {0, 1, -1}."""
        # Choose sparsity level
        u_sparsity = rng.choice([1, 2, 3, 4, 5])
        v_sparsity = rng.choice([1, 2, 3, 4, 5])
        w_sparsity = rng.choice([1, 2, 3, 4, 5])
        
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        u_support = rng.choice(n*m, size=min(u_sparsity, n*m), replace=False)
        v_support = rng.choice(m*p, size=min(v_sparsity, m*p), replace=False)
        w_support = rng.choice(p*n, size=min(w_sparsity, p*n), replace=False)
        
        for idx in u_support:
            u[idx] = rng.choice([1, -1])
        for idx in v_support:
            v[idx] = rng.choice([1, -1])
        for idx in w_support:
            w[idx] = rng.choice([1, -1])
        
        return u, v, w
    
    def term_contribution(u, v, w):
        """Return dict: entry -> contribution value."""
        contrib = {}
        for (i,j,k), (ui, vi, wi) in entry_indices.items():
            val = u[ui] * v[vi] * w[wi]
            if val != 0:
                contrib[(i,j,k)] = val
        return contrib
    
    # Greedy set cover with random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Start with empty coverage
        coverage = {entry: 0 for entry in all_entries}
        selected_terms = []
        uncovered = set(all_entries)
        
        # Generate a pool of candidate terms
        pool_size = 200
        candidates = []
        for _ in range(pool_size):
            u, v, w = generate_random_term(rng)
            contrib = term_contribution(u, v, w)
            if contrib:
                candidates.append((u, v, w, contrib))
        
        # Greedily select terms
        while uncovered and len(selected_terms) < best_rank:
            if time.time() - start_time > time_limit:
                break
            
            best_gain = 0
            best_candidate = None
            
            for u, v, w, contrib in candidates:
                # Compute net gain: how many uncovered entries this term helps
                gain = 0
                for entry, val in contrib.items():
                    if entry in uncovered:
                        # If adding this term would make coverage[entry] closer to 1
                        current = coverage[entry]
                        new_val = current + val
                        if new_val == 1:
                            gain += 2  # Perfect: covers this entry
                        elif abs(new_val - 1) < abs(current - 1):
                            gain += 1  # Partial improvement
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = (u, v, w, contrib)
            
            if best_candidate is None or best_gain == 0:
                break
            
            u, v, w, contrib = best_candidate
            selected_terms.append((u, v, w))
            for entry, val in contrib.items():
                coverage[entry] += val
            uncovered = {e for e in all_entries if coverage[e] != 1}
            
            # Remove used candidate to avoid reusing
            candidates.remove(best_candidate)
        
        if not uncovered and len(selected_terms) < best_rank:
            best_rank = len(selected_terms)
            best_solution = selected_terms
    
    if best_solution is not None:
        U = [[0] * best_rank for _ in range(n * m)]
        V = [[0] * best_rank for _ in range(m * p)]
        W = [[0] * best_rank for _ in range(p * n)]
        for c, (u, v, w) in enumerate(best_solution):
            for i in range(n*m):
                if u[i] != 0:
                    U[i][c] = int(u[i])
            for i in range(m*p):
                if v[i] != 0:
                    V[i][c] = int(v[i])
            for i in range(p*n):
                if w[i] != 0:
                    W[i][c] = int(w[i])
        return {"U": U, "V": V, "W": W}
    
    return None


def _construct_winograd_fusion(n, m, p):
    """
    Try a Winograd-style fusion: instead of independent blocks,
    use overlapping transform matrices to share multiplications.
    
    For the 3x5x7 case, we can try to find a global transform-based
    decomposition. The key idea: if we can write the multiplication as
    (T_A * A) * (T_B * B) with shared intermediate products, we save rank.
    
    Specifically, use the fact that for small matrices, we can find
    low-rank factorizations of the "multiplication map".
    
    This is a heuristic: try random sparse transforms and check if
    the resulting decomposition has lower rank.
    """
    start_time = time.time()
    
    # Try a specific structured approach:
    # Split the 5x7 multiplication into overlapping 3x3 and 2x2 blocks
    # with shared Strassen multiplications.
    
    # Actually, let's try a different structural idea:
    # The 3x5 multiplication: A is 3x5. We can write A = [A1 | A2] where
    # A1 is 3x2 and A2 is 3x3. Then C = A1*B1 + A2*B2 where B1 is 2x7, B2 is 3x7.
    # This doesn't directly help with rank.
    
    # Let's try: use the fact that rank(3x5x7) <= rank(3x3x3) + rank(3x2x7) + ...
    # But we need a better decomposition.
    
    # Try: partition j into [3, 2], and for the 3x3x7 part, use
    # Winograd for the 3x3 part (rank 5 per 3x3x3 block) times 7/3...
    # This doesn't work cleanly.
    
    # Alternative: use the "border rank" concept. The border rank of
    # matrix multiplication can be lower than the rank. But we need
    # exact decomposition, so border rank doesn't directly apply.
    
    # Let's try a concrete structured decomposition:
    # n=3, m=5, p=7
    # Split: i=[1,1,1], j=[2,2,1], k=[2,2,3]
    # For each (i,j,k) block of size 1xhjxhk, use schoolbook: hj*hk terms
    # Total: sum over j-blocks and k-blocks of hj*hk = (2*2 + 2*2 + 2*3) * 3 = (4+4+6)*3 = 42
    # Wait, that's for each i. So total = 3 * (2*2 + 2*2 + 2*3) = 3*14 = 42? No...
    
    # Actually for 1xhjxhk block, schoolbook rank = 1*hj*hk = hj*hk
    # Total = sum_{jj,kk} splits_j[jj] * splits_k[kk] = (2*2 + 2*2 + 2*3) * 3
    # = (4+4+6) * 3 = 42
    # That's way less than 105! But wait, that's wrong.
    # For a 1xhjxhk block, we need hj*hk multiplications, one for each (j,k) pair.
    # Total over all blocks: sum_{jj} sum_{kk} 1 * splits_j[jj] * splits_k[kk]
    # = (sum_jj splits_j[jj]) * (sum_kk splits_k[kk]) = 5 * 7 = 35
    # Plus we have 3 i-values, so 3 * 35 = 105. Same as schoolbook.
    
    # The issue is that splitting i doesn't help since each i is independent.
    # We need to find structure ACROSS i values.
    
    # Key insight: if two rows of A share the same column pattern in B,
    # we can share multiplications. But in general matrix multiplication,
    # all entries are independent, so there's no such sharing possible
    # with a simple block decomposition.
    
    # The only way to beat schoolbook is to use algebraic identities
    # like Strassen, which exploit the bilinear structure.
    
    # For 3x5x7, let's try to embed it in a larger structured problem.
    # 3x5x7 can be seen as a sub-problem of 4x8x8 (or similar).
    # If we have a rank-r decomposition of 4x8x8, we can restrict it
    # to the 3x5x7 sub-problem, potentially getting rank < 105.
    
    # But finding a good 4x8x8 decomposition is also hard.
    
    # Let's try a different approach: use the known result that
    # rank(3x3x3) = 23 (Bini et al.), and build up from there.
    # 3x5x7: split as (3x3x3) + (3x2x3) + (3x3x4) + (3x2x4)
    # = 23 + 6 + 12 + 8 = 49? No, that's not how it works.
    
    # Actually, rank(3x5x7) <= rank(3x3x7) + rank(3x2x7)
    # rank(3x3x7) <= 7 * rank(3x3x1) = 7*3 = 21 (schoolbook)
    # But rank(3x3x3) = 23, so rank(3x3x7) <= 7*23/3... no, that's not right either.
    
    # The correct bound: rank(n x m x p) <= rank(n x m' x p) + rank(n x (m-m') x p)
    # for any m'. This is just additivity.
    
    # For our purposes, let's just try the best block decomposition we can find
    # and verify it.
    
    return None  # Fall through to other methods


def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Start with schoolbook as fallback
    best_result = _construct_schoolbook()
    best_rank = 3 * 5 * 7
    
    # Try Strassen block decomposition
    if time.time() < deadline:
        strassen_result = _construct_strassen_block()
        strassen_rank = len(strassen_result['U'][0])
        if _verify(strassen_result['U'], strassen_result['V'], strassen_result['W'], 3, 5, 7):
            if strassen_rank < best_rank:
                best_result = strassen_result
                best_rank = strassen_rank
    
    # Try direct sparse search (the new strategy)
    if time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            direct_result = _construct_direct_sparse(3, 5, 7, time_limit=remaining_time - 2)
            if direct_result is not None:
                direct_rank = len(direct_result['U'][0])
                if _verify(direct_result['U'], direct_result['V'], direct_result['W'], 3, 5, 7):
                    if direct_rank < best_rank:
                        best_result = direct_result
                        best_rank = direct_rank
    
    return best_result
# EVOLVE-BLOCK-END

