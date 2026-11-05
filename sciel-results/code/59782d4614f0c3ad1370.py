"""Matrix multiplication 3x3x5 rank-minimal exact tensor decomposition search."""
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import random
import numpy as np
from itertools import combinations, product

def _build_tensor(n, m, p):
    """Return dict T[(a,b,c)] = 1 for the multiplication tensor.
    a = i*m + j, b = j*p + k, c = k*n + i.
    """
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    return T

def _verify(U, V, W, T, n, m, p):
    """Check that sum_c U[a,c]*V[b,c]*W[c] == T[(a,b,c)] for all a,b,c."""
    rank = len(U[0]) if U else 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                s = Fraction(0)
                for col in range(rank):
                    s += U[a][col] * V[b][col] * W[c][col]
                expected = T.get((a, b, c), 0)
                if s != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W

def _to_output(U, V, W):
    """Convert Fraction matrices to int/half-int lists for output."""
    def conv(M):
        out = []
        for row in M:
            r = []
            for x in row:
                if x.denominator == 1:
                    r.append(int(x.numerator))
                elif x.denominator == 2:
                    r.append(Fraction(x.numerator, 2))
                else:
                    r.append(x)
            out.append(r)
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}

def _try_merge_two_terms(n, m, p, term1, term2, U, V, W):
    """
    Try to merge two schoolbook terms into one by finding a rank-1 tensor
    that equals the sum of the two original rank-1 tensors, OR by adjusting
    coefficients. Returns modified (U,V,W) with one fewer column, or None.
    """
    i1, j1, k1 = term1
    i2, j2, k2 = term2
    
    a1, b1, c1 = i1*m+j1, j1*p+k1, k1*n+i1
    a2, b2, c2 = i2*m+j2, j2*p+k2, k2*n+i2
    
    if a1 == a2 and b1 == b2 and c1 == c2:
        return None
    
    u_support = {a1} if a1 == a2 else {a1, a2}
    v_support = {b1} if b1 == b2 else {b1, b2}
    w_support = {c1} if c1 == c2 else {c1, c2}
    
    for a in u_support:
        for b in v_support:
            for c in w_support:
                if (a, b, c) != (a1, b1, c1) and (a, b, c) != (a2, b2, c2):
                    return None
    
    candidates = [Fraction(1), Fraction(-1), Fraction(2), Fraction(-2), 
                  Fraction(1,2), Fraction(-1,2)]
    
    for u_val1 in candidates:
        for u_val2 in candidates:
            for v_val1 in candidates:
                for v_val2 in candidates:
                    try:
                        w_c1 = Fraction(1) / (u_val1 * v_val1)
                        w_c2 = Fraction(1) / (u_val2 * v_val2)
                    except ZeroDivisionError:
                        continue
                    
                    if w_c1 not in candidates and w_c2 not in candidates:
                        continue
                    
                    U_new = [row[:] for row in U]
                    V_new = [row[:] for row in V]
                    W_new = [row[:] for row in W]
                    
                    col1 = None
                    col2 = None
                    for col in range(len(U[0])):
                        if U[a1][col] == 1 and V[b1][col] == 1 and W[c1][col] == 1:
                            if col1 is None:
                                col1 = col
                        if U[a2][col] == 1 and V[b2][col] == 1 and W[c2][col] == 1:
                            if col2 is None:
                                col2 = col
                    
                    if col1 is None or col2 is None or col1 == col2:
                        continue
                    
                    for a in range(n*m):
                        U_new[a][col1] = Fraction(0)
                    for b in range(m*p):
                        V_new[b][col1] = Fraction(0)
                    for c in range(p*n):
                        W_new[c][col1] = Fraction(0)
                    
                    for a in u_support:
                        if a == a1 and a == a2:
                            U_new[a][col1] = u_val1
                        elif a == a1:
                            U_new[a][col1] = u_val1
                        elif a == a2:
                            U_new[a][col1] = u_val2
                    
                    for b in v_support:
                        if b == b1 and b == b2:
                            V_new[b][col1] = v_val1
                        elif b == b1:
                            V_new[b][col1] = v_val1
                        elif b == b2:
                            V_new[b][col1] = v_val2
                    
                    for c in w_support:
                        if c == c1 and c == c2:
                            W_new[c][col1] = w_c1
                        elif c == c1:
                            W_new[c][col1] = w_c1
                        elif c == c2:
                            W_new[c][col1] = w_c2
                    
                    for a in range(n*m):
                        U_new[a][col2] = Fraction(0)
                    for b in range(m*p):
                        V_new[b][col2] = Fraction(0)
                    for c in range(p*n):
                        W_new[c][col2] = Fraction(0)
                    
                    return U_new, V_new, W_new, col2
    
    return None

def _try_strassen_like(n, m, p):
    """
    Try to use a Strassen-like decomposition for the 3x3x3 core
    and handle the extra 2 columns of the 5-column matrix.
    """
    return None

def _gauss_solve(A, b):
    """Solve A x = b using Fraction Gaussian elimination.
    Returns solution vector x or None if inconsistent.
    """
    n_rows = len(A)
    n_cols = len(A[0]) if n_rows > 0 else 0
    if n_rows == 0:
        return [Fraction(0)] * n_cols
    
    aug = []
    for i in range(n_rows):
        aug.append(list(A[i]) + [b[i]])
    
    rank = 0
    pivot_cols = []
    for col in range(n_cols):
        if rank >= n_rows:
            break
        pivot_row = -1
        for row in range(rank, n_rows):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
        pivot_val = aug[rank][col]
        for j in range(col, n_cols + 1):
            aug[rank][j] = aug[rank][j] / pivot_val
        for row in range(n_rows):
            if row != rank and aug[row][col] != 0:
                factor = aug[row][col]
                for j in range(col, n_cols + 1):
                    aug[row][j] -= factor * aug[rank][j]
        rank += 1
        pivot_cols.append(col)
    
    for row in range(rank, n_rows):
        if aug[row][n_cols] != 0:
            return None
    
    solution = [Fraction(0)] * n_cols
    for i in range(rank):
        col = pivot_cols[i]
        solution[col] = aug[i][n_cols]
    
    return solution

def _structured_blocks(n, m, p):
    """
    Generate structured rank-1 blocks based on tensor slicing.
    For each slice type, generate blocks that are nonzero on structured subsets.
    
    Slice types:
    - U-slice: fixed i, vary j,k -> U nonzero on {i*m+j for j}, V on {j*p+k for j,k}, W on {k*n+i for k}
    - V-slice: fixed j, vary i,k -> similar
    - W-slice: fixed k, vary i,j -> similar
    - Row-slice: fixed i,j, vary k
    - Col-slice: fixed j,k, vary i
    - Diag-slice: fixed i=k or other diagonal patterns
    """
    blocks = []
    nm, mp, pn = n*m, m*p, p*n
    
    # Type 1: U-slice blocks - fix i, use structured V,W for the remaining
    for i in range(n):
        # U vector: nonzero on row i (positions i*m .. i*m+m-1)
        for j_mask in range(1, 1 << m):  # which j's are active
            u = [Fraction(0)] * nm
            for j in range(m):
                if j_mask & (1 << j):
                    u[i*m + j] = Fraction(1)
            
            # For each active j, we need V and W to cover the k dimension
            # V: for each active j, positions j*p..j*p+p-1
            # W: for each active j, positions (k*n+i) for k in range(p)
            # This is essentially a p-dimensional problem for each j
            
            # Generate V,W that work for all active j's simultaneously
            # V[b] must be nonzero at b = j*p+k for active j and some k
            # W[c] must be nonzero at c = k*n+i for active j and some k
            
            # Simplest: V and W are "full" on the relevant slices
            for k_mask in range(1, 1 << p):  # which k's are active
                v = [Fraction(0)] * mp
                w = [Fraction(0)] * pn
                for j in range(m):
                    if j_mask & (1 << j):
                        for k in range(p):
                            if k_mask & (1 << k):
                                v[j*p + k] = Fraction(1)
                                w[k*n + i] = Fraction(1)
                blocks.append((tuple(u), tuple(v), tuple(w)))
    
    # Type 2: V-slice blocks - fix j, use structured U,W
    for j in range(m):
        for i_mask in range(1, 1 << n):
            v = [Fraction(0)] * mp
            for i in range(n):
                if i_mask & (1 << i):
                    v[j*p + 0] = Fraction(0)  # placeholder
            # Actually V is indexed by (j,k), so for fixed j, V[j*p+k] for k in range(p)
            # U is indexed by (i,j), so for fixed j, U[i*m+j] for i in range(n)
            # W is indexed by (k,i), so for fixed j... W doesn't depend on j directly
            
            # Let me redo: for fixed j, the tensor entries are T[(i*m+j, j*p+k, k*n+i)]
            # U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all i,k
            # This factors as U_i * V_k * W_{k,i} which isn't separable unless we use multiple terms
            
            # Better: for fixed j, generate blocks where U is nonzero on {i*m+j}, 
            # V on {j*p+k}, W on {k*n+i}
            for k_mask in range(1, 1 << p):
                u = [Fraction(0)] * nm
                v = [Fraction(0)] * mp
                w = [Fraction(0)] * pn
                for i in range(n):
                    if i_mask & (1 << i):
                        u[i*m + j] = Fraction(1)
                for k in range(p):
                    if k_mask & (1 << k):
                        v[j*p + k] = Fraction(1)
                        for i in range(n):
                            if i_mask & (1 << i):
                                w[k*n + i] = Fraction(1)
                blocks.append((tuple(u), tuple(v), tuple(w)))
    
    # Type 3: W-slice blocks - fix k
    for k in range(p):
        for i_mask in range(1, 1 << n):
            for j_mask in range(1, 1 << m):
                u = [Fraction(0)] * nm
                v = [Fraction(0)] * mp
                w = [Fraction(0)] * pn
                for i in range(n):
                    if i_mask & (1 << i):
                        for j in range(m):
                            if j_mask & (1 << j):
                                u[i*m + j] = Fraction(1)
                                v[j*p + k] = Fraction(1)
                for i in range(n):
                    if i_mask & (1 << i):
                        w[k*n + i] = Fraction(1)
                blocks.append((tuple(u), tuple(v), tuple(w)))
    
    # Type 4: Single-term blocks (schoolbook-like but with signs)
    signs = [Fraction(1), Fraction(-1)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for s1 in signs:
                    for s2 in signs:
                        u = [Fraction(0)] * nm
                        v = [Fraction(0)] * mp
                        w = [Fraction(0)] * pn
                        u[i*m + j] = s1
                        v[j*p + k] = s2
                        w[k*n + i] = Fraction(1) / (s1 * s2)
                        blocks.append((tuple(u), tuple(v), tuple(w)))
    
    # Type 5: Pairs of terms merged (2-term blocks)
    # For pairs sharing an index, try to merge
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(k1+1, p):
                    # Terms (i,j,k1) and (i,j,k2) share i,j
                    # U: nonzero at i*m+j
                    # V: nonzero at j*p+k1 and j*p+k2
                    # W: nonzero at k1*n+i and k2*n+i
                    u = [Fraction(0)] * nm
                    v = [Fraction(0)] * mp
                    w = [Fraction(0)] * pn
                    u[i*m + j] = Fraction(1)
                    for k in [k1, k2]:
                        v[j*p + k] = Fraction(1)
                        w[k*n + i] = Fraction(1)
                    blocks.append((tuple(u), tuple(v), tuple(w)))
                    
                    # With signs
                    for s in [Fraction(1), Fraction(-1)]:
                        v2 = [Fraction(0)] * mp
                        w2 = [Fraction(0)] * pn
                        v2[j*p + k1] = Fraction(1)
                        v2[j*p + k2] = s
                        w2[k1*n + i] = Fraction(1)
                        w2[k2*n + i] = Fraction(1) / s
                        blocks.append((tuple(u), tuple(v2), tuple(w2)))
    
    for i in range(n):
        for j1 in range(m):
            for j2 in range(j1+1, m):
                for k in range(p):
                    # Terms (i,j1,k) and (i,j2,k) share i,k
                    u = [Fraction(0)] * nm
                    v = [Fraction(0)] * mp
                    w = [Fraction(0)] * pn
                    for j in [j1, j2]:
                        u[i*m + j] = Fraction(1)
                        v[j*p + k] = Fraction(1)
                    w[k*n + i] = Fraction(1)
                    blocks.append((tuple(u), tuple(v), tuple(w)))
                    
                    for s in [Fraction(1), Fraction(-1)]:
                        u2 = [Fraction(0)] * nm
                        v2 = [Fraction(0)] * mp
                        u2[i*m + j1] = Fraction(1)
                        u2[i*m + j2] = s
                        v2[j1*p + k] = Fraction(1)
                        v2[j2*p + k] = Fraction(1)
                        w2 = [Fraction(0)] * pn
                        w2[k*n + i] = Fraction(1) / s
                        blocks.append((tuple(u2), tuple(v2), tuple(w2)))
    
    for j in range(m):
        for i1 in range(n):
            for i2 in range(i1+1, n):
                for k in range(p):
                    # Terms (i1,j,k) and (i2,j,k) share j,k
                    u = [Fraction(0)] * nm
                    v = [Fraction(0)] * mp
                    w = [Fraction(0)] * pn
                    for i in [i1, i2]:
                        u[i*m + j] = Fraction(1)
                    v[j*p + k] = Fraction(1)
                    for i in [i1, i2]:
                        w[k*n + i] = Fraction(1)
                    blocks.append((tuple(u), tuple(v), tuple(w)))
                    
                    for s in [Fraction(1), Fraction(-1)]:
                        u2 = [Fraction(0)] * nm
                        w2 = [Fraction(0)] * pn
                        u2[i1*m + j] = Fraction(1)
                        u2[i2*m + j] = s
                        w2[k*n + i1] = Fraction(1)
                        w2[k*n + i2] = Fraction(1) / s
                        v2 = [Fraction(0)] * mp
                        v2[j*p + k] = Fraction(1)
                        blocks.append((tuple(u2), tuple(v2), tuple(w2)))
    
    # Deduplicate
    seen = set()
    unique_blocks = []
    for block in blocks:
        if block not in seen:
            seen.add(block)
            unique_blocks.append(block)
    
    return unique_blocks

def _block_contribution(block, n, m, p):
    """
    Compute the set of (a,b,c) positions where this block contributes non-zero,
    and what value it contributes.
    """
    u, v, w = block
    contrib = {}
    for a in range(n*m):
        ua = u[a]
        if ua == 0: continue
        for b in range(m*p):
            vb = v[b]
            if vb == 0: continue
            for c in range(p*n):
                wc = w[c]
                if wc == 0: continue
                val = ua * vb * wc
                contrib[(a, b, c)] = val
    return contrib

def _try_structured_decomposition(n, m, p, T, deadline):
    """
    Use structured blocks and solve for coefficients via linear algebra.
    Select a subset of blocks that can span the tensor, then solve.
    """
    start = time.time()
    
    # Build the set of required positions
    required = set(T.keys())
    required_vals = dict(T)
    
    # Generate structured blocks
    blocks = _structured_blocks(n, m, p)
    print(f"Generated {len(blocks)} structured blocks")
    
    # Precompute contributions
    block_contribs = []
    for block in blocks:
        contrib = _block_contribution(block, n, m, p)
        if contrib:
            block_contribs.append((block, contrib))
    
    print(f"Blocks with non-zero contribution: {len(block_contribs)}")
    
    # Greedy selection: pick blocks that cover the most uncovered positions
    selected = []
    remaining = set(required)
    
    for iteration in range(60):  # max rank to try
        if time.time() > deadline:
            break
        if not remaining:
            break
        
        best_idx = -1
        best_cov = 0
        for idx, (block, contrib) in enumerate(block_contribs):
            cov = len(contrib.keys() & remaining)
            if cov > best_cov:
                best_cov = cov
                best_idx = idx
        
        if best_idx == -1 or best_cov == 0:
            break
        
        selected.append(block_contribs[best_idx])
        remaining -= block_contribs[best_idx][1].keys()
        block_contribs[best_idx] = (None, {})  # remove from pool
    
    if remaining:
        # Try adding more blocks even if coverage is 0 (for coefficient flexibility)
        # This allows for cancellations
        extra_added = 0
        for idx, (block, contrib) in enumerate(block_contribs):
            if time.time() > deadline:
                break
            if block is None:
                continue
            # Check if this block can help with cancellation
            # Add it if it overlaps with already-selected blocks
            overlap = len(contrib.keys() & set(T.keys()))
            if overlap > 0:
                selected.append(block_contribs[idx])
                block_contribs[idx] = (None, {})
                extra_added += 1
                if extra_added >= 10:
                    break
        
        if remaining:
            return None
    
    # Solve for coefficients
    num_blocks = len(selected)
    
    # Build linear system
    eq_positions = list(required)
    rows = []
    rhs = []
    for pos in eq_positions:
        a, b, c = pos
        row = []
        for block, contrib in selected:
            if block is None:
                row.append(Fraction(0))
                continue
            u, v, w = block
            val = u[a] * v[b] * w[c]
            row.append(val)
        rows.append(row)
        rhs.append(required_vals[pos])
    
    # Also add equations for positions NOT in required (must be 0)
    all_positions = set()
    for block, contrib in selected:
        if block is not None:
            all_positions.update(contrib.keys())
    
    zero_positions = all_positions - required
    for pos in zero_positions:
        a, b, c = pos
        row = []
        for block, contrib in selected:
            if block is None:
                row.append(Fraction(0))
                continue
            u, v, w = block
            val = u[a] * v[b] * w[c]
            row.append(val)
        rows.append(row)
        rhs.append(Fraction(0))
    
    # Solve
    solution = _gauss_solve(rows, rhs)
    if solution is None:
        return None
    
    # Build U, V, W
    U = [[Fraction(0)] * num_blocks for _ in range(n*m)]
    V = [[Fraction(0)] * num_blocks for _ in range(m*p)]
    W = [[Fraction(0)] * num_blocks for _ in range(p*n)]
    
    for r in range(num_blocks):
        block, contrib = selected[r]
        if block is None:
            continue
        u, v, w = block
        coeff = solution[r]
        if coeff == 0:
            continue
        for a in range(n*m):
            U[a][r] = u[a] * coeff
        for b in range(m*p):
            V[b][r] = v[b]
        for c in range(p*n):
            W[c][r] = w[c]
    
    if _verify(U, V, W, T, n, m, p):
        return U, V, W, num_blocks
    
    return None

def _try_remove_column(n, m, p, U, V, W, col_idx, T):
    """Try removing a single column and see if the decomposition still works."""
    rank = len(U[0])
    if rank <= 1:
        return None
    
    U_new = [row[:col_idx] + row[col_idx+1:] for row in U]
    V_new = [row[:col_idx] + row[col_idx+1:] for row in V]
    W_new = [row[:col_idx] + row[col_idx+1:] for row in W]
    
    if _verify(U_new, V_new, W_new, T, n, m, p):
        return U_new, V_new, W_new
    
    return None

def _try_replace_column(n, m, p, U, V, W, col_idx, T, candidates):
    """Try replacing a column with a different rank-1 tensor."""
    rank = len(U[0])
    if rank <= 1:
        return None
    
    for cand in candidates:
        u_vals, v_vals, w_vals = cand
        U_new = [row[:] for row in U]
        V_new = [row[:] for row in V]
        W_new = [row[:] for row in W]
        
        for a in range(n*m):
            U_new[a][col_idx] = u_vals[a]
        for b in range(m*p):
            V_new[b][col_idx] = v_vals[b]
        for c in range(p*n):
            W_new[c][col_idx] = w_vals[c]
        
        if _verify(U_new, V_new, W_new, T, n, m, p):
            return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill

    T = _build_tensor(n, m, p)
    rank0 = n * m * p

    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = rank0

    # Strategy 1: Structured decomposition with coefficient solving
    if time.time() < deadline - 20:
        result = _try_structured_decomposition(n, m, p, T, deadline - 10)
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            if new_rank < best_rank:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_rank
                print(f"Structured decomposition found rank {best_rank}!")

    # Strategy 2: Pair merging from schoolbook
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k))
    
    candidate_pairs = []
    for idx1, t1 in enumerate(terms):
        for idx2 in range(idx1 + 1, len(terms)):
            t2 = terms[idx2]
            if t1[0] == t2[0] or t1[1] == t2[1] or t1[2] == t2[2]:
                candidate_pairs.append((t1, t2))
    
    random.seed(42)
    random.shuffle(candidate_pairs)
    
    # First pass: try merging pairs
    for t1, t2 in candidate_pairs:
        if time.time() > deadline:
            break
        result = _try_merge_two_terms(n, m, p, t1, t2, best_U, best_V, best_W)
        if result is not None:
            U_new, V_new, W_new, col_to_remove = result
            U_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in U_new]
            V_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in V_new]
            W_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in W_new]
            
            if _verify(U_final, V_final, W_final, T, n, m, p):
                best_U, best_V, best_W = U_final, V_final, W_final
                best_rank = best_rank - 1
                print(f"Found rank {best_rank} decomposition!")
                break
    
    # Second pass: try removing any column
    if time.time() < deadline:
        for col in range(len(best_U[0])):
            if time.time() > deadline:
                break
            result = _try_remove_column(n, m, p, best_U, best_V, best_W, col, T)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank -= 1
                print(f"Removed column, rank {best_rank}!")
                break
    
    # Third pass: try more pair merges with updated matrices
    if time.time() < deadline:
        for t1, t2 in candidate_pairs:
            if time.time() > deadline:
                break
            result = _try_merge_two_terms(n, m, p, t1, t2, best_U, best_V, best_W)
            if result is not None:
                U_new, V_new, W_new, col_to_remove = result
                U_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in U_new]
                V_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in V_new]
                W_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in W_new]
                
                if _verify(U_final, V_final, W_final, T, n, m, p):
                    best_U, best_V, best_W = U_final, V_final, W_final
                    best_rank = best_rank - 1
                    print(f"Found rank {best_rank} decomposition in pass 3!")
                    break

    return _to_output(best_U, best_V, best_W)
# EVOLVE-BLOCK-END
