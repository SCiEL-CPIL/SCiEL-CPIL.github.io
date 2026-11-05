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

def _generate_building_blocks(n, m, p, max_blocks=20000):
    """
    Generate a pool of candidate rank-1 building blocks.
    Each block is (U_vec, V_vec, W_vec) where each vec has entries in {-2,-1,0,1,2}
    (or small half-integers). We focus on sparse blocks with small support.
    """
    blocks = []
    # Small value set
    vals = [Fraction(-2), Fraction(-1), Fraction(0), Fraction(1), Fraction(2)]
    
    # Generate sparse U vectors (support size 1-3)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute all sparse vectors for each dimension
    def gen_sparse_vecs(dim, max_support):
        vecs = []
        # support 1
        for pos in range(dim):
            for v in vals:
                if v != 0:
                    vec = [Fraction(0)] * dim
                    vec[pos] = v
                    vecs.append(tuple(vec))
        # support 2
        for pos1 in range(dim):
            for pos2 in range(pos1+1, dim):
                for v1 in vals:
                    if v1 == 0: continue
                    for v2 in vals:
                        if v2 == 0: continue
                        vec = [Fraction(0)] * dim
                        vec[pos1] = v1
                        vec[pos2] = v2
                        vecs.append(tuple(vec))
        # support 3
        for pos1 in range(dim):
            for pos2 in range(pos1+1, dim):
                for pos3 in range(pos2+1, dim):
                    for v1 in vals:
                        if v1 == 0: continue
                        for v2 in vals:
                            if v2 == 0: continue
                            for v3 in vals:
                                if v3 == 0: continue
                                vec = [Fraction(0)] * dim
                                vec[pos1] = v1
                                vec[pos2] = v2
                                vec[pos3] = v3
                                vecs.append(tuple(vec))
        return vecs
    
    U_vecs = gen_sparse_vecs(nm, 3)
    V_vecs = gen_sparse_vecs(mp, 3)
    W_vecs = gen_sparse_vecs(pn, 3)
    
    # Randomly sample combinations to stay within budget
    random.seed(123)
    seen = set()
    count = 0
    attempts = 0
    while count < max_blocks and attempts < max_blocks * 10:
        attempts += 1
        u = random.choice(U_vecs)
        v = random.choice(V_vecs)
        w = random.choice(W_vecs)
        key = (u, v, w)
        if key in seen:
            continue
        seen.add(key)
        blocks.append((u, v, w))
        count += 1
    
    return blocks

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

def _try_tensor_slicing_set_cover(n, m, p, T, deadline):
    """
    Tensor slicing + greedy set-cover approach.
    Treat each of the p output columns as a slice.
    Use greedy set-cover to select minimal blocks that span all required positions.
    """
    start = time.time()
    
    # Build the set of required positions
    required = set(T.keys())
    required_vals = dict(T)
    
    # Generate building blocks
    blocks = _generate_building_blocks(n, m, p, max_blocks=5000)
    
    # Precompute contributions for each block
    block_contribs = []
    for block in blocks:
        contrib = _block_contribution(block, n, m, p)
        if contrib:
            block_contribs.append((block, contrib))
    
    # Greedy set-cover with coefficient solving
    # We need to find a subset of blocks and coefficients such that
    # sum_r alpha_r * contrib_r = T
    
    # Start with empty selection
    selected = []
    selected_coefficients = []
    residual = {pos: Fraction(0) for pos in required}
    for pos in required:
        residual[pos] = required_vals[pos]
    
    remaining = set(required)
    
    max_iterations = 45  # schoolbook rank
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        if not remaining:
            break
        
        # Find the block that covers the most remaining positions
        best_block_idx = -1
        best_coverage = 0
        best_contrib = None
        
        for idx, (block, contrib) in enumerate(block_contribs):
            coverage = 0
            for pos in contrib:
                if pos in remaining:
                    coverage += 1
            if coverage > best_coverage:
                best_coverage = coverage
                best_block_idx = idx
                best_contrib = contrib
        
        if best_block_idx == -1 or best_coverage == 0:
            break
        
        # Add this block to selection
        selected.append(block_contribs[best_block_idx][0])
        selected_coefficients.append(None)  # to be determined
        remaining -= set(best_contrib.keys())
        
        # Remove from consideration to avoid re-selecting
        block_contribs[best_block_idx] = (None, {})
    
    # If we couldn't cover everything, return None
    if remaining:
        return None
    
    # Now solve for coefficients using linear algebra
    # We have len(selected) blocks and need to satisfy |required| equations
    num_blocks = len(selected)
    num_eqs = len(required)
    
    if num_blocks > num_eqs:
        # Too many blocks, try to prune
        return None
    
    if num_blocks < num_eqs:
        return None
    
    # Build linear system: for each position, sum_r alpha_r * contrib_r[pos] = T[pos]
    # Only consider positions where at least one selected block has non-zero contribution
    eq_positions = []
    for pos in required:
        # Check if any selected block contributes to this position
        has_contrib = False
        for block in selected:
            u, v, w = block
            # Check if this position can get contribution
            a, b, c = pos
            if u[a] != 0 and v[b] != 0 and w[c] != 0:
                has_contrib = True
                break
        if has_contrib:
            eq_positions.append(pos)
    
    # Build matrix A and vector b
    # A[i][j] = contribution of block j to position i
    # b[i] = T[position_i]
    
    rows = []
    rhs = []
    for pos in eq_positions:
        a, b, c = pos
        row = []
        for block in selected:
            u, v, w = block
            val = u[a] * v[b] * w[c]
            row.append(val)
        rows.append(row)
        rhs.append(required_vals[pos])
    
    num_rows = len(rows)
    num_cols = num_blocks
    
    # Solve using fraction-based Gaussian elimination
    # Augmented matrix
    aug = []
    for i in range(num_rows):
        aug.append(rows[i] + [rhs[i]])
    
    # Gaussian elimination
    rank = 0
    for col in range(num_cols):
        if rank >= num_rows:
            break
        # Find pivot
        pivot_row = -1
        for row in range(rank, num_rows):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        # Swap
        aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
        # Normalize
        pivot_val = aug[rank][col]
        for j in range(col, num_cols + 1):
            aug[rank][j] = aug[rank][j] / pivot_val
        # Eliminate
        for row in range(num_rows):
            if row != rank and aug[row][col] != 0:
                factor = aug[row][col]
                for j in range(col, num_cols + 1):
                    aug[row][j] -= factor * aug[rank][j]
        rank += 1
    
    # Check for consistency
    for row in range(rank, num_rows):
        if aug[row][num_cols] != 0:
            return None  # Inconsistent
    
    # Extract solution (take first solution, free variables = 0)
    solution = [Fraction(0)] * num_cols
    for i in range(rank):
        # Find which column this row corresponds to
        for col in range(num_cols):
            if aug[i][col] != 0:
                solution[col] = aug[i][num_cols]
                break
    
    # Verify the solution
    U = [[Fraction(0)] * num_blocks for _ in range(n*m)]
    V = [[Fraction(0)] * num_blocks for _ in range(m*p)]
    W = [[Fraction(0)] * num_blocks for _ in range(p*n)]
    
    for r in range(num_blocks):
        u, v, w = selected[r]
        for a in range(n*m):
            U[a][r] = u[a]
        for b in range(m*p):
            V[b][r] = v[b]
        for c in range(p*n):
            W[c][r] = w[c]
    
    # Scale by coefficients
    for r in range(num_blocks):
        coeff = solution[r]
        if coeff != 0:
            for a in range(n*m):
                U[a][r] = U[a][r] * coeff
    
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

    # Strategy 1: Tensor slicing + greedy set-cover
    if time.time() < deadline - 10:
        result = _try_tensor_slicing_set_cover(n, m, p, T, deadline - 5)
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            if new_rank < best_rank:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_rank
                print(f"Tensor slicing found rank {best_rank}!")

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
