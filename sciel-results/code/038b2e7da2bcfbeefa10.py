"""Matrix multiplication 3x5x7 - rank-minimal tensor decomposition search."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Known optimal ranks for small block sizes:
    # M(2,2,2) = 7 (Strassen)
    # M(2,2,3) = 11
    # M(2,3,3) = 12 (known result)
    # M(1,*,*) or M(*,1,*) = product of other dims
    # Schoolbook: a*b*c
    
    def get_block_rank(a, b, c):
        """Get the best known rank for a x b x c matrix multiplication."""
        # If any dimension is 1, it's just the product of the other two
        if a == 1:
            return b * c
        if b == 1:
            return a * c
        if c == 1:
            return a * b
        
        # Known special cases
        if (a, b, c) == (2, 2, 2):
            return 7
        if (a, b, c) == (2, 2, 3) or (a, b, c) == (2, 3, 2) or (a, b, c) == (3, 2, 2):
            return 11
        if (a, b, c) == (2, 3, 3) or (a, b, c) == (3, 2, 3) or (a, b, c) == (3, 3, 2):
            return 12
        if (a, b, c) == (3, 3, 3):
            return 23  # Best known
        
        # Try to decompose using Strassen-like approaches
        # For 2x2xN: rank is 7 + 2*(N-2) for N >= 2
        if a == 2 and b == 2:
            if c >= 2:
                return 7 + 2 * (c - 2)
            return a * b * c
        if a == 2 and c == 2:
            if b >= 2:
                return 7 + 2 * (b - 2)
            return a * b * c
        if b == 2 and c == 2:
            if a >= 2:
                return 7 + 2 * (a - 2)
            return a * b * c
        
        # For 2xNxM where N,M >= 3: try to use 2x2x2 Strassen on pairs
        # A(2,N,M) <= A(2,2,2) + (N-2)*(M) + 2*(M) ... complex
        # Use a simple bound: split into 2x2xM blocks
        if a == 2 and b >= 2 and c >= 2:
            # Use Strassen on 2x2 part, schoolbook on rest
            # A(2,b,c) <= A(2,2,c) + (b-2)*c
            return get_block_rank(2, 2, c) + (b - 2) * c if b >= 2 else a*b*c
        
        # General fallback: use schoolbook or try simple splits
        best = a * b * c
        
        # Try splitting one dimension
        for i in range(1, a):
            r1 = get_block_rank(i, b, c)
            r2 = get_block_rank(a - i, b, c)
            best = min(best, r1 + r2)
        for j in range(1, b):
            r1 = get_block_rank(a, j, c)
            r2 = get_block_rank(a, b - j, c)
            best = min(best, r1 + r2)
        for k in range(1, c):
            r1 = get_block_rank(a, b, k)
            r2 = get_block_rank(a, b, c - k)
            best = min(best, r1 + r2)
            
        return best
    
    # Now search over all possible splits of n, m, p
    # A split of size S into parts means we partition the indices
    # For n=3: splits are [[3]], [[1,2]], [[2,1]], [[1,1,1]]
    # For m=5: splits are [[5]], [[1,4]], [[2,3]], [[3,2]], [[4,1]], [[1,1,3]], [[1,3,1]], [[3,1,1]], [[1,2,2]], [[2,1,2]], [[2,2,1]], [[1,1,1,2]], etc.
    
    def get_all_splits(S):
        """Get all compositions of S."""
        if S == 1:
            return [[1]]
        splits = []
        # Use recursive approach
        for first in range(1, S + 1):
            if first == S:
                splits.append([first])
            else:
                for rest in get_all_splits(S - first):
                    splits.append([first] + rest)
        return splits
    
    # Precompute all splits
    n_splits = get_all_splits(n)
    m_splits = get_all_splits(m)
    p_splits = get_all_splits(p)
    
    # Find the best total rank
    best_rank = n * m * p  # schoolbook fallback
    best_splits = (n_splits[0], m_splits[0], p_splits[0])
    
    # Search for minimum rank combination
    count = 0
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                if time.time() > deadline - 10:
                    break
                total_rank = 0
                valid = True
                for ni in ns:
                    for mj in ms:
                        for pk in ps:
                            r = get_block_rank(ni, mj, pk)
                            total_rank += r
                            if total_rank >= best_rank:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid and total_rank < best_rank:
                    best_rank = total_rank
                    best_splits = (ns, ms, ps)
                count += 1
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Build the decomposition based on best_splits
    ns, ms, ps = best_splits
    
    # Now we need to actually construct the decomposition
    # This requires knowing the explicit tensor decomposition for each block
    # For simplicity, if we can't construct explicit decompositions, use schoolbook
    # But for 2x2x2, we have Strassen. For 2x3x3 with rank 12, we need explicit terms.
    
    # Let's check if we can actually construct the decomposition
    # If not, fall back to a constructible decomposition
    
    def can_construct_block(a, b, c):
        """Check if we have an explicit decomposition for this block size."""
        if a == 1 or b == 1 or c == 1:
            return True
        if (a, b, c) == (2, 2, 2):
            return True
        # For other blocks, we'd need explicit decompositions
        # We can construct schoolbook always
        return True  # We'll use schoolbook for non-2x2x2 blocks
    
    # Since constructing explicit rank-12 for 2x3x3 is complex,
    # let's use a pragmatic approach: use Strassen where possible, schoolbook elsewhere
    # and pick splits that minimize rank with constructible blocks
    
    def get_constructible_rank(a, b, c):
        """Get rank for blocks we can explicitly construct."""
        if a == 1:
            return b * c
        if b == 1:
            return a * c
        if c == 1:
            return a * b
        if (a, b, c) == (2, 2, 2):
            return 7
        # For 2x2xN: use Strassen + schoolbook
        if a == 2 and b == 2 and c >= 2:
            return 7 + 2 * (c - 2)
        if a == 2 and c == 2 and b >= 2:
            return 7 + 2 * (b - 2)
        if b == 2 and c == 2 and a >= 2:
            return 7 + 2 * (a - 2)
        # Schoolbook
        return a * b * c
    
    # Search with constructible ranks
    best_rank2 = n * m * p
    best_splits2 = (n_splits[0], m_splits[0], p_splits[0])
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                if time.time() > deadline - 10:
                    break
                total_rank = 0
                valid = True
                for ni in ns:
                    for mj in ms:
                        for pk in ps:
                            r = get_constructible_rank(ni, mj, pk)
                            total_rank += r
                            if total_rank >= best_rank2:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid and total_rank < best_rank2:
                    best_rank2 = total_rank
                    best_splits2 = (ns, ms, ps)
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    ns, ms, ps = best_splits2
    
    # Build explicit decomposition
    terms = []  # Each term is (u_dict, v_dict, w_dict)
    
    def add_term(u_coeff, v_coeff, w_coeff):
        terms.append((u_coeff, v_coeff, w_coeff))
    
    def strassen_222_terms(n_rows, m_cols, p_rows):
        """Add Strassen terms for 2x2x2 block."""
        nr = list(n_rows)
        mc = list(m_cols)
        pr = list(p_rows)
        
        # p1 = (a00+a11)(b00+b11)
        u = {(nr[0], mc[0]): 1, (nr[1], mc[1]): 1}
        v = {(mc[0], pr[0]): 1, (mc[1], pr[1]): 1}
        w = {(pr[0], nr[0]): 1, (pr[1], nr[1]): 1}
        add_term(u, v, w)
        
        # p2 = (a10+a11)b00
        u = {(nr[1], mc[0]): 1, (nr[1], mc[1]): 1}
        v = {(mc[0], pr[0]): 1}
        w = {(pr[0], nr[1]): 1}
        add_term(u, v, w)
        
        # p3 = a00(b01-b11)
        u = {(nr[0], mc[0]): 1}
        v = {(mc[0], pr[1]): 1, (mc[1], pr[1]): -1}
        w = {(pr[1], nr[0]): 1, (pr[1], nr[1]): 1}
        add_term(u, v, w)
        
        # p4 = a11(b10-b00)
        u = {(nr[1], mc[1]): 1}
        v = {(mc[1], pr[0]): 1, (mc[0], pr[0]): -1}
        w = {(pr[0], nr[0]): -1, (pr[0], nr[1]): 1}
        add_term(u, v, w)
        
        # p5 = (a00+a01)b11
        u = {(nr[0], mc[0]): 1, (nr[0], mc[1]): 1}
        v = {(mc[1], pr[1]): 1}
        w = {(pr[0], nr[0]): 1, (pr[1], nr[0]): 1}
        add_term(u, v, w)
        
        # p6 = (a10-a11)(b01+b10)
        u = {(nr[1], mc[0]): 1, (nr[1], mc[1]): -1}
        v = {(mc[0], pr[1]): 1, (mc[1], pr[0]): 1}
        w = {(pr[1], nr[1]): 1}
        add_term(u, v, w)
        
        # p7 = (a01-a11)(b00+b11)
        u = {(nr[0], mc[1]): 1, (nr[1], mc[1]): -1}
        v = {(mc[0], pr[0]): 1, (mc[1], pr[1]): 1}
        w = {(pr[0], nr[0]): -1}
        add_term(u, v, w)
    
    def schoolbook_terms(n_rows, m_cols, p_rows):
        """Add schoolbook terms for a block."""
        for i in n_rows:
            for j in m_cols:
                for k in p_rows:
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    add_term(u, v, w)
    
    def strassen_22c_terms(n_rows, m_cols, p_rows):
        """Add terms for 2x2xC block: Strassen for first 2 p-rows, schoolbook for rest."""
        pr = list(p_rows)
        if len(pr) <= 2:
            strassen_222_terms(n_rows, m_cols, pr)
            return
        # Use Strassen on 2x2x2 part
        strassen_222_terms(n_rows, m_cols, pr[:2])
        # Schoolbook on remaining
        for k_idx in range(2, len(pr)):
            for i in n_rows:
                for j in m_cols:
                    u = {(i, j): 1}
                    v = {(j, pr[k_idx]): 1}
                    w = {(pr[k_idx], i): 1}
                    add_term(u, v, w)
    
    def handle_block(n_rows, m_cols, p_rows):
        """Handle a block with optimal constructible decomposition."""
        a, b, c = len(n_rows), len(m_cols), len(p_rows)
        
        if a == 1:
            schoolbook_terms(n_rows, m_cols, p_rows)
            return
        if b == 1:
            schoolbook_terms(n_rows, m_cols, p_rows)
            return
        if c == 1:
            schoolbook_terms(n_rows, m_cols, p_rows)
            return
        
        if a == 2 and b == 2 and c == 2:
            strassen_222_terms(n_rows, m_cols, p_rows)
        elif a == 2 and b == 2:
            strassen_22c_terms(n_rows, m_cols, p_rows)
        elif a == 2 and c == 2:
            # Use Strassen-like: treat as 2x2x2 on m,p then extend
            # Actually, for 2xNx2, we can use: for each pair of m columns, use Strassen
            # This is complex; use schoolbook for safety
            schoolbook_terms(n_rows, m_cols, p_rows)
        elif b == 2 and c == 2:
            schoolbook_terms(n_rows, m_cols, p_rows)
        else:
            schoolbook_terms(n_rows, m_cols, p_rows)
    
    # Build index ranges from splits
    def make_index_range(sizes, total):
        """Create list of (indices, size) from split sizes."""
        result = []
        start = 0
        for s in sizes:
            result.append((list(range(start, start + s)), s))
            start += s
        return result
    
    n_parts = make_index_range(ns, n)
    m_parts = make_index_range(ms, m)
    p_parts = make_index_range(ps, p)
    
    for n_rows, n_size in n_parts:
        for m_cols, m_size in m_parts:
            for p_rows, p_size in p_parts:
                handle_block(n_rows, m_cols, p_rows)
    
    rank = len(terms)
    
    # Build factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_coeff, v_coeff, w_coeff) in enumerate(terms):
        for (i, j), coeff in u_coeff.items():
            U[i * m + j, c] = coeff
        for (j, k), coeff in v_coeff.items():
            V[j * p + k, c] = coeff
        for (k, i), coeff in w_coeff.items():
            W[k * n + i, c] = coeff
    
    # Verify correctness
    if time.time() < deadline:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
