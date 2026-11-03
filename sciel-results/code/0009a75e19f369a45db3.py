import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

def _strassen_2x2_terms(i0, j0, k0, m, p, n):
    """Generate the 7 rank-1 terms for a 2x2x2 Strassen block."""
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    a00, a01 = i0*m+j0, i0*m+j1
    a10, a11 = i1*m+j0, i1*m+j1
    b00, b01 = j0*p+k0, j0*p+k1
    b10, b11 = j1*p+k0, j1*p+k1
    c00, c01 = k0*n+i0, k1*n+i0
    c10, c11 = k0*n+i1, k1*n+i1
    
    terms = [
        ({a00:1, a11:1}, {b00:1, b11:1}, {c00:1, c11:1}),
        ({a10:1, a11:1}, {b00:1}, {c10:1, c11:-1}),
        ({a00:1}, {b01:1, b11:-1}, {c01:1, c11:1}),
        ({a11:1}, {b10:1, b00:-1}, {c00:1, c10:1}),
        ({a00:1, a01:1}, {b11:1}, {c00:-1, c01:1}),
        ({a10:1, a00:-1}, {b00:1, b01:1}, {c11:1}),
        ({a01:1, a11:-1}, {b10:1, b11:1}, {c00:1}),
    ]
    return terms

def _strassen_3x3_terms(i0, j0, k0, m, p, n):
    """
    Generate a rank-19 decomposition for a 3x3x3 block.
    Uses a known rank-19 decomposition for 3x3 matrix multiplication.
    """
    i1, i2 = i0+1, i0+2
    j1, j2 = j0+1, j0+2
    k1, k2 = k0+1, k0+2
    
    # Index helpers
    def A(i,j): return i*m + j
    def B(j,k): return j*p + k
    def C(k,i): return k*n + i
    
    # We use a structured decomposition based on splitting into
    # a 2x2x2 Strassen core (on indices 0,1) plus corrections for index 2.
    # This gives rank 7 + 12 = 19 terms.
    
    terms = []
    
    # Core 2x2x2 Strassen on (i0,i1) x (j0,j1) x (k0,k1)
    core_terms = _strassen_2x2_terms(i0, j0, k0, m, p, n)
    for u, v, w in core_terms:
        terms.append((u, v, w))
    
    # Now handle all terms involving index 2 in any dimension.
    # There are 27 - 8 = 19 such terms, but we need to be careful
    # not to double-count the 8 core terms.
    # 
    # Strategy: For each pair of "outer" indices, use rank-1 terms
    # that cover the remaining structure.
    
    # Terms with i=i2, j in {j0,j1}, k in {k0,k1}: 4 terms
    # Terms with i in {i0,i1}, j=j2, k in {k0,k1}: 4 terms  
    # Terms with i in {i0,i1}, j in {j0,j1}, k=k2: 4 terms
    # Terms with i=i2, j=j2, k in {k0,k1}: 2 terms
    # Terms with i=i2, j in {j0,j1}, k=k2: 2 terms
    # Terms with i in {i0,i1}, j=j2, k=k2: 2 terms
    # Term with i=i2, j=j2, k=k2: 1 term
    # Total: 4+4+4+2+2+2+1 = 19... but we need rank-19 total,
    # and we already have 7 from the core. So we need 12 more.
    # 
    # Let's use a different approach: group the 19 non-core terms
    # into 12 rank-1 terms using a clever factorization.
    
    # Actually, let me use a known approach: the 3x3x3 tensor can be
    # decomposed with rank 19. One construction:
    # - 7 terms for the 2x2x2 core (done)
    # - For the remaining 19 terms, use the fact that each "slab"
    #   (fixing one index to 2) is a 2x2 or 2x3 or 3x2 matrix product.
    
    # Slab i=i2: A[i2,j] * B[j,k] for j in {0,1,2}, k in {0,1,2}
    #   This is a 1x3 * 3x3 = 1x3 product, rank at most 3.
    #   But we can do better by sharing with other slabs.
    
    # Let me just use a direct rank-19 construction:
    # Split into:
    #   7 terms: 2x2x2 core (i0,i1)x(j0,j1)x(k0,k1)
    #   3 terms: i2 x (j0,j1) x (k0,k1) -> 3 rank-1 terms for 2x2x1
    #   3 terms: (i0,i1) x j2 x (k0,k1) -> 3 rank-1 terms for 2x1x2
    #   3 terms: (i0,i1) x (j0,j1) x k2 -> 3 rank-1 terms for 2x2x1
    #   That's 7+3+3+3 = 16, covering all terms where at most one index is 2.
    #   Remaining: terms with two or three indices equal to 2:
    #     i2,j2,k0; i2,j2,k1; i2,j0,k2; i2,j1,k2; i0,j2,k2; i1,j2,k2; i2,j2,k2
    #     That's 7 more terms. We need 3 more rank-1 terms to cover these 7.
    #     Use: {i2,j2,k0},{i2,j2,k1} -> 1 term
    #          {i2,j0,k2},{i2,j1,k2} -> 1 term  
    #          {i0,j2,k2},{i1,j2,k2},{i2,j2,k2} -> 1 term
    #     Total: 7+3+3+3+3 = 19. 
    #     But wait, we need to check that the 3-term groups are valid rank-1.
    
    # Let me just be explicit and careful:
    
    # Group 1: i=i2, j in {j0,j1}, k in {k0,k1} (4 terms)
    # Use 2 rank-1 terms:
    #   T1: U={A(i2,j0):1}, V={B(j0,k0):1, B(j1,k0):1}, W={C(k0,i2):1}
    #   T2: U={A(i2,j0):1, A(i2,j1):1}, V={B(j1,k1):1}, W={C(k1,i2):1}
    #   Check: T1 gives (i2,j0,k0) and (i2,j1,k0). T2 gives (i2,j0,k1) and (i2,j1,k1).
    #   But we need all 4: (i2,j0,k0),(i2,j0,k1),(i2,j1,k0),(i2,j1,k1).
    #   T1 covers (i2,j0,k0) and (i2,j1,k0). T2 covers (i2,j0,k1) and (i2,j1,k1). 
    #   That works! 2 terms for 4 entries.
    
    terms.append(({A(i2,j0):1}, {B(j0,k0):1, B(j1,k0):1}, {C(k0,i2):1}))
    terms.append(({A(i2,j0):1, A(i2,j1):1}, {B(j1,k1):1}, {C(k1,i2):1}))
    
    # Group 2: i in {i0,i1}, j=j2, k in {k0,k1} (4 terms)
    # Use 2 rank-1 terms:
    terms.append(({A(i0,j2):1}, {B(j2,k0):1, B(j2,k1):1}, {C(k0,i0):1, C(k1,i0):1}))
    # Wait, that's not right. Let me think more carefully.
    # We need: (i0,j2,k0), (i0,j2,k1), (i1,j2,k0), (i1,j2,k1)
    # T1: U={A(i0,j2):1}, V={B(j2,k0):1}, W={C(k0,i0):1} -> covers (i0,j2,k0)
    # T2: U={A(i0,j2):1}, V={B(j2,k1):1}, W={C(k1,i0):1} -> covers (i0,j2,k1)
    # T3: U={A(i1,j2):1}, V={B(j2,k0):1}, W={C(k0,i1):1} -> covers (i1,j2,k0)
    # T4: U={A(i1,j2):1}, V={B(j2,k1):1}, W={C(k1,i1):1} -> covers (i1,j2,k1)
    # That's 4 terms, not 2. Let me use a better grouping.
    # Actually for a 2x1x2 tensor, the rank is at most 2.
    # T1: U={A(i0,j2):1, A(i1,j2):1}, V={B(j2,k0):1}, W={C(k0,i0):1, C(k0,i1):1}
    #   This gives: (i0,j2,k0) with coeff 1*1*1=1, (i1,j2,k0) with coeff 1*1*1=1. Good.
    # T2: U={A(i0,j2):1, A(i1,j2):1}, V={B(j2,k1):1}, W={C(k1,i0):1, C(k1,i1):1}
    #   This gives: (i0,j2,k1) and (i1,j2,k1). Good.
    # But wait, W has two entries, so this is valid.
    # However, this might conflict with other terms. Let me just use 4 simple terms
    # and accept rank 21 if needed, or find a better decomposition.
    
    # Let me restart with a cleaner approach. I'll use a known rank-19
    # decomposition from the literature. The key idea is to use:
    # - 7 terms for the 2x2x2 sub-block
    # - 12 terms for the remaining 19 entries
    # 
    # For the remaining entries, I'll use a systematic approach:
    # Each "line" of the tensor (fixing 2 indices) can be covered efficiently.
    
    # Let me just use a simple but valid decomposition with rank up to 27
    # and then optimize. Actually, for this problem size, let me just
    # use schoolbook for the 3x3x3 case and focus on 2x2x2 blocks.
    # 
    # Revised strategy: Only use 2x2x2 Strassen blocks, but use a
    # smarter selection algorithm (greedy set cover with scoring).
    
    return None  # Fall back to 2x2x2 only

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Generate all 2x2x2 blocks
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
                entries = frozenset((i, j, k) for i in (i0, i1) for j in (j0, j1) for k in (k0, k1))
                all_blocks.append((i0, i1, j0, j1, k0, k1, entries))
    
    num_blocks = len(all_blocks)
    
    # Build conflict sets
    conflict_sets = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][6] & all_blocks[j][6]:
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)
    
    # Greedy set cover: repeatedly pick the block that covers the most
    # uncovered terms per unit rank cost (7 rank for 8 terms = 0.875)
    all_terms = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_terms.add((i, j, k))
    
    uncovered = set(all_terms)
    selected = []
    available = set(range(num_blocks))
    
    while available and time.time() < deadline - 20:
        best_block = None
        best_score = 0
        for b in available:
            new_cover = len(all_blocks[b][6] & uncovered)
            if new_cover > 0:
                # Score: number of new terms covered
                score = new_cover
                if score > best_score or (score == best_score and best_block is not None and b < best_block):
                    best_score = score
                    best_block = b
        
        if best_block is None:
            break
        
        selected.append(best_block)
        uncovered -= all_blocks[best_block][6]
        
        # Remove conflicting blocks
        to_remove = set()
        for b in available:
            if best_block in conflict_sets[b]:
                to_remove.add(b)
        to_remove.add(best_block)
        available -= to_remove
    
    # Build decomposition
    covered = set()
    U_cols = []
    V_cols = []
    W_cols = []
    
    for idx in selected:
        i0, i1, j0, j1, k0, k1, entries = all_blocks[idx]
        covered |= entries
        terms = _strassen_2x2_terms(i0, j0, k0, m, p, n)
        for u, v, w in terms:
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    # Add schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_cols.append({i * m + j: 1})
                    V_cols.append({j * p + k: 1})
                    W_cols.append({k * n + i: 1})
    
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val
    
    # Verify
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)
    
    if error > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
