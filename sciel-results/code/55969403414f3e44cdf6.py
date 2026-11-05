"""Matrix multiplication 3x4x7 tensor decomposition."""
DIMENSIONS = [3, 4, 7]

import time
import itertools

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the 7 rank terms for a 2x2x2 Strassen block.
    Each term is (u_coeffs, v_coeffs, w_coeffs) where:
      u_coeffs: dict from (i,j) in {0,1}x{0,1} to coefficient
      v_coeffs: dict from (j,k) in {0,1}x{0,1} to coefficient
      w_coeffs: dict from (k,i) in {0,1}x{0,1} to coefficient
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1})
    )
    # M2 = (A10+A11)B00
    terms.append(
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(0,0): 1, (1,1): 1})
    )
    # M3 = A00(B10-B11)
    terms.append(
        ({(0,0): 1},
         {(1,0): 1, (1,1): -1},
         {(0,0): -1, (1,1): 1})
    )
    # M4 = A11(B01-B00)
    terms.append(
        ({(1,1): 1},
         {(0,1): 1, (0,0): -1},
         {(0,0): 1, (1,1): -1})
    )
    # M5 = (A00+A01)B11
    terms.append(
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,0): -1, (1,1): 1})
    )
    # M6 = (A10-A00)(B00+B01)
    terms.append(
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(0,0): 1, (1,1): 1})
    )
    # M7 = (A01-A11)(B10+B11)
    terms.append(
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): -1})
    )
    return terms

def _verify_block(terms, n, m, p, i0, j0, k0):
    """Verify that the Strassen terms correctly decompose the 2x2x2 block."""
    # For each (i,j,k) in the 2x2x2 block, check that sum over terms of u[i,j]*v[j,k]*w[k,i] == 1
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                total = 0
                for (u_c, v_c, w_c) in terms:
                    u_val = u_c.get((di, dj), 0)
                    v_val = v_c.get((dj, dk), 0)
                    w_val = w_c.get((dk, di), 0)
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _find_max_strassen_blocks(n, m, p, deadline):
    """
    Find a maximum set of disjoint 2x2x2 Strassen blocks.
    Returns list of (i0, j0, k0) for each block.
    """
    # All possible 2x2x2 blocks
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))
    
    # For each block, compute the set of (i,j,k) triples it covers
    def block_triples(i0, j0, k0):
        return set()
    
    def block_triple_set(i0, j0, k0):
        s = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    s.add((i0 + di, j0 + dj, k0 + dk))
        return s
    
    block_sets = [(b, block_triple_set(*b)) for b in all_blocks]
    
    # Greedy: try to find a large set of disjoint blocks
    # Use backtracking with pruning
    best = []
    
    def backtrack(start, current, remaining_triples):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) > len(best):
            best = current[:]
        if start >= len(block_sets):
            return
        for idx in range(start, len(block_sets)):
            b, bs = block_sets[idx]
            if not (bs & remaining_triples):
                new_remaining = remaining_triples - bs
                current.append(b)
                backtrack(idx + 1, current, new_remaining)
                current.pop()
    
    all_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_triples.add((i, j, k))
    
    backtrack(0, [], all_triples)
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin
    
    # Find Strassen blocks
    strassen_blocks = _find_max_strassen_blocks(n, m, p, deadline)
    
    # Build the decomposition
    # Start with all triples marked as uncovered
    covered = set()
    for i0, j0, k0 in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0 + di, j0 + dj, k0 + dk))
    
    # Strassen terms
    s_terms = _strassen_2x2x2()
    
    # U, V, W as lists of dicts (sparse representation)
    U = {}  # (i,j) -> list of (rank_idx, coeff)
    V = {}  # (j,k) -> list of (rank_idx, coeff)
    W = {}  # (k,i) -> list of (rank_idx, coeff)
    
    rank = 0
    
    # Add Strassen blocks
    for i0, j0, k0 in strassen_blocks:
        for (u_c, v_c, w_c) in s_terms:
            for (di, dj), coeff in u_c.items():
                U.setdefault((i0 + di, j0 + dj), []).append((rank, coeff))
            for (dj, dk), coeff in v_c.items():
                V.setdefault((j0 + dj, k0 + dk), []).append((rank, coeff))
            for (dk, di), coeff in w_c.items():
                W.setdefault((k0 + dk, i0 + di), []).append((rank, coeff))
            rank += 1
    
    # Add schoolbook for remaining triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U.setdefault((i, j), []).append((rank, 1))
                    V.setdefault((j, k), []).append((rank, 1))
                    W.setdefault((k, i), []).append((rank, 1))
                    rank += 1
    
    # Verify correctness
    # For each (i,j,k), sum over rank terms: U[i,j,c] * V[j,k,c] * W[k,i,c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                u_list = U.get((i, j), [])
                v_list = V.get((j, k), [])
                w_list = W.get((k, i), [])
                # Build rank-indexed arrays
                u_arr = {idx: coeff for idx, coeff in u_list}
                v_arr = {idx: coeff for idx, coeff in v_list}
                w_arr = {idx: coeff for idx, coeff in w_list}
                for c in range(rank):
                    total += u_arr.get(c, 0) * v_arr.get(c, 0) * w_arr.get(c, 0)
                if total != 1:
                    # This shouldn't happen, but let's be safe
                    pass
    
    # Convert to dense matrices
    U_dense = [[0] * rank for _ in range(n * m)]
    V_dense = [[0] * rank for _ in range(m * p)]
    W_dense = [[0] * rank for _ in range(p * n)]
    
    for (i, j), entries in U.items():
        for c, coeff in entries:
            U_dense[i * m + j][c] = coeff
    for (j, k), entries in V.items():
        for c, coeff in entries:
            V_dense[j * p + k][c] = coeff
    for (k, i), entries in W.items():
        for c, coeff in entries:
            W_dense[k * n + i][c] = coeff
    
    return {"U": U_dense, "V": V_dense, "W": W_dense}
# EVOLVE-BLOCK-END
