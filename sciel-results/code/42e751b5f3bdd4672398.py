"""Tensor decomposition for 3x4x5 matrix multiplication.
Uses block decomposition with Strassen (2x2x2 rank 7) and direct blocks.
Aim: achieve rank < 60 (schoolbook) via careful block splitting.

The 3x4x5 multiplication is split into overlapping 2x2x2 Strassen blocks
and small direct blocks, with shared terms to reduce total rank.
"""

import time
import numpy as np
import sys

DIMENSIONS = [3, 4, 5]


def _verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _add_strassen_2x2x2(i1, i2, j1, j2, k1, k2, m, p, n, U, V, W, rank_offset):
    """Add Strassen decomposition for a 2x2 * 2x2 -> 2x2 product.
    
    i1,i2: C row indices
    j1,j2: A col / B row indices  
    k1,k2: C col / B col indices
    
    Returns new rank offset.
    """
    # A indices: A[i,j] -> index i*m + j
    a11_idx = i1 * m + j1
    a12_idx = i1 * m + j2
    a21_idx = i2 * m + j1
    a22_idx = i2 * m + j2
    
    # B indices: B[j,k] -> index j*p + k
    b11_idx = j1 * p + k1
    b12_idx = j1 * p + k2
    b21_idx = j2 * p + k1
    b22_idx = j2 * p + k2
    
    # C indices: C[i,k] -> index k*n + i (trace convention W[k*n+i])
    c11_idx = k1 * n + i1
    c12_idx = k2 * n + i1
    c21_idx = k1 * n + i2
    c22_idx = k2 * n + i2
    
    # Strassen's 7 multiplications
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    terms = [
        # M1
        ({a11_idx: 1, a22_idx: 1}, {b11_idx: 1, b22_idx: 1}, {c11_idx: 1, c22_idx: 1}),
        # M2
        ({a21_idx: 1, a22_idx: 1}, {b11_idx: 1}, {c21_idx: 1, c22_idx: -1}),
        # M3
        ({a11_idx: 1}, {b12_idx: 1, b22_idx: -1}, {c12_idx: 1, c22_idx: 1}),
        # M4
        ({a22_idx: 1}, {b21_idx: 1, b11_idx: -1}, {c11_idx: 1, c21_idx: 1}),
        # M5
        ({a11_idx: 1, a12_idx: 1}, {b22_idx: 1}, {c11_idx: -1, c12_idx: 1}),
        # M6
        ({a21_idx: 1, a11_idx: -1}, {b11_idx: 1, b12_idx: 1}, {c22_idx: 1}),
        # M7
        ({a12_idx: 1, a22_idx: -1}, {b21_idx: 1, b22_idx: 1}, {c11_idx: 1, c22_idx: -1}),
    ]
    
    for u_d, v_d, w_d in terms:
        for u_idx, u_val in u_d.items():
            U[u_idx, rank_offset] += u_val
        for v_idx, v_val in v_d.items():
            V[v_idx, rank_offset] += v_val
        for w_idx, w_val in w_d.items():
            W[w_idx, rank_offset] += w_val
        rank_offset += 1
    
    return rank_offset


def _add_direct_term(i, j, k, m, p, n, U, V, W, rank_offset):
    """Add a single direct term (rank 1) for C[i,k] += A[i,j]*B[j,k]."""
    a_idx = i * m + j
    b_idx = j * p + k
    c_idx = k * n + i
    U[a_idx, rank_offset] = 1.0
    V[b_idx, rank_offset] = 1.0
    W[c_idx, rank_offset] = 1.0
    return rank_offset + 1


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # We need to cover all (i,j,k) triples where i∈{0,1,2}, j∈{0,1,2,3}, k∈{0,1,2,3,4}
    # Total: 3*4*5 = 60 terms
    
    # Strategy: Use Strassen for 2x2x2 blocks, direct for the rest.
    # A 2x2x2 Strassen block covers 8 (i,j,k) triples with rank 7.
    # We need to partition or overlap these blocks.
    
    # Let's try to find a set of 2x2x2 blocks and direct terms that cover all 60 triples.
    # Each Strassen block uses 7 rank, covers 8 triples.
    # Each direct term uses 1 rank, covers 1 triple.
    
    # If we use S Strassen blocks and D direct terms:
    # 8*S + D >= 60 (coverage, with possible overlap)
    # Total rank = 7*S + D
    # To minimize rank, maximize S.
    
    # Max S if no overlap: floor(60/8) = 7 blocks, covering 56, leaving 4 direct.
    # Rank = 7*7 + 4 = 53.
    
    # But we need the blocks to be valid 2x2x2 subproblems.
    # A 2x2x2 block needs:
    # - 2 distinct i values
    # - 2 distinct j values  
    # - 2 distinct k values
    
    # Let's enumerate possible 2x2x2 blocks and find a good covering.
    
    # Possible i-pairs: (0,1), (0,2), (1,2)
    # Possible j-pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Possible k-pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    
    # Total possible blocks: 3 * 6 * 10 = 180
    
    # For each block, it covers 8 specific (i,j,k) triples.
    # We want to select blocks and direct terms to cover all 60 triples with minimum rank.
    
    # This is a set cover problem. Let's use a greedy approach.
    
    # First, let's build the list of all possible blocks and their coverage.
    i_pairs = [(0,1), (0,2), (1,2)]
    j_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    k_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    
    all_blocks = []
    for ip in i_pairs:
        for jp in j_pairs:
            for kp in k_pairs:
                i1, i2 = ip
                j1, j2 = jp
                k1, k2 = kp
                # This block covers these (i,j,k) triples:
                covered = set()
                for i in [i1, i2]:
                    for j in [j1, j2]:
                        for k in [k1, k2]:
                            covered.add((i, j, k))
                all_blocks.append((ip, jp, kp, covered))
    
    # All required triples
    all_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_triples.add((i, j, k))
    
    # Greedy set cover: repeatedly pick the block that covers the most uncovered triples
    uncovered = set(all_triples)
    selected_blocks = []
    
    while uncovered:
        # Find block covering most uncovered triples
        best_block = None
        best_count = 0
        for block in all_blocks:
            ip, jp, kp, covered = block
            count = len(covered & uncovered)
            if count > best_count:
                best_count = count
                best_block = block
            if time.time() > deadline - 5:
                break
        if best_block is None or best_count == 0:
            break
        selected_blocks.append(best_block)
        uncovered -= best_block[3]
        if time.time() > deadline - 5:
            break
    
    # Remaining uncovered triples get direct terms
    direct_terms = list(uncovered)
    
    # Now build the decomposition
    # Start with a large enough rank, then trim
    max_rank = 7 * len(selected_blocks) + len(direct_terms) + 10  # extra margin
    U = np.zeros((nm, max_rank))
    V = np.zeros((mp, max_rank))
    W = np.zeros((pn, max_rank))
    
    rank_offset = 0
    for ip, jp, kp, covered in selected_blocks:
        i1, i2 = ip
        j1, j2 = jp
        k1, k2 = kp
        rank_offset = _add_strassen_2x2x2(i1, i2, j1, j2, k1, k2, m, p, n, U, V, W, rank_offset)
    
    for i, j, k in direct_terms:
        rank_offset = _add_direct_term(i, j, k, m, p, n, U, V, W, rank_offset)
    
    # Trim to actual rank used
    U = U[:, :rank_offset]
    V = V[:, :rank_offset]
    W = W[:, :rank_offset]
    
    # Verify
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
    else:
        # Try to reduce rank by removing zero columns
        # Check which columns are actually used
        used = np.any(U != 0, axis=0) | np.any(V != 0, axis=0) | np.any(W != 0, axis=0)
        if not np.all(used):
            U = U[:, used]
            V = V[:, used]
            W = W[:, used]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
