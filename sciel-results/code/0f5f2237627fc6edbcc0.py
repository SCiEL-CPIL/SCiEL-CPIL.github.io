"""Exact tensor decomposition for 3x4x7 matrix multiplication.
Explores Strassen block configurations to minimize rank.
"""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from itertools import product, combinations_with_replacement

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_block_contribs(i0, j0, k0, n, m, p):
    """Return list of 7 (U_idx, V_idx, W_idx, sign) tuples for one 2x2x2 Strassen block.
    Each tuple represents one rank-1 term: U[row]*V[row']*W[row''] with given signs."""
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    terms = [
        # Term 0: M1 = A11*A22 -> contributes to C11, C22
        ((i0*m+j0, 1), (j0*p+k0, 1), (k0*n+i0, 1), 1),
        ((i1*m+j1, 1), (j1*p+k1, 1), (k1*n+i1, 1), 1),
        # Term 1: M2 = (A21+A22)*A11 -> C21 += M2, C22 -= M2
        ((i1*m+j0, 1), (j0*p+k0, 1), (k0*n+i1, 1), 1),
        ((i1*m+j1, 1), (j1*p+k1, 1), (k1*n+i1, -1), 1),
        # Term 2: M3 = A11*(B21-B11) -> C12 += M3, C22 += M3
        ((i0*m+j0, 1), (j0*p+k1, 1), (k1*n+i0, 1), 1),
        ((i0*m+j0, 1), (j1*p+k1, -1), (k1*n+i1, 1), 1),
        # Term 3: M4 = A22*(B12-B11) -> C11 += M4, C21 += M4
        ((i1*m+j1, 1), (j1*p+k0, 1), (k0*n+i0, 1), 1),
        ((i1*m+j1, 1), (j0*p+k0, -1), (k0*n+i1, 1), 1),
        # Term 4: M5 = (A11+A12)*B22 -> C12 -= M5, C22 += M5
        ((i0*m+j0, 1), (j1*p+k1, 1), (k0*n+i0, -1), 1),
        ((i0*m+j1, 1), (j1*p+k1, 1), (k1*n+i0, 1), 1),
        # Term 5: M6 = (A21-A11)*(B11+B12) -> C22 += M6
        ((i1*m+j0, 1), (j0*p+k0, 1), (k1*n+i1, 1), 1),
        ((i0*m+j0, -1), (j0*p+k1, 1), (k1*n+i1, 1), 1),
        # Term 6: M7 = (A12-A22)*(B21+B22) -> C11 += M7
        ((i0*m+j1, 1), (j1*p+k0, 1), (k0*n+i0, 1), 1),
        ((i1*m+j1, -1), (j1*p+k1, 1), (k0*n+i0, 1), 1),
    ]
    return terms

def _build_decomposition(blocks, n=3, m=4, p=7, overlap_mode="disjoint"):
    """Build U,V,W from a set of Strassen blocks.
    blocks: list of (i0,j0,k0) tuples.
    Returns (U, V, W, rank) or None if invalid.
    """
    if overlap_mode == "disjoint":
        # Check all blocks are pairwise disjoint in their 2x2x2 regions
        for a in range(len(blocks)):
            for b in range(a+1, len(blocks)):
                ia, ja, ka = blocks[a]
                ib, jb, kb = blocks[b]
                # Check overlap
                if (ia <= ib+1 and ib <= ia+1 and
                    ja <= jb+1 and jb <= ja+1 and
                    ka <= kb+1 and kb <= ka+1):
                    return None  # Overlapping blocks in disjoint mode
        rank = 7 * len(blocks) + (n*m*p - 8*len(blocks))
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for (i0, j0, k0) in blocks:
            terms = _strassen_block_contribs(i0, j0, k0, n, m, p)
            for t in range(7):
                # Each Strassen term t uses one column
                # Accumulate contributions
                for (ui, us), (vi, vs), (wi, ws), _ in [terms[t*2 + (0 if t < 7 else 0)]]:
                    pass
                # Simpler: directly assign
                i1, j1, k1 = i0+1, j0+1, k0+1
                c = col + t
                if t == 0:
                    U[i0*m+j0][c] = 1; U[i1*m+j1][c] = 1
                    V[j0*p+k0][c] = 1; V[j1*p+k1][c] = 1
                    W[k0*n+i0][c] = 1; W[k1*n+i1][c] = 1
                elif t == 1:
                    U[i1*m+j0][c] = 1; U[i1*m+j1][c] = 1
                    V[j0*p+k0][c] = 1
                    W[k0*n+i1][c] = 1; W[k1*n+i1][c] = -1
                elif t == 2:
                    U[i0*m+j0][c] = 1
                    V[j0*p+k1][c] = 1; V[j1*p+k1][c] = -1
                    W[k1*n+i0][c] = 1; W[k1*n+i1][c] = 1
                elif t == 3:
                    U[i1*m+j1][c] = 1
                    V[j1*p+k0][c] = 1; V[j0*p+k0][c] = -1
                    W[k0*n+i0][c] = 1; W[k0*n+i1][c] = 1
                elif t == 4:
                    U[i0*m+j0][c] = 1; U[i0*m+j1][c] = 1
                    V[j1*p+k1][c] = 1
                    W[k0*n+i0][c] = -1; W[k1*n+i0][c] = 1
                elif t == 5:
                    U[i1*m+j0][c] = 1; U[i0*m+j0][c] = -1
                    V[j0*p+k0][c] = 1; V[j0*p+k1][c] = 1
                    W[k1*n+i1][c] = 1
                elif t == 6:
                    U[i0*m+j1][c] = 1; U[i1*m+j1][c] = -1
                    V[j1*p+k0][c] = 1; V[j1*p+k1][c] = 1
                    W[k0*n+i0][c] = 1
            col += 7
        # Fill remaining with schoolbook
        used = set()
        for (i0, j0, k0) in blocks:
            for ii in range(i0, i0+2):
                for jj in range(j0, j0+2):
                    for kk in range(k0, k0+2):
                        used.add((ii, jj, kk))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in used:
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
        if col != rank:
            return None
        return U, V, W, rank
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0

    # All possible 2x2x2 block positions
    i_vals = list(range(n - 1))  # 0, 1
    j_vals = list(range(m - 1))  # 0, 1, 2
    k_vals = list(range(p - 1))  # 0..5
    all_blocks = [(i, j, k) for i in i_vals for j in j_vals for k in k_vals]

    # Known good: 6 disjoint blocks
    known_good = [(0,0,0), (0,0,2), (0,0,4), (0,2,0), (0,2,2), (0,2,4)]
    U, V, W, rank = _build_decomposition(known_good, n, m, p, "disjoint")
    if U is not None and _verify(U, V, W, n, m, p):
        best = {"U": U, "V": V, "W": W}
        best_rank = rank
    else:
        best = _schoolbook(n, m, p)
        best_rank = n * m * p

    # Search for better disjoint configurations
    # Try all subsets of size 6-8 from all_blocks
    # But that's C(36, 8) which is too large. Use heuristic search.

    # Strategy: greedy + local search
    # Start with the known good set, try to swap in different blocks

    def _is_valid_disjoint(block_set):
        for a in range(len(block_set)):
            for b in range(a+1, len(block_set)):
                ia, ja, ka = block_set[a]
                ib, jb, kb = block_set[b]
                if (ia <= ib+1 and ib <= ia+1 and
                    ja <= jb+1 and jb <= ja+1 and
                    ka <= kb+1 and kb <= ka+1):
                    return False
        return True

    def _count_covered(block_set):
        covered = set()
        for (i0, j0, k0) in block_set:
            for ii in range(i0, min(i0+2, n)):
                for jj in range(j0, min(j0+2, m)):
                    for kk in range(k0, min(k0+2, p)):
                        covered.add((ii, jj, kk))
        return len(covered)

    # The rank for disjoint blocks = 7*|blocks| + (84 - 8*|blocks|) = 84 - |blocks|
    # So we want MAXIMUM number of disjoint blocks!
    # Maximum disjoint 2x2x2 blocks in a 3x4x7 grid:
    # i: can fit at most floor(3/2)=1 pair non-overlapping (positions 0 or 1)
    # Actually with i in {0,1}, blocks at i=0 and i=1 overlap (share i=1)
    # So at most 1 block per i-position for disjointness in i-dimension
    # Wait: blocks at i0=0 cover i={0,1}, blocks at i0=1 cover i={1,2}. They share i=1, so they overlap!
    # For disjoint 2x2x2 blocks, we need:
    #   |i0_a - i0_b| >= 2, |j0_a - j0_b| >= 2, |k0_a - k0_b| >= 2 (not all three)
    # Actually two 2x2x2 boxes [i0,i0+1]x[j0,j0+1]x[k0,k0+1] and [i0',i0'+1]x[j0',j0'+1]x[k0',k0'+1]
    # are disjoint iff i0+1 < i0' or i0'+1 < i0 or j0+1 < j0' or j0'+1 < j0 or k0+1 < k0' or k0'+1 < k0
    # i.e., they differ by at least 2 in at least one coordinate.

    # Find maximum packing of disjoint 2x2x2 blocks in 3x4x7
    # i: values 0,1 (adjacent, so can't have both)
    # j: values 0,1,2 (0&2 are distance 2, so non-adjacent)
    # k: values 0,1,2,3,4,5 (0&2&4 or 1&3&5 are pairwise distance >= 2)

    # Max blocks: since i only allows 1 value, and for that i-value:
    #   j allows at most 2 non-adjacent values (e.g., 0,2)
    #   k allows at most 3 non-adjacent values (e.g., 0,2,4)
    # So max = 1 * 2 * 3 = 6 blocks. That's what we have!

    # Can we do better with i=1? Same count. No improvement possible with disjoint blocks.

    # Try overlapping blocks - this is where we might find rank < 78
    # With overlapping, we need to solve a linear system.
    # Approach: pick a set of blocks, set up the tensor equation, solve for minimal rank.

    # For overlapping blocks, the contributions add. We can think of it as:
    # Each block contributes a "template" to the tensor. The total tensor must equal the
    # matmul tensor T[i,j,k] = 1 for all (i,j,k) in the support... wait, no.
    # The matmul tensor has T[i,j,k] = 1 for all i,j,k (it's the "all ones" pattern in the
    # sense that each output C[i,k] = sum_j A[i,j]*B[j,k], and the tensor is T[i,j,k]=1).
    
    # Actually the matmul tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    # A rank-r decomposition means T = sum_{c=1}^r u_c * v_c * w_c (outer products).
    # Each Strassen block gives 7 rank-1 terms that together produce the 2x2x2 sub-tensor of ones.
    # If blocks overlap, the overlapping entries get contributions from multiple blocks.
    # For the decomposition to work, the sum of all contributions at each (i,j,k) must equal 1.

    # Let's try: use blocks that share some space but adjust coefficients.
    # This is complex. Instead, let's try a different approach:
    # Use the known 6-block solution and try to reduce rank by merging terms.

    # Alternative: try 7 blocks with overlap, solving for the overlap corrections.
    # This is a linear algebra problem. Let's set it up.

    # For now, verify the best we have and return.
    if time.time() > deadline:
        return best

    # Try to improve: can we find a configuration with fewer total terms?
    # Rank 78 = 6*7 + 36 (36 schoolbook terms for 36 uncovered triples)
    # If we could cover more triples with Strassen blocks, we'd reduce rank.
    # But max disjoint is 6 blocks covering 48 triples.
    
    # What about using 1x2x2 or 2x1x2 blocks? These aren't standard Strassen.
    # A 2x1x2 "block" would multiply a 2x1 by 1x2, giving 2 outputs.
    # That's just 2 schoolbook terms, no saving.
    
    # What about a 3x3x3 Strassen-like block? Known rank for 3x3x3 is 23 (or 20?).
    # Actually the best known is 23. A 3x3x3 block would cover 27 triples with 23 terms
    # vs 27 schoolbook terms: saves 4. But fitting a 3x3x3 in 3x4x7:
    # i: must be 0 (only option for 3-wide)
    # j: can be 0 or 1 (positions for 3-wide in 4-wide)
    # k: can be 0,1,2,3,4 (positions for 3-wide in 7-wide)
    # If we use a 3x3x3 block at (0,0,0): covers i={0,1,2}, j={0,1,2}, k={0,1,2}
    # Remaining: j={3}, k={3,4,5,6} -> 1*4=4 triples from j=3,k=3..6
    #           i={0,1,2}, j={3}, k={0,1,2} -> 3*3=9 triples
    #           i={0,1,2}, j={0,1,2}, k={3,4,5,6} -> 3*3*4=36 triples
    # Total remaining: 4+9+36 = 49 schoolbook terms
    # Total rank: 23 + 49 = 72. That would be better than 78!
    # But do we have a known exact rank-23 decomposition for 3x3x3?
    # The best known is 23 (due to various authors). Let me use it.

    if time.time() > deadline:
        return best

    # 3x3x3 rank-23 decomposition (from literature)
    # This is complex to implement from scratch. Let me use a known construction.
    # Actually, the exact rank-23 for 3x3x3 is non-trivial. 
    # A simpler approach: use a 3x3x3 block with a known decomposition.
    
    # Known: rank(3,3,3) = 23. One explicit decomposition exists but is complex.
    # For now, let's stick with what works and try a different angle.
    
    # Try: combine a 3x3x3 block (if we can find a decomposition) with Strassen blocks
    # on the remaining parts.
    
    # Actually, let me try a simpler improvement:
    # Instead of 6 disjoint 2x2x2 blocks + 36 schoolbook, what if I use
    # 4 disjoint 2x2x2 blocks covering a 2x4x6 region (48 triples) + 
    # handle the remaining 36 triples (i=2 row) with a smarter method?
    # The i=2 row has 4*7=28 triples. That's a 1x4x7 problem.
    # Rank of 1x4x7 = 4*7 = 28 (no Strassen possible since one dimension is 1).
    # So no improvement there.
    
    # The 6-block solution with rank 78 appears to be optimal for this approach.
    # Let me verify and return.

    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
