"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations, product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin
    
    def validate(U, V, W):
        """Check if the decomposition is exact using vectorized numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)
    
    def get_strassen_factors(i0, j0, k0):
        """Return the 7 Strassen factors for a 2x2x2 block at (i0,j0,k0)."""
        factors = []
        # M1 = (A00+A11)(B00+B11)
        factors.append(({i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
                        {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                        {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}))
        # M2 = (A10+A11)B00
        factors.append(({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                        {j0*p+k0: 1},
                        {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): -1}))
        # M3 = A00(B01-B11)
        factors.append(({i0*m+j0: 1},
                        {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
                        {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}))
        # M4 = A11(B10-B00)
        factors.append(({(i0+1)*m+(j0+1): 1},
                        {(j0+1)*p+k0: 1, j0*p+k0: -1},
                        {k0*n+i0: 1, k0*n+(i0+1): 1}))
        # M5 = (A00+A01)B11
        factors.append(({i0*m+j0: 1, i0*m+(j0+1): 1},
                        {(j0+1)*p+(k0+1): 1},
                        {k0*n+i0: -1, (k0+1)*n+i0: 1}))
        # M6 = (A10-A00)(B00+B01)
        factors.append(({(i0+1)*m+j0: 1, i0*m+j0: -1},
                        {j0*p+k0: 1, j0*p+(k0+1): 1},
                        {(k0+1)*n+(i0+1): 1}))
        # M7 = (A01-A11)(B10+B11)
        factors.append(({i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                        {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                        {k0*n+i0: 1}))
        return factors
    
    def build_from_factors(factors):
        """Build U, V, W matrices from a list of (u_dict, v_dict, w_dict) factors."""
        total_rank = len(factors)
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        for t, (u, v, w) in enumerate(factors):
            for idx, val in u.items():
                U[idx][t] += val
            for idx, val in v.items():
                V[idx][t] += val
            for idx, val in w.items():
                W[idx][t] += val
        return U, V, W
    
    def get_covered_terms(blocks):
        """Get set of (i,j,k) terms covered by Strassen blocks."""
        covered = set()
        for (i0, j0, k0) in blocks:
            for i in (i0, i0+1):
                for j in (j0, j0+1):
                    for k in (k0, k0+1):
                        covered.add((i, j, k))
        return covered
    
    def are_disjoint(b1, b2):
        """Check if two Strassen blocks are disjoint."""
        return not (b1[0] in (b2[0], b2[0]+1) and b2[0] in (b1[0], b1[0]+1) and
                    b1[1] in (b2[1], b2[1]+1) and b2[1] in (b1[1], b1[1]+1) and
                    b1[2] in (b2[2], b2[2]+1) and b2[2] in (b1[2], b1[2]+1))
    
    def is_set_disjoint(blocks):
        """Check if a set of blocks are pairwise disjoint."""
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                if not are_disjoint(blocks[a], blocks[b]):
                    return False
        return True
    
    # Generate all possible 2x2x2 block positions
    all_possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_possible_blocks.append((i0, j0, k0))
    
    # Current best
    best_rank = 75
    best_solution = None
    
    # Initial: 4 disjoint blocks for rows 0-1
    initial_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    factors = []
    for blk in initial_blocks:
        factors.extend(get_strassen_factors(*blk))
    covered = get_covered_terms(initial_blocks)
    residual = [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]
    for (i, j, k) in residual:
        factors.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
    
    U, V, W = build_from_factors(factors)
    if validate(U, V, W):
        best_rank = len(factors)
        best_solution = (U, V, W)
    
    # Search over all disjoint block combinations
    for size in range(1, 8):
        if time.time() > deadline - 15:
            break
        for combo in combinations(all_possible_blocks, size):
            if not is_set_disjoint(list(combo)):
                continue
            covered = get_covered_terms(list(combo))
            rank = size * 7 + (n*m*p - len(covered))
            if rank >= best_rank:
                continue
            factors = []
            for blk in combo:
                factors.extend(get_strassen_factors(*blk))
            residual = [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]
            for (i, j, k) in residual:
                factors.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
            U, V, W = build_from_factors(factors)
            if validate(U, V, W):
                best_rank = rank
                best_solution = (U, V, W)
    
    # Now try: overlapping blocks with linear combination to cancel overlaps
    # Strategy: use blocks that share i-indices but solve for coefficients
    # to make the overlap contributions cancel out.
    # 
    # For example, use blocks at (0,0,0), (0,2,0), (0,0,2), (0,2,2) AND
    # (1,0,0), (1,2,0), (1,0,2), (1,2,2).
    # The overlap is in row i=1. We need to find coefficients for the second set
    # such that the net contribution to row 1 from overlapping terms is zero,
    # while still contributing correctly to row 2.
    
    # Let's try a different approach: use 3x2x2 sub-tensors.
    # For each pair of j,k columns, we have a 3x2x2 sub-problem (3 rows of A, 2 cols of B, 2 cols of C).
    # The 3x2x2 matrix multiplication tensor can be decomposed in rank 12 (schoolbook) or possibly less.
    
    # Actually, let's try: partition the j-k space into 2x2 blocks and handle each with a 3x2x2 decomposition.
    # 5 = 2+2+1, so we have 4 disjoint 2x2 blocks in j-k space plus a 1x5 and 5x1 residual.
    
    # For each 2x2 block in j-k at position (j0,k0), we need to compute:
    # C[i,k0+r] += sum_s A[i,j0+s] * B[j0+s, k0+r] for i=0,1,2, r=0,1, s=0,1
    # This is a 3x2x2 matrix multiplication: rank 12 in schoolbook.
    
    # Can we do 3x2x2 in less than 12? Let's try rank 11 or 10.
    # The tensor rank of 3x2x2 matrix mult: schoolbook is 12.
    # Let's just use schoolbook for now: 4 blocks * 12 = 48, plus residual.
    # Residual: j=4 or k=4 terms = 3*1*5 + 3*5*1 - 3*1*1 = 15+15-3 = 27
    # Total: 48 + 27 = 75. No improvement.
    
    # What if we use Strassen for the 2x2 part within each 3x2x2?
    # For fixed i, the 2x2x2 part can use Strassen (rank 7). But we have 3 values of i.
    # If we use Strassen per i: 3*7 = 21 per 2x2 block. Worse than 12.
    
    # Alternative: use a shared Strassen-like decomposition across all 3 rows.
    # Define M1 = (A[0,j0]+A[1,j0+1]+A[2,j0]) * (B[j0,k0]+B[j0+1,k0+1]) ... this gets messy.
    
    # Let's try yet another approach: use the fact that for 3x5x5, we can write
    # C = A * B where A is 3x5 and B is 5x5.
    # Partition A into A1 (first 2 rows) and A2 (last row).
    # C1 = A1 * B (2x5), C2 = A2 * B (1x5).
    # For C1: use 4 Strassen blocks (rank 28) + 18 residual = 46.
    # For C2: 25 schoolbook terms.
    # Total: 71.
    
    # To improve, we need to share factors between C1 and C2 computations.
    # The key: modify Strassen factors to also contribute to C2.
    
    # Let's try: for each Strassen factor in C1, add a small contribution to C2.
    # A Strassen factor t has U_t (2x2), V_t (2x2), W_t (2x2) for the 2x2x2 sub-problem.
    # We can extend U_t to 3x2 by adding a row for i=2.
    # The contribution to C2 from factor t would be: A2[j] * V_t[j,k] * W_t[k,2]
    # We want this to be part of the correct C2 computation.
    
    # This requires solving a system. Let's try a simpler heuristic:
    # Use 3 blocks (not 4) for C1, leaving more room for sharing.
    
    # Actually, let's try a completely different tactic:
    # Use rank-7 Strassen for a 2x2x2 block, but place it to cover terms from all 3 rows
    # by using a 3x2x2 "Strassen" with rank 7 or 8.
    
    # For 3x2x2: we want to compute C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1,2; j=0,1; k=0,1.
    # 12 terms total. Can we do it in rank < 12?
    
    # Try rank 11: remove one schoolbook term and see if it can be expressed via others.
    # This is a linear algebra problem. Let's set it up.
    
    # Actually, let's just try to find a good decomposition by search.
    # We'll use the existing best and try to improve by finding better block placements.
    
    if best_solution is not None:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    # Fallback: full schoolbook
    factors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                factors.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
    U, V, W = build_from_factors(factors)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
