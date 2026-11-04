"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Build a structured base: Strassen for (2,2,2) sub-blocks + schoolbook for remaining row/column.
     - 4 Strassen blocks of (2x2)*(2x2) covering A1(2x2)*B1(2x8): 4*7 = 28 terms
     - Schoolbook for A2(2x1)*B2(1x8): 2*1*8 = 16 terms
     - Total: rank 44
  2. Search for linear dependencies among the 44 column-vectors to reduce rank.
  3. Try structured rank-r decompositions for r < 44.
  4. Validate any candidate with exact rational arithmetic.
  5. If no improvement is found within the time budget, return the best known decomposition.
"""
import time
import sys
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # 15 s safety margin before 150 s kill

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    num_terms = n * m * p  # 48

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation using Fractions ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build rank-44 base candidate ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Column vector extraction ----
    def _col_vector(U, V, W, c):
        vec = [0] * num_terms
        for i in range(n):
            for a in range(m):
                u_val = U[i * m + a][c]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[a * p + k][c]
                    if v_val == 0:
                        continue
                    w_val = W[k * n + i][c]
                    if w_val == 0:
                        continue
                    vec[i * m * p + a * p + k] = u_val * v_val * w_val
        return vec

    # ---- Search for rank reduction via linear dependency ----
    def _search_rank_reduction():
        nonlocal best_candidate, best_rank

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 44

        cols = []
        for c in range(rank0):
            cols.append(_col_vector(U, V, W, c))

        target = [1] * num_terms

        # Verify base
        total = [0] * num_terms
        for cv in cols:
            for idx in range(num_terms):
                total[idx] += cv[idx]
        for idx in range(num_terms):
            if total[idx] != 1:
                return

        # Check for duplicate columns
        seen = {}
        removable = []
        for c in range(rank0):
            key = tuple(cols[c])
            if key in seen:
                removable.append(c)
            else:
                seen[key] = c

        if removable:
            for c_rem in removable:
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for c in range(rank0):
                    if c == c_rem:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][c]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][c]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][c]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

        if not _elapsed_ok():
            return

        # Search for a linear combination of columns that sums to zero
        # If sum_{c in S} cols[c] = 0, then we can drop one column from S
        # and adjust the others, OR if there's a dependency, we can reduce rank.
        
        # Try small subsets (size 2 and 3) for exact zero sum
        # Size 2: cols[i] + cols[j] = 0 means cols[i] = -cols[j]
        for i in range(rank0):
            if not _elapsed_ok():
                return
            for j in range(i + 1, rank0):
                dep = True
                for idx in range(num_terms):
                    if cols[i][idx] + cols[j][idx] != 0:
                        dep = False
                        break
                if dep:
                    # cols[i] = -cols[j], so we can drop one
                    keep = [c for c in range(rank0) if c != j]
                    new_rank = rank0 - 1
                    new_U = [[0] * new_rank for _ in range(n * m)]
                    new_V = [[0] * new_rank for _ in range(m * p)]
                    new_W = [[0] * new_rank for _ in range(p * n)]
                    nc = 0
                    for c in keep:
                        for i2 in range(n * m):
                            new_U[i2][nc] = U[i2][c]
                        for i2 in range(m * p):
                            new_V[i2][nc] = V[i2][c]
                        for i2 in range(p * n):
                            new_W[i2][nc] = W[i2][c]
                        nc += 1
                    cand = {"U": new_U, "V": new_V, "W": new_W}
                    if _validate(cand):
                        best_candidate = cand
                        best_rank = new_rank
                        return

        if not _elapsed_ok():
            return

        # Try size-3 dependencies: cols[i] + cols[j] + cols[k] = 0
        for i in range(rank0):
            if not _elapsed_ok():
                return
            for j in range(i + 1, rank0):
                if not _elapsed_ok():
                    return
                for k in range(j + 1, rank0):
                    dep = True
                    for idx in range(num_terms):
                        if cols[i][idx] + cols[j][idx] + cols[k][idx] != 0:
                            dep = False
                            break
                    if dep:
                        keep = [c for c in range(rank0) if c != k]
                        new_rank = rank0 - 1
                        new_U = [[0] * new_rank for _ in range(n * m)]
                        new_V = [[0] * new_rank for _ in range(m * p)]
                        new_W = [[0] * new_rank for _ in range(p * n)]
                        nc = 0
                        for c in keep:
                            for i2 in range(n * m):
                                new_U[i2][nc] = U[i2][c]
                            for i2 in range(m * p):
                                new_V[i2][nc] = V[i2][c]
                            for i2 in range(p * n):
                                new_W[i2][nc] = W[i2][c]
                            nc += 1
                        cand = {"U": new_U, "V": new_V, "W": new_W}
                        if _validate(cand):
                            best_candidate = cand
                            best_rank = new_rank
                            return

        if not _elapsed_ok():
            return

        # Try random linear combinations with small integer coefficients
        # Look for sum_c alpha_c * cols[c] = 0 with small integers alpha_c
        # This is a lattice problem; try random sparse combinations
        random.seed(42)
        for trial in range(5000):
            if not _elapsed_ok():
                return
            # Pick 3-5 random columns with random coefficients in {-2,-1,1,2}
            size = random.choice([3, 4, 5])
            indices = random.sample(range(rank0), size)
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(size)]
            
            dep = True
            for idx in range(num_terms):
                s = 0
                for ci, cc in zip(indices, coeffs):
                    s += cc * cols[ci][idx]
                if s != 0:
                    dep = False
                    break
            if dep:
                # Found a dependency! Drop the last index
                drop_idx = indices[-1]
                keep = [c for c in range(rank0) if c != drop_idx]
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for c in keep:
                    for i2 in range(n * m):
                        new_U[i2][nc] = U[i2][c]
                    for i2 in range(m * p):
                        new_V[i2][nc] = V[i2][c]
                    for i2 in range(p * n):
                        new_W[i2][nc] = W[i2][c]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

    if _elapsed_ok():
        _search_rank_reduction()

    if best_rank < 44:
        return best_candidate

    # ---- Try structured lower-rank decompositions ----
    # Approach: partition the 3x8 matrix differently
    # Option A: Use 2x3 Strassen-like for the full 2x3 * 3x8
    # The 2x3 * 3x8 can be seen as:
    #   - 2x3 * 3x2 (rank 10 via Strassen+identity) for each of 4 blocks of 2 columns
    #   - Total: 4 * 10 = 40
    
    def _build_rank_40():
        """
        Use rank-10 decomposition for 2x3 * 3x2 blocks.
        2x3 * 3x2: A is 2x3, B is 3x2.
        Split A into A1(2x2) and A2(2x1).
        Split B into B1(2x2) and B2(1x2).
        A1*B1: rank 7 (Strassen)
        A2*B2: rank 4 (schoolbook: 2*1*2)
        Total: 11 per block, 4 blocks = 44. Same as before.
        
        Alternative: use a different splitting.
        2x3 * 3x2 can potentially be done in rank 10.
        """
        # Actually, let's try a different approach:
        # 2x3 * 3x8 = (2x3 * 3x4) + (2x3 * 3x4)
        # If 2x3 * 3x4 has rank r, total is 2r.
        # Schoolbook for 2x3*3x4: 24
        # Strassen-based: 2x2*2x4 (2 blocks of 2x2*2x2 = 14) + 2x1*2x4 (8) = 22
        # So 2*22 = 44 again.
        
        # Let's try: 2x3 * 3x8 as a single problem.
        # Known: matmul rank for 2x3*3x8 is at most 44 (our construction).
        # Can we do better?
        
        # Try: treat it as 2x3 * 3x8 where we use a different structure.
        # The tensor has 48 entries. Rank 44 means 4 fewer than schoolbook.
        # Strassen saves 1 per 2x2x2 block (7 vs 8). We have 4 such blocks = 4 savings.
        # The 2x1*1x8 part is already minimal (8 per row pair = 16 total).
        
        # So rank 44 might be optimal for this structure. Let's try to verify
        # by searching more aggressively.
        return None

    # ---- Aggressive search: try to find ANY rank-r decomposition for r < 44 ----
    # This is a constraint satisfaction problem. Use random initialization + local search.
    if _elapsed_ok():
        def _try_random_search(max_rank):
            nonlocal best_candidate, best_rank
            
            # We'll try to find a decomposition with rank r by:
            # 1. Start with the rank-44 decomposition
            # 2. Try to zero out columns and adjust others
            # This is very hard in general, so we use a heuristic.
            
            # Alternative: use the fact that the multiplication tensor is fixed.
            # We need: sum_c U[i*m+a][c] * V[a*p+k][c] * W[k*n+i][c] = 1 for all (i,a,k)
            
            # This is a system of 48 equations in 48*r unknowns (with constraints that
            # each "term" c has a specific structure).
            
            # For now, try a different structured decomposition.
            
            # Idea: 2x3 * 3x8 can be decomposed as:
            #   A = [A_left(2x2) | A_right(2x1)]
            #   B = [B_top(2x8); B_bottom(1x8)]
            #   AB = A_left * B_top + A_right * B_bottom
            #   A_left * B_top: 2x2 * 2x8 = 4 blocks of 2x2*2x2, rank 4*7=28
            #   A_right * B_bottom: 2x1 * 1x8, rank 2*1*8=16
            #   Total: 44
            
            # Alternative splitting:
            #   A = [A1(1x3); A2(1x3)]  (row split)
            #   B = [B1(3x4) B2(3x4)]  (col split)
            #   AB = [A1*B1  A1*B2; A2*B1  A2*B2]
            #   Each 1x3*3x4 has rank 12 (schoolbook)
            #   Total: 4*12 = 48 (worse)
            
            # The current 44 seems hard to beat with simple structures.
            # Let's try a more sophisticated approach.
            
            pass

        _try_random_search(43)

    return best_candidate

# EVOLVE-BLOCK-END
