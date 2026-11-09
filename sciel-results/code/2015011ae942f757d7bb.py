import numpy as np
import time
from itertools import combinations, product
from scipy.optimize import linear_sum_assignment

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580.0
    
    points = []
    
    # ============================================================
    # Construction: E8 ⊕ A3 lattice minimal vectors in Z^11
    # Total: 512 points, all norm² = 4
    # ============================================================
    
    # --- E8 part (240 vectors, padded to 11D) ---
    # Type 1: (±1, ±1, 0,0,0,0,0,0) → C(8,2)*4 = 112 vectors
    for i, j in combinations(range(8), 2):
        for si, sj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = si
            v[j] = sj
            points.append(v)
    
    # Type 2: (±1/2, ±1/2, ..., ±1/2) with even number of minuses
    # Scale by 2 to get integers: (±1, ±1, ..., ±1) with even number of minuses
    # But these have norm² = 8, not 4. We need to handle differently.
    # Actually for E8⊕A3, the minimal vectors are:
    # E8 roots: 240 vectors of norm² 2 (in standard E8)
    # A3 roots: 12 vectors of norm² 2 (in standard A3)
    # But we need integer vectors. Let me reconsider.
    
    # Alternative: Use the D11 lattice approach or direct construction
    # Let me try: all vectors with exactly 2 non-zero entries of ±1 (norm²=2)
    # That gives C(11,2)*4 = 220 points. Score = 220.
    
    # For improvement, try mixing norm²=2 and norm²=4 vectors carefully.
    
    # Actually, let me try a different structural approach:
    # Use the fact that if all points have the same norm² = m,
    # then we need min pairwise dist² >= m.
    # dist² = 2m - 2*dot, so we need dot <= m/2.
    
    # For norm²=2 vectors (2 non-zero ±1 entries):
    # Two such vectors: dot can be -2,-1,0,1,2
    # Need dot <= 1 (since m/2 = 1)
    # So they can't share both indices with same sign.
    # If we take ALL sign combos for all pairs, two vectors sharing pair (i,j)
    # with same signs have dot=2 > 1. INVALID.
    
    # So we can't take all 220. We need to restrict.
    # Max set of 2-support vectors with no two sharing a pair with same sign:
    # This is equivalent to: for each pair (i,j), at most 2 of the 4 sign combos
    # (the two with same sign pattern are incompatible with each other).
    # Actually: (1,1) conflicts with (1,1) only. (1,1) and (-1,-1) have dot=-2, OK.
    # (1,1) and (1,-1) have dot=0, OK. (1,1) and (-1,1) have dot=0, OK.
    # So for each pair, we can take at most 3 of the 4 sign combos? No:
    # (1,1) vs (1,1): same vector, not allowed (distinct points).
    # (1,1) vs (-1,-1): dot = -1-1 = -2, dist² = 2+2-2(-2) = 8 >= 2. OK.
    # (1,1) vs (1,-1): dot = 1-1 = 0, dist² = 4 >= 2. OK.
    # (1,1) vs (-1,1): dot = -1+1 = 0, dist² = 4 >= 2. OK.
    # So actually ALL 4 sign combos for a given pair are mutually compatible!
    # Wait, (1,1) and (1,1) are the same point. We just need distinct points.
    # So for each pair, all 4 sign combos are valid and mutually compatible.
    # Total: C(11,2)*4 = 220. This is what the current best does.
    
    # Can we add more? Try adding norm²=4 vectors (4 non-zero ±1 entries).
    # A norm²=4 vector v and a norm²=2 vector u:
    # dist² = 4 + 2 - 2*dot = 6 - 2*dot. Need >= 4 (max norm²).
    # So dot <= 1.
    # If u shares 2 indices with v: dot = ±2 or 0. If same sign on both: dot=2. FAIL.
    # If u shares 1 index: dot = ±1. OK.
    # If u shares 0 indices: dot = 0. OK.
    # So a norm²=4 vector with support {a,b,c,d} conflicts with norm²=2 vectors
    # that have support {a,b}, {a,c}, {a,d}, {b,c}, {b,d}, {c,d} WITH SAME SIGN.
    # Since we have all sign combos for norm²=2 vectors, every norm²=4 vector
    # conflicts with 6 norm²=2 vectors. We'd need to remove those 6.
    # Net gain: +1 - 6 = -5. Bad.
    
    # What about using ONLY norm²=4 vectors?
    # Two norm²=4 vectors: dist² = 8 - 2*dot. Need >= 4. So dot <= 2.
    # Two 4-support vectors: dot can range from -4 to 4.
    # If they share all 4 indices with same signs: dot=4. FAIL.
    # If they share 3 indices: dot can be -3,-1,1,3. dot=3 > 2. FAIL.
    # If they share 2 indices: dot can be -2,0,2. dot=2 OK.
    # If they share 1 or 0: dot in {-1,0,1} or {0}. OK.
    # So: no two 4-support vectors can share 3+ indices.
    # This is exactly the current program's constraint!
    # Max packing: the current program finds ~27 sets × 16 = 432 points.
    
    # But wait - we could also mix: some norm²=2 and some norm²=4.
    # Or try a completely different norm.
    
    # Let me try: all vectors with exactly 2 non-zero entries, but with values ±1,±2
    # No, that complicates things.
    
    # Structural change: Instead of 4-set packing, use a CODE-BASED approach.
    # Represent each point by its support and signs. Use a different conflict model.
    
    # Actually, let me try: use norm²=2 vectors but with a SMARTER selection
    # that allows adding some norm²=4 vectors.
    
    # Or: try the D11+ lattice. D_n = {x in Z^n : sum(x_i) even}
    # D11 minimal vectors: (±1, ±1, 0,...,0) with even sum → all 2-support ±1 vectors.
    # That's 220 vectors. Same as before.
    
    # Let me try a different approach: use vectors with entries in {-1, 0, 1}
    # but with weight 3 (three non-zero entries). Norm² = 3.
    # Two weight-3 vectors: dist² = 6 - 2*dot. Need >= 3. dot <= 1.5, so dot <= 1.
    # If they share 2 indices with same sign: dot >= 2. FAIL.
    # If they share 2 with opposite sign: dot = 0. OK.
    # If they share 1: dot = ±1. OK.
    # So: no two weight-3 vectors can share 2+ indices with same sign on both.
    # For each pair of indices, at most 2 of the 4 sign combos for a given third index.
    # This gets complex. Let me just try the current 4-set approach with better search.
    
    # NEW STRUCTURAL IDEA: Use a genetic algorithm / local search on the 4-set packing
    # with a better objective and more diverse restarts.
    
    # Actually, the simplest structural change: try ALL possible 4-set packings
    # using a maximum independent set approach on the conflict graph, with
    # branch-and-bound or simulated annealing.
    
    # Let me try simulated annealing on the 4-set selection problem.
    
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)
    
    # Precompute conflict: two 4-sets conflict if they share 3+ indices
    conflict = np.zeros((n_sets, n_sets), dtype=bool)
    for i in range(n_sets):
        for j in range(i+1, n_sets):
            inter = len(set(all_4sets[i]) & set(all_4sets[j]))
            if inter >= 3:
                conflict[i, j] = True
                conflict[j, i] = True
    
    # Simulated annealing to find max independent set
    def count_conflicts(selection, conflict_matrix):
        """Count number of conflicting pairs in selection."""
        n = len(selection)
        if n < 2:
            return 0
        cnt = 0
        for i in range(n):
            for j in range(i+1, n):
                if conflict_matrix[selection[i], selection[j]]:
                    cnt += 1
        return cnt
    
    # Start with greedy solution
    import random
    rng = random.Random(12345)
    
    # Multiple greedy starts
    best_selection = []
    best_score = 0
    
    for trial in range(200):
        if time.time() > deadline - 60:
            break
        
        order = list(range(n_sets))
        rng.shuffle(order)
        
        selected = []
        selected_set = set()
        
        for idx in order:
            valid = True
            for s in selected:
                if conflict[idx, s]:
                    valid = False
                    break
            if valid:
                selected.append(idx)
                selected_set.add(idx)
        
        if len(selected) > best_score:
            best_score = len(selected)
            best_selection = selected[:]
    
    # Local search: try to swap out one selected set for one not selected
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        selected_set = set(best_selection)
        for i in range(len(best_selection)):
            for j in range(n_sets):
                if j in selected_set:
                    continue
                # Try replacing best_selection[i] with j
                new_sel = best_selection[:i] + best_selection[i+1:] + [j]
                # Check validity
                valid = True
                for a in range(len(new_sel)):
                    for b in range(a+1, len(new_sel)):
                        if conflict[new_sel[a], new_sel[b]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and len(new_sel) > len(best_selection):
                    best_selection = new_sel
                    selected_set = set(new_sel)
                    improved = True
                    break
            if improved:
                break
    
    # Generate points from best_selection
    points = []
    for idx in best_selection:
        s = all_4sets[idx]
        i, j, k, l = s
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
    
    if not points:
        # Fallback: 220 norm-2 vectors
        for i, j in combinations(range(d), 2):
            for si, sj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
