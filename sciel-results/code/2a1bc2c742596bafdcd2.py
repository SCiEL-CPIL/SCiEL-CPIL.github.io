"""Ternary covering code K_3(7,1): vectorized greedy + optimized local search."""
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 112.0  # Safety margin before 120s wall-clock

    # Fallback for non-(3,7,1): simple product construction (valid, not optimal)
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q]
                     for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]

    N = 3 ** n  # 2187
    # all_words[i] = digits of base-3 representation of i
    all_words = np.zeros((N, n), dtype=np.int8)
    for idx in range(N):
        t = idx
        for pos in range(n - 1, -1, -1):
            all_words[idx, pos] = t % 3
            t //= 3

    # Precompute ball indices for all words: ball_indices[i] = array of indices j such that dist(i,j) <= radius
    # For q=3, n=7, radius=1, each ball has 1 + 7*2 = 15 elements.
    ball_indices = [None] * N
    for i in range(N):
        neigh = [i]
        w = all_words[i]
        for pos in range(n):
            base = 3 ** (n - 1 - pos)
            orig_val = w[pos]
            for val in range(3):
                if val == orig_val:
                    continue
                neigh.append(i + (val - orig_val) * base)
        ball_indices[i] = np.array(neigh, dtype=np.intp)
    
    # Also precompute sets for fast intersection in local search
    ball_sets = [set(arr) for arr in ball_indices]

    # Build the full ball matrix B for vectorized greedy operations
    # B[i, j] = 1 iff j is in ball_indices[i]
    B = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        B[i, ball_indices[i]] = 1

    def is_valid(code):
        if not code:
            return False
        # Check if union of balls covers all N words
        return bool((B[np.asarray(code, dtype=np.intp)].any(axis=0)).all())

    def greedy_fill(code, covered):
        """Add codewords until every word is covered. Returns (code, covered)."""
        code = list(code)
        covered = covered.copy()
        code_set = set(code)
        while not covered.all():
            if time.time() > deadline - 10:
                break
            gain_vec = (~covered).astype(np.int32)
            gains = B @ gain_vec
            # Mask out already selected codewords
            for c in code_set:
                gains[c] = -1
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                break
            code.append(best)
            code_set.add(best)
            covered |= B[best].astype(bool)
        return code, covered

    def local_search(code):
        """Optimized deletion + swap local search. Returns the (possibly smaller) code."""
        code = list(code)
        cs = set(code)
        # counts[p] = number of codewords in code whose ball contains p
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[ball_indices[c]] += 1
        
        improved = True
        while improved and time.time() < deadline - 6:
            improved = False

            # ---- Deletion pass: drop any codeword whose ball is fully covered (counts > 1 everywhere in its ball) ----
            to_remove = []
            for c in code:
                if counts[ball_indices[c]].min() > 1:
                    to_remove.append(c)
            if to_remove:
                improved = True
                for c in to_remove:
                    code.remove(c)
                    cs.discard(c)
                    counts[ball_indices[c]] -= 1

            # ---- Swap pass: relocate codewords to reduce critical points ----
            if time.time() > deadline - 6:
                break
            shuffled = list(code)
            random.shuffle(shuffled)
            for c in shuffled:
                if time.time() > deadline - 6:
                    break
                if c not in cs:
                    continue
                
                c_ball_idx = ball_indices[c]
                c_ball_set = ball_sets[c]
                # Critical points: points in c's ball that are covered only by c
                crit_mask = counts[c_ball_idx] == 1
                if not np.any(crit_mask):
                    continue  # This codeword is redundant and should have been removed
                
                crit_points = c_ball_idx[crit_mask]
                if len(crit_points) > 3:
                    continue  # Too hard to cover all with one new position efficiently

                # Find candidates that cover all critical points
                # Candidate w must be in the intersection of balls of all crit_points
                cand_set = ball_sets[crit_points[0]]
                for p in crit_points[1:]:
                    cand_set &= ball_sets[p]
                
                # Exclude current codewords
                candidates = [x for x in cand_set if x not in cs]
                if not candidates:
                    continue

                best_w, best_score = -1, 0
                for w in candidates:
                    w_ball_idx = ball_indices[w]
                    w_ball_set = ball_sets[w]
                    score = 0
                    # Gain: points that were critical (count=1) and are now covered by w but not by c
                    # Actually, we are moving c to w.
                    # Points in w_ball_idx that have count 1 and are NOT in c_ball_idx will gain coverage from w?
                    # No, we are replacing c with w.
                    # New counts:
                    # For p in c_ball_idx: count decreases by 1. If it was 1, it becomes 0 (bad, unless covered by w).
                    # For p in w_ball_idx: count increases by 1.
                    
                    # Simulate:
                    # p in c_ball_idx:
                    #   if count[p] == 1:
                    #     if p in w_ball_idx: new count is 1 (ok)
                    #     else: new count is 0 (bad, penalty)
                    # p in w_ball_idx:
                    #   if p not in c_ball_idx:
                    #     if count[p] == 1: new count is 2 (good, reduces criticality elsewhere? no, just increases coverage)
                    #     Actually, the goal is to not increase the number of critical points or reduce total code size.
                    #     A swap is beneficial if it allows further deletions or if it reduces the "critical load".
                    
                    # Heuristic score:
                    # Penalty for points in c_ball_idx that become uncovered (count 1 and not in w_ball_idx)
                    for p in c_ball_idx:
                        if counts[p] == 1 and p not in w_ball_set:
                            score -= 10  # High penalty for creating uncovered points
                    # Benefit for points in w_ball_idx that are currently critical (count 1) and not in c_ball_idx
                    # This reduces the number of critical points for OTHER codewords?
                    # If p is critical for some other codeword d, and we add w, p is no longer critical for d.
                    for p in w_ball_idx:
                        if p not in c_ball_set:
                            if counts[p] == 1:
                                score += 1
                            elif counts[p] == 0:
                                score += 2  # Covering an uncovered point? Shouldn't happen in valid code, but safe.
                    
                    if score > best_score:
                        best_score, best_w = score, w

                if best_w >= 0 and best_score > 0:
                    # Perform swap
                    counts[c_ball_idx] -= 1
                    w_ball_idx = ball_indices[best_w]
                    counts[w_ball_idx] += 1
                    code.remove(c)
                    cs.discard(c)
                    code.append(best_w)
                    cs.add(best_w)
                    improved = True
        return code

    # ---- Initial greedy construction ----
    covered = np.zeros(N, dtype=bool)
    code, _ = greedy_fill([], covered)
    code = local_search(code)
    if not is_valid(code):
        code, _ = greedy_fill(code, np.zeros(N, dtype=bool))
        code = local_search(code)
    best = code if is_valid(code) else []

    # ---- Kick / re-greedy / local search, keep best-so-far ----
    while time.time() < deadline - 8:
        cur = list(best)
        if not cur:
            break
        # Use a slightly larger kick size to escape local minima more effectively
        k = max(1, min(5, len(cur) // 5))
        to_remove = set(random.sample(cur, k))
        kicked = [c for c in cur if c not in to_remove]
        covered = np.zeros(N, dtype=bool)
        if kicked:
            covered |= B[np.asarray(kicked, dtype=np.intp)].any(axis=0)
        new_code, _ = greedy_fill(kicked, covered)
        new_code = local_search(new_code)
        if is_valid(new_code) and (not best or len(new_code) < len(best)):
            best = new_code

    if not best or not is_valid(best):
        # Safety fallback: full greedy then local search
        covered = np.zeros(N, dtype=bool)
        code, _ = greedy_fill([], covered)
        code = local_search(code)
        best = code
    
    # Final polish pass with more aggressive local search
    if time.time() < deadline - 2:
        final_code = local_search(best)
        if is_valid(final_code) and len(final_code) < len(best):
            best = final_code

    if not is_valid(best):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(all_words[c]) for c in best]
