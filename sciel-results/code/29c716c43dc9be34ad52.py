"""
Strategy: Deterministic Criticality-Greedy Construction + Exhaustive Local Search.

1. Criticality-Greedy Initialization:
   Instead of standard greedy (max coverage), we select codewords that cover
   uncovered words with the "highest criticality". Criticality is defined as
   the number of available (unselected) codewords that can cover a word.
   Words with fewest options are most constrained; covering them early
   prevents them from becoming isolated. This is a different structural
   heuristic than standard set cover.

2. Exhaustive Deletion Phase:
   Repeatedly attempt to remove any codeword that does not uniquely cover
   any word. This runs until no such codeword exists (local minimum w.r.t. deletion).

3. Exhaustive Swap Phase:
   For every pair of codewords (c1, c2), check if removing c1 and adding c2
   maintains coverage. This is O(N^2 * BallSize) but with N~186 and BallSize~14,
   it is ~ 186^2 * 14 * 5000 (coverage check) which might be too slow in pure Python.
   We optimize coverage checking using numpy bitmasks or simple array ops.
   
   Actually, N=186, pairs ~ 17k. Ball size 14. 
   If we use numpy for coverage counts, updating is O(BallSize). 
   Checking validity is O(N) or O(1) if we track uncovered count.
   Let's use a fast numpy-based swap check.

4. Randomized Restart / Perturbation:
   If time remains, perform random swaps to escape local minima, followed by
   another deletion/swap refinement cycle.
"""

import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    total_words = q ** n
    
    # Precompute all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    
    # Precompute coverage indices (balls of radius 1)
    # For each word i, list of words within distance 1
    cover_indices = []
    for i in range(total_words):
        w = all_words[i]
        inds = [i]
        for j in range(n):
            orig_val = w[j]
            for v in range(q):
                if v == orig_val:
                    continue
                new_w = w.copy()
                new_w[j] = v
                # Convert to index: base-q representation
                # Since all_words is generated via product, index is just the base-q number
                # But we need to map tuple to index. 
                # We can compute index directly: sum(new_w[k] * q^(n-1-k))
                idx = 0
                for k in range(n):
                    idx = idx * q + new_w[k]
                inds.append(idx)
        cover_indices.append(inds)
    
    cover_matrix = np.array(cover_indices, dtype=np.int32)
    ball_size = cover_matrix.shape[1]
    
    # Helper to check time
    def time_ok():
        return time.time() < deadline - 5.0
    
    # --- Phase 1: Criticality Greedy Initialization ---
    # For each word, count how many codewords cover it.
    # Initially, all words are uncovered.
    # We want to pick a codeword that covers the "most critical" uncovered words.
    # Criticality of an uncovered word = number of codewords that cover it.
    # (Lower count = more critical, because if we don't pick one of them soon, 
    #  we might lose the chance to cover it efficiently? Actually, all codewords are 
    #  "available" initially. Criticality is static initially: every word is covered 
    #  by 1 + n*(q-1) codewords. 
    #  Wait, the "available" set changes. 
    #  Let's define criticality dynamically: 
    #  For each uncovered word w, let C(w) be the set of codewords c such that w in Ball(c).
    #  If we pick c, we cover all w in Ball(c).
    #  Standard greedy picks c maximizing |Ball(c) \ Covered|.
    #  Criticality greedy: 
    #  1. Compute for each uncovered word w, the number of *remaining* codewords that cover it.
    #     Initially, all codewords are remaining. So this count is constant (ball_size).
    #     This suggests standard greedy is equivalent initially.
    #  2. As we pick codewords, the set of "remaining" codewords shrinks.
    #     A word w is critical if few remaining codewords can cover it.
    #     We should prioritize covering words with low "remaining coverage count".
    
    # Implementation:
    # remaining_cov_count[w] = number of selected codewords that cover w? No.
    # We need: for each word w, how many *unselected* codewords cover it?
    # This is expensive to update.
    
    # Alternative Structural Approach: 
    # Use a different initialization: 
    # 1. Start with a random valid code? No, hard to find.
    # 2. Use a known good construction (e.g. product code) as a seed, then optimize.
    #    The product code for K_3(7,1) has size 3^4 * K_3(3,1)? 
    #    K_3(3,1) = 3 (000, 111, 222? No, distance 1. 000 covers 000, 100, 200, 010, 020, 001, 002. 
    #    Size of K_3(3,1) is 3? No, 1+3*2=7 words covered by 1. 
    #    Lower bound: 3^3/7 = 27/7 = 3.8 -> 4. 
    #    Actually K_3(3,1) = 3 is not enough. 
    #    Let's just use the standard Greedy as base, but improve the optimization.
    
    # Let's stick to the "Criticality" idea but implement it efficiently.
    # We maintain:
    # - selected: list of indices
    # - covered: bool array
    # - rem_count[w]: number of *unselected* codewords that cover w.
    #   Initially, rem_count[w] = ball_size for all w.
    #   When we select c, for all w in Ball(c), rem_count[w] -= 1.
    #   When we select c, we mark w in Ball(c) as covered.
    #   The "score" of a candidate c is:
    #   Sum over w in Ball(c) \ Covered of (1 / rem_count[w])? 
    #   Or simply: Prioritize c that covers the most "critical" words.
    #   Criticality of w = 1 / rem_count[w].
    #   Score(c) = sum_{w in Ball(c) \ Covered} (1 / rem_count[w])
    
    selected = []
    covered = np.zeros(total_words, dtype=bool)
    rem_count = np.full(total_words, ball_size, dtype=np.int32)
    
    # Precompute inverse rem_count for scoring? 
    # rem_count changes, so we need to update scores.
    # This is O(N * BallSize) per step. N=2187, BallSize=14. 
    # Steps ~ 186. 
    # 186 * 2187 * 14 ~ 5.7M operations. Fast enough.
    
    # To make it faster, we can use numpy.
    # But rem_count is a vector. 
    # Let's just use a list for selected and arrays for state.
    
    # Optimization: 
    # Instead of recalculating scores for all N candidates every step,
    # we can just pick the candidate that covers the word with the current 
    # minimum rem_count.
    # 1. Find uncovered word w with min rem_count[w].
    # 2. Among all unselected codewords c that cover w, pick one that covers 
    #    the most *other* uncovered words (standard greedy tie-breaker).
    
    # This is a "Most Constrained Variable First" (MRV) heuristic from CSP.
    
    unselected_mask = np.ones(total_words, dtype=bool)
    
    while time_ok():
        if covered.all():
            break
            
        # Find uncovered words
        uncovered_idx = np.where(~covered)[0]
        if len(uncovered_idx) == 0:
            break
            
        # Find the most critical uncovered word: min rem_count
        # rem_count is updated only when we select a codeword.
        # We need the min rem_count among uncovered words.
        # Since rem_count only decreases, we can just find min.
        min_rem = np.min(rem_count[uncovered_idx])
        
        # Find all uncovered words with this min rem count
        critical_words = uncovered_idx[rem_count[uncovered_idx] == min_rem]
        
        # Pick one randomly to break symmetry (or just the first)
        # Randomness helps escape bad local structures
        w_star = critical_words[random.randint(0, len(critical_words)-1)]
        
        # Find all unselected codewords that cover w_star
        # A codeword c covers w_star if w_star is in Ball(c).
        # This is equivalent to: c is in Ball(w_star).
        # So candidates = unselected words in Ball(w_star).
        
        # Get ball of w_star
        ball_w_star = cover_matrix[w_star]
        
        # Filter to unselected
        candidates = ball_w_star[unselected_mask[ball_w_star]]
        
        if len(candidates) == 0:
            # Should not happen if rem_count is correct
            break
            
        # Among candidates, pick the one that covers the most uncovered words
        # Score(c) = sum_{w in Ball(c) \ Covered} 1
        
        # Vectorized scoring
        # For each candidate c, count uncovered words in Ball(c)
        # cover_matrix[candidates] gives the balls
        # We need to sum (~covered) over each row
        
        # This is O(len(candidates) * ball_size)
        # len(candidates) <= ball_size = 14.
        # So it's very fast.
        
        best_c = candidates[0]
        best_score = -1
        
        for c in candidates:
            ball_c = cover_matrix[c]
            score = (~covered[ball_c]).sum()
            if score > best_score:
                best_score = score
                best_c = c
            elif score == best_score:
                # Tie break: pick randomly or by index
                pass # Keep first found
        
        # Select best_c
        selected.append(best_c)
        unselected_mask[best_c] = False
        
        # Update covered and rem_count
        ball_c = cover_matrix[best_c]
        covered[ball_c] = True
        rem_count[ball_c] -= 1
        
    # --- Phase 2: Exhaustive Deletion ---
    # Compute coverage counts
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for c in selected:
        coverage_count[cover_matrix[c]] += 1
        
    changed = True
    while changed and time_ok():
        changed = False
        new_selected = []
        for c in selected:
            # Check if c uniquely covers any word
            ball_c = cover_matrix[c]
            uniquely_covered = (coverage_count[ball_c] == 1)
            if not uniquely_covered.any():
                # Remove c
                coverage_count[ball_c] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
        
    # --- Phase 3: Exhaustive Swap ---
    # Try to swap any c1 in selected with c2 not in selected
    # such that coverage is maintained.
    
    # Precompute coverage_count again if it changed? 
    # It didn't change in deletion phase if we kept the list consistent.
    # Wait, in deletion loop, we updated coverage_count.
    
    # For swap:
    # Remove c1, add c2.
    # Valid if: 
    # 1. All words uniquely covered by c1 are covered by c2.
    # 2. (Adding c2 doesn't break anything, it only adds coverage).
    
    # So condition: Ball(c1) \ Ball(c2) subset of (Words covered by others).
    # Equivalent: For all w in Ball(c1), if coverage_count[w] == 1, then w must be in Ball(c2).
    
    in_selected = np.zeros(total_words, dtype=bool)
    for c in selected:
        in_selected[c] = True
        
    unselected_indices = np.where(~in_selected)[0]
    
    improved = True
    while improved and time_ok():
        improved = False
        new_selected = list(selected)
        
        for i, c1 in enumerate(selected):
            if not time_ok(): break
            ball_c1 = cover_matrix[c1]
            
            # Words uniquely covered by c1
            unique_to_c1 = ball_c1[coverage_count[ball_c1] == 1]
            
            if len(unique_to_c1) == 0:
                # Should have been deleted in Phase 2, but maybe not if we just added?
                # No, we just did Phase 2.
                continue
                
            # We need a c2 that covers all unique_to_c1
            # And c2 must be unselected.
            
            # Find candidates c2 that cover all unique_to_c1
            # Since unique_to_c1 is small (usually 1 or 2), we can intersect balls.
            
            # Start with all unselected
            candidates = unselected_indices.copy()
            
            for w in unique_to_c1:
                # c2 must be in Ball(w)
                ball_w = cover_matrix[w]
                # Intersect candidates with ball_w
                # candidates is an array of indices
                # ball_w is an array of indices
                # Use set intersection or numpy in1d
                # Since sizes are small, list comprehension or set is fine
                candidates_set = set(candidates)
                ball_w_set = set(ball_w.tolist())
                candidates = np.array(list(candidates_set & ball_w_set), dtype=np.int32)
                
                if len(candidates) == 0:
                    break
                    
            if len(candidates) > 0:
                # Pick one candidate. 
                # Heuristic: pick the one that covers the most uncovered words? 
                # Or just the first one?
                # To improve code quality, pick the one that minimizes "waste" or maximizes coverage.
                # For now, pick the first one found.
                c2 = candidates[0]
                
                # Perform swap
                # Remove c1
                coverage_count[ball_c1] -= 1
                # Add c2
                coverage_count[cover_matrix[c2]] += 1
                
                new_selected[i] = c2
                in_selected[c1] = False
                in_selected[c2] = True
                
                # Update unselected_indices? 
                # It's a numpy array. We can just update in_selected and recompute unselected if needed.
                # Or just maintain a set.
                # For simplicity, recompute unselected_indices if it gets stale? 
                # No, we can just use in_selected to filter in the next iteration.
                # But the `candidates` calculation used `unselected_indices`.
                # Let's update unselected_indices efficiently.
                # Actually, just recompute it if we swapped.
                # It's O(N). N=2187. Fast.
                unselected_indices = np.where(~in_selected)[0]
                
                improved = True
                selected = new_selected
                break # Restart loop with new state
        
        # If no improvement in this pass, we are at a local minimum w.r.t. single swaps.
        
    # --- Phase 4: Randomized Perturbation & Refinement ---
    # If time remains, try random swaps to escape local minima.
    # Then run Deletion + Swap again.
    
    best_selected = list(selected)
    best_size = len(selected)
    
    rng = random.Random(42)
    
    while time.time() < deadline - 10.0:
        # Perturb: Swap a random pair
        if len(selected) < 2:
            break
            
        # Pick random c1 in selected
        i1 = rng.randint(0, len(selected)-1)
        c1 = selected[i1]
        
        # Pick random c2 not in selected
        # Use in_selected to filter
        unselected_list = np.where(~in_selected)[0]
        if len(unselected_list) == 0:
            break
        c2 = unselected_list[rng.randint(0, len(unselected_list)-1)]
        
        # Perform swap blindly (might break coverage)
        # Then try to fix it? 
        # Or only swap if it's valid? 
        # If we only swap if valid, we are just doing random valid swaps.
        # Let's do: Random swap, then run Deletion+Swap to repair/optimize.
        
        # Save state
        old_selected = list(selected)
        old_in_selected = in_selected.copy()
        old_coverage_count = coverage_count.copy()
        
        # Swap
        selected[i1] = c2
        in_selected[c1] = False
        in_selected[c2] = True
        
        # Update coverage
        coverage_count[cover_matrix[c1]] -= 1
        coverage_count[cover_matrix[c2]] += 1
        
        # Check if valid
        if not covered.all():
             # We don't have a 'covered' array here, we have coverage_count.
             # Valid if min(coverage_count) >= 1
             if coverage_count.min() < 1:
                 # Invalid. Revert? Or try to fix?
                 # Revert is safer for stability.
                 selected = old_selected
                 in_selected = old_in_selected
                 coverage_count = old_coverage_count
                 continue
                 
        # Run Deletion
        changed = True
        while changed and time_ok():
            changed = False
            new_selected = []
            for c in selected:
                ball_c = cover_matrix[c]
                if not (coverage_count[ball_c] == 1).any():
                    coverage_count[ball_c] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            # Update in_selected
            in_selected = np.zeros(total_words, dtype=bool)
            for c in selected:
                in_selected[c] = True
            
        # Run Swap (limited steps to save time)
        # Just one pass of exhaustive swap
        unselected_indices = np.where(~in_selected)[0]
        improved = True
        while improved and time_ok():
            improved = False
            for i, c1 in enumerate(selected):
                if not time_ok(): break
                ball_c1 = cover_matrix[c1]
                unique_to_c1 = ball_c1[coverage_count[ball_c1] == 1]
                if len(unique_to_c1) == 0:
                    continue
                candidates = unselected_indices.copy()
                for w in unique_to_c1:
                    ball_w = cover_matrix[w]
                    candidates_set = set(candidates)
                    ball_w_set = set(ball_w.tolist())
                    candidates = np.array(list(candidates_set & ball_w_set), dtype=np.int32)
                    if len(candidates) == 0:
                        break
                if len(candidates) > 0:
                    c2 = candidates[0]
                    coverage_count[ball_c1] -= 1
                    coverage_count[cover_matrix[c2]] += 1
                    selected[i] = c2
                    in_selected[c1] = False
                    in_selected[c2] = True
                    unselected_indices = np.where(~in_selected)[0]
                    improved = True
                    break
                    
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
            
    # Return best code
    result = [all_words[i].tolist() for i in best_selected]
    return result

# EVOLVE-BLOCK-END
