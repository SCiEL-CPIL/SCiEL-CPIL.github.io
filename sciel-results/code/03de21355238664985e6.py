import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Weights for positions (big-endian)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls for all words
    # balls[w, 0] = w, balls[w, 1:17] = neighbors
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # 1. Greedy Construction
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    # Main greedy loop
    while np.any(current_uncovered):
        if time.time() > deadline - 40.0:
            break
            
        # Count uncovered neighbors for each word
        # balls has shape (total, 17), current_uncovered[balls] gives a (total, 17) bool array
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        
        if max_count == 0:
            break
        
        # Deterministic: pick smallest index among max-count candidates
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
    # If timed out during greedy, force completion (greedy is fast enough usually, but just in case)
    # Actually, greedy on 6561 items with 17 neighbors is very fast. 
    # The bottleneck is usually the pruning or local search.
    # Let's ensure we finish the greedy if we broke early due to time, but only if we have plenty of time left.
    # If we broke early, we might have uncovered words. We need to cover them.
    if np.any(current_uncovered) and time.time() < deadline - 10.0:
        # Simple completion: pick any uncovered word, add it, update.
        while np.any(current_uncovered):
            uncovered_indices = np.where(current_uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            # Pick the first one
            best_idx = uncovered_indices[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            if time.time() > deadline - 10.0:
                break
    
    # 2. Aggressive Pruning (Greedy Removal)
    # We want to remove as many codewords as possible.
    # A codeword can be removed if all words in its ball are covered by at least one other codeword.
    # i.e., min(cover_count[balls[c]]) > 1
    
    # To make this efficient, we can iterate multiple times.
    # However, a single pass removing all removable nodes simultaneously is not safe (interdependencies).
    # We must do it sequentially or in careful batches.
    # Given N~500, sequential is fine.
    
    # Optimization: Instead of checking min() which is O(17), we can precompute.
    # But 500 * 17 is trivial.
    
    changed = True
    while changed and time.time() < deadline - 20.0:
        changed = False
        new_selected = []
        for c in selected:
            # Check if removing c is safe
            # cover_count[balls[c]] gives counts for all 17 words in the ball
            # If all are > 1, we can remove c
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
        
        # If no changes in a full pass, we are at a local minimum for this heuristic.
        # We can try to shuffle the order of removal?
        # Deterministic order is fine for reproducibility, but might get stuck.
        # Let's try a second pass with reversed order if we have time.
        
    # Try reverse order pass
    if time.time() < deadline - 20.0:
        changed = True
        while changed and time.time() < deadline - 20.0:
            changed = False
            new_selected = []
            for c in reversed(selected):
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            # Note: reversed iteration changes the order in new_selected? 
            # No, new_selected is built in reverse order of iteration, so it ends up in original order?
            # Iteration: last, second_last...
            # new_selected: [last_kept, second_last_kept...]
            # So new_selected is reversed.
            # We should reverse it back to maintain consistent state?
            # Or just accept the order change. The set is what matters.
            # Let's reverse it back to keep 'selected' order consistent for debugging, though it doesn't matter for validity.
            selected = list(reversed(new_selected))

    # 3. Local Search: Swap-based improvement
    # Try to replace a codeword with another to allow further removals.
    # This is complex. A simpler heuristic: 
    # For each codeword c, check if there's a word w (not in selected) such that 
    # replacing c with w allows removing another codeword d.
    # This is O(N^2 * 17). 500^2 * 17 ~ 4M ops. Feasible.
    
    # However, a simpler and often effective method is "Greedy with Backtracking" or "Simulated Annealing".
    # Given the time limit, let's stick to a robust deterministic pruning + a few random restarts if time permits?
    # No, determinism is preferred.
    
    # Let's try a "Swap" heuristic:
    # 1. Identify "critical" codewords: those where min(cover_count[balls[c]]) == 1.
    # 2. For each critical c, find the word u in balls[c] that is covered ONLY by c (cover_count[u] == 1).
    #    There might be multiple such u.
    # 3. Try to find a new codeword w (not selected) that covers u.
    # 4. If we add w, does it allow us to remove c?
    #    Actually, if we replace c with w:
    #    - Remove c: cover_count[balls[c]] -= 1
    #    - Add w: cover_count[balls[w]] += 1
    #    - Check if all words still covered (min cover_count == 1? No, >0)
    #    - Check if any other codeword becomes removable.
    
    # This is getting complicated. Let's try a different approach:
    # **LP Relaxation Rounding** as inspired by the prompt?
    # The prompt suggested LP. But scipy linprog on 6561 variables and 6561 constraints might be slow or memory heavy.
    # A_c is 6561x6561. That's 43 million entries. Too big for standard dense LP in 120s.
    
    # Let's stick to improving the heuristic.
    # **Iterated Local Search (ILS)**
    # 1. Perturb the current solution (remove k random codewords, re-cover greedily).
    # 2. Prune.
    # 3. Accept if better.
    
    # But we need determinism? The prompt doesn't explicitly forbid randomness, but "deterministic tie-breaking" was in the seed.
    # "Deterministic" is safer.
    
    # Let's try **Weighted Greedy** for the initial construction.
    # Standard greedy picks max coverage.
    # What if we pick based on a score that considers "overlap"?
    # Actually, the standard greedy is quite strong for this size.
    
    # Let's focus on **Better Pruning**.
    # Current pruning: remove if min count > 1.
    # This is a "First Fit Decreasing" style removal? No, it's just "Remove if possible".
    
    # **Pairwise Swap Search**:
    # For every pair (c1, c2) in selected:
    #   Try removing c1. If valid, check if we can remove c2? No.
    #   Try replacing c1 with w.
    
    # Given the time constraints and the fact that 486 is the target, and we are likely getting ~490-500,
    # a more sophisticated local search is needed.
    
    # **Strategy**: 
    # 1. Greedy
    # 2. Prune (multiple passes)
    # 3. **2-Opt Swap**:
    #    For each c in selected:
    #       For each w in balls[c] (neighbors of c):
    #          If w is not in selected:
    #             # Try swap: remove c, add w
    #             # Check validity
    #             # Check if this enables removal of another codeword
    #             # If yes, do it.
    
    # This is O(N * 17 * N) = O(N^2 * 17). 500 * 17 * 500 = 4.25M checks.
    # Each check involves:
    # - Temporarily modify cover_count
    # - Check min cover_count > 0 (validity)
    # - Check if any other codeword can be removed (min count > 1)
    
    # Let's implement this.
    
    if time.time() < deadline - 5.0:
        selected_set = set(selected)
        improved = True
        max_iterations = 10 # Safety
        iter_count = 0
        while improved and time.time() < deadline - 5.0 and iter_count < max_iterations:
            improved = False
            iter_count += 1
            new_selected = list(selected)
            new_selected_set = set(new_selected)
            
            # We need to update cover_count if we make a change.
            # It's safer to do one change at a time and update state.
            
            for i in range(len(new_selected)):
                if time.time() > deadline - 5.0:
                    break
                c = new_selected[i]
                # Try to replace c with a neighbor w
                for w in balls[c]:
                    if w in new_selected_set:
                        continue
                    
                    # Candidate swap: remove c, add w
                    # 1. Remove c
                    cover_count[balls[c]] -= 1
                    # Check if any word is now uncovered
                    if np.any(cover_count == 0):
                        cover_count[balls[c]] += 1 # Restore
                        continue
                    
                    # 2. Add w
                    cover_count[balls[w]] += 1
                    
                    # 3. Check if we can remove another codeword d (d != c, d != w)
                    #    We look for a d in new_selected (excluding c) such that min(cover_count[balls[d]]) > 1
                    #    Note: w is not in new_selected yet.
                    
                    removable_d = None
                    # Optimization: only check codewords that might be affected?
                    # Adding w increases counts for balls[w].
                    # Removing c decreases counts for balls[c].
                    # A codeword d is removable if min(cover_count[balls[d]]) > 1.
                    # Before swap, d was NOT removable (min <= 1).
                    # After swap:
                    # - If d is in balls[w] \ balls[c], counts increased.
                    # - If d is in balls[c] \ balls[w], counts decreased.
                    # - If d is in intersection, net change 0.
                    
                    # To be safe and simple, we check all d in new_selected (except c).
                    # This is O(N).
                    
                    # Actually, we can be smarter.
                    # Only d that were "critical" (min=1) before might become removable if their critical word got covered by w.
                    # And d that were "safe" (min>1) remain safe or become more safe.
                    
                    # Let's just do the full check for simplicity, N is small.
                    
                    for j in range(len(new_selected)):
                        if j == i: # c
                            continue
                        d = new_selected[j]
                        if np.min(cover_count[balls[d]]) > 1:
                            removable_d = d
                            break
                    
                    if removable_d is not None:
                        # Perform the swap: Remove c, Add w, Remove d
                        # We already did: Remove c, Add w
                        # Now Remove d
                        cover_count[balls[removable_d]] -= 1
                        
                        # Update selected list
                        new_selected[i] = w # Replace c with w
                        new_selected_set.remove(c)
                        new_selected_set.add(w)
                        
                        new_selected.remove(removable_d)
                        new_selected_set.remove(removable_d)
                        
                        improved = True
                        # Break inner loops to restart? Or continue?
                        # Restarting is safer for state consistency.
                        break
                    else:
                        # Swap didn't help remove another node.
                        # Restore state
                        cover_count[balls[w]] -= 1
                        cover_count[balls[c]] += 1
                    
                    if improved:
                        break
                
                if improved:
                    # Reset loop to restart with new state
                    selected = new_selected
                    selected_set = new_selected_set
                    break
            
            # If no improvement in this pass, break
            # (The 'improved' flag handles this)

    # 4. Final Pruning Pass
    # After swaps, we might have new removable nodes.
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected

    # 5. Final Verification and Conversion
    # Verify coverage
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        # If invalid, add missing words
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1

    # Convert to digit lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
