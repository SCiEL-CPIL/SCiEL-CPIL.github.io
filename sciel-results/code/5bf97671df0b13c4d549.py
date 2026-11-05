import numpy as np
import itertools
import time

def construct(q, n, radius):
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Build incidence matrix
    # incidence[i, j] = 1 if j is within distance <= radius of i
    incidence = np.zeros((N, N), dtype=np.uint8)
    
    for i, w in enumerate(all_words):
        # Self
        incidence[i, i] = 1
        if radius == 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    # Create new word
                    # Convert to list, modify, convert back to tuple for dict lookup
                    w_list = list(w)
                    w_list[pos] = new_val
                    neighbor = tuple(w_list)
                    j = word_to_idx[neighbor]
                    incidence[i, j] = 1
        elif radius > 1:
            # General case (slower, but robust)
            # Generate all words within distance radius
            # For small N, we can just check all pairs, but N=2187, N^2=4M, which is fine for setup
            pass # We assume radius=1 for the specific task, but let's add a fallback
            
    # If radius > 1, rebuild incidence by checking all pairs (N^2 is 4.7M, which is fast)
    if radius > 1:
         for i in range(N):
             for j in range(N):
                 # Hamming distance
                 dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                 if dist <= radius:
                     incidence[i, j] = 1

    # Heuristic: Greedy Set Cover
    def greedy_cover():
        covered = np.zeros(N, dtype=bool)
        code = []
        remaining = np.ones(N, dtype=bool)
        
        for _ in range(N):
            if not np.any(remaining):
                break
            # Find best candidate
            # scores[i] = number of remaining points covered by i
            scores = incidence @ remaining.astype(np.uint8)
            # We want max score.
            # If all scores are 0, we are stuck (should not happen)
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            code.append(best_idx)
            # Update covered and remaining
            covered[incidence[best_idx] == 1] = True
            remaining[incidence[best_idx] == 1] = False
            
        return code

    # Local Search: Deletion and Swap
    def local_search(code):
        code = list(code)
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts += incidence[c]
            
        improved = True
        while improved:
            improved = False
            
            # Deletion Pass
            changed = True
            while changed:
                changed = False
                new_code = []
                for c in code:
                    counts_without_c = counts - incidence[c]
                    if np.all(counts_without_c > 0):
                        # Can remove c
                        counts = counts_without_c
                        improved = True
                        changed = True
                    else:
                        new_code.append(c)
                if changed:
                    code = new_code
                    # Restart swap pass if deletion happened? 
                    # Usually yes, as counts changed.
                    # But we can continue to swap pass in same loop iteration.
                    # Let's just set improved=True and break to restart the while loop 
                    # to ensure counts are consistent for swap logic.
                    break 
            if not improved:
                # No deletion happened, try swap
                swapped = False
                for idx, c in enumerate(code):
                    counts_without_c = counts - incidence[c]
                    uncovered_mask = counts_without_c == 0
                    if not np.any(uncovered_mask):
                        # This is a deletion case, should have been caught.
                        # But if it happens, handle it.
                        code.pop(idx)
                        counts = counts_without_c
                        improved = True
                        swapped = True
                        break
                    if np.any(uncovered_mask):
                        # Need to find j not in code that covers uncovered_mask
                        required = uncovered_mask
                        # Find candidates j
                        # scores[j] = sum(incidence[j, k] for k where required[k])
                        scores = incidence @ required.astype(np.uint8)
                        max_score = scores.max()
                        total_required = np.sum(required)
                        if max_score == total_required:
                            # Find all j with max score
                            candidates = np.where(scores == max_score)[0]
                            # Exclude those in code
                            code_set = set(code)
                            valid_candidates = [j for j in candidates if j not in code_set]
                            if valid_candidates:
                                # Pick one, maybe random or first
                                new_c = valid_candidates[0]
                                # Replace c with new_c
                                code[idx] = new_c
                                # Update counts
                                counts = counts - incidence[c] + incidence[new_c]
                                improved = True
                                swapped = True
                                break
                if swapped:
                    # Restart loop to allow further deletions/swaps
                    continue
        return code

    # Multi-start
    best_code = None
    best_size = float('inf')
    
    start_time = time.time()
    time_limit = 110.0
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        # Randomize the order of ties in greedy?
        # Or just run greedy.
        code = greedy_cover()
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # If we found a very good code, we can break early? 
        # Best known is 186. If we hit 186, we might still try to find 185? 
        # Lower bound is 156. 
        # Let's keep running to try for better.

    return [list(all_words[c]) for c in best_code]
