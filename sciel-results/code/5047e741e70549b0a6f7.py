import time
import numpy as np
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors: neighbors[w] = list of 17 indices (w itself + 16 neighbors)
    # Shape: (6561, 17)
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1
                
    # Precompute inverse neighbors for faster swap search? 
    # neighbors_inv[w] = list of codewords that cover w. 
    # This is the transpose of the incidence matrix.
    # Since the graph is regular and symmetric, neighbors_inv is just the same structure, 
    # but we might need it for "who covers w" queries.
    # Actually, for swaps, we need: given a set of words U, find a codeword c such that U subset of neighbors[c].
    # This is equivalent to: intersection of neighbors_inv[u] for u in U.
    
    # Let's build an incidence matrix or sparse representation for faster set operations?
    # For n=8, 17 neighbors is small. 
    # neighbors_inv can be constructed by inverting neighbors.
    # But 6561 * 17 is small enough that we can just use the neighbors array directly 
    # if we organize our data structures well.
    
    best_size = total
    best_set = None
    
    def run_greedy_and_local_search(seed_val):
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        # 1. Greedy Initialization
        # Start with empty set
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        # Greedy Step
        while uncovered_count > 0 and time.time() < deadline - 20.0:
            # Count how many uncovered words each candidate covers
            # neighbors shape (6561, 17), uncovered_mask shape (6561,)
            # uncovered_mask[neighbors] gives (6561, 17) boolean array
            # .sum(axis=1) gives count per codeword
            counts = uncovered_mask[neighbors].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Tie-breaking: pick randomly among max candidates
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
            
            # Early stop if we already exceeded best known
            if len(selected) >= best_size:
                break
        
        if uncovered_count > 0:
            return None, None, len(selected)
            
        # 2. Local Search: Removal and Swaps
        # We iterate until no improvement is found or time runs out
        
        # Optimization: 
        # Removal: A codeword c is removable if for all w in neighbors[c], cover_count[w] > 1.
        # Equivalently, min(cover_count[neighbors[c]]) > 1.
        
        # Swaps: 
        # Remove c, let U be words with cover_count 1 in neighbors[c].
        # Find c' not in selected such that U is subset of neighbors[c'].
        # This means c' must be in neighbors_inv[u] for all u in U.
        
        # To speed up swap search, we can precompute a "coverage profile" or just use the neighbors array.
        # Given small size of U (usually small), we can find candidates by intersecting lists.
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # --- Phase 1: Batch Removal ---
            # Compute min coverage in each ball
            # min_cover_in_ball[i] = min(cover_count[neighbors[i]])
            min_cover_in_ball = np.min(cover_count[neighbors], axis=1)
            
            # Candidates for removal: min_cover_in_ball > 1
            removable_indices = np.where(min_cover_in_ball > 1)[0]
            
            if len(removable_indices) > 0:
                # Remove all at once? 
                # Careful: removing one might make another non-removable? 
                # No, removing a codeword only *decreases* cover counts.
                # If min_cover > 1, it means every word in the ball is covered by at least 2 codewords.
                # If we remove one, they are covered by at least 1.
                # However, if two codewords are "interdependent" (e.g., they are the only two covering a specific word w),
                # then min_cover for both would be 1 (because w has count 2, but if we remove one, w has count 1).
                # Wait. If w is covered by c1 and c2. cover_count[w] = 2.
                # min_cover_in_ball[c1] includes w, so min is 2? No, min is the minimum count in the ball.
                # If w is in ball(c1), count is 2. If another word v in ball(c1) is covered by only c1, count is 1.
                # So min_cover_in_ball[c1] would be 1.
                # So the condition min_cover > 1 correctly identifies codewords that do NOT have any unique words.
                # Removing any such codeword is safe.
                # Can we remove them all at once?
                # If c1 and c2 are both removable, and they share a word w.
                # count[w] >= 2.
                # If we remove c1, count[w] decreases by 1.
                # If we remove c2, count[w] decreases by 1.
                # If count[w] was exactly 2, and we remove both, count[w] becomes 0.
                # So we CANNOT remove them all at once if they share words with count exactly 2.
                
                # Safe strategy: Remove one by one, but use vectorized check to find candidates.
                # Or, remove a maximal independent set of removable codewords?
                # Simpler: Just remove one at a time, but recompute min_cover_in_ball only when necessary?
                # Actually, recomputing min_cover_in_ball is O(N*17). N=6561. 111k ops. Very fast.
                
                # Let's remove them one by one to be safe, but pick the one that frees up the most "slack"?
                # Or just pick the first one.
                # To speed up, we can remove a subset that are guaranteed not to conflict.
                # Two codewords c1, c2 conflict if they share a word w with cover_count[w] == 2.
                
                # Let's just pick one removable codeword, remove it, and mark improved=True.
                # The loop will catch the rest.
                # To make it faster, we can try to remove a batch of non-conflicting ones.
                # But for simplicity and correctness, let's do single removals but vectorized detection.
                
                # Pick a random removable one to avoid bias
                idx_to_remove = removable_indices[np.random.randint(len(removable_indices))]
                
                selected.remove(idx_to_remove)
                cover_count[neighbors[idx_to_remove]] -= 1
                improved = True
                # Continue loop to check for more removals
                continue
            
            # --- Phase 2: Swaps ---
            # Find a codeword to swap out.
            # Prefer codewords with many "unique" words? No, prefer those with fewest unique words to make swap easier?
            # Or just iterate through all.
            
            # To speed up, we can identify "bad" codewords: those with high number of unique words?
            # Actually, we want to find a swap that results in a valid code.
            # Removing c leaves U = {w in neighbors[c] | cover_count[w] == 1}.
            # We need c' such that U subset neighbors[c'].
            
            # Let's try a few random codewords for swapping to save time, or all?
            # N is ~400-500. Iterating all is 500 * (cost of swap check).
            # Cost of swap check:
            # 1. Identify U. (17 ops)
            # 2. If U empty, remove (handled in Phase 1).
            # 3. Find candidates c'. 
            #    Candidates must cover all u in U.
            #    c' is in intersection of neighbors_inv[u] for u in U.
            
            # Let's precompute neighbors_inv? 
            # neighbors_inv[w] is list of codewords covering w.
            # Since graph is symmetric, neighbors_inv[w] = [c | w in neighbors[c]].
            # We can build this.
            # neighbors_inv shape (6561, variable length). Max length 17.
            # Let's use a list of lists or a padded array.
            
            # Actually, we can just use the neighbors array.
            # To find c' covering u, we look at all c' and check if u is in neighbors[c'].
            # This is slow.
            
            # Better: Precompute an inverse adjacency matrix.
            # inv_neighbors[w] = list of c such that dist(c,w) <= 1.
            
            # Let's construct inv_neighbors once.
            # It's the same as neighbors because the relation is symmetric!
            # dist(c,w) <= 1 iff dist(w,c) <= 1.
            # So neighbors_inv is just neighbors.
            
            # So, to find c' covering all u in U:
            # We need c' such that for all u in U, u is in neighbors[c'].
            # This is equivalent to: c' is in the intersection of the sets {c | u in neighbors[c]} for each u in U.
            # Let S_u = {c | u in neighbors[c]}.
            # Note that S_u is exactly neighbors_inv[u], which is neighbors[u] (symmetric).
            # Wait. neighbors[u] is the list of words at dist <= 1 from u.
            # Is c in neighbors[u] iff u in neighbors[c]? Yes.
            # So the set of codewords covering u is exactly neighbors[u].
            
            # So we need c' in intersection of neighbors[u] for u in U.
            
            # Algorithm for Swap:
            # 1. Pick a candidate c_out from selected.
            # 2. Compute U = [w for w in neighbors[c_out] if cover_count[w] == 1]
            # 3. If U is empty, remove c_out (should have been caught in Phase 1, but maybe race condition or just safe).
            # 4. If U is not empty:
            #    - Start with candidates = neighbors[U[0]]
            #    - For each subsequent u in U[1:]:
            #        - Filter candidates to only those in neighbors[u]
            #        - If candidates becomes empty, break.
            #    - Filter candidates to exclude selected.
            #    - If candidates not empty, pick one (randomly or best heuristic) and swap.
            
            # To speed up, we can vectorize the intersection step?
            # U is small (usually 1-5).
            
            # Let's try swaps for a limited number of random codewords to save time.
            # If we don't find a swap, we might be stuck.
            
            # Heuristic: Try codewords with smallest "redundancy"? 
            # Or just random.
            
            num_swaps_tried = 0
            max_swaps_to_try = 50 # Limit per iteration to avoid stalling
            
            while num_swaps_tried < max_swaps_tried and time.time() < deadline - 5.0:
                num_swaps_tried += 1
                if len(selected) == 0: break
                
                idx_c_out = np.random.randint(len(selected))
                c_out = selected[idx_c_out]
                
                # Find unique words
                ball = neighbors[c_out]
                unique_words = ball[cover_count[ball] == 1]
                
                if len(unique_words) == 0:
                    # Should be removable, but maybe min_cover check missed it?
                    # Or maybe it's just redundant.
                    selected.pop(idx_c_out)
                    cover_count[neighbors[c_out]] -= 1
                    improved = True
                    break # Restart removal phase
                
                # Find replacement
                # Intersection of neighbors[u] for u in unique_words
                # Start with neighbors of first unique word
                # We need to check if a codeword c' is in neighbors[u].
                # Since neighbors is a matrix, we can't easily do set intersection.
                # But we can iterate through neighbors[u0] and check if they cover others.
                
                candidates = neighbors[unique_words[0]]
                
                # Filter candidates
                # For each candidate c' in candidates:
                #   Check if all u in unique_words[1:] are in neighbors[c']
                
                # Vectorized check:
                # candidates shape (17,)
                # We want to keep c' if for all u in unique_words[1:], u is in neighbors[c']
                
                # This is a bit tricky to vectorize directly without loops over u.
                # But len(unique_words) is small.
                
                valid_candidates = []
                for c_prime in candidates:
                    # c_prime is an int index
                    # Check if unique_words[1:] subset neighbors[c_prime]
                    # neighbors[c_prime] is an array of 17 ints.
                    # We can check if all u are in it.
                    # Since 17 is small, a loop or set is fast.
                    # Or: np.isin(unique_words[1:], neighbors[c_prime]).all()
                    
                    if len(unique_words) == 1:
                        valid_candidates.append(c_prime)
                    else:
                        # Check coverage
                        # unique_words[1:] shape (k,)
                        # neighbors[c_prime] shape (17,)
                        # np.isin returns boolean array of size k
                        if np.all(np.isin(unique_words[1:], neighbors[c_prime])):
                            valid_candidates.append(c_prime)
                
                if not valid_candidates:
                    continue
                
                # Filter out already selected
                selected_set = set(selected)
                valid_candidates = [c for c in valid_candidates if c not in selected_set]
                
                if not valid_candidates:
                    continue
                
                # Pick a replacement
                c_in = valid_candidates[np.random.randint(len(valid_candidates))]
                
                # Perform swap
                selected[idx_c_out] = c_in
                cover_count[neighbors[c_out]] -= 1
                cover_count[neighbors[c_in]] += 1
                improved = True
                break # Restart loop to check for more removals/swaps
        
        return selected, cover_count, len(selected)

    # Main Loop: Multi-start
    # We have ~100s. 
    # Each run takes maybe 5-10s?
    # So we can do ~10-20 runs.
    
    # Initialize with a good heuristic start?
    # Greedy from scratch is O(N * 6561 * 17) ~ 600 * 110k ~ 66M ops. ~0.1-0.5s in NumPy.
    # Local search is the main cost.
    
    num_runs = 20
    for i in range(num_runs):
        if time.time() > deadline - 10.0:
            break
            
        seed_val = i * 1000 + 42
        selected, cover_count, size = run_greedy_and_local_search(seed_val)
        
        if selected is not None and size < best_size:
            best_size = size
            best_set = selected.copy()
            
            if best_size <= 486:
                break

    # If no best_set found (shouldn't happen), fallback to greedy
    if best_set is None:
        # Just run one greedy
        selected, _, _ = run_greedy_and_local_search(0)
        best_set = selected

    # Convert best_set to list of digit lists
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
