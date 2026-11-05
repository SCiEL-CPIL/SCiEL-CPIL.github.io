import numpy as np
import time
import itertools
import random
import sys

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        # Simple fallback for other cases
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        # Generic fallback: random or full space (too slow for large N, but this branch is rarely hit in this task)
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_points = q ** n          # 6561
    ball_size = 1 + n * (q - 1)    # 17
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (ball of radius 1) for each index
    # Using arithmetic on the index to avoid string conversions
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        neighbors[idx, 0] = idx
        nb = 1
        for pos in range(n):
            orig = word[pos]
            pw = powers[pos]
            # +1
            nd = (orig + 1) % q
            delta = -2 * pw if nd == 0 else pw
            neighbors[idx, nb] = idx + delta
            nb += 1
            # -1
            nd = (orig - 1) % q
            delta = 2 * pw if nd == 2 else -pw
            neighbors[idx, nb] = idx + delta
            nb += 1

    # Precompute neighbor sets for fast intersection in local search if needed
    # But we will primarily use the array for counts.
    # Let's keep a list of sets for the swap candidate intersection logic
    neighbor_sets = [set(neigh.tolist()) for neigh in neighbors]

    def get_counts(code):
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        return counts

    def reduce_code(code, counts):
        """
        Reduces code by removing redundant words.
        Returns new_code and new_counts.
        """
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                # If all neighbors have count > 1, we can remove c
                if np.all(current_counts[neighbors[c]] > 1):
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    # --- Fast Greedy Initialization with Potential Tracking ---
    def fast_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        # potential[i] = number of points in neighbors[i] that are currently uncovered
        potential = np.full(total_points, ball_size, dtype=np.int32)
        selected = []
        
        # We need to efficiently find the max potential.
        # Since we only decrement, we can just scan or use a heap.
        # Scanning 6561 items is fast in numpy.
        while True:
            if time.time() > deadline - 20.0:
                break
            maxp = potential.max()
            if maxp <= 0:
                break
            
            # Find candidates with max potential
            cands = np.where(potential == maxp)[0]
            if len(cands) == 1:
                c = int(cands[0])
            else:
                # Random tie-break
                c = int(cands[rng.integers(len(cands))])
            
            selected.append(c)
            potential[c] = 0 # Mark as used/selected so we don't pick it again (though it will drop anyway)
            
            # Update potentials for all balls covering the points in neighbors[c]
            # For each point x in neighbors[c]:
            #   If x was uncovered, it is now covered.
            #   For every ball y that covers x, potential[y] -= 1
            for x in neighbors[c]:
                if not is_covered[x]:
                    is_covered[x] = True
                    # For every ball y in neighbors[x], reduce potential
                    # Note: neighbors[x] is the ball centered at x
                    potential[neighbors[x]] -= 1
                    # Clip to 0 to handle any logic errors or double counting (shouldn't happen)
                    np.maximum(potential, 0, out=potential)
            
            if is_covered.all():
                break
        
        return selected

    # --- Local Search ---
    def local_search(code, counts, deadline):
        code = list(code)
        counts = counts.copy()
        code_set = set(code)
        best_len = len(code)
        best_code = code[:]
        best_counts = counts.copy()
        rng = np.random.default_rng(7)
        
        no_improve = 0
        max_no_improve = 5000 # Limit iterations to avoid infinite loops on stuck states
        
        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            
            # 1. Reduction Pass
            old_len = len(code)
            code, counts = reduce_code(code, counts)
            if len(code) < best_len:
                best_len = len(code)
                best_code = code[:]
                best_counts = counts.copy()
                no_improve = 0
                continue
            elif len(code) == best_len and len(code) < old_len:
                # We reduced but didn't beat best, still reset no_improve slightly?
                pass
            
            if len(code) == old_len:
                # No reduction happened. Try swaps.
                
                # Sample a few codewords to try to swap
                sample_size = min(10, len(code))
                if sample_size > 0:
                    indices = rng.choice(len(code), size=sample_size, replace=False)
                    
                    for idx in indices:
                        if time.time() > deadline:
                            break
                        c = code[idx]
                        # Find unique points (count == 1) in c's ball
                        u_mask = counts[neighbors[c]] == 1
                        unique_points = neighbors[c][u_mask]
                        
                        if len(unique_points) == 0:
                            # Should have been removed, but maybe logic missed it?
                            # Force removal
                            code.remove(c)
                            code_set.discard(c)
                            counts[neighbors[c]] -= 1
                            improved = True
                            no_improve = 0
                            break
                        
                        # If too many unique points, unlikely to find a single replacement
                        if len(unique_points) > 3:
                            continue
                        
                        # Find candidates c' that cover all unique_points
                        # Intersection of neighbors[u] for u in unique_points
                        if len(unique_points) == 1:
                            candidates_set = set(neighbor_sets[unique_points[0]])
                        else:
                            candidates_set = set(neighbor_sets[unique_points[0]])
                            for u in unique_points[1:]:
                                candidates_set &= neighbor_sets[u]
                                if not candidates_set:
                                    break
                        
                        if not candidates_set:
                            continue
                        
                        # Filter candidates: must not be in code
                        candidates_set.difference_update(code_set)
                        
                        if not candidates_set:
                            continue
                        
                        # Pick a candidate
                        # To speed up, we can just pick random one or one with max overlap with current code?
                        # Actually, we want to replace c with c'.
                        # We need to ensure no other point becomes uncovered.
                        # Points in neighbors[c] that are NOT in neighbors[c'] will have count -1.
                        # Points in neighbors[c'] that are NOT in neighbors[c] will have count +1.
                        # Points in intersection will have count unchanged.
                        # So we only need to check points in neighbors[c] \ neighbors[c'].
                        # If any of those have count == 1, the swap is invalid.
                        
                        # Let's iterate through a few candidates
                        cand_list = list(candidates_set)
                        random.shuffle(cand_list)
                        
                        for c_prime in cand_list[:5]: # Limit candidates tried per swap attempt
                            # Check validity
                            # Points at risk: neighbors[c] where count is 1 AND not in neighbors[c_prime]
                            # We already know unique_points are covered by c_prime.
                            # So we just need to check if there are OTHER points in neighbors[c] with count 1 not in neighbors[c_prime].
                            
                            # Efficient check:
                            # Create a boolean array for neighbors[c_prime]
                            # Or just check:
                            # Are there any points in neighbors[c] with count==1 that are NOT in neighbors[c_prime]?
                            
                            # Let's use sets for this specific check as it's small (17 elements)
                            ball_c = set(neighbors[c].tolist())
                            ball_cp = set(neighbors[c_prime].tolist())
                            
                            # Points in ball_c with count 1
                            risky_in_c = ball_c & set(neighbors[c][counts[neighbors[c]] == 1].tolist())
                            
                            # If all risky points are in ball_cp, the swap is valid
                            if risky_in_c.issubset(ball_cp):
                                # Apply swap
                                old_code = code[:]
                                old_counts = counts.copy()
                                
                                code.remove(c)
                                code_set.discard(c)
                                code.append(c_prime)
                                code_set.add(c_prime)
                                
                                counts[neighbors[c]] -= 1
                                counts[neighbors[c_prime]] += 1
                                
                                # Verify no count < 1 (should be guaranteed by check, but safe)
                                if np.any(counts < 1):
                                    code = old_code
                                    counts = old_counts
                                    code_set = set(code)
                                    continue
                                
                                # Try to reduce
                                new_code, new_counts = reduce_code(code, counts)
                                
                                if len(new_code) < len(old_code):
                                    code = new_code
                                    counts = new_counts
                                    code_set = set(code)
                                    if len(code) < best_len:
                                        best_len = len(code)
                                        best_code = code[:]
                                        best_counts = counts.copy()
                                    improved = True
                                    no_improve = 0
                                    break # Restart loop
                                else:
                                    # No reduction. Keep the swap if it's neutral?
                                    # Usually we want to accept neutral moves to explore.
                                    code = new_code
                                    counts = new_counts
                                    code_set = set(code)
                                    improved = True
                                    no_improve = 0
                                    break # Restart loop
                            
                        if improved:
                            break # Break out of indices loop
                    
                    if not improved:
                        no_improve += 1
                        # If stuck, do a random kick
                        if no_improve % 100 == 0 and code:
                            c_idx = rng.integers(len(code))
                            c = code[c_idx]
                            cand = neighbors[c]
                            c_prime = cand[rng.integers(len(cand))]
                            if c_prime not in code_set:
                                old_code = code[:]
                                old_counts = counts.copy()
                                
                                code.remove(c)
                                code_set.discard(c)
                                code.append(c_prime)
                                code_set.add(c_prime)
                                
                                counts[neighbors[c]] -= 1
                                counts[neighbors[c_prime]] += 1
                                
                                if not np.any(counts < 1):
                                    code, counts = reduce_code(code, counts)
                                    code_set = set(code)
                                else:
                                    code = old_code
                                    counts = old_counts
                                    code_set = set(code)
                            no_improve = 0 # Reset after kick

        # Final Reduction
        best_code, _ = reduce_code(best_code, get_counts(best_code))
        
        return best_code

    # 1. Fast Greedy Initialization
    C = fast_greedy(42)
    
    # 2. Local Search
    counts = get_counts(C)
    C = local_search(C, counts, deadline)
    
    # Convert indices back to words
    all_words = list(itertools.product(range(q), repeat=n))
    # This conversion is slow. Let's do it efficiently.
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    return [idx_to_word(idx) for idx in C]
