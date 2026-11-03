import time
import random
import itertools

def construct(q, n, radius):
    # Specific logic for q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        # Simple greedy fallback
        all_words = list(itertools.product(range(q), repeat=n))
        word_index = {w: i for i, w in enumerate(all_words)}
        N = len(all_words)
        balls = []
        for i, w in enumerate(all_words):
            neighbors = [i]
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    nw = list(w); nw[pos] = val
                    neighbors.append(word_index[tuple(nw)])
            balls.append(neighbors)
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best = -1
            best_count = -1
            for i in range(N):
                if i in solution: continue
                count = sum(1 for nb in balls[i] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best = i
            if best == -1: break
            solution.append(best)
            for nb in balls[best]:
                uncovered.discard(nb)
        return [all_words[i] for i in solution]

    all_words = list(itertools.product(range(3), repeat=7))
    word_index = {w: i for i, w in enumerate(all_words)}
    N = len(all_words) # 2187
    
    # Precompute balls as lists of indices
    balls = []
    for i, w in enumerate(all_words):
        neighbors = [i]
        for pos in range(7):
            for val in range(3):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                neighbors.append(word_index[tuple(nw)])
        balls.append(neighbors)

    # Bitmask representation for fast set operations?
    # 2187 bits is ~274 bytes. Python int is efficient.
    # Let's use integer bitmasks for the "covered" set in local search checks if needed,
    # but for counting, arrays are better.
    
    # Helper: Greedy construction
    def greedy_construct():
        uncovered = [True] * N
        covered_count = 0
        solution = []
        
        # Precompute coverage potential? No, just iterate.
        # To speed up, we can keep a list of "candidates" but N=2187 is small enough
        # to scan all if we do it smartly.
        
        while covered_count < N:
            best_idx = -1
            best_gain = -1
            
            # Optimization: Only check candidates that are not in solution?
            # We need to check all to find the one with max gain.
            for i in range(N):
                if i in solution:
                    continue
                gain = 0
                for nb in balls[i]:
                    if uncovered[nb]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
                
            solution.append(best_idx)
            for nb in balls[best_idx]:
                if uncovered[nb]:
                    uncovered[nb] = False
                    covered_count += 1
        return solution

    # Helper: Local Search to minimize solution size
    def local_search(solution):
        # Ensure valid
        # We assume input is valid.
        sol_set = set(solution)
        
        # Count coverage
        cover_count = [0] * N
        for idx in solution:
            for nb in balls[idx]:
                cover_count[nb] += 1
        
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Remove redundant codewords
            # A codeword is redundant if all its neighbors are covered by at least one other codeword
            # i.e., cover_count[nb] >= 2 for all nb in balls[idx]
            
            i = 0
            while i < len(solution):
                idx = solution[i]
                redundant = True
                for nb in balls[idx]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    solution.pop(i)
                    sol_set.discard(idx)
                    for nb in balls[idx]:
                        cover_count[nb] -= 1
                    improved = True
                    # Do not increment i, check the new element at position i
                else:
                    i += 1
            
            if not improved:
                # Phase 2: Try swaps to enable removals
                # For each codeword c_i in solution, try to replace it with c_j
                # such that the resulting set allows removing some codewords.
                
                # This is expensive. We limit it.
                # Strategy:
                # 1. Pick a random codeword c_i from solution.
                # 2. Temporarily remove it.
                # 3. Find uncovered words.
                # 4. Find best c_j to cover them.
                # 5. Add c_j.
                # 6. Run Phase 1 (Remove Redundant).
                # 7. If size decreased, keep. Else revert.
                
                # To avoid getting stuck, we do a few random swap attempts.
                # If no improvement after a few attempts, break?
                # Or just try all swaps? N=2187, K~180. 180 * 2187 is ~400k checks.
                # Each check involves updating cover_count and scanning for redundants.
                # This might be too slow per iteration if we do it in a tight loop.
                
                # Let's try a limited number of random swaps.
                swaps_attempted = 0
                max_swaps_per_pass = 50
                
                while swaps_attempted < max_swaps_per_pass:
                    if not solution:
                        break
                    i = random.randint(0, len(solution) - 1)
                    c_i = solution[i]
                    
                    # Temporarily remove c_i
                    # We need to track the change to revert if bad.
                    # Instead of modifying the main array, let's simulate or use a copy?
                    # Copying is O(K). K~180. 50 * 180 = 9000 ops. Cheap.
                    
                    # But we need to check if adding c_j helps.
                    # Let's find the "deficit" caused by removing c_i.
                    # deficit = {nb in balls[c_i] | cover_count[nb] == 1}
                    # If deficit is empty, it was redundant (should have been caught in Phase 1).
                    
                    deficit = []
                    for nb in balls[c_i]:
                        if cover_count[nb] == 1:
                            deficit.append(nb)
                    
                    if not deficit:
                        # Should have been removed in Phase 1. 
                        # If we are here, maybe cover_count was updated incorrectly?
                        # Or maybe it's redundant but we missed it?
                        # Let's just remove it.
                        solution.pop(i)
                        sol_set.discard(c_i)
                        for nb in balls[c_i]:
                            cover_count[nb] -= 1
                        improved = True
                        break # Restart while loop to re-run Phase 1
                    
                    # Find best c_j to cover deficit
                    best_c_j = -1
                    best_cover_count = 0
                    # Only consider c_j that cover at least one deficit word
                    # To speed up, we can precompute for each word the list of words that cover it?
                    # That's the "reverse ball".
                    # Let's just scan all c_j not in sol_set.
                    
                    # Optimization: We can break early if we find a c_j that covers ALL deficit words?
                    # Not necessarily best, but good.
                    
                    for c_j in range(N):
                        if c_j in sol_set:
                            continue
                        # Count how many deficit words c_j covers
                        # We only care about words in deficit
                        # deficit is small (size of ball[c_i] is 15, so deficit <= 15)
                        count = 0
                        for nb in deficit:
                            if nb in balls[c_j]: # This is O(1) if balls[c_j] is a set? No, it's a list.
                                # We need faster membership check.
                                # Precompute ball_sets?
                                pass
                        # This inner loop is too slow if we do it for all N.
                        
                    # We need a better way to find c_j.
                    # Let's precompute `covering_words[nb]` = list of words that cover nb.
                    # Then for each nb in deficit, we have candidates.
                    # Intersection of candidates?
                    
                    # Let's build `covering_words` once.
                    # This is a change to the outer scope.
                    pass

        return solution

    # Precompute covering_words for faster swap search
    covering_words = [[] for _ in range(N)]
    for i in range(N):
        for nb in balls[i]:
            covering_words[nb].append(i)
    
    # Refine local_search with covering_words
    def local_search_v2(solution):
        sol_set = set(solution)
        cover_count = [0] * N
        for idx in solution:
            for nb in balls[idx]:
                cover_count[nb] += 1
        
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Remove redundant
            i = 0
            while i < len(solution):
                idx = solution[i]
                redundant = True
                for nb in balls[idx]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    solution.pop(i)
                    sol_set.discard(idx)
                    for nb in balls[idx]:
                        cover_count[nb] -= 1
                    improved = True
                else:
                    i += 1
            
            if not improved:
                # Phase 2: Swap search
                # Try to find a swap (c_i -> c_j) that allows removal of >= 1 other word
                # or simply reduces size? No, swap keeps size same.
                # Reduction happens if after swap, we can remove redundants.
                
                # We will try a limited number of random swaps.
                found_improvement = False
                for _ in range(20): # Limit attempts per phase
                    if not solution:
                        break
                    i = random.randint(0, len(solution) - 1)
                    c_i = solution[i]
                    
                    # Calculate deficit
                    deficit = []
                    for nb in balls[c_i]:
                        if cover_count[nb] == 1:
                            deficit.append(nb)
                    
                    if not deficit:
                        # Redundant, should be caught in Phase 1
                        # But maybe cover_count is stale? No, we update it.
                        # If we are here, it means cover_count[nb] > 1 for all nb?
                        # Then it IS redundant.
                        # Why wasn't it caught?
                        # Maybe because we are in the middle of a loop?
                        # Anyway, remove it.
                        solution.pop(i)
                        sol_set.discard(c_i)
                        for nb in balls[c_i]:
                            cover_count[nb] -= 1
                        improved = True
                        found_improvement = True
                        break
                    
                    # Find best c_j
                    # Candidates are words that cover at least one word in deficit.
                    # We can score them by how many deficit words they cover.
                    # Since deficit is small, we can gather candidates from covering_words[deficit_word]
                    
                    candidate_counts = {}
                    for nb in deficit:
                        for c_j in covering_words[nb]:
                            if c_j in sol_set:
                                continue
                            if c_j == c_i:
                                continue
                            candidate_counts[c_j] = candidate_counts.get(c_j, 0) + 1
                    
                    if not candidate_counts:
                        continue
                    
                    # Pick best candidate
                    best_c_j = max(candidate_counts, key=candidate_counts.get)
                    
                    # Perform swap: Remove c_i, Add best_c_j
                    # 1. Remove c_i
                    solution[i] = best_c_j # Replace in place
                    sol_set.discard(c_i)
                    sol_set.add(best_c_j)
                    
                    # Update cover_count
                    for nb in balls[c_i]:
                        cover_count[nb] -= 1
                    for nb in balls[best_c_j]:
                        cover_count[nb] += 1
                    
                    # Now try to remove redundants
                    # We don't know if this was a good swap.
                    # Let's run a quick redundant check.
                    # If we can remove any word, it's a good swap.
                    # If not, we should revert.
                    
                    # To avoid complex revert logic, let's just run Phase 1 logic here?
                    # But Phase 1 modifies the list.
                    
                    # Let's check if we can remove any word from the NEW solution.
                    # We need to check if any word in solution is redundant.
                    # This is O(K * B).
                    
                    # If we find one, we remove it, and the swap is "accepted" (improvement).
                    # If we don't find one, we revert the swap.
                    
                    removed_any = False
                    for j in range(len(solution)):
                        idx = solution[j]
                        redundant = True
                        for nb in balls[idx]:
                            if cover_count[nb] <= 1:
                                redundant = False
                                break
                        if redundant:
                            # Remove it
                            solution.pop(j)
                            sol_set.discard(idx)
                            for nb in balls[idx]:
                                cover_count[nb] -= 1
                            removed_any = True
                            break
                    
                    if removed_any:
                        improved = True
                        found_improvement = True
                        break # Restart while loop
                    else:
                        # Revert swap
                        solution[i] = c_i
                        sol_set.discard(best_c_j)
                        sol_set.add(c_i)
                        
                        for nb in balls[best_c_j]:
                            cover_count[nb] -= 1
                        for nb in balls[c_i]:
                            cover_count[nb] += 1

        return solution

    start_time = time.time()
    max_time = 115.0
    
    # Initial Greedy
    best_sol = greedy_construct()
    best_sol = local_search_v2(best_sol)
    best_size = len(best_sol)
    
    # Iterated Local Search
    while time.time() - start_time < max_time:
        # Perturb best_sol
        # Remove K random words, then repair?
        # Or just add random words and let local search clean up?
        # Adding words increases size. We want to explore smaller sizes.
        # Let's do: Take best_sol, remove 5 random words, add 5 random words.
        # Then run local search.
        
        current_sol = best_sol[:]
        
        # Perturbation
        # Remove a few
        num_remove = min(5, len(current_sol))
        if num_remove > 0:
            to_remove = random.sample(current_sol, num_remove)
            for idx in to_remove:
                current_sol.remove(idx)
        
        # Add some random
        num_add = min(5, N - len(current_sol))
        if num_add > 0:
            # Ensure we don't add duplicates
            existing = set(current_sol)
            to_add = []
            while len(to_add) < num_add:
                r = random.randint(0, N-1)
                if r not in existing:
                    to_add.append(r)
                    existing.add(r)
            current_sol.extend(to_add)
        
        # Local Search
        current_sol = local_search_v2(current_sol)
        
        if len(current_sol) < best_size:
            best_size = len(current_sol)
            best_sol = current_sol
            # Optional: restart perturbation strength?
            # print(f"Improved to {best_size}")

    return [all_words[i] for i in best_sol]
