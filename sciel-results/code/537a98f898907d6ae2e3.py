import itertools
import random
import sys
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # 1. Precomputation
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # ball_indices[i]: list of word indices in ball of words[i]
    # inverse_ball_indices[p]: list of word indices whose ball contains p
    # Note: p is a word index as well, since the space to cover is the same as the space of codewords.
    
    ball_indices = [[] for _ in range(N)]
    inverse_ball_indices = [[] for _ in range(N)]
    
    # Precompute neighbors (distance 1)
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        w = words[i]
        neighbors[i].append(i) # self
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    j = word_to_idx[new_w]
                    neighbors[i].append(j)
        
        for j in neighbors[i]:
            ball_indices[i].append(j)
            inverse_ball_indices[j].append(i)

    # 2. Greedy Set Cover
    def greedy_cover():
        uncovered = set(range(N))
        selected = []
        
        # Sort words? No, just iterate.
        # To speed up, we can maintain a list of uncovered points.
        # But N is small enough that scanning all candidates each time is O(N^2) ~ 40M ops, which is ok in Py if optimized?
        # 6561^2 = 43M. Python might be slow.
        # Better: Iterate candidates, count coverage in uncovered.
        
        while uncovered:
            best_idx = -1
            max_cover = 0
            # Randomize order to break ties differently in different runs?
            # For initial greedy, deterministic is fine, but randomized helps ILS.
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                # Count how many points in ball_indices[i] are in uncovered
                count = 0
                for p in ball_indices[i]:
                    if p in uncovered:
                        count += 1
                        if count > max_cover:
                            max_cover = count
                            best_idx = i
                            # Early stop if we found a perfect cover (unlikely)
                
                # Optimization: if max_cover is 17 (full ball) and we just started, etc.
                
            if best_idx == -1:
                break # Should not happen
                
            selected.append(best_idx)
            # Remove covered points
            for p in ball_indices[best_idx]:
                uncovered.discard(p)
                
        return selected

    # 3. Local Search Optimizer
    def optimize(selected):
        selected_set = set(selected)
        cover_count = [0] * N
        
        # Init cover counts
        for v in selected:
            for p in ball_indices[v]:
                cover_count[p] += 1
                
        current_len = len(selected)
        
        # Loop for a while or until no change
        while True:
            changed = False
            
            # Try to delete redundant codewords
            # A codeword v is redundant if for all p in ball(v), cover_count[p] > 1
            redundant_candidates = []
            for v in selected:
                is_redundant = True
                for p in ball_indices[v]:
                    if cover_count[p] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_candidates.append(v)
                    
            if redundant_candidates:
                # Remove one (random choice to diversify)
                v_remove = random.choice(redundant_candidates)
                selected_set.remove(v_remove)
                for p in ball_indices[v_remove]:
                    cover_count[p] -= 1
                current_len -= 1
                changed = True
                # Continue to check for more deletions immediately
                continue 
            else:
                # Try to swap
                # Pick a random codeword to try swapping
                if not selected:
                    break
                v_old = random.choice(selected)
                
                # Find points p in ball(v_old) with cover_count[p] == 1
                # These MUST be covered by the new codeword
                unique_points = []
                for p in ball_indices[v_old]:
                    if cover_count[p] == 1:
                        unique_points.append(p)
                        
                if not unique_points:
                    # This v_old is redundant? Why wasn't it caught?
                    # Ah, if unique_points is empty, it means all p have count > 1.
                    # Then it is redundant. The logic above should have caught it.
                    # Unless selected is empty?
                    # If selected is not empty, and no redundant, then every v has at least one unique point.
                    pass
                
                # Find candidates for v_new
                # v_new must be in intersection of inverse_ball_indices[p] for p in unique_points
                # And v_new not in selected_set
                
                # Start with first unique point's inverses
                candidates = set(inverse_ball_indices[unique_points[0]])
                for p in unique_points[1:]:
                    candidates &= set(inverse_ball_indices[p])
                
                # Filter out already selected
                candidates -= selected_set
                
                if candidates:
                    # Pick a candidate. 
                    # Heuristic: Pick one that covers the most "highly covered" points?
                    # Or just random.
                    v_new = random.choice(list(candidates))
                    
                    # Perform Swap
                    selected_set.remove(v_old)
                    selected_set.add(v_new)
                    
                    # Update cover counts
                    for p in ball_indices[v_old]:
                        cover_count[p] -= 1
                    for p in ball_indices[v_new]:
                        cover_count[p] += 1
                        
                    changed = True
                    # After swap, re-evaluate deletions in next iteration
                    continue
                
                # If no candidates, try next v_old?
                # To avoid getting stuck on one bad v_old, we can try a few
                # But for simplicity, if we can't swap this one, we might be stuck.
                # Break to restart? Or try another v_old.
                # Let's try a few random v_old's before giving up on this state.
                # We already picked random v_old. If it fails, try another?
                # Let's add a counter.
                pass
            
            if not changed:
                break
                
        return list(selected_set)

    # Main Driver
    start_time = time.time()
    time_limit = 115
    
    best_solution = None
    best_len = float('inf')
    
    # Run multiple iterations of Greedy + Optimize
    while time.time() - start_time < time_limit:
        # Random seed for greedy tie-breaking
        # The greedy function above shuffles indices, so it's randomized per call.
        sol = greedy_cover()
        
        # Optimize
        optimized_sol = optimize(sol)
        
        if len(optimized_sol) < best_len:
            best_len = len(optimized_sol)
            best_solution = optimized_sol[:]
            
        # Optional: Perturb best_solution and re-optimize (Simulated Annealing style)
        if best_solution and random.random() < 0.5:
            # Perturb: remove 1-2, add 1-2 random
            perturbed = best_solution[:]
            if len(perturbed) > 2:
                # Remove 1
                v_rm = random.choice(perturbed)
                perturbed.remove(v_rm)
                # Add 1 random
                v_add = random.randint(0, N-1)
                if v_add not in perturbed:
                    perturbed.append(v_add)
            
            # Check if valid? If not, optimize will try to fix it?
            # No, optimize assumes it's a valid cover.
            # A random perturbation likely breaks validity.
            # We need a "Repair" step.
            # If we add random words, it's likely still valid (superset).
            # If we remove, it might be invalid.
            # Let's stick to valid perturbations:
            # Swap v_old with v_new where validity is maintained? Hard to guarantee.
            # Better: Just re-run Greedy from scratch. The randomness in Greedy is enough.
            
    if best_solution is None:
        return [list(words[i]) for i in range(N)]
        
    return [list(words[i]) for i in best_solution]
