import itertools
import random
import time
import copy

def construct(q, n, radius):
    # Specialized for K_3(7,1) but robust for small q^n
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters if needed, though task is specific
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

    # Specific logic for q=3, n=7, radius=1
    all_words = list(itertools.product(range(3), repeat=7))
    word_index = {w: i for i, w in enumerate(all_words)}
    N = len(all_words) # 2187
    
    # Precompute balls as bitmasks for fast set operations
    # ball_mask[i] is a bitmask where bit j is 1 if word j is within distance 1 of word i
    ball_masks = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                j = word_index[tuple(nw)]
                mask |= (1 << j)
        ball_masks[i] = mask
        
    FULL_MASK = (1 << N) - 1
    
    # Helper: Check if a set of indices covers all words
    # Returns (is_valid, uncovered_mask)
    def check_coverage(indices):
        covered = 0
        for idx in indices:
            covered |= ball_masks[idx]
        if covered == FULL_MASK:
            return True, 0
        return False, FULL_MASK ^ covered

    # Helper: Repair a solution to make it valid
    # Adds codewords to cover any uncovered points
    def repair_solution(indices):
        indices = list(indices)
        covered = 0
        for idx in indices:
            covered |= ball_masks[idx]
        
        uncovered = FULL_MASK ^ covered
        while uncovered:
            best_idx = -1
            best_count = -1
            # Find codeword covering most uncovered
            # Optimization: Only check candidates not already in solution
            for i in range(N):
                if i in indices: continue
                # Count how many bits in uncovered are set in ball_masks[i]
                # (uncovered & ball_masks[i]).bit_count()
                count = (uncovered & ball_masks[i]).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            indices.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return indices

    # Helper: Local Search - Remove redundant codewords
    def optimize_solution(indices):
        indices = list(indices)
        # Count coverage for each word
        # We can maintain a count array or just recompute masks
        # For efficiency, let's use a coverage count array
        cover_count = [0] * N
        for idx in indices:
            mask = ball_masks[idx]
            # Iterate set bits
            m = mask
            while m:
                lsb = m & -m
                idx_bit = lsb.bit_length() - 1
                cover_count[idx_bit] += 1
                m ^= lsb
                
        changed = True
        while changed:
            changed = False
            # Try to remove each codeword
            for i in range(len(indices)):
                idx = indices[i]
                mask = ball_masks[idx]
                # Check if all neighbors of idx are covered by others (count > 1)
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    if cover_count[idx_bit] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    # Remove it
                    indices.pop(i)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_bit = lsb.bit_length() - 1
                        cover_count[idx_bit] -= 1
                        m ^= lsb
                    changed = True
                    break # Restart loop after removal
        return indices

    # Helper: Swap local search
    # Try replacing one codeword with another to potentially allow further removals
    def swap_local_search(indices, max_iters=100):
        indices = list(indices)
        current_size = len(indices)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try all pairs (i, j) where i is in solution, j is not
            # This is O(K * N). K~180, N=2187. ~400k ops per pass.
            # We can optimize by only trying swaps that cover at least one new point
            # or free up a point.
            
            # Calculate coverage counts
            cover_count = [0] * N
            for idx in indices:
                mask = ball_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    cover_count[idx_bit] += 1
                    m ^= lsb
            
            for i in range(len(indices)):
                idx_remove = indices[i]
                mask_remove = ball_masks[idx_remove]
                
                # Points that become uncovered if we remove idx_remove
                # These are points where cover_count == 1
                # We need to find a replacement that covers these points.
                
                # For each candidate j not in indices:
                # If we swap i with j:
                # 1. Remove idx_remove: cover_count decreases for bits in mask_remove
                # 2. Add j: cover_count increases for bits in ball_masks[j]
                # 3. Check if valid (no point has cover_count 0)
                # 4. If valid, try to optimize (remove redundant)
                
                # Optimization: Only consider j that covers at least one "critical" point
                # Critical points: those with cover_count == 1 in mask_remove
                critical_mask = 0
                m = mask_remove
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    if cover_count[idx_bit] == 1:
                        critical_mask |= lsb
                    m ^= lsb
                
                if critical_mask == 0:
                    # idx_remove is redundant, should have been caught by optimize_solution
                    continue
                
                for j in range(N):
                    if j in indices: continue
                    # Does j cover any critical point?
                    if (ball_masks[j] & critical_mask) == 0:
                        continue
                    
                    # Potential swap. Check if it's valid.
                    # We don't want to build full cover_count every time.
                    # Just check if adding j covers all critical points that would be lost?
                    # No, removing i might uncover points that j doesn't cover.
                    # But if j covers all points in mask_remove that had count 1, 
                    # and j doesn't "break" anything else (which it can't, it only adds),
                    # then the swap is valid IF j covers all points in mask_remove that had count 1.
                    # Wait, if j covers a point that had count 1, that point stays covered.
                    # If j does NOT cover a point that had count 1, that point becomes uncovered.
                    # So we need: (mask_remove & ~ball_masks[j]) & critical_mask == 0
                    # i.e., all critical points in mask_remove must be in ball_masks[j].
                    
                    if (mask_remove & critical_mask & ~ball_masks[j]) == 0:
                        # Swap is valid!
                        new_indices = indices[:]
                        new_indices.pop(i)
                        new_indices.append(j)
                        
                        # Try to optimize the new solution
                        optimized = optimize_solution(new_indices)
                        if len(optimized) < current_size:
                            indices = optimized
                            current_size = len(indices)
                            improved = True
                            break # Restart the outer loop? Or just break inner?
                            # Breaking inner loop is enough to restart the "improved" check
                        elif len(optimized) == current_size:
                            # Same size, but maybe better structure? 
                            # For now, only accept if size reduces.
                            pass
                if improved:
                    break
        return indices

    # Genetic Algorithm Setup
    start_time = time.time()
    max_time = 115.0
    
    # Initialize population with diverse solutions
    # 1. Greedy solution
    # 2. Random valid solutions (repaired)
    population = []
    
    # Generate initial greedy solution
    def greedy_init():
        uncovered = FULL_MASK
        sol = []
        while uncovered:
            best = -1
            best_count = -1
            # Randomize tie breaking
            candidates = []
            max_c = 0
            for i in range(N):
                if i in sol: continue
                count = (uncovered & ball_masks[i]).bit_count()
                if count > max_c:
                    max_c = count
                    candidates = [i]
                elif count == max_c and count > 0:
                    candidates.append(i)
            if not candidates: break
            chosen = random.choice(candidates)
            sol.append(chosen)
            uncovered &= ~ball_masks[chosen]
        return optimize_solution(sol)
        
    pop_size = 20
    for _ in range(pop_size):
        # Mix of greedy and random
        if random.random() < 0.5:
            sol = greedy_init()
        else:
            # Random set of ~200 words, then repair
            k = random.randint(150, 250)
            random_sol = random.sample(range(N), k)
            sol = repair_solution(random_sol)
            sol = optimize_solution(sol)
        population.append(sol)
        
    best_solution = min(population, key=len)
    
    while time.time() - start_time < max_time:
        # Selection: Tournament
        new_population = []
        
        # Elitism: Keep best
        new_population.append(best_solution)
        
        while len(new_population) < pop_size:
            # Pick two parents
            p1, p2 = random.sample(population, 2)
            
            # Crossover: Uniform
            # Represent as sets for ease? No, lists are fine.
            # We need to ensure the child is roughly the same size or valid.
            # Simple approach: Take random mix of parents, then repair.
            
            # Determine target size? Or just mix?
            # Let's create a candidate by taking elements from p1 and p2.
            # To avoid explosion, we can limit the size.
            
            # A better crossover for set cover:
            # Start with empty set.
            # For each word in p1, if it covers many uncovered words in the context of p2?
            # Too complex.
            
            # Let's do:
            # 1. Take a random subset of p1 and p2.
            # 2. Repair.
            # 3. Optimize.
            
            # Simpler: Randomly pick a size K between len(p1) and len(p2) + noise.
            # Then pick K words from the union of p1 and p2.
            
            union_parents = set(p1) | set(p2)
            target_size = random.randint(min(len(p1), len(p2)), max(len(p1), len(p2)) + 10)
            target_size = min(target_size, N)
            
            if random.random() < 0.8:
                # Mix from union
                if len(union_parents) >= target_size:
                    child = random.sample(list(union_parents), target_size)
                else:
                    child = list(union_parents)
                    # Fill with random
                    while len(child) < target_size:
                        r = random.randint(0, N-1)
                        if r not in child:
                            child.append(r)
            else:
                # Pure mutation of best
                child = best_solution[:]
                # Add/Remove random
                for _ in range(3):
                    if random.random() < 0.5:
                        r = random.randint(0, N-1)
                        if r not in child:
                            child.append(r)
                    else:
                        if child:
                            child.pop(random.randint(0, len(child)-1))
                            
            # Repair and Optimize
            child = repair_solution(child)
            child = optimize_solution(child)
            
            # Occasional deep local search on the best child
            if random.random() < 0.2:
                 # Try swap local search on this child? 
                 # swap_local_search is expensive. 
                 # Let's just do another round of removal.
                 child = optimize_solution(child)
                 
            new_population.append(child)
            
        # Update best
        for sol in new_population:
            if len(sol) < len(best_solution):
                best_solution = sol
                
        # Update population (keep some diversity)
        # Sort by size
        new_population.sort(key=len)
        # Keep top 80%
        cutoff = int(pop_size * 0.8)
        population = new_population[:cutoff]
        # Fill rest with random mutations of best
        while len(population) < pop_size:
            mut = best_solution[:]
            for _ in range(5):
                if random.random() < 0.5:
                    r = random.randint(0, N-1)
                    if r not in mut:
                        mut.append(r)
                else:
                    if mut:
                        mut.pop(random.randint(0, len(mut)-1))
            mut = repair_solution(mut)
            mut = optimize_solution(mut)
            population.append(mut)
            
    return [all_words[i] for i in best_solution]
