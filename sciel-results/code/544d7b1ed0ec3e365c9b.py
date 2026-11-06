import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(16))
    all_triples_list = list(itertools.combinations(all_points, 3))
    num_triples = len(all_triples_list)
    
    # Precompute masks and triple-to-index mapping
    triple_masks = {}
    triple_to_idx = {}
    for i, t in enumerate(all_triples_list):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[t] = mask
        triple_to_idx[t] = i
        
    # Precompute conflict bitsets
    conflict_bits = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[all_triples_list[i]]
        for j in range(i+1, num_triples):
            mj = triple_masks[all_triples_list[j]]
            inter = mi & mj
            if inter & (inter - 1):  # share > 1 point
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
    
    def calculate_alpha(active_mask):
        """
        Calculate the independence number (alpha) of the triple system defined by active_mask.
        Alpha is the size of the largest subset of points containing no triple.
        This is equivalent to 16 - (minimum hitting set size).
        We use branch and bound to find the minimum hitting set size tau.
        """
        if active_mask == 0:
            return 16
            
        active_indices = [i for i in range(num_triples) if (active_mask >> i) & 1]
        if not active_indices:
            return 16
        
        # Get the masks of active triples
        active_masks = [triple_masks[all_triples_list[i]] for i in active_indices]
        
        # Find minimum hitting set size tau
        # A hitting set H intersects every triple in active_masks.
        # We want to minimize |H|.
        
        # Branch and bound state:
        # remaining_triples: list of masks of triples not yet hit
        # current_size: number of points in H so far
        # best_tau: best found so far
        
        # Optimization: Sort active triples by some heuristic? 
        # Or just use a standard recursive search.
        # Since N=16, the search space for H is small if we prune well.
        
        # We can also iterate k from 1 to 8 (since tau <= 8 usually for good systems, alpha >= 8? 
        # Actually, for Steiner Triple Systems, alpha is around 7-8. 
        # If we find alpha=5, tau=11.
        # Let's just find min tau directly.
        
        # To speed up, we can check if a set of size k is a hitting set.
        # But B&B on points is usually faster for small N.
        
        # Let's try a different approach for speed:
        # Iterate k from 1 to 16. Check if there exists an independent set of size k.
        # Independent set of size k means NO triple is contained in it.
        # This is what the old code did, but it was slow.
        
        # Let's optimize the "is there an independent set of size k" check.
        # We can use a recursive backtracking to build an independent set.
        # State: current_set_mask, available_points_mask.
        # If we add point p, we must remove all points q such that there is a triple {p, q, r} in the active set?
        # No, if we add p, we just ensure no triple is *fully* in the set.
        # This is equivalent to: The set S is independent iff for all active triples T, T is not a subset of S.
        
        # Let's use the Hitting Set approach (Min Tau).
        # If we find a hitting set of size tau, alpha = 16 - tau.
        # We want to minimize tau.
        
        # Recursive function to find min hitting set
        def min_hitting_set_size(triples_masks, best_so_far):
            if not triples_masks:
                return 0
            if best_so_far == 1:
                return 1 # Can't be smaller than 1 if non-empty
            
            # Heuristic: pick the triple with the fewest points? All have 3.
            # Pick the first triple.
            t_mask = triples_masks[0]
            rest = triples_masks[1:]
            
            # Points in this triple:
            points = []
            m = t_mask
            while m:
                lsb = m & -m
                points.append(lsb.bit_length() - 1)
                m -= lsb
            
            min_val = best_so_far
            
            for p in points:
                # If we pick point p, we hit this triple.
                # We also hit all other triples that contain p.
                new_triples = []
                for tm in rest:
                    if (tm & (1 << p)) == 0:
                        new_triples.append(tm)
                
                # Recurse
                # Pruning: if current_depth + 1 >= best_so_far, we might prune, 
                # but we are returning the size, so we need to track depth or pass remaining budget.
                
                # Let's pass a limit: we want to know if we can find a hitting set of size <= best_so_far - 1.
                # Actually, simpler: just compute the min size for this branch and take min.
                
                # Optimization: If the number of remaining triples is 0, cost is 1.
                if not new_triples:
                    branch_cost = 1
                else:
                    # We need to find min hitting set for new_triples.
                    # To avoid deep recursion, we can use a helper that checks if a hitting set of size <= K exists.
                    # But let's stick to the direct min calculation with memoization if possible? 
                    # State is set of triples. Too many.
                    
                    # Let's use a simple recursive min.
                    branch_cost = 1 + min_hitting_set_size(new_triples, best_so_far - 1)
                
                if branch_cost < min_val:
                    min_val = branch_cost
                    if min_val == 1:
                        break # Can't do better than 1 for non-empty set? No, 1 is min if non-empty.
            
            return min_val

        # This recursive approach might be too slow without strong pruning.
        # Let's try the "Check Independent Set of Size K" approach again, but optimized.
        # We want the MAX K such that an independent set exists.
        # We can binary search K? No, independence is monotonic? 
        # If there is an independent set of size K, there is one of size K-1.
        # So we can check K=8, 9, 10... 
        # If we find one for K, we try K+1.
        
        # Let's implement a fast check for "Does an independent set of size K exist?"
        # Using backtracking with pruning.
        # We try to build a set S of size K.
        # Pruning: 
        # 1. If we have chosen 'chosen' points, and we need 'needed' more.
        # 2. If the remaining available points are less than 'needed', fail.
        # 3. If adding a point p creates a "conflict" that makes it impossible to fill the rest?
        #    A conflict is not direct. We just need to ensure no triple is fully in S.
        #    If we add p, we "block" all triples containing p? No, we just can't add the other 2 points of any triple containing p.
        #    Actually, if we add p, we must ensure that for every triple {p, x, y} in the system, we do NOT add both x and y.
        #    This is a constraint satisfaction problem.
        
        # Given the small size (16), a simple iterative deepening DFS for K might work if we order points well.
        
        # Let's try a different heuristic for Alpha:
        # Random sampling of subsets? No, need exact.
        
        # Let's go back to Min Hitting Set (Tau).
        # We can use an ILP solver? No, only standard library/NumPy/SciPy.
        # SciPy has `scipy.optimize.milp`? No, that's not standard in all versions or might be slow to setup.
        
        # Let's optimize the B&B for Hitting Set.
        # Key: Sort triples by "degree" (how many other triples they conflict with? No, hitting set doesn't care about conflict, it cares about coverage).
        # Actually, for Hitting Set, we want to pick points that cover many triples.
        
        # Let's implement a greedy + local search for Hitting Set to get a good upper bound, then B&B to verify optimality?
        # Or just B&B with a good upper bound from greedy.
        
        # Greedy Hitting Set:
        def greedy_hitting_set(triples_masks):
            remaining = list(triples_masks)
            hits = 0
            while remaining:
                # Find point that covers most remaining triples
                counts = [0] * 16
                for tm in remaining:
                    m = tm
                    while m:
                        lsb = m & -m
                        p = lsb.bit_length() - 1
                        counts[p] += 1
                        m -= lsb
                best_p = max(range(16), key=lambda x: counts[x])
                if counts[best_p] == 0:
                    break
                hits += 1
                # Remove all triples containing best_p
                new_remaining = []
                for tm in remaining:
                    if (tm & (1 << best_p)) == 0:
                        new_remaining.append(tm)
                remaining = new_remaining
            return hits
        
        ub = greedy_hitting_set(active_masks)
        
        # Now B&B to find exact min, using ub as initial best.
        # We can also use a lower bound: ceil(len(remaining) / 3) ? No, a point can hit 1 triple. 
        # Lower bound: max(1, ceil(len(remaining) / max_cover))? 
        # Simple LB: 1 if remaining else 0.
        
        def exact_min_hitting_set(triples_masks, current_best):
            if not triples_masks:
                return 0
            if current_best == 1:
                return 1
            
            # Lower bound
            if len(triples_masks) > 0 and current_best <= 1:
                 return 1 # We only care if we can improve current_best. If current_best is 1, we can't improve.
            
            # Pick a triple
            # Heuristic: Pick the triple that appears in the fewest "candidate" points? 
            # Or just the first one.
            t_mask = triples_masks[0]
            rest = triples_masks[1:]
            
            points = []
            m = t_mask
            while m:
                lsb = m & -m
                points.append(lsb.bit_length() - 1)
                m -= lsb
            
            best_val = current_best
            
            for p in points:
                # If we pick p, cost is 1 + min_hitting_set(rest_covered_by_p)
                # Prune: if 1 >= best_val, skip? No, 1 + ... >= 2.
                # If best_val is 1, we can return 1 immediately? 
                # Yes, if we found a hitting set of size 1, it's optimal for non-empty set.
                
                new_triples = [tm for tm in rest if (tm & (1 << p)) == 0]
                
                # Recursive call
                # We can pass best_val - 1 as the limit? 
                # We want to know if we can find a hitting set of size <= best_val - 1 for new_triples.
                # If yes, total is 1 + (something < best_val).
                
                # To keep it simple, we compute the min for the branch.
                branch_cost = 1 + exact_min_hitting_set(new_triples, best_val - 1)
                
                if branch_cost < best_val:
                    best_val = branch_cost
                    if best_val == 1:
                        break
            
            return best_val
        
        tau = exact_min_hitting_set(active_masks, ub)
        return 16 - tau

    def find_witness_independent_set(active_mask, target_size):
        """
        Find a specific independent set of size target_size.
        Returns the mask of the set or 0 if not found.
        """
        if target_size > 16:
            return 0
        if target_size == 0:
            return 0
            
        active_indices = [i for i in range(num_triples) if (active_mask >> i) & 1]
        active_masks = [triple_masks[all_triples_list[i]] for i in active_indices]
        
        # Backtracking to find a set of size target_size
        # We try to include points 0..15.
        # State: current_mask, remaining_points_mask, size_needed
        # Pruning: 
        # 1. If popcount(remaining_points_mask) < size_needed: fail
        # 2. If we have size_needed points, check if any triple is fully in current_mask.
        #    If yes, fail. If no, success.
        
        # To speed up, we can check conflicts incrementally.
        # But for target_size ~ 7-8, the search space is small.
        
        def backtrack(curr_mask, rem_mask, need):
            if need == 0:
                # Check if curr_mask is independent
                for tm in active_masks:
                    if (tm & curr_mask) == tm:
                        return 0 # Not independent
                return curr_mask
            
            if rem_mask == 0:
                return 0
                
            # Count bits in rem_mask
            if bin(rem_mask).count('1') < need:
                return 0
                
            # Try adding the lowest bit in rem_mask
            lsb = rem_mask & -rem_mask
            p = lsb.bit_length() - 1
            
            # Option 1: Include p
            new_curr = curr_mask | lsb
            new_rem = rem_mask & ~lsb
            
            # Check if adding p creates an immediate "impossibility"?
            # Not really, we just can't complete a triple.
            # But we can prune if adding p makes it impossible to reach 'need' without violating independence?
            # Hard to check cheaply.
            
            res = backtrack(new_curr, new_rem, need - 1)
            if res:
                return res
            
            # Option 2: Exclude p
            res = backtrack(curr_mask, new_rem, need)
            if res:
                return res
            
            return 0
        
        return backtrack(0, (1 << 16) - 1, target_size)

    def generate_initial():
        """Generate a random linear triple system using greedy method."""
        active_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
            if (conflict_bits[idx] & active_mask) == 0:
                active_mask |= (1 << idx)
        return active_mask

    def simulated_annealing(initial_mask, time_limit):
        """
        Simulated Annealing to minimize alpha.
        """
        current_mask = initial_mask
        current_alpha = calculate_alpha(current_mask)
        best_mask = current_mask
        best_alpha = current_alpha
        
        # Initial temperature
        T = 5.0
        cooling_rate = 0.995
        
        while time.time() < time_limit:
            if current_alpha == 5: # Theoretical minimum?
                break
                
            # Find witness independent set
            witness_mask = find_witness_independent_set(current_mask, current_alpha)
            
            # Generate a move
            # Strategy: 
            # 1. Pick a random triple from the witness set? No, witness set has NO triples.
            # 2. We want to ADD a triple that intersects the witness set.
            # 3. We might need to REMOVE a conflicting triple.
            
            # Get points in witness set
            witness_points = []
            m = witness_mask
            while m:
                lsb = m & -m
                witness_points.append(lsb.bit_length() - 1)
                m -= lsb
            
            # Find all triples that intersect witness set
            candidate_adds = []
            for i in range(num_triples):
                if (current_mask >> i) & 1:
                    continue
                tm = triple_masks[all_triples_list[i]]
                if tm & witness_mask:
                    candidate_adds.append(i)
            
            if not candidate_adds:
                # No triple can be added that hits the witness set? 
                # This means every triple that hits the witness set is already in the system?
                # Or maybe we need to remove a triple to make room?
                # If all candidates are already in the system, we can't add new ones.
                # We must swap.
                # Pick a random triple from current system to remove.
                # Then pick a random triple from candidates (which are now all in system? No, if they are in system, we removed one, so it's available? No, we removed a DIFFERENT one).
                # Actually, if candidate_adds is empty, it means no *inactive* triple hits the witness set.
                # This implies that all triples hitting the witness set are *active*.
                # To change the system, we must remove one of them and add another?
                # If we remove a triple that hits the witness set, the witness set might shrink or stay same.
                # Let's just do a random swap in this case.
                
                active_indices = [i for i in range(num_triples) if (current_mask >> i) & 1]
                if not active_indices:
                    break
                remove_idx = random.choice(active_indices)
                
                # Find a triple to add that doesn't conflict with the new system
                # Remove remove_idx from mask
                temp_mask = current_mask & ~(1 << remove_idx)
                
                # Find inactive triples that don't conflict
                add_candidates = []
                for i in range(num_triples):
                    if (temp_mask >> i) & 1:
                        continue
                    if (conflict_bits[i] & temp_mask) == 0:
                        add_candidates.append(i)
                
                if not add_candidates:
                    # Can't add anything, just revert or skip
                    continue
                    
                add_idx = random.choice(add_candidates)
                new_mask = temp_mask | (1 << add_idx)
                
            else:
                # Pick a random candidate to add
                add_idx = random.choice(candidate_adds)
                
                # Check conflicts
                conflicts = conflict_bits[add_idx] & current_mask
                if conflicts == 0:
                    new_mask = current_mask | (1 << add_idx)
                else:
                    # Need to remove conflicting triples
                    # Pick one random conflicting triple to remove? 
                    # Or remove all? Removing all might be too destructive.
                    # Let's remove one random conflicting triple.
                    conflict_indices = []
                    m = conflicts
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        conflict_indices.append(idx)
                        m -= lsb
                    
                    remove_idx = random.choice(conflict_indices)
                    new_mask = (current_mask & ~(1 << remove_idx)) | (1 << add_idx)
                    
                    # Ensure linear: check if new_mask is valid? 
                    # The construction ensures no conflicts with the rest if we only remove one?
                    # No, removing one might expose conflicts? No, removing a triple only reduces conflicts.
                    # But adding add_idx might conflict with others? 
                    # We checked conflicts of add_idx against current_mask.
                    # We removed one of them. Are there others? 
                    # If there were multiple conflicts, we only removed one. 
                    # So add_idx still conflicts with the others.
                    # This move is INVALID if we don't remove ALL conflicts.
                    
                    # So we must remove ALL conflicts to add add_idx.
                    # This is a "swap" of 1 for many.
                    # Let's do that.
                    new_mask = (current_mask & ~conflicts) | (1 << add_idx)
            
            # Calculate new alpha
            new_alpha = calculate_alpha(new_mask)
            
            # Acceptance criterion
            if new_alpha < current_alpha:
                current_mask = new_mask
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_mask = current_mask
            else:
                # Probabilistic acceptance
                delta = new_alpha - current_alpha
                if delta == 0:
                    # Accept with some probability to escape plateaus
                    if random.random() < 0.5:
                        current_mask = new_mask
                        current_alpha = new_alpha
                else:
                    # delta > 0
                    prob = 1.0 / (1.0 + delta) # Simple heuristic
                    # Or exp(-delta/T)
                    if T > 0.01:
                        prob = pow(2.718, -delta / T)
                    else:
                        prob = 0.0
                    
                    if random.random() < prob:
                        current_mask = new_mask
                        current_alpha = new_alpha
            
            T *= cooling_rate
            
            # If T is very low, reheat?
            if T < 0.01:
                T = 1.0
        
        return best_mask, best_alpha

    best_alpha = 16
    best_mask = 0
    
    # Run SA for a few restarts
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        current_mask = generate_initial()
        # Run SA with time limit
        sa_limit = deadline - 25
        best_mask, best_alpha = simulated_annealing(current_mask, sa_limit)
        
        # If we found a very good solution, try to polish it further
        if best_alpha <= 6:
             # Run SA again on the best found
             sa_limit = deadline - 10
             best_mask, best_alpha = simulated_annealing(best_mask, sa_limit)

    # Final conversion
    result = []
    for i in range(num_triples):
        if (best_mask >> i) & 1:
            result.append(list(all_triples_list[i]))
    
    return result
