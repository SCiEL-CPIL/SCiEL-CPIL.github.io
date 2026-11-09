import itertools
import random
import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0   # Safety margin under the 90s kill
    N = 4096

    # Helper to check if v can be added to chosen set with existing sums
    def check_add(v, chosen, sums):
        """
        Checks if v can be added to the set 'chosen' without creating duplicate triple sums.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        new_sums = set()
        # Triple (v, v, v)
        new_sums.add(3 * v)
        
        # Triples (v, v, u) and (v, u, u) for u in chosen
        for u in chosen:
            new_sums.add(2 * v + u)
            new_sums.add(v + 2 * u)
            
        # Triples (v, u, w) for u, w in chosen with u <= w
        for i in range(len(chosen)):
            u = chosen[i]
            for j in range(i, len(chosen)):
                w = chosen[j]
                new_sums.add(v + u + w)
                
        # Check for conflicts with existing sums
        if new_sums & sums:
            return False, None
        return True, new_sums

    # NumPy-vectorized collision check for a candidate v against existing sums
    def check_add_numpy(v, chosen_arr, sums_np):
        """
        Fast check using numpy arrays.
        chosen_arr: numpy array of current elements
        sums_np: numpy array of all existing triple sums
        Returns True if v can be added without collision.
        """
        m = len(chosen_arr)
        if m == 0:
            return True
        
        # Generate all new triple sums involving v
        # (v,v,v), (v,v,u), (v,u,u), (v,u,w) for u<=w in chosen
        new_sums_list = [3*v]
        
        # (v,v,u) and (v,u,u)
        new_sums_list.extend(2*v + chosen_arr)
        new_sums_list.extend(v + 2*chosen_arr)
        
        # (v,u,w) for u<=w
        for i in range(m):
            for j in range(i, m):
                new_sums_list.append(v + chosen_arr[i] + chosen_arr[j])
        
        new_sums_arr = np.array(new_sums_list, dtype=np.int64)
        
        # Check for any overlap with existing sums
        # Use set intersection for efficiency
        new_sums_set = set(new_sums_list)
        existing_sums_set = set(sums_np.tolist())
        
        return len(new_sums_set & existing_sums_set) == 0

    # Compute all triple sums for a set
    def compute_triple_sums(chosen):
        """Compute all unique triple sums (with replacement) for a set."""
        m = len(chosen)
        sums = set()
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    sums.add(chosen[i] + chosen[j] + chosen[k])
        return sums

    # Full validation for safety
    def is_valid(vals):
        m = len(vals)
        seen = set()
        for i in range(m):
            a = vals[i]
            for j in range(i, m):
                b = vals[j]
                for k in range(j, m):
                    c = vals[k]
                    s = a + b + c
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    # Greedy construction from a specific order of candidates
    def greedy_build(order, limit):
        chosen = []
        sums = set()
        for v in order:
            if time.time() > limit:
                break
            ok, ns = check_add(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        return chosen

    # Compute collision density for candidate v given current set
    def collision_density(v, chosen, sums):
        """
        Estimate how many triple-sum collisions would occur if v were added.
        Returns the number of new sums that conflict with existing sums.
        """
        new_sums = set()
        new_sums.add(3 * v)
        for u in chosen:
            new_sums.add(2 * v + u)
            new_sums.add(v + 2 * u)
        for i in range(len(chosen)):
            u = chosen[i]
            for j in range(i, len(chosen)):
                w = chosen[j]
                new_sums.add(v + u + w)
        return len(new_sums & sums)

    # Simulated annealing to improve the set
    def simulated_annealing(chosen, limit, max_iter=500):
        """
        Uses simulated annealing to escape local optima.
        Moves: add, remove, or swap elements.
        Accepts non-improving moves with probability based on temperature.
        """
        current = list(chosen)
        current_sums = set()
        for x in itertools.combinations_with_replacement(current, 3):
            current_sums.add(sum(x))
        
        current_score = len(current)
        best = list(current)
        best_sums = set(current_sums)
        best_score = current_score
        
        T = 10.0  # Initial temperature
        cooling = 0.995
        
        for iteration in range(max_iter):
            if time.time() > limit:
                break
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random element
                candidates = [v for v in range(1, N + 1) if v not in set(current)]
                if not candidates:
                    continue
                v = random.choice(candidates)
                
                # Check if valid
                ok, ns = check_add(v, current, current_sums)
                if not ok:
                    # Compute collision density for scoring
                    col = collision_density(v, current, current_sums)
                    new_score = current_score  # Size unchanged if we don't add
                    delta = new_score - current_score - col * 0.1
                    if delta > 0 or random.random() < np.exp(delta / T):
                        # Accept but don't actually add (it's invalid)
                        pass
                    continue
                else:
                    new_current = current + [v]
                    new_sums = current_sums | ns
                    new_score = len(new_current)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current = new_current
                        current_sums = new_sums
                        current_score = new_score
                        if current_score > best_score:
                            best = list(current)
                            best_sums = set(current_sums)
                            best_score = current_score
                            
            elif move_type < 0.7:
                # Remove a random element
                if not current:
                    continue
                idx = random.randrange(len(current))
                v = current[idx]
                new_current = current[:idx] + current[idx+1:]
                
                # Recompute sums
                new_sums = set()
                for x in itertools.combinations_with_replacement(new_current, 3):
                    new_sums.add(sum(x))
                
                new_score = len(new_current)
                delta = new_score - current_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    current = new_current
                    current_sums = new_sums
                    current_score = new_score
                    if current_score > best_score:
                        best = list(current)
                        best_sums = set(current_sums)
                        best_score = current_score
                        
            else:
                # Swap: remove one, add another
                if not current:
                    continue
                idx = random.randrange(len(current))
                v_remove = current[idx]
                new_current = current[:idx] + current[idx+1:]
                
                candidates = [v for v in range(1, N + 1) if v not in set(new_current)]
                if not candidates:
                    continue
                v_add = random.choice(candidates)
                
                # Check if v_add can be added to new_current
                ok, ns = check_add(v_add, new_current, set())
                # Need to recompute sums for new_current first
                new_current_sums = set()
                for x in itertools.combinations_with_replacement(new_current, 3):
                    new_current_sums.add(sum(x))
                
                ok, ns = check_add(v_add, new_current, new_current_sums)
                if not ok:
                    continue
                    
                final_current = new_current + [v_add]
                final_sums = new_current_sums | ns
                final_score = len(final_current)
                delta = final_score - current_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    current = final_current
                    current_sums = final_sums
                    current_score = final_score
                    if current_score > best_score:
                        best = list(current)
                        best_sums = set(current_sums)
                        best_score = current_score
            
            T *= cooling
            if T < 0.01:
                break
        
        return best

    best = []
    
    # Strategy 1: Base-3-like construction with large digit gaps
    # Use elements that are multiples of powers of 3 with gaps to ensure uniqueness
    # Then compress via local search
    
    def base3_construction():
        """
        Construct a set using base-3-like representation.
        Elements: a_i = c * 3^i for some constant c, then compress.
        """
        # Start with powers of 3 scaled to fit in [1, 4096]
        # 3^0=1, 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243, 3^6=729, 3^7=2187
        # 3^8=6561 > 4096, so we can have at most 8 pure powers
        
        # But we can do better: use a_i = floor(alpha * 3^i) for some alpha
        # Or use a_i = c_i * 3^i where c_i are small digits
        
        # Simple approach: start with powers of 3, then greedily add more
        chosen = [1, 3, 9, 27, 81, 243, 729, 2187]
        
        # Verify this is valid
        if not is_valid(chosen):
            return []
        
        sums = compute_triple_sums(chosen)
        
        # Now greedily add more elements, trying small numbers first
        # to maximize density
        for v in range(1, N + 1):
            if time.time() > DEADLINE - 15:
                break
            if v in chosen:
                continue
            ok, ns = check_add(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        
        return chosen

    base3_result = base3_construction()
    if is_valid(base3_result) and len(base3_result) > len(best):
        best = base3_result

    # Strategy 2: Randomized Greedy Restarts
    random.seed(42)
    
    # We will run multiple trials with different random orderings
    num_trials = 50
    trial_time = (DEADLINE - 5) / num_trials
    
    for t in range(num_trials):
        if time.time() > DEADLINE - 5:
            break
        
        # Generate a random order
        # Mix of small numbers and random numbers
        # Option A: Pure random permutation
        # Option B: Sorted, but with random swaps
        # Option C: Random weights
        
        # Let's try a "Randomized Ascending" approach:
        # Pick the next candidate from the smallest K available, with some probability picking a larger one.
        
        # Simpler: Random permutation of 1..N
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Bias towards smaller numbers?
        # Sort by (random() * 1000 + v) ?
        
        # Let's just use a shuffled list.
        # To improve, we can try "Greedy with Randomized Tie-Breaking"
        # Or "Greedy from Random Subsets"
        
        # Try 1: Pure Random
        res = greedy_build(order, DEADLINE - 5)
        if is_valid(res) and len(res) > len(best):
            best = res
            
        # Try 2: Random but biased to small
        # Create a list where small numbers appear more often?
        # No, greedy takes first valid.
        # Let's try a different order: Sort by v * random()
        order2 = list(range(1, N + 1))
        keys = [v * random.uniform(0.5, 2.0) for v in order2]
        order2 = [v for v, k in sorted(zip(order2, keys))]
        
        if time.time() < DEADLINE - 5:
            res2 = greedy_build(order2, DEADLINE - 5)
            if is_valid(res2) and len(res2) > len(best):
                best = res2

    # Strategy 3: Dense Local Search - try to compress spacing
    def dense_local_search(chosen, limit):
        """
        Try to replace elements with smaller values to compress the set,
        while maintaining uniqueness. This can allow adding more elements.
        """
        current = sorted(chosen)
        current_sums = compute_triple_sums(current)
        current_score = len(current)
        best_local = list(current)
        best_local_score = current_score
        
        # Try to decrease each element to a smaller unused value
        improved = True
        while improved and time.time() < limit:
            improved = False
            for idx in range(len(current)):
                old_val = current[idx]
                # Try all values smaller than old_val that aren't in the set
                for new_val in range(1, old_val):
                    if new_val in current:
                        continue
                    
                    # Check if we can replace old_val with new_val
                    new_set = current[:idx] + [new_val] + current[idx+1:]
                    new_set_sorted = sorted(new_set)
                    
                    # Quick check: verify uniqueness
                    if is_valid(new_set_sorted):
                        # Accept the improvement
                        current = new_set_sorted
                        current_sums = compute_triple_sums(current)
                        if current_score > best_local_score:
                            best_local = list(current)
                            best_local_score = current_score
                        improved = True
                        break  # Move to next element
        
        # Now try to add more elements
        for v in range(1, N + 1):
            if time.time() > limit:
                break
            if v in current:
                continue
            ok, ns = check_add(v, current, current_sums)
            if ok:
                current_sums |= ns
                current.append(v)
                current_score += 1
                if current_score > best_local_score:
                    best_local = list(current)
                    best_local_score = current_score
        
        return best_local

    if time.time() < DEADLINE - 20 and best:
        dls_result = dense_local_search(best, DEADLINE - 10)
        if is_valid(dls_result) and len(dls_result) > len(best):
            best = dls_result

    # Strategy 4: Final greedy pass with multiple orderings
    if time.time() < DEADLINE - 5 and best:
        # Try several different orderings to squeeze out more elements
        orderings = [
            list(range(1, N + 1)),  # Ascending
            list(range(N, 0, -1)),  # Descending
        ]
        
        # Add a few random orderings
        for _ in range(3):
            rand_order = list(range(1, N + 1))
            random.shuffle(rand_order)
            orderings.append(rand_order)
        
        for order in orderings:
            if time.time() > DEADLINE - 2:
                break
            
            chosen = list(best)
            chosen_set = set(chosen)
            sums = compute_triple_sums(chosen)
            
            for v in order:
                if time.time() > DEADLINE - 1:
                    break
                if v in chosen_set:
                    continue
                ok, ns = check_add(v, chosen, sums)
                if ok:
                    sums |= ns
                    chosen.append(v)
                    chosen_set.add(v)
            
            if is_valid(chosen) and len(chosen) > len(best):
                best = chosen

    # Final sanity check
    best = sorted(set(best))
    return best
