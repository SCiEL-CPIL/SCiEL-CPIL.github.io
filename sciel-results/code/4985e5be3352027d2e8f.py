import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    all_points = list(range(16))
    all_triples_list = list(itertools.combinations(all_points, 3))
    num_triples = len(all_triples_list)
    
    # Precompute masks
    triple_masks = np.array([(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples_list], dtype=np.int64)
    
    # Precompute conflict matrix (boolean)
    # conflict[i, j] is True if triple i and j share > 1 point
    # We can compute this efficiently
    # For each pair, check intersection size
    # N = 560. 560^2 = 313600. Fast enough.
    
    # Alternative: Use bitwise operations on the mask array
    # But numpy doesn't have direct pairwise bitwise AND with intersection count easily without loops or broadcasting.
    # Broadcasting: 560x560x16 might be memory heavy? 560*560*8 bytes ~ 2.5MB. Fine.
    
    # Let's stick to a simple loop for conflict generation to be safe and clear, or optimize it.
    # Since N=560 is small, a Python loop is okay, but let's try to vectorize the conflict check.
    
    # Actually, let's just use the bitset approach from the seed but with numpy for the alpha calculation.
    # The bottleneck was alpha calculation.
    
    # Let's build the conflict matrix properly.
    # conflict[i] is a bitset of j's that conflict with i.
    conflict_bits = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[i]
        for j in range(i+1, num_triples):
            mj = triple_masks[j]
            inter = mi & mj
            if inter & (inter - 1):  # share > 1 point
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
                
    # Convert conflict_bits to a numpy array of int64 for faster masking?
    # Or just keep as list of ints.
    
    # Alpha calculation using NumPy vectorization
    # We want to find the largest subset S of points such that no triple is contained in S.
    # This is equivalent to finding the minimum hitting set H (size tau) such that every triple intersects H.
    # alpha = 16 - tau.
    
    # We can check if a specific set of points H is a hitting set:
    # For all active triples T, T & H != 0.
    
    # To find min tau, we can iterate k from 1 to 16 and check if a hitting set of size k exists.
    # Or we can use the "independent set" perspective:
    # Is there an independent set of size k?
    # S is independent if for all active triples T, T is NOT a subset of S.
    # i.e., T & S != T for all T.
    
    # Let's implement a fast check for "Does an independent set of size K exist?"
    # We can use a heuristic search or a smarter check.
    # Given the small N=16, we can just iterate all subsets of size K?
    # C(16, 8) = 12870. C(16, 7) = 11440.
    # For each subset, check if it's independent.
    # Checking if a subset S is independent:
    # For each active triple T, check if T & S == T.
    # This can be vectorized with numpy!
    
    # Let's precompute the active triple masks as a numpy array.
    
    def calculate_alpha_fast(active_mask):
        # active_mask is a Python int (bitset of triples)
        # We need to extract the active triples.
        # Instead of extracting, we can just use the full triple_masks array and a validity mask.
        
        # Create a boolean mask for active triples
        # active_mask is a 560-bit integer.
        # We can convert it to a numpy array of bytes or just loop?
        # Looping 560 times is fast.
        
        # Let's try to keep it simple and fast.
        # We will check for independent sets of size k from 8 down to 1.
        # If we find one, return k.
        # Wait, we want the MAX k. So start from 16 down?
        # No, alpha is usually small (5-7).
        # So check k=8, 9, 10... until we find one?
        # Or check k=1, 2... until we fail?
        # If we find an independent set of size k, we can try k+1.
        # But checking existence is the hard part.
        
        # Let's use the Hitting Set approach again but with a twist.
        # We want min tau.
        # We can use a greedy algorithm to get an upper bound on tau, then verify if tau-1 is possible?
        
        # Actually, the most robust and fast way for N=16 is to iterate all subsets of points?
        # 2^16 = 65536 subsets.
        # For each subset S, check if it's independent.
        # If S is independent, update max_size.
        # Checking if S is independent:
        # For all active triples T, T & S != T.
        
        # Let's vectorize this check.
        # active_triples_masks is a numpy array of shape (num_active,).
        # We want to check if any active triple is a subset of S.
        # (T & S) == T  <=>  (T & ~S) == 0.
        
        # Let's precompute the masks for all 2^16 subsets? No, that's memory.
        # Let's just loop over subsets? 65536 iterations is fine in Python if the inner check is fast.
        # But the inner check involves 560 triples. 65536 * 560 = 36M operations. Might be slow in Python.
        
        # Let's use the NumPy vectorization for the check.
        # For a given S (int), we want to know if any active triple T is a subset of S.
        # This is equivalent to: max((active_masks & S) == active_masks) == True?
        # No, we want to know if ALL active triples are NOT subsets.
        # So: np.all((active_masks & S) != active_masks)
        
        # Let's implement this.
        
        # Extract active masks
        # This part is slow in Python if done naively.
        # Let's precompute the list of active indices?
        
        # Alternative:
        # We can represent the system by the list of active triples.
        # Let's just pass the list of active triple masks to the alpha calculator.
        
        pass # Placeholder
    
    # Let's refine the alpha calculation strategy.
    # The seed's alpha calculation was the bottleneck.
    # The "inspiration" mentioned NumPy vectorized alpha computation.
    
    # Let's implement `calculate_alpha` using NumPy.
    
    def calculate_alpha(active_mask):
        if active_mask == 0:
            return 16
        
        # Get active triple masks
        # We can do this efficiently by precomputing a mapping or just looping.
        # Since we call this many times, let's optimize.
        # We can store the active triples in a list.
        
        # Let's use a different approach:
        # We know the system is linear.
        # We want the largest independent set.
        
        # Let's try the "Check all subsets of size k" approach with NumPy.
        # We iterate k from 8 down to 1 (since alpha is likely <= 8).
        # Actually, let's iterate k from 16 down to 1, but break early.
        # No, alpha is small. Let's start from 8 and go up?
        # If we find an independent set of size 8, we try 9.
        
        # Let's precompute the masks for all subsets of points of size k?
        # No, let's just generate them on the fly or use a smart search.
        
        # Given the time constraints, let's use the Hitting Set B&B but optimized.
        # Or, let's use the fact that N=16 is small.
        
        # Let's try this:
        # 1. Get the list of active triple masks (numpy array).
        # 2. For k from 8 down to 1:
        #    - Generate all subsets of points of size k? No, too many.
        #    - Use a backtracking search to find an independent set of size k.
        
        # Let's go back to the Hitting Set approach, but use NumPy to speed up the "remove triples" step.
        
        # Actually, the simplest speedup is to use the NumPy array of all triple masks and a boolean mask of active ones.
        
        # Let's do this:
        # active_bool = np.zeros(num_triples, dtype=bool)
        # for i in range(num_triples):
        #     if (active_mask >> i) & 1:
        #         active_bool[i] = True
        
        # This loop is 560 iterations. Very fast.
        
        active_bool = np.zeros(num_triples, dtype=bool)
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            active_bool[idx] = True
            m -= lsb
            
        active_masks = triple_masks[active_bool]
        
        # Now find min hitting set size tau.
        # We can use a greedy heuristic to get an upper bound, then B&B.
        
        # Greedy Hitting Set
        def greedy_hitting_set(triples):
            if len(triples) == 0:
                return 0
            remaining = list(triples)
            hits = 0
            while remaining:
                # Count coverage for each point
                counts = np.zeros(16, dtype=int)
                for tm in remaining:
                    # Extract points
                    # This is slow in Python loop.
                    # Let's use bit operations.
                    m = tm
                    while m:
                        lsb = m & -m
                        p = lsb.bit_length() - 1
                        counts[p] += 1
                        m -= lsb
                best_p = np.argmax(counts)
                if counts[best_p] == 0:
                    break
                hits += 1
                # Remove triples containing best_p
                mask_p = 1 << best_p
                remaining = [tm for tm in remaining if (tm & mask_p) == 0]
            return hits
        
        ub = greedy_hitting_set(active_masks)
        
        # B&B to find exact min
        # This might still be slow.
        
        # Let's try a different tactic:
        # Since we only need to MINIMIZE alpha, and we are doing local search,
        # we can use a FAST approximate alpha?
        # No, we need exact to know if we improved.
        
        # Let's optimize the B&B.
        # Key: Sort triples by "impact" (number of points they cover? No, all 3).
        # Sort by how many other triples they intersect?
        
        # Let's just use the B&B from the seed but with the NumPy active_masks.
        
        def min_hitting_set_size(triples, best_so_far):
            if not triples:
                return 0
            if best_so_far == 1:
                return 1
            
            # Pick a triple
            # Heuristic: Pick the triple that is contained in the fewest "candidate" hitting sets?
            # Or just the first one.
            t_mask = triples[0]
            rest = triples[1:]
            
            points = []
            m = t_mask
            while m:
                lsb = m & -m
                points.append(lsb.bit_length() - 1)
                m -= lsb
            
            min_val = best_so_far
            
            for p in points:
                # If we pick p, we hit t_mask and any other triple in rest containing p.
                new_triples = [tm for tm in rest if (tm & (1 << p)) == 0]
                
                # Pruning:
                # If 1 + len(new_triples) // 3 >= min_val? No.
                # Simple bound: 1 + 0 >= min_val?
                
                branch_cost = 1 + min_hitting_set_size(new_triples, min_val - 1)
                
                if branch_cost < min_val:
                    min_val = branch_cost
                    if min_val == 1:
                        break
            
            return min_val
        
        tau = min_hitting_set_size(list(active_masks), ub)
        return 16 - tau

    # This is still potentially slow.
    # Let's look at the "inspiration" again: "NumPy vectorized alpha computation enables ~10x faster..."
    # How to vectorize the alpha calculation?
    
    # Idea:
    # We want to know if there exists a subset S of size k such that no triple is in S.
    # This is equivalent to: The complement H = Points \ S is a hitting set of size 16-k.
    # So we want to know if there is a hitting set of size 16-k.
    
    # Let's check for hitting sets of size t from 1 to 8.
    # For a given t, we want to know if there is a set H of size t that hits all active triples.
    # This is a constraint satisfaction problem.
    
    # Given the difficulty of vectorizing the B&B, let's stick with the optimized B&B and hope it's fast enough for 50k iterations.
    # 50k iterations * (time for alpha calc).
    # If alpha calc takes 1ms, 50k * 1ms = 50s. Too slow.
    # If alpha calc takes 0.1ms, 5s. Good.
    
    # The B&B above is recursive and Python-based. It might be slow.
    
    # Let's try to improve the B&B with better pruning.
    # Or use a different algorithm for alpha.
    
    # Alternative: Use `itertools.combinations` to generate all subsets of size k and check if any is independent.
    # For k=8, C(16,8) = 12870.
    # For each subset, check if it's independent using NumPy.
    # Check: np.all((active_masks & subset_mask) != active_masks)
    # This is a vectorized operation on 560 elements. Very fast!
    
    # So the strategy is:
    # 1. Get active_masks (numpy array).
    # 2. For k from 8 down to 1 (or 16 down to 1, but start high to find max alpha? No, we want min alpha, so we want to know the MAX independent set size).
    #    Wait, we want to MINIMIZE alpha. So we want to find the LARGEST independent set.
    #    So we check k=16, 15, ... until we find an independent set.
    #    But if alpha is small (e.g., 5), we check 16, 15, ..., 6 (fail), 5 (success).
    #    This is 11 checks.
    #    Each check involves generating C(16, k) subsets and checking them.
    #    C(16, 8) is the largest.
    
    # Let's implement this.
    
    def calculate_alpha_v2(active_mask):
        if active_mask == 0:
            return 16
        
        active_bool = np.zeros(num_triples, dtype=bool)
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            active_bool[idx] = True
            m -= lsb
            
        active_masks = triple_masks[active_bool]
        
        # Check for independent sets of size k
        # We start from k=16 down to 1
        for k in range(16, 0, -1):
            # Generate all subsets of size k
            # We can use itertools.combinations
            found = False
            for subset in itertools.combinations(range(16), k):
                # Create mask for subset
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                
                # Check if any active triple is contained in s_mask
                # (active_masks & s_mask) == active_masks
                # If any is True, then subset is NOT independent.
                # We want to know if ALL are False.
                
                # Vectorized check
                # np.all((active_masks & s_mask) != active_masks)
                if np.all((active_masks & s_mask) != active_masks):
                    found = True
                    break
            
            if found:
                return k
        
        return 0 # Should not happen

    # This is O(sum(C(16, k)) * N_triples).
    # Sum(C(16, k)) = 2^16 = 65536.
    # 65536 * 560 = 36M operations.
    # With NumPy, the inner check is fast, but the loop over subsets is Python.
    # 65536 iterations in Python might take ~1s?
    # If so, 50k iterations would take 50,000s. Too slow.
    
    # We need a faster alpha check.
    
    # Let's go back to the Hitting Set B&B.
    # The B&B is much faster on average if the hitting set is small.
    # For good systems, tau is around 9-11.
    # The B&B depth is tau.
    # The branching factor is 3.
    # 3^11 = 177k.
    # With pruning, it's much less.
    
    # Let's optimize the B&B implementation.
    # Use lists of masks, not numpy arrays, for the recursion.
    # Use a greedy upper bound.
    
    # Let's try to combine:
    # 1. Greedy Hitting Set to get ub.
    # 2. B&B to find exact min.
    
    # Let's implement this carefully.
    
    def calculate_alpha_final(active_mask):
        if active_mask == 0:
            return 16
        
        active_bool = np.zeros(num_triples, dtype=bool)
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            active_bool[idx] = True
            m -= lsb
            
        active_masks = list(triple_masks[active_bool])
        
        if not active_masks:
            return 16
            
        # Greedy Hitting Set
        def greedy_hitting_set(triples):
            if not triples:
                return 0
            remaining = list(triples)
            hits = 0
            while remaining:
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
                mask_p = 1 << best_p
                remaining = [tm for tm in remaining if (tm & mask_p) == 0]
            return hits
        
        ub = greedy_hitting_set(active_masks)
        
        # B&B
        def min_hitting_set_size(triples, best_so_far):
            if not triples:
                return 0
            if best_so_far == 1:
                return 1
            
            # Heuristic: Sort triples by "constraint" (e.g., number of points in common with others?)
            # For now, just pick the first one.
            t_mask = triples[0]
            rest = triples[1:]
            
            points = []
            m = t_mask
            while m:
                lsb = m & -m
                points.append(lsb.bit_length() - 1)
                m -= lsb
            
            min_val = best_so_far
            
            for p in points:
                mask_p = 1 << p
                new_triples = [tm for tm in rest if (tm & mask_p) == 0]
                
                # Pruning
                # If 1 + len(new_triples) > min_val? No.
                # Simple bound: 1 + 0 >= min_val?
                
                branch_cost = 1 + min_hitting_set_size(new_triples, min_val - 1)
                
                if branch_cost < min_val:
                    min_val = branch_cost
                    if min_val == 1:
                        break
            
            return min_val
        
        tau = min_hitting_set_size(active_masks, ub)
        return 16 - tau

    # This should be faster.
    
    # Now, the local search.
    # We want to minimize alpha.
    # Move: Swap a triple in the system with a triple not in the system.
    # Or: Add a triple, remove conflicting ones.
    
    # Let's use a simple local search:
    # 1. Find the largest independent set (witness).
    # 2. Try to add a triple that hits this witness set.
    # 3. If it conflicts, remove the conflicting triples.
    # 4. Check if alpha decreased.
    
    def find_witness_independent_set(active_mask, target_size):
        # We want to find an independent set of size target_size.
        # We can use the same "check all subsets" approach, but stop at the first one found.
        # This is fast if target_size is large (few subsets to check? No, C(16, 8) is large).
        # But we only need ONE witness.
        
        active_bool = np.zeros(num_triples, dtype=bool)
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            active_bool[idx] = True
            m -= lsb
            
        active_masks = triple_masks[active_bool]
        
        # Shuffle the points to randomize the search?
        # Or just iterate combinations.
        # To speed up, we can randomize the order of points in the combinations?
        # itertools.combinations doesn't support random order easily.
        # We can generate random subsets of size target_size and check them.
        
        for _ in range(1000): # Try 1000 random subsets
            subset = random.sample(range(16), target_size)
            s_mask = 0
            for p in subset:
                s_mask |= (1 << p)
            
            if np.all((active_masks & s_mask) != active_masks):
                return s_mask
        
        # If not found, fall back to exhaustive?
        # Or return 0.
        return 0

    def local_search(initial_mask, time_limit):
        current_mask = initial_mask
        current_alpha = calculate_alpha_final(current_mask)
        best_mask = current_mask
        best_alpha = current_alpha
        
        T = 5.0
        cooling_rate = 0.995
        
        while time.time() < time_limit:
            if current_alpha <= 5:
                break
            
            witness_mask = find_witness_independent_set(current_mask, current_alpha)
            
            if witness_mask == 0:
                # No witness found? Maybe alpha is smaller?
                # Recalculate?
                continue
            
            # Find candidates to add
            # We want to add a triple that hits the witness set.
            candidate_adds = []
            for i in range(num_triples):
                if (current_mask >> i) & 1:
                    continue
                tm = triple_masks[i]
                if tm & witness_mask:
                    candidate_adds.append(i)
            
            if not candidate_adds:
                # No candidate hits the witness set.
                # This means the witness set is "stable" w.r.t additions?
                # We need a different move.
                # Random swap?
                active_indices = [i for i in range(num_triples) if (current_mask >> i) & 1]
                if not active_indices:
                    break
                remove_idx = random.choice(active_indices)
                temp_mask = current_mask & ~(1 << remove_idx)
                
                add_candidates = []
                for i in range(num_triples):
                    if (temp_mask >> i) & 1:
                        continue
                    if (conflict_bits[i] & temp_mask) == 0:
                        add_candidates.append(i)
                
                if not add_candidates:
                    continue
                add_idx = random.choice(add_candidates)
                new_mask = temp_mask | (1 << add_idx)
            else:
                add_idx = random.choice(candidate_adds)
                conflicts = conflict_bits[add_idx] & current_mask
                if conflicts == 0:
                    new_mask = current_mask | (1 << add_idx)
                else:
                    new_mask = (current_mask & ~conflicts) | (1 << add_idx)
            
            new_alpha = calculate_alpha_final(new_mask)
            
            if new_alpha < current_alpha:
                current_mask = new_mask
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_mask = current_mask
            else:
                delta = new_alpha - current_alpha
                if delta == 0:
                    if random.random() < 0.5:
                        current_mask = new_mask
                else:
                    if T > 0.01:
                        prob = pow(2.718, -delta / T)
                        if random.random() < prob:
                            current_mask = new_mask
                            current_alpha = new_alpha
            
            T *= cooling_rate
            if T < 0.01:
                T = 1.0
        
        return best_mask, best_alpha

    def generate_initial():
        active_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
            if (conflict_bits[idx] & active_mask) == 0:
                active_mask |= (1 << idx)
        return active_mask

    best_alpha = 16
    best_mask = 0
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        current_mask = generate_initial()
        sa_limit = deadline - 25
        best_mask, best_alpha = local_search(current_mask, sa_limit)
        
        if best_alpha <= 6:
            sa_limit = deadline - 10
            best_mask, best_alpha = local_search(best_mask, sa_limit)

    result = []
    for i in range(num_triples):
        if (best_mask >> i) & 1:
            result.append(list(all_triples_list[i]))
    
    return result
