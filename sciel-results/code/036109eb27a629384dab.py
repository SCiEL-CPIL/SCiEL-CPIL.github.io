"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    def get_sums(selected):
        sums = set()
        for i in range(len(selected)):
            a = selected[i]
            sums.add(2 * a)
            for j in range(i + 1, len(selected)):
                sums.add(a + selected[j])
        return sums

    def is_sidon(selected):
        if not selected:
            return True
        sums = get_sums(selected)
        expected = len(selected) * (len(selected) + 1) // 2
        return len(sums) == expected

    def get_extendable(selected, sums):
        """Returns list of elements in [1, N] not in selected that can be added without violating Sidon."""
        extendable = []
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if x can be added
            # New sums: 2x, x+s for s in selected
            # Collision if 2x in sums OR (x+s) in sums for any s
            if 2 * x in sums:
                continue
            valid = True
            for s in selected:
                if (x + s) in sums:
                    valid = False
                    break
            if valid:
                extendable.append(x)
        return extendable

    def is_maximal(selected):
        if not is_sidon(selected):
            return False
        sums = get_sums(selected)
        extendable = get_extendable(selected, sums)
        return len(extendable) == 0

    def objective(selected):
        """
        Objective for local search.
        We want:
        1. Valid Sidon set (penalty if not)
        2. Maximal (penalty for each element that could be added)
        3. Small size (fixed in this approach, but penalize if we allow size changes? 
           Actually, we fix size K. So we just minimize 'non-maximality' and 'invalidity'.)
        
        Returns a cost value. Lower is better.
        """
        if not is_sidon(selected):
            # Heavy penalty for invalid Sidon sets
            sums = get_sums(selected)
            invalid_count = len(selected) * (len(selected) + 1) // 2 - len(sums)
            return 1000 + invalid_count
        
        sums = get_sums(selected)
        extendable = get_extendable(selected, sums)
        # Cost is number of elements that can still be added.
        # If 0, it's a valid maximal Sidon set.
        return len(extendable)

    def local_search(k_target, time_limit):
        """
        Run simulated annealing to find a maximal Sidon set of size k_target.
        """
        # Initialize with random set of size k_target
        # Ensure it's Sidon. If not, repair or restart.
        # Simple init: random sample
        current_set = set(random.sample(range(1, N + 1), k_target))
        
        # Check if initial is Sidon
        while not is_sidon(list(current_set)):
            # If not Sidon, try to fix by removing conflicting elements? 
            # Easier: just restart with a new random sample until we get a Sidon set.
            # Probability of random set being Sidon is low for large K.
            # For K=12, N=256, it's possible but rare.
            # Better: Construct a valid Sidon set first, then optimize.
            # Let's use a greedy valid Sidon construction of size K to start.
            break
        
        # Greedy construction of a valid Sidon set of size K
        S = []
        S_set = set()
        Sums = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= k_target:
                break
            new_sums = {2 * x}
            new_sums.update(x + s for s in S)
            if not (new_sums & Sums):
                S.append(x)
                S_set.add(x)
                Sums.update(new_sums)
        
        if len(S) < k_target:
            # Could not find K valid elements? Unlikely for K < sqrt(N)
            # But if so, return what we have (might be maximal or not)
            if is_maximal(S):
                return S
            # Try to continue search with smaller set?
            # For now, just return S if it's maximal, else None
            return S if is_maximal(S) else None

        current_set = set(S)
        current_cost = objective(list(current_set))
        
        if current_cost == 0:
            return list(current_set)

        best_set = list(current_set)
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.1
        alpha = 0.995
        steps = 0
        
        while time.time() < time_limit:
            # Generate neighbor: pick one element to swap or remove/add?
            # Since we fix size K, we do swaps.
            # Pick random element in set, random element outside set.
            if random.random() < 0.1:
                # Occasionally try to change size? 
                # The problem asks to minimize cardinality. 
                # If we find a maximal set of size K, we can try to shrink it.
                # But shrinking a maximal set is hard.
                # Let's stick to fixed K for the SA phase, then try to shrink in a post-processing step.
                pass
            
            elem_in = random.choice(list(current_set))
            # Pick an element from [1, N] not in current_set
            # Efficient pick:
            while True:
                elem_out = random.randint(1, N)
                if elem_out not in current_set:
                    break
            
            new_set = set(current_set)
            new_set.remove(elem_in)
            new_set.add(elem_out)
            
            new_cost = objective(list(new_set))
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_set = new_set
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_set = list(current_set)
                    
                    if best_cost == 0:
                        # Found a maximal Sidon set of size K
                        return best_set
            
            T *= alpha
            if T < T_min:
                T = T_min
                # Reheat occasionally?
                if random.random() < 0.05:
                    T = 5.0
            
            steps += 1
            if steps % 1000 == 0 and time.time() > time_limit - 5:
                break

        return best_set

    def try_shrink(S):
        """
        Given a maximal Sidon set S, try to remove elements to see if it remains maximal.
        If removing x makes it non-maximal (i.e., some y becomes addable), then x is 'essential'.
        We want to find a subset that is still maximal.
        This is like finding a minimal vertex cover in the blocking graph.
        """
        # Build blocking graph?
        # For each x in S, check if S \ {x} is maximal.
        # If yes, x is not essential.
        # We can iteratively remove non-essential elements.
        
        S_list = list(S)
        changed = True
        while changed:
            changed = False
            # Random order to avoid bias
            random.shuffle(S_list)
            for x in S_list:
                if len(S_list) <= 1:
                    break
                trial_set = [y for y in S_list if y != x]
                if is_maximal(trial_set):
                    # x can be removed
                    S_list = trial_set
                    changed = True
                    break # Restart loop with new list
        return S_list

    best_global_set = None
    best_global_size = N + 1

    # Strategy:
    # 1. Try fixed sizes K from small to large.
    # 2. For each K, run local search to find a maximal Sidon set of size K.
    # 3. If found, try to shrink it.
    # 4. Keep the smallest maximal set found.
    
    # Lower bound for maximal Sidon set size?
    # A Sidon set of size k has ~k^2/2 sums.
    # Maximal means it blocks N-k elements.
    # Each element in S blocks some elements.
    # Empirically, for N=256, minimal maximal sets are around 12-16.
    
    # We will try K = 10, 11, 12, 13, 14, 15, 16
    # Allocate time budget to each K.
    
    k_values = [10, 11, 12, 13, 14, 15, 16]
    
    # If we find a set of size 12, we don't need to try 15.
    # So we iterate from small K upwards.
    
    for K in k_values:
        if time.time() > deadline - 5:
            break
            
        # Allocate time for this K
        # Give more time to likely sizes (12-14)
        if K in [12, 13, 14]:
            time_alloc = 10.0
        else:
            time_alloc = 5.0
            
        end_time = time.time() + time_alloc
        
        # Run multiple restarts for this K
        found_maximal = False
        while time.time() < end_time:
            S = local_search(K, end_time)
            if S is not None and is_maximal(S):
                # Try to shrink
                S_shrunk = try_shrink(S)
                if is_maximal(S_shrunk):
                    if len(S_shrunk) < best_global_size:
                        best_global_size = len(S_shrunk)
                        best_global_set = S_shrunk
                        found_maximal = True
                        # If we found a very small set, we can break early?
                        # If best_global_size <= 10, it's very good.
                        if best_global_size <= 10:
                            break
                # Even if not shrunk, S is a candidate
                if len(S) < best_global_size:
                    best_global_size = len(S)
                    best_global_set = S
            
        if found_maximal and best_global_size <= K:
            # If we found a maximal set of size <= K, and we are iterating K upwards,
            # we might have found a smaller one via shrinking.
            # If best_global_size is small, we can stop trying larger K?
            # Actually, if we found a maximal set of size 12, we don't need to try K=15 
            # to find a size 12 set. But we might find a size 11 set by trying K=11.
            # Since we iterate K=10, 11, 12..., if we find a size 12 set at K=12,
            # we should still try K=11? No, K is the target size for construction.
            # If we found size 12, it's possible size 11 exists. 
            # But local search for K=11 is harder (more constrained).
            # Let's just run through all K values if time permits.
            pass

    if best_global_set is None:
        # Fallback: Greedy maximal
        S = []
        S_set = set()
        Sums = set()
        alive = set(range(1, N + 1))
        while alive:
            # Pick smallest available
            x = min(alive)
            new_sums_x = {2 * x}
            new_sums_x.update(x + s for s in S)
            if new_sums_x & Sums:
                alive.remove(x)
                continue
            
            S.append(x)
            S_set.add(x)
            Sums.update(new_sums_x)
            alive.remove(x)
            
            # Remove blocked
            to_remove = []
            for y in alive:
                if 2 * y in Sums:
                    to_remove.append(y)
                    continue
                blocked = False
                for s in S:
                    if (y + s) in Sums:
                        blocked = True
                        break
                if blocked:
                    to_remove.append(y)
            for y in to_remove:
                alive.remove(y)
        
        best_global_set = S

    return best_global_set

# EVOLVE-BLOCK-END
