import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Precomputation for divisibility checks might not be necessary if incremental check is fast enough
    # But let's precompute multiples for small numbers to speed up "does a divide b" checks if needed.
    # Actually, just b % a == 0 is fast in Python.
    
    def get_subset_sums(elements):
        """Compute all non-empty subset sums of elements."""
        n = len(elements)
        sums = set()
        # Use DP-like approach or bitmask
        # For n <= 15, bitmask is fine.
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    def is_valid_full(elements):
        """Check if the set of elements forms a valid configuration."""
        if not elements:
            return True
        sums = get_subset_sums(elements)
        # Check distinctness is handled by set, but we need to ensure 
        # that no two *subsets* produce the same sum. 
        # The problem states: "Two different subsets with the same sum still make the set invalid".
        # So if len(sums) != (1 << len(elements)) - 1, it's invalid.
        expected_count = (1 << len(elements)) - 1
        if len(sums) != expected_count:
            return False
        
        # Check divisibility
        sums_list = sorted(sums)
        for i in range(len(sums_list)):
            a = sums_list[i]
            for j in range(i + 1, len(sums_list)):
                b = sums_list[j]
                if b % a == 0:
                    return False
        return True

    def try_add_incremental(elements, current_sums, x):
        """
        Try to add x to elements.
        Returns (new_elements, new_sums) if valid, else None.
        elements: list of current elements
        current_sums: set of current subset sums
        x: new element to add
        """
        # 1. Check x is not in current_sums
        if x in current_sums:
            return None
        
        # 2. Generate new sums: s + x for all s in current_sums
        shifted_sums = set()
        for s in current_sums:
            ns = s + x
            # Check collision with existing sums
            if ns in current_sums:
                return None
            # Check collision within shifted sums (shouldn't happen if current_sums is unique)
            if ns in shifted_sums:
                return None
            shifted_sums.add(ns)
        
        # 3. Check x against shifted_sums? 
        # x is a new sum. 
        # Is x in shifted_sums? 
        # If x = s + x for some s in current_sums, then s=0, which is not in non-empty sums.
        # So x is distinct from shifted_sums.
        
        new_sums = current_sums | shifted_sums | {x}
        
        # 4. Check Divisibility
        # We need to check pairs in new_sums.
        # We already know current_sums is internally valid.
        # We need to check:
        # a) x vs current_sums
        # b) x vs shifted_sums
        # c) current_sums vs shifted_sums
        # d) shifted_sums internally
        
        # Optimization: Only check pairs where one is from the "new" part or between new and old.
        # But shifted_sums is a transformed version of current_sums.
        
        # Let's just check all pairs involving at least one element from (shifted_sums U {x})
        # against (current_sums U shifted_sums U {x})
        
        # To be safe and simple, let's sort and check, but we can prune.
        # Actually, for small sets (k < 12), full check on new_sums is okay.
        # new_sums size is 2 * len(current_sums) + 1.
        
        sums_list = sorted(new_sums)
        n_sums = len(sums_list)
        
        for i in range(n_sums):
            a = sums_list[i]
            for j in range(i + 1, n_sums):
                b = sums_list[j]
                if b % a == 0:
                    return None
        return elements + [x], new_sums

    def greedy_construct(order, max_size=30):
        """Build a valid set by adding elements in the given order."""
        current = []
        current_sums = set()
        
        for x in order:
            if x in current:
                continue
            result = try_add_incremental(current, current_sums, x)
            if result is not None:
                current, current_sums = result
                if len(current) >= max_size:
                    break
        return current, current_sums

    def local_search(elements, current_sums, candidates, max_iterations=100, time_limit=5.0):
        """
        Local search to improve the set.
        Moves: Insert, Swap.
        """
        best_elements = list(elements)
        best_sums = set(current_sums)
        best_size = len(best_elements)
        start_ls = time.time()
        
        for _ in range(max_iterations):
            if time.time() - start_ls > time_limit:
                break
            
            improved = False
            
            # 1. Try Insert
            # Shuffle candidates to randomize
            cand_copy = candidates[:]
            random.shuffle(cand_copy)
            for x in cand_copy:
                if x in best_elements:
                    continue
                result = try_add_incremental(best_elements, best_sums, x)
                if result is not None:
                    best_elements, best_sums = result
                    best_size = len(best_elements)
                    improved = True
                    break
            
            if improved:
                # Update candidates list to exclude new best_elements
                candidates = [x for x in range(1, 257) if x not in best_elements]
                continue
            
            # 2. Try Swap
            # Remove one element, try to add a better one
            elem_copy = best_elements[:]
            random.shuffle(elem_copy)
            for old_x in elem_copy:
                # Remove old_x
                remaining = [e for e in best_elements if e != old_x]
                remaining_sums = get_subset_sums(remaining)
                
                # Try to add a candidate
                cand_copy = candidates[:]
                random.shuffle(cand_copy)
                for new_x in cand_copy:
                    if new_x in remaining:
                        continue
                    result = try_add_incremental(remaining, remaining_sums, new_x)
                    if result is not None:
                        # We swapped old_x for new_x. Size remains same.
                        # Does this help? Only if it allows future inserts.
                        # For now, accept swap only if it leads to a "better" state?
                        # Since size is same, we accept it to escape local minima.
                        # But we need to track if it leads to improvement later.
                        # Simple strategy: Accept swap with some probability or if it changes the sum structure.
                        # Let's just accept it to explore.
                        best_elements, best_sums = result
                        # Note: size didn't change, so best_size stays same
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_elements, best_sums, best_size

    # Main Search Loop
    best_set = []
    best_sums = set()
    best_size = 0
    
    # Phase 1: Greedy with different orderings
    candidates_all = list(range(1, 257))
    
    # Ordering 1: Descending
    orderings = [
        sorted(candidates_all, reverse=True),
        sorted(candidates_all),
        sorted(candidates_all, key=lambda x: (x % 2, x)), # Even then odd
        sorted(candidates_all, key=lambda x: (x % 3, x))
    ]
    
    # Random orderings
    rng = random.Random(42)
    for _ in range(50):
        shuffled = candidates_all[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() > deadline:
            break
        elements, sums = greedy_construct(order)
        if len(elements) > best_size:
            best_size = len(elements)
            best_set = elements
            best_sums = sums
            
    # Phase 2: Local Search from best found
    if time.time() < deadline:
        remaining = [x for x in candidates_all if x not in best_set]
        improved_set, improved_sums, improved_size = local_search(
            best_set, best_sums, remaining, max_iterations=50, time_limit=5.0
        )
        if improved_size > best_size:
            best_set = improved_set
            best_sums = improved_sums
            best_size = improved_size

    # Phase 3: Random Restart + Local Search
    # Try to find larger sets by starting from scratch or perturbing the best set
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Start with a random subset of the best set or a new random start
        if best_size > 0 and rng.random() < 0.5:
            # Perturb best set: remove 1-2 random elements
            current_elements = best_set[:]
            num_remove = rng.randint(1, min(2, len(current_elements)))
            for _ in range(num_remove):
                if current_elements:
                    idx = rng.randrange(len(current_elements))
                    current_elements.pop(idx)
            current_sums = get_subset_sums(current_elements)
        else:
            # Random start
            current_elements = []
            current_sums = set()
            # Greedily add a few random elements
            candidates = candidates_all[:]
            rng.shuffle(candidates)
            for x in candidates[:10]:
                res = try_add_incremental(current_elements, current_sums, x)
                if res:
                    current_elements, current_sums = res
        
        # Local search from this state
        remaining = [x for x in candidates_all if x not in current_elements]
        improved_set, improved_sums, improved_size = local_search(
            current_elements, current_sums, remaining, max_iterations=30, time_limit=3.0
        )
        
        if improved_size > best_size:
            best_size = improved_size
            best_set = improved_set
            best_sums = improved_sums

    # Final verification
    if not is_valid_full(best_set):
        # Fallback
        best_set = [31, 32, 34]
        
    return list(best_set)
