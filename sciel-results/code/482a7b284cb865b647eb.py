import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    
    # Precompute: for each possible sum value (1..30*256=7680), 
    # which larger values it divides
    MAX_SUM = 30 * 256  # 7680
    
    def get_divisibility_conflicts(sums_set):
        """Return count of divisibility conflicts among sums_set."""
        if len(sums_set) < 2:
            return 0
        sorted_sums = sorted(sums_set)
        max_s = sorted_sums[-1]
        conflicts = 0
        sums_set_copy = set(sums_set)
        for u in sorted_sums:
            if u * 2 > max_s:
                break
            for v in range(u * 2, max_s + 1, u):
                if v in sums_set_copy:
                    conflicts += 1
        return conflicts
    
    def compute_subset_sums(elements):
        """Compute all nonempty subset sums as a set."""
        if not elements:
            return set()
        sums = set()
        for e in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + e)
            new_sums.add(e)
            sums = sums | new_sums
        return sums
    
    def is_valid_set(elements):
        """Check if set of elements satisfies all constraints."""
        if len(elements) != len(set(elements)):
            return False
        if any(e < 1 or e > 256 for e in elements):
            return False
        sums = compute_subset_sums(elements)
        # Check for duplicate subset sums (which would cause divisibility issues)
        # Actually, duplicate subset sums are the real problem
        # Let's check: if two different subsets have same sum, that's invalid
        # But we can detect this by checking if the number of unique sums 
        # equals 2^n - 1 where n = len(elements)
        expected_unique = (1 << len(elements)) - 1
        if len(sums) != expected_unique:
            return False
        # Check divisibility
        return get_divisibility_conflicts(sums) == 0
    
    # Simulated annealing approach
    # State: a set of elements (as a frozenset or sorted tuple)
    # Move: replace one element with another, or add/remove an element
    
    best_set = []
    best_size = 0
    
    # Start with a good initial set using greedy
    def greedy_initial():
        """Build an initial set using greedy with randomization."""
        current = []
        current_sums = set()
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for c in candidates:
            if len(current) >= 20:
                break
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + c)
            new_sums.add(c)
            
            # Check for duplicate sums
            if len(new_sums) != 2 * len(current_sums) + 1:
                continue
            
            # Check divisibility
            if get_divisibility_conflicts(new_sums) == 0:
                current.append(c)
                current_sums = new_sums
        
        return current, current_sums
    
    # Run greedy to get initial solution
    init_set, init_sums = greedy_initial()
    best_set = init_set[:]
    best_size = len(init_set)
    
    if best_size >= 30:
        return [int(x) for x in best_set]
    
    # Simulated annealing
    current_set = init_set[:]
    current_sums = init_sums
    current_conflicts = 0
    
    # Temperature schedule
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    while time.time() < deadline - 2.0 and T > min_T:
        # Generate a move
        move_type = random.random()
        
        if len(current_set) < 30 and move_type < 0.3:
            # Add a random element
            candidates = [x for x in range(1, 257) if x not in current_set]
            if not candidates:
                T *= cooling
                continue
            new_elem = random.choice(candidates)
            new_set = current_set + [new_elem]
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + new_elem)
            new_sums.add(new_elem)
            
            # Check for duplicates
            if len(new_sums) != 2 * len(current_sums) + 1:
                T *= cooling
                continue
            
            new_conflicts = get_divisibility_conflicts(new_sums)
            
            if new_conflicts < current_conflicts:
                current_set = new_set
                current_sums = new_sums
                current_conflicts = new_conflicts
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    if best_size >= 30:
                        return [int(x) for x in best_set]
            else:
                # Accept with probability exp(-delta/T)
                delta = new_conflicts - current_conflicts
                if random.random() < (2.718281828 ** (-delta / T)):
                    current_set = new_set
                    current_sums = new_sums
                    current_conflicts = new_conflicts
        
        elif len(current_set) > 5 and move_type < 0.6:
            # Remove a random element
            idx = random.randint(0, len(current_set) - 1)
            removed = current_set[idx]
            new_set = current_set[:idx] + current_set[idx+1:]
            new_sums = compute_subset_sums(new_set)
            new_conflicts = get_divisibility_conflicts(new_sums)
            
            if new_conflicts < current_conflicts:
                current_set = new_set
                current_sums = new_sums
                current_conflicts = new_conflicts
            else:
                delta = new_conflicts - current_conflicts
                if random.random() < (2.718281828 ** (-delta / T)):
                    current_set = new_set
                    current_sums = new_sums
                    current_conflicts = new_conflicts
        
        else:
            # Swap: replace one element with another
            if not current_set:
                T *= cooling
                continue
            idx = random.randint(0, len(current_set) - 1)
            old_elem = current_set[idx]
            candidates = [x for x in range(1, 257) if x != old_elem and x not in current_set]
            if not candidates:
                T *= cooling
                continue
            new_elem = random.choice(candidates)
            
            new_set = current_set[:]
            new_set[idx] = new_elem
            new_sums = compute_subset_sums(new_set)
            new_conflicts = get_divisibility_conflicts(new_sums)
            
            if new_conflicts < current_conflicts:
                current_set = new_set
                current_sums = new_sums
                current_conflicts = new_conflicts
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    if best_size >= 30:
                        return [int(x) for x in best_set]
            else:
                delta = new_conflicts - current_conflicts
                if random.random() < (2.718281828 ** (-delta / T)):
                    current_set = new_set
                    current_sums = new_sums
                    current_conflicts = new_conflicts
        
        T *= cooling
    
    # Final check and return
    # Verify the best set is valid
    if best_set:
        if is_valid_set(best_set):
            return [int(x) for x in best_set]
    
    # If no valid set found, return empty
    return [int(x) for x in best_set]
