"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    # Helper to compute all sums a+b for a<=b in the set
    def get_sums(selected):
        sums = set()
        for i in range(len(selected)):
            a = selected[i]
            sums.add(2 * a)
            for j in range(i + 1, len(selected)):
                sums.add(a + selected[j])
        return sums

    # Check if a set is a valid Sidon set
    def is_sidon(selected):
        if not selected:
            return True
        sums = get_sums(selected)
        # If number of sums is less than expected (k(k+1)/2), there are duplicates
        expected = len(selected) * (len(selected) + 1) // 2
        return len(sums) == expected

    # Check if a set is inclusion-maximal
    # Returns list of missing elements that can be added
    def get_extendable(selected, sums):
        extendable = []
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if adding x creates a collision
            # New sums: 2x, x+y for y in selected
            new_sums = {2 * x}
            new_sums.update(x + y for y in selected)
            
            # If any new sum is in existing sums, collision
            if new_sums & sums:
                continue
            
            # Also need to check if new_sums have internal collisions
            # Since x is not in selected, x+y1 == x+y2 implies y1==y2 (distinct y)
            # 2x == x+y implies x==y (impossible as x not in selected)
            # So internal collisions in new_sums are impossible if selected is distinct.
            # Wait, is it possible that 2x == x+y? No, x!=y.
            # Is it possible x+y1 == x+y2? No, y1!=y2.
            # So we only need to check against existing sums.
            
            extendable.append(x)
        return extendable

    # Objective function for minimization
    # We want: 1. Valid Sidon (hard constraint), 2. Maximal (hard constraint), 3. Min Size
    # Penalty:
    # - Violations: (ExpectedSums - ActualSums) * Large_Penalty
    # - Non-Maximal: Count of extendable elements * Medium_Penalty
    # - Size: len(selected)
    
    LARGE_PENALTY = 1000
    MED_PENALTY = 100

    def objective(selected):
        if not selected:
            return float('inf')
        sums = get_sums(selected)
        expected = len(selected) * (len(selected) + 1) // 2
        violations = expected - len(sums)
        
        if violations > 0:
            # Heavily penalize invalid sets
            return violations * LARGE_PENALTY + len(selected)
            
        # Valid Sidon set
        # Check maximality
        extendable = get_extendable(selected, sums)
        non_max_count = len(extendable)
        
        if non_max_count > 0:
            # Penalize non-maximality
            return non_max_count * MED_PENALTY + len(selected)
            
        # Valid and Maximal
        return len(selected)

    # Initial State: Random valid Sidon set
    # Greedy construction is fast and provides a valid start
    def random_valid_sidon():
        selected = []
        sums = set()
        # Shuffle to get different starting points
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            added = [x + a for a in selected] + [2 * x]
            # Check internal collisions of added
            if len(set(added)) != len(added):
                continue
            # Check against existing
            if sums.intersection(added):
                continue
            selected.append(x)
            sums.update(added)
            
        return selected

    # Simulated Annealing
    best_set = None
    best_obj = float('inf')
    
    # Run multiple restarts to escape local minima
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        current = random_valid_sidon()
        # Ensure current is valid (it should be by construction)
        if not is_sidon(current):
            continue
            
        cur_sums = get_sums(current)
        cur_obj = objective(current)
        
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = list(current)
            
        # If we already have a very small set, maybe stop early?
        # Theoretical lower bound for N=256 is around sqrt(256)=16.
        # Known max size is ~16. Min maximal size is likely slightly higher.
        # If we find size <= 18, it's good.
        
        T = 50.0
        T_min = 0.1
        cooling_rate = 0.995
        
        while time.time() < deadline:
            if T < T_min:
                break
                
            # Generate neighbor
            # Move types:
            # 1. Remove an element
            # 2. Add an element (if valid)
            # 3. Swap an element
            
            move_type = random.random()
            
            new_set = list(current)
            new_sums = None # Lazy eval
            
            if move_type < 0.3 and len(current) > 1:
                # Remove
                idx = random.randint(0, len(current) - 1)
                removed_val = current[idx]
                new_set.pop(idx)
                
            elif move_type < 0.6:
                # Add
                # Find a candidate to add
                # Pick a random missing element
                # Optimization: pick from extendable if we are maximal? 
                # If we are not maximal, picking an extendable one fixes non-maximality.
                # If we are maximal, adding breaks Sidon property (unless we remove something else first, but this is single move).
                # So adding is only useful if we are NOT maximal.
                # If we ARE maximal, adding is always invalid (by definition of maximal Sidon).
                # Wait, "Maximal Sidon" means no element can be added while preserving Sidon.
                # So if current is Maximal, any add is invalid.
                # If current is NOT Maximal, there exist elements that can be added.
                
                # Let's check if current is maximal
                # This is expensive. 
                # Alternative: Just try a random add. If invalid, penalty handles it.
                
                missing = [x for x in range(1, N+1) if x not in current]
                if missing:
                    add_val = random.choice(missing)
                    new_set.append(add_val)
                    new_set.sort()
                    
            else:
                # Swap
                if len(current) > 1:
                    # Pick one to remove
                    idx_rem = random.randint(0, len(current) - 1)
                    rem_val = current[idx_rem]
                    new_set.pop(idx_rem)
                    
                    # Pick one to add
                    missing = [x for x in range(1, N+1) if x not in current]
                    if missing:
                        add_val = random.choice(missing)
                        new_set.append(add_val)
                        new_set.sort()
                    else:
                        # No missing, just removal?
                        pass

            # Evaluate new state
            # Optimization: if new_set is same length or small change, can we update sums incrementally?
            # For simplicity and correctness first, recompute.
            # N=256, set size ~16-20. Sums ~ 200. Recomputing is fast.
            
            if not new_set:
                new_obj = float('inf')
            else:
                new_obj = objective(new_set)
                
            # Acceptance
            if new_obj < cur_obj:
                current = new_set
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_set = list(current)
            else:
                delta = new_obj - cur_obj
                # Avoid log of negative or zero
                if T > 0:
                    accept_prob = math.exp(-delta / T)
                    if random.random() < accept_prob:
                        current = new_set
                        cur_obj = new_obj
                        
            T *= cooling_rate

    if best_set is None:
        # Fallback to greedy if SA failed completely
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
        
    return best_set
# EVOLVE-BLOCK-END
